use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::types::{BatchKind, TestId, DEFAULT_PREDICTED_DURATION};

/// Message sent from workers to the scheduler
pub enum SchedulerMessage {
    /// Worker requests a batch to execute.
    /// The scheduler decides what batch to give based on GPU availability.
    GetBatch {
        response_tx: crossbeam_channel::Sender<Option<BatchAssignment>>,
    },
    /// Worker completed a batch
    CompleteBatch { batch_id: usize },
    /// Worker wants to retry/repool tests (e.g., after crash)
    AddTests { tests: Vec<String> },
    /// Query: get atomic status snapshot (for progress display and termination check)
    GetStatus {
        num_workers: usize,
        response_tx: crossbeam_channel::Sender<SchedulerStatus>,
    },
    /// Shutdown the scheduler
    Shutdown,
}

/// Snapshot of scheduler state for progress display.
/// Returned atomically to avoid TOCTOU issues.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStatus {
    /// Is the scheduler completely empty?
    pub is_empty: bool,
    /// Number of tests remaining (in pool + in-flight)
    pub remaining: usize,
    /// Calculated ETA in seconds
    pub eta: f64,
    /// Are there batches waiting (not in-flight)?
    pub has_pending_batches: bool,
    /// Debug info: (batches_count, pending_count, in_flight_ids)
    pub debug_state: (usize, usize, Vec<usize>),
}

/// A batch assignment returned to a worker
#[derive(Debug, Clone)]
pub struct BatchAssignment {
    pub batch_id: usize,
    pub tests: Vec<String>,
    pub kind: BatchKind,
}

/// Tracks an in-flight batch
struct InFlightBatch {
    start_time: Instant,
    predicted_duration: f64,
    test_count: usize,
    kind: BatchKind,
}

/// A batch with its kind classification
struct BatchWithKind {
    tests: Vec<String>,
    kind: BatchKind,
}

/// The scheduler owns all scheduling state and runs in a dedicated thread.
/// No mutexes needed - all state is owned by the single scheduler thread.
pub struct Scheduler {
    /// Channel to receive messages from workers
    rx: crossbeam_channel::Receiver<SchedulerMessage>,

    /// Pre-built batches of tests, ready to be handed to workers
    batches: Vec<BatchWithKind>,
    /// Pending files that need to be rebuilt into batches
    pending_files: Vec<String>,
    /// In-flight batches keyed by batch ID
    in_flight: HashMap<usize, InFlightBatch>,
    /// Counter for generating batch IDs
    next_batch_id: usize,
    /// Predicted time for tests still in the pool
    pool_predicted: f64,

    /// Configuration
    max_batch_size: usize,
    target_batch_duration: f64,
    predictions: HashMap<String, f64>,
    has_timing_data: bool,
    gpu_jobs: Option<usize>,
    gpu_in_flight: usize,
    /// Number of workers (for parallelism optimization)
    num_workers: usize,
}

impl Scheduler {
    /// Create a new scheduler and return (scheduler, handle).
    /// The scheduler should be run in a dedicated thread via `scheduler.run()`.
    pub fn new(
        files: Vec<String>,
        max_batch_size: usize,
        num_workers: usize,
        predictions: HashMap<String, f64>,
        has_timing_data: bool,
        target_batch_duration: f64,
        gpu_jobs: Option<usize>,
        gpu_stagger_ms: u64,
    ) -> (Self, SchedulerHandle) {
        let (tx, rx) = crossbeam_channel::unbounded();

        let total_predicted: f64 = files.iter()
            .map(|f| predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
            .sum();

        let mut batches = Self::build_batches(
            files,
            &predictions,
            has_timing_data,
            max_batch_size,
            target_batch_duration,
            gpu_jobs,
        );

        // Stagger GPU test starts in the first N batches to avoid Vulkan context contention
        if gpu_stagger_ms > 0 {
            Self::stagger_gpu_starts(&mut batches, num_workers, &predictions, gpu_stagger_ms);
        }

        let predictions_arc = Arc::new(predictions.clone());

        let scheduler = Self {
            rx,
            batches,
            pending_files: Vec::new(),
            in_flight: HashMap::new(),
            next_batch_id: 0,
            pool_predicted: total_predicted,
            max_batch_size,
            target_batch_duration,
            predictions,
            has_timing_data,
            gpu_jobs,
            gpu_in_flight: 0,
            num_workers,
        };

        let handle = SchedulerHandle {
            tx,
            predictions: predictions_arc,
            has_timing_data,
            num_workers,
        };

        (scheduler, handle)
    }

    /// Run the scheduler message loop. Call this in a dedicated thread.
    /// Returns when Shutdown message is received or all senders are dropped.
    pub fn run(&mut self) {
        while let Ok(msg) = self.rx.recv() {
            match msg {
                SchedulerMessage::GetBatch { response_tx } => {
                    let batch = self.try_get_batch();
                    let _ = response_tx.send(batch);
                }
                SchedulerMessage::CompleteBatch { batch_id } => {
                    self.complete_batch(batch_id);
                }
                SchedulerMessage::AddTests { tests } => {
                    self.add_tests(tests);
                }
                SchedulerMessage::GetStatus { num_workers, response_tx } => {
                    let status = self.get_status(num_workers);
                    let _ = response_tx.send(status);
                }
                SchedulerMessage::Shutdown => {
                    break;
                }
            }
        }
    }

    /// Get atomic status snapshot
    fn get_status(&self, num_workers: usize) -> SchedulerStatus {
        let pool_count: usize = self.batches.iter()
            .map(|b| b.tests.len()).sum::<usize>()
            + self.pending_files.len();
        let in_flight_count: usize = self.in_flight.values()
            .map(|b| b.test_count).sum();

        SchedulerStatus {
            is_empty: self.pending_files.is_empty()
                && self.batches.is_empty()
                && self.in_flight.is_empty(),
            remaining: pool_count + in_flight_count,
            eta: self.calculate_eta(num_workers),
            has_pending_batches: !self.pending_files.is_empty() || !self.batches.is_empty(),
            debug_state: (
                self.batches.len(),
                self.pending_files.len(),
                self.in_flight.keys().copied().collect(),
            ),
        }
    }

    /// Build batches from a list of files.
    fn build_batches(
        files: Vec<String>,
        predictions: &HashMap<String, f64>,
        has_timing_data: bool,
        max_batch_size: usize,
        target_batch_duration: f64,
        gpu_jobs: Option<usize>,
    ) -> Vec<BatchWithKind> {
        if files.is_empty() {
            return Vec::new();
        }

        // If gpu_jobs is set, separate files into GPU and CPU lists first
        let (gpu_files, cpu_files): (Vec<String>, Vec<String>) = if gpu_jobs.is_some() {
            files.into_iter().partition(|f| TestId::parse(f).is_gpu_test())
        } else {
            (Vec::new(), files)
        };

        let mut batches = Vec::new();

        // Helper to build batches from a list of files with a given kind.
        // If sort_slowest_first is true, sorts tests by duration before batching
        // (useful for CPU-only batches where we want slow tests dispatched first).
        // If false, preserves input order (important for GPU/mixed batches to maintain
        // the constrained random shuffle that prevents GPU contention).
        let build_from_files = |files: Vec<String>, kind: BatchKind, sort_slowest_first: bool, batches: &mut Vec<BatchWithKind>| {
            if files.is_empty() {
                return;
            }

            let mut files = files;

            if has_timing_data {
                if sort_slowest_first {
                    // Sort by predicted duration, slowest first
                    files.sort_by(|a, b| {
                        let dur_a = predictions.get(a).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                        let dur_b = predictions.get(b).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                        dur_b.partial_cmp(&dur_a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                // Build batches using duration-based grouping
                let mut current_batch = Vec::new();
                let mut current_duration = 0.0;

                for file in files {
                    let duration = predictions.get(&file).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);

                    // If this test exceeds target duration, it gets its own batch
                    if duration > target_batch_duration {
                        // Flush current batch first
                        if !current_batch.is_empty() {
                            batches.push(BatchWithKind {
                                tests: std::mem::take(&mut current_batch),
                                kind,
                            });
                            current_duration = 0.0;
                        }
                        batches.push(BatchWithKind {
                            tests: vec![file],
                            kind,
                        });
                        continue;
                    }

                    // If adding this test would exceed target duration or max size, start new batch
                    if !current_batch.is_empty()
                        && (current_duration + duration > target_batch_duration
                            || current_batch.len() >= max_batch_size)
                    {
                        batches.push(BatchWithKind {
                            tests: std::mem::take(&mut current_batch),
                            kind,
                        });
                        current_duration = 0.0;
                    }

                    current_batch.push(file);
                    current_duration += duration;
                }

                // Don't forget the last batch
                if !current_batch.is_empty() {
                    batches.push(BatchWithKind {
                        tests: current_batch,
                        kind,
                    });
                }
            } else {
                // No timing data - just chunk into fixed-size batches
                for chunk in files.chunks(max_batch_size) {
                    batches.push(BatchWithKind {
                        tests: chunk.to_vec(),
                        kind,
                    });
                }
            }
        };

        if gpu_jobs.is_some() {
            // Segmented mode: separate GPU and CPU batches
            // GPU batches: preserve random order (constrained shuffle handles long-pole)
            build_from_files(gpu_files, BatchKind::Gpu, false, &mut batches);

            // CPU batches: sort slowest-first, then reverse so slow batches are at the
            // end of the vector and get .pop()'d first
            let cpu_start_idx = batches.len();
            build_from_files(cpu_files, BatchKind::Cpu, true, &mut batches);
            batches[cpu_start_idx..].reverse();
        } else {
            // Mixed mode: preserve the constrained random order from the shuffle.
            // The shuffle already prevents slow tests from clustering at the end
            // while keeping GPU tests randomly distributed to avoid contention.
            build_from_files(cpu_files, BatchKind::Mixed, false, &mut batches);
        }

        batches
    }

    /// Stagger GPU test starts in the first N batches (N = num_workers).
    ///
    /// When all workers start simultaneously, they'd all hit GPU tests at once,
    /// causing Vulkan context creation contention. This reorders tests within
    /// the first N batches so each has an increasing CPU "prefix":
    /// - Batch 0: ~stagger_ms of CPU tests first
    /// - Batch 1: ~2*stagger_ms of CPU tests first
    /// - Batch N-1: ~N*stagger_ms of CPU tests first
    ///
    /// This naturally staggers GPU test starts over ~N*stagger_ms instead of all at once.
    fn stagger_gpu_starts(
        batches: &mut Vec<BatchWithKind>,
        num_workers: usize,
        predictions: &HashMap<String, f64>,
        stagger_increment_ms: u64,
    ) {
        for (batch_idx, batch) in batches.iter_mut().take(num_workers).enumerate() {
            // Target CPU prefix duration increases with batch index
            let target_cpu_prefix_secs = ((batch_idx + 1) * stagger_increment_ms as usize) as f64 / 1000.0;

            // Partition tests into CPU and GPU
            let (cpu_tests, gpu_tests): (Vec<_>, Vec<_>) = batch.tests
                .drain(..)
                .partition(|t| !TestId::parse(t).is_gpu_test());

            // Rebuild batch: CPU tests first (up to target), then GPU, then remaining CPU
            let mut new_order = Vec::with_capacity(cpu_tests.len() + gpu_tests.len());
            let mut cpu_prefix_duration = 0.0;
            let mut remaining_cpu = Vec::new();

            for test in cpu_tests {
                let dur = predictions.get(&test).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                if cpu_prefix_duration < target_cpu_prefix_secs {
                    new_order.push(test);
                    cpu_prefix_duration += dur;
                } else {
                    remaining_cpu.push(test);
                }
            }

            // Add GPU tests after the CPU prefix
            new_order.extend(gpu_tests);
            // Add remaining CPU tests at the end
            new_order.extend(remaining_cpu);

            batch.tests = new_order;
        }
    }

    /// Add tests back to the pool (for retries or repooled tests after crash).
    fn add_tests(&mut self, tests: Vec<String>) {
        for file in tests {
            let predicted = self.predictions.get(&file).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
            self.pool_predicted += predicted;
            self.pending_files.push(file);
        }
    }

    /// Ensure we have enough batches to keep all workers busy.
    /// Splits large batches when we have fewer batches than potentially idle workers.
    /// This prevents tail latency where one worker runs many tests while others sit idle.
    fn ensure_parallelism(&mut self) {
        // How many workers could potentially be looking for work?
        let idle_workers = self.num_workers.saturating_sub(self.in_flight.len());

        // If we have fewer batches than idle workers, split the largest batches
        while self.batches.len() < idle_workers {
            if !self.split_largest_batch() {
                break; // No more splittable batches
            }
        }
    }

    /// Split the largest batch into two roughly equal parts (by predicted duration).
    /// Returns true if a split occurred, false if no splittable batch exists.
    fn split_largest_batch(&mut self) -> bool {
        // Find the largest batch (by test count) that can be split
        // Batches are sorted smallest-first, so largest splittable is near the end
        let split_idx = self.batches.iter()
            .enumerate()
            .rev()
            .find(|(_, b)| b.tests.len() > 1)
            .map(|(i, _)| i);

        let Some(idx) = split_idx else {
            return false;
        };

        let batch = self.batches.remove(idx);
        let kind = batch.kind;

        // Calculate total predicted time for this batch
        let total_predicted: f64 = batch.tests.iter()
            .map(|t| self.predictions.get(t).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
            .sum();

        // Split by predicted duration to balance work, not just count
        let target = total_predicted / 2.0;
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut left_time = 0.0;

        for test in batch.tests {
            let pred = self.predictions.get(&test).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
            if left_time < target && (left.is_empty() || left_time + pred <= target * 1.2) {
                // Add to left if under target (with 20% tolerance to avoid bad splits)
                left_time += pred;
                left.push(test);
            } else {
                right.push(test);
            }
        }

        // Ensure we actually split (don't create empty batches)
        if left.is_empty() || right.is_empty() {
            // Fallback: split by count
            let mut all_tests = left;
            all_tests.extend(right);
            let mid = all_tests.len() / 2;
            left = all_tests[..mid].to_vec();
            right = all_tests[mid..].to_vec();
        }

        // Re-insert both batches (no sorting - tail prioritization handles selection)
        if !left.is_empty() {
            self.batches.push(BatchWithKind { tests: left, kind });
        }
        if !right.is_empty() {
            self.batches.push(BatchWithKind { tests: right, kind });
        }

        true
    }

    /// Rebuild batches if there are pending files, then get the next batch.
    /// The scheduler decides internally whether to give a GPU or CPU batch
    /// based on current gpu_in_flight count vs gpu_jobs limit.
    fn try_get_batch(&mut self) -> Option<BatchAssignment> {
        // Rebuild batches if we have pending files
        if !self.pending_files.is_empty() {
            // Collect all remaining tests: pending + already batched
            let mut all_files: Vec<String> = std::mem::take(&mut self.pending_files);
            for batch in self.batches.drain(..) {
                all_files.extend(batch.tests);
            }

            // Recalculate pool_predicted from all_files
            self.pool_predicted = all_files.iter()
                .map(|f| self.predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
                .sum();

            // Rebuild batches
            self.batches = Self::build_batches(
                all_files,
                &self.predictions,
                self.has_timing_data,
                self.max_batch_size,
                self.target_batch_duration,
                self.gpu_jobs,
            );
        }

        // Ensure enough batches to keep workers busy (split large batches at end)
        self.ensure_parallelism();

        // Simple batch selection: just respect GPU limits if set
        // The constrained random shuffle already ensures slow tests aren't at the end.
        let batch_with_kind = if let Some(max_gpu) = self.gpu_jobs {
            if self.gpu_in_flight >= max_gpu {
                // GPU slots full - must find a CPU batch
                let cpu_idx = self.batches.iter()
                    .enumerate()
                    .rev()
                    .find(|(_, b)| b.kind == BatchKind::Cpu)
                    .map(|(i, _)| i);
                cpu_idx.map(|i| self.batches.remove(i))
            } else {
                self.batches.pop()
            }
        } else {
            self.batches.pop()
        }?;

        let tests = batch_with_kind.tests;
        let kind = batch_with_kind.kind;

        // Calculate predicted duration for this batch
        let predicted_duration: f64 = tests.iter()
            .map(|f| self.predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
            .sum();

        // Subtract from pool predicted
        self.pool_predicted -= predicted_duration;

        // Track GPU batch count
        if kind == BatchKind::Gpu {
            self.gpu_in_flight += 1;
        }

        // Track as in-flight
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;
        let test_count = tests.len();
        self.in_flight.insert(batch_id, InFlightBatch {
            start_time: Instant::now(),
            predicted_duration,
            test_count,
            kind,
        });

        Some(BatchAssignment { batch_id, tests, kind })
    }

    /// Mark a batch as complete
    fn complete_batch(&mut self, batch_id: usize) {
        if let Some(batch) = self.in_flight.remove(&batch_id) {
            if batch.kind == BatchKind::Gpu {
                self.gpu_in_flight -= 1;
            }
        }
    }

    /// Calculate estimated time remaining
    fn calculate_eta(&self, num_workers: usize) -> f64 {
        let mut in_flight_remaining = 0.0f64;
        let mut longest_remaining = 0.0f64;

        for batch in self.in_flight.values() {
            let elapsed = batch.start_time.elapsed().as_secs_f64();
            let remaining = (batch.predicted_duration - elapsed).max(0.0);
            in_flight_remaining += remaining;
            longest_remaining = longest_remaining.max(remaining);
        }

        let total_remaining = self.pool_predicted + in_flight_remaining;
        let parallel_eta = total_remaining / num_workers.max(1) as f64;

        parallel_eta.max(longest_remaining)
    }
}

/// Handle for workers to communicate with the scheduler.
/// This is cheap to clone and can be shared across threads.
#[derive(Clone)]
pub struct SchedulerHandle {
    tx: crossbeam_channel::Sender<SchedulerMessage>,
    /// Read-only predictions for workers (avoids round-trip for lookups)
    pub predictions: Arc<HashMap<String, f64>>,
    pub has_timing_data: bool,
    /// Number of workers (for ETA calculation in status queries)
    num_workers: usize,
}

impl SchedulerHandle {
    /// Request a batch from the scheduler. Returns None if no batches available.
    /// The scheduler decides atomically whether to give a GPU or CPU batch
    /// based on current state - no TOCTOU issues.
    pub fn get_batch(&self) -> Option<BatchAssignment> {
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        self.tx.send(SchedulerMessage::GetBatch { response_tx }).ok()?;
        response_rx.recv().ok().flatten()
    }

    /// Notify the scheduler that a batch is complete
    pub fn complete_batch(&self, batch_id: usize) {
        let _ = self.tx.send(SchedulerMessage::CompleteBatch { batch_id });
    }

    /// Add tests back to the pool (for retries or repooled tests)
    pub fn add_tests(&self, tests: Vec<String>) {
        if !tests.is_empty() {
            let _ = self.tx.send(SchedulerMessage::AddTests { tests });
        }
    }

    /// Add a single test back to the pool
    pub fn add_test(&self, test: String) {
        self.add_tests(vec![test]);
    }

    /// Get atomic status snapshot from scheduler.
    /// This is the only way to query scheduler state - all fields are
    /// computed atomically to avoid TOCTOU issues.
    pub fn get_status(&self) -> SchedulerStatus {
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        if self.tx.send(SchedulerMessage::GetStatus {
            num_workers: self.num_workers,
            response_tx,
        }).is_err() {
            // Scheduler is gone, return "empty" status
            return SchedulerStatus {
                is_empty: true,
                remaining: 0,
                eta: 0.0,
                has_pending_batches: false,
                debug_state: (0, 0, vec![]),
            };
        }
        response_rx.recv().unwrap_or(SchedulerStatus {
            is_empty: true,
            remaining: 0,
            eta: 0.0,
            has_pending_batches: false,
            debug_state: (0, 0, vec![]),
        })
    }

    /// Shutdown the scheduler
    pub fn shutdown(&self) {
        let _ = self.tx.send(SchedulerMessage::Shutdown);
    }
}
