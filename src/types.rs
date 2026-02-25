use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::{Deserialize, Serialize};
use sysinfo::{CpuRefreshKind, RefreshKind, System};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

// ============================================================================
// Test Identifier
// ============================================================================

/// Represents a test identifier from slang-test -dry-run
/// Formats:
/// - Simple: "tests/path/file.slang"
/// - With variant: "tests/path/file.slang.0 (vk)"
/// - Synthesized: "tests/path/file.slang.1 syn (llvm)"
/// - Internal: "slang-unit-test-tool/modulePtr.internal"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TestId {
    /// Base file path (e.g., "tests/path/file.slang")
    pub path: String,
    /// Variant number if present (e.g., 0, 1, 2)
    pub variant: Option<u32>,
    /// Whether this is a synthesized test
    pub synthesized: bool,
    /// Backend/API if specified (e.g., "vk", "llvm", "cpu")
    pub api: Option<String>,
}

impl TestId {
    /// Parse a test identifier string from slang-test -dry-run output
    pub fn parse(s: &str) -> Self {
        let s = s.trim();

        // Extract API from parentheses at the end: "... (vk)"
        let (rest, api) = if let Some(paren_start) = s.rfind('(') {
            if let Some(paren_end) = s.rfind(')') {
                if paren_end > paren_start {
                    let api = s[paren_start + 1..paren_end].trim().to_string();
                    let rest = s[..paren_start].trim();
                    (rest, Some(api))
                } else {
                    (s, None)
                }
            } else {
                (s, None)
            }
        } else {
            (s, None)
        };

        // Check for "syn" suffix
        let (rest, synthesized) = if rest.ends_with(" syn") {
            (&rest[..rest.len() - 4], true)
        } else {
            (rest, false)
        };

        // Check for variant number: "path.slang.0" -> path="path.slang", variant=0
        // Need to find the last ".N" where N is a number
        let (path, variant) = if let Some(last_dot) = rest.rfind('.') {
            let potential_num = &rest[last_dot + 1..];
            if let Ok(num) = potential_num.parse::<u32>() {
                // Make sure this isn't the file extension
                let base = &rest[..last_dot];
                if base.ends_with(".slang") || base.ends_with(".hlsl")
                    || base.ends_with(".glsl") || base.ends_with(".c")
                    || base.contains(".internal") {
                    (base.to_string(), Some(num))
                } else {
                    (rest.to_string(), None)
                }
            } else {
                (rest.to_string(), None)
            }
        } else {
            (rest.to_string(), None)
        };

        TestId {
            path,
            variant,
            synthesized,
            api,
        }
    }

    /// Convert back to the full string format (for display)
    pub fn to_test_string(&self) -> String {
        let mut s = self.path.clone();
        if let Some(v) = self.variant {
            s.push_str(&format!(".{}", v));
        }
        if self.synthesized {
            s.push_str(" syn");
        }
        if let Some(ref api) = self.api {
            s.push_str(&format!(" ({})", api));
        }
        s
    }

    /// Convert to the format that slang-test accepts as input (path + variant only)
    /// This strips the API suffix and synthesized flag which are only for display
    pub fn to_slang_test_arg(&self) -> String {
        let mut s = self.path.clone();
        if let Some(v) = self.variant {
            s.push_str(&format!(".{}", v));
        }
        // Note: slang-test doesn't accept " syn" or "(api)" suffixes as input
        s
    }
}

impl TestId {
    /// Convert to the timing cache key format: "path.variant"
    /// This strips the synthesized flag and API suffix, keeping only the file path and variant.
    /// Note: slang-test omits ".0" in output but includes it in dry-run, so we normalize
    /// by treating no-variant source files as .0
    pub fn to_timing_key(&self) -> String {
        if let Some(v) = self.variant {
            format!("{}.{}", self.path, v)
        } else if self.path.ends_with(".slang")
            || self.path.ends_with(".hlsl")
            || self.path.ends_with(".glsl")
            || self.path.ends_with(".c")
        {
            // slang-test omits .0 variant in output, normalize to .0
            format!("{}.0", self.path)
        } else {
            self.path.clone()
        }
    }
}

impl std::fmt::Display for TestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_test_string())
    }
}

/// Convert a test string to its timing cache key.
/// "tests/foo.slang.4 syn (vk)" -> "tests/foo.slang.4"
pub fn test_to_timing_key(test: &str) -> String {
    TestId::parse(test).to_timing_key()
}

impl PartialOrd for TestId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TestId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_test_string().cmp(&other.to_test_string())
    }
}

// Constants
pub const DEFAULT_PREDICTED_DURATION: f64 = 0.5;
pub const EMA_NEW_WEIGHT: f64 = 0.7;
pub const OUTPUT_TRUNCATE_LINES: usize = 30;

// ============================================================================
// Batch Kind (for GPU job limiting)
// ============================================================================

/// Classification of a batch for GPU job limiting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    /// Batch contains only CPU tests (cpu, llvm backends or no API)
    Cpu,
    /// Batch contains only GPU tests (vk, cuda, dx11, dx12, metal or gfx-unit-test-tool)
    Gpu,
    /// Batch contains mixed tests or gpu_jobs is not set (no segmentation)
    Mixed,
}

/// Check if a test is a GPU test based on its API or test name
/// GPU tests: vk, cuda, dx11, dx12, metal backends, or gfx-unit-test-tool internal tests
pub fn is_gpu_test(test: &str) -> bool {
    let test_id = TestId::parse(test);

    // gfx-unit-test-tool internal tests are GPU tests
    if test_id.path.starts_with("gfx-unit-test-tool/") {
        return true;
    }

    // Check API
    match test_id.api.as_deref() {
        Some(api) => {
            let api_lower = api.to_lowercase();
            matches!(api_lower.as_str(), "vk" | "vulkan" | "cuda" | "dx11" | "dx12" | "metal")
        }
        // No API specified - assume CPU (e.g., internal tests without API suffix)
        None => false,
    }
}

// ============================================================================
// Event Logging
// ============================================================================

#[derive(Debug)]
pub struct EventLog {
    writer: Mutex<BufWriter<File>>,
    start_time: Instant,
}

impl EventLog {
    pub fn new(path: &PathBuf) -> Result<Self> {
        let file = File::create(path).context("Failed to create event log file")?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "timestamp_ms,event,details")?;
        Ok(Self {
            writer: Mutex::new(writer),
            start_time: Instant::now(),
        })
    }

    pub fn log(&self, event: &str, details: &str) {
        let elapsed_ms = self.start_time.elapsed().as_millis();
        if let Ok(mut writer) = self.writer.lock() {
            let mut csv_writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(Vec::new());
            let record = [
                elapsed_ms.to_string(),
                event.to_string(),
                details.to_string(),
            ];
            if csv_writer.write_record(&record).is_ok() {
                if let Ok(data) = csv_writer.into_inner() {
                    let _ = writer.write_all(&data);
                }
            }
        }
    }

    pub fn flush(&self) {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.flush();
        }
    }
}

/// Global event log (set once at startup if --event-log is provided)
static EVENT_LOG: LazyLock<Mutex<Option<Arc<EventLog>>>> = LazyLock::new(|| Mutex::new(None));

pub fn log_event(event: &str, details: &str) {
    if let Ok(guard) = EVENT_LOG.lock() {
        if let Some(ref log) = *guard {
            log.log(event, details);
        }
    }
}

pub fn init_event_log(path: &PathBuf) -> Result<()> {
    let log = EventLog::new(path)?;
    if let Ok(mut guard) = EVENT_LOG.lock() {
        *guard = Some(Arc::new(log));
    }
    Ok(())
}

pub fn flush_event_log() {
    if let Ok(guard) = EVENT_LOG.lock() {
        if let Some(ref log) = *guard {
            log.flush();
        }
    }
}

// ============================================================================
// State Directory
// ============================================================================

/// Get the state directory path (~/.local/state/slang-test-interceptor or XDG equivalent)
pub fn get_state_dir() -> Option<PathBuf> {
    dirs::state_dir()
        .or_else(|| dirs::data_local_dir()) // Fallback for macOS/Windows
        .map(|p| p.join("slang-test-interceptor"))
}

// ============================================================================
// Timing Cache
// ============================================================================

/// Build type for timing cache segmentation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BuildType {
    Debug,
    Release,
    RelWithDebInfo,
}

impl BuildType {
    /// Detect build type from slang-test path
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        let path_str = path.to_string_lossy().to_lowercase();
        if path_str.contains("relwithdebinfo") {
            Some(BuildType::RelWithDebInfo)
        } else if path_str.contains("debug") {
            Some(BuildType::Debug)
        } else if path_str.contains("release") {
            Some(BuildType::Release)
        } else {
            None
        }
    }
}

impl std::fmt::Display for BuildType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildType::Debug => write!(f, "debug"),
            BuildType::Release => write!(f, "release"),
            BuildType::RelWithDebInfo => write!(f, "relwithdebinfo"),
        }
    }
}

/// Timing cache stores per-test durations, segmented by build type.
/// Keys are test identifiers like "tests/foo.slang" or "tests/foo.slang.4" (with variant suffix).
/// The variant suffix is included when there are multiple tests from the same file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingCache {
    pub version: u32,
    /// Map from build type to (test identifier -> duration in seconds)
    /// Using String keys for JSON serialization compatibility
    #[serde(default)]
    pub timings_by_build: HashMap<String, HashMap<String, f64>>,
    /// Legacy: flat timings map (for migration from version 2)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub timings: HashMap<String, f64>,
}

impl TimingCache {
    pub fn load() -> Self {
        if let Some(state_dir) = get_state_dir() {
            let path = state_dir.join("timing.json");
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(mut cache) = serde_json::from_str::<TimingCache>(&contents) {
                    // Migrate from version 2 (flat timings) to version 3 (segmented)
                    if cache.version == 2 && !cache.timings.is_empty() {
                        // Move old timings to "release" bucket as a reasonable default
                        cache.timings_by_build.insert("release".to_string(), cache.timings.clone());
                        cache.timings.clear();
                        cache.version = 3;
                    }
                    if cache.version >= 3 {
                        return cache;
                    }
                }
                // Old format or version 1 - just start fresh
            }
        }
        Self {
            version: 3,
            timings_by_build: HashMap::new(),
            timings: HashMap::new(),
        }
    }

    pub fn save(&self) {
        if let Some(state_dir) = get_state_dir() {
            let _ = std::fs::create_dir_all(&state_dir);
            let path = state_dir.join("timing.json");
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = std::fs::write(&path, json);
            }
        }
    }

    /// Get the timings map for a specific build type
    fn get_timings(&self, build_type: BuildType) -> Option<&HashMap<String, f64>> {
        self.timings_by_build.get(&build_type.to_string())
    }

    /// Get or create the timings map for a specific build type
    fn get_timings_mut(&mut self, build_type: BuildType) -> &mut HashMap<String, f64> {
        self.timings_by_build
            .entry(build_type.to_string())
            .or_insert_with(HashMap::new)
    }

    /// Record a test's duration for a specific build type. Uses EMA to smooth out variations.
    pub fn record(&mut self, build_type: BuildType, test_id: &str, duration: f64) {
        let timings = self.get_timings_mut(build_type);
        let existing = timings.entry(test_id.to_string()).or_insert(0.0);
        if *existing == 0.0 {
            *existing = duration;
        } else {
            *existing = duration * EMA_NEW_WEIGHT + *existing * (1.0 - EMA_NEW_WEIGHT);
        }
    }

    /// Predict duration for a test with a specific build type.
    /// Returns DEFAULT_PREDICTED_DURATION if unknown.
    pub fn predict(&self, build_type: BuildType, test_id: &str) -> f64 {
        self.get_timings(build_type)
            .and_then(|t| t.get(test_id).copied())
            .unwrap_or(DEFAULT_PREDICTED_DURATION)
    }

    /// Check if there's timing data for a specific build type
    pub fn has_timing_data(&self, build_type: BuildType) -> bool {
        self.get_timings(build_type)
            .map(|t| !t.is_empty())
            .unwrap_or(false)
    }

    /// Merge observed timings into the cache for a specific build type
    pub fn merge(&mut self, build_type: BuildType, observed: &HashMap<String, f64>) {
        for (test_id, duration) in observed {
            self.record(build_type, test_id, *duration);
        }
    }
}

// ============================================================================
// Test Results and Statistics
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestResult {
    Passed,
    Failed,
    Ignored,
}

#[derive(Debug, Clone)]
pub struct TestOutcome {
    pub name: String,
    pub result: TestResult,
    pub duration: Option<Duration>,
    pub failure_output: Vec<String>,
}

#[derive(Debug, Default)]
pub struct TestStats {
    pub passed: AtomicUsize,
    pub failed: AtomicUsize,
    pub ignored: AtomicUsize,
    pub retried_and_passed: AtomicUsize,
    pub files_seen: Mutex<HashSet<String>>,
    pub last_test_output: Mutex<Option<Instant>>,
    pub compiling_since: Mutex<Option<Instant>>,
    /// Observed timings keyed by test identifier (e.g., "tests/foo.slang.4")
    pub observed_timings: Mutex<HashMap<String, f64>>,
}

impl TestStats {
    pub fn record_file(&self, file: &str) {
        self.files_seen.lock().unwrap().insert(file.to_string());
    }

    pub fn files_completed(&self) -> usize {
        self.files_seen.lock().unwrap().len()
    }

    pub fn set_compiling(&self, is_compiling: bool) {
        let mut compiling = self.compiling_since.lock().unwrap();
        if is_compiling {
            if compiling.is_none() {
                *compiling = Some(Instant::now());
            }
        } else {
            *compiling = None;
        }
    }

    pub fn get_compiling_time(&self) -> Option<f64> {
        self.compiling_since.lock().unwrap().map(|t| t.elapsed().as_secs_f64())
    }

    pub fn mark_execution_started(&self) {
        let mut last = self.last_test_output.lock().unwrap();
        if last.is_none() {
            *last = Some(Instant::now());
        }
    }

    pub fn record_test_output(&self) {
        *self.last_test_output.lock().unwrap() = Some(Instant::now());
    }

    pub fn seconds_since_last_output(&self) -> Option<f64> {
        self.last_test_output.lock().unwrap().map(|t| t.elapsed().as_secs_f64())
    }

    /// Record observed timing for a test identifier (e.g., "tests/foo.slang.4")
    pub fn record_observed_timing(&self, test_id: &str, duration: f64) {
        let mut observed = self.observed_timings.lock().unwrap();
        observed.insert(test_id.to_string(), duration);
    }

    pub fn get_observed_timings(&self) -> HashMap<String, f64> {
        self.observed_timings.lock().unwrap().clone()
    }
}

#[derive(Debug, Clone)]
pub struct FailureInfo {
    pub test_name: String,
    pub output_lines: Vec<String>,
    pub expected: Option<String>,
    pub actual: Option<String>,
}

// ============================================================================
// Work Pool
// ============================================================================

/// Tracks an in-flight batch: when it started, its predicted duration, and test count
#[derive(Clone)]
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

/// Work pool for distributing test batches to workers.
///
/// LOCK ORDERING: To avoid deadlocks, locks must always be acquired in this order:
///   1. pending_files
///   2. batches
///   3. pool_predicted
///   4. in_flight
///
/// Any method acquiring multiple locks must follow this order.
pub struct WorkPool {
    /// Pre-built batches of tests, ready to be popped by workers
    batches: Mutex<Vec<BatchWithKind>>,
    /// Pending files that need to be rebuilt into batches (retries, repooled after crash)
    pending_files: Mutex<Vec<String>>,
    /// In-flight batches keyed by batch ID (for tracking remaining time)
    in_flight: Mutex<HashMap<usize, InFlightBatch>>,
    /// Counter for generating batch IDs
    next_batch_id: AtomicUsize,
    /// Predicted time for tests still in the pool (not in-flight, not completed)
    pool_predicted: Mutex<f64>,
    /// Configuration for batch building
    max_batch_size: usize,
    target_batch_duration: f64,
    /// Predictions keyed by test identifier (e.g., "tests/foo.slang.4")
    pub predictions: HashMap<String, f64>,
    pub has_timing_data: bool,
    /// Maximum concurrent GPU batches (None = no limit, batches are Mixed)
    gpu_jobs: Option<usize>,
    /// Current number of in-flight GPU batches
    gpu_in_flight: AtomicUsize,
}

impl WorkPool {
    pub fn new(
        files: Vec<String>,
        max_batch_size: usize,
        _num_workers: usize,
        predictions: HashMap<String, f64>,
        has_timing_data: bool,
        target_batch_duration: f64,
        gpu_jobs: Option<usize>,
    ) -> Self {
        let total_predicted: f64 = files.iter()
            .map(|f| predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
            .sum();

        let batches = Self::build_batches(
            files,
            &predictions,
            has_timing_data,
            max_batch_size,
            target_batch_duration,
            gpu_jobs,
        );

        Self {
            batches: Mutex::new(batches),
            pending_files: Mutex::new(Vec::new()),
            in_flight: Mutex::new(HashMap::new()),
            next_batch_id: AtomicUsize::new(0),
            pool_predicted: Mutex::new(total_predicted),
            max_batch_size,
            target_batch_duration,
            predictions,
            has_timing_data,
            gpu_jobs,
            gpu_in_flight: AtomicUsize::new(0),
        }
    }

    /// Build batches from a list of files.
    /// Files should be pre-sorted by duration (slowest first) if timing data is available.
    /// When gpu_jobs is Some, tests are segmented into GPU-only and CPU-only batches.
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
            files.into_iter().partition(|f| is_gpu_test(f))
        } else {
            (Vec::new(), files)
        };

        let mut batches = Vec::new();

        // Helper to build batches from a list of files with a given kind
        let build_from_files = |files: Vec<String>, kind: BatchKind, batches: &mut Vec<BatchWithKind>| {
            if files.is_empty() {
                return;
            }

            let mut files = files;

            if has_timing_data {
                // Sort by predicted duration, slowest first
                files.sort_by(|a, b| {
                    let dur_a = predictions.get(a).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                    let dur_b = predictions.get(b).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                    dur_b.partial_cmp(&dur_a).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Build batches: slow tests get their own batch, fast tests are grouped
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
            // Build separate GPU and CPU batches
            build_from_files(gpu_files, BatchKind::Gpu, &mut batches);
            build_from_files(cpu_files, BatchKind::Cpu, &mut batches);
        } else {
            // No GPU limiting - all batches are Mixed
            build_from_files(cpu_files, BatchKind::Mixed, &mut batches);
        }

        // Sort by predicted duration (slowest first) across all batches
        batches.sort_by(|a, b| {
            let dur_a: f64 = a.tests.iter()
                .map(|f| predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
                .sum();
            let dur_b: f64 = b.tests.iter()
                .map(|f| predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
                .sum();
            dur_b.partial_cmp(&dur_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Reverse so we can pop from the end (O(1)) and get slowest batches first
        batches.reverse();
        batches
    }

    /// Add a file back to the pool (for retries or repooled tests after crash).
    /// This triggers a rebuild of batches on the next try_get_batch call.
    pub fn add_file(&self, file: String) {
        let predicted = self.predictions.get(&file).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
        // Lock order must match try_get_batch: pending_files first, then pool_predicted
        let mut pending = self.pending_files.lock().unwrap();
        *self.pool_predicted.lock().unwrap() += predicted;
        pending.push(file);
    }

    /// Check if the pool is empty.
    /// IMPORTANT: Lock order must be pending_files -> batches -> in_flight (same as try_get_batch)
    pub fn is_empty(&self) -> bool {
        // Lock in same order as try_get_batch to avoid deadlock
        let pending = self.pending_files.lock().unwrap();
        let batches = self.batches.lock().unwrap();
        let in_flight = self.in_flight.lock().unwrap();
        pending.is_empty() && batches.is_empty() && in_flight.is_empty()
    }

    /// Get debug info about pool state: (batches_count, pending_count, in_flight_ids)
    /// IMPORTANT: Lock order must be pending_files -> batches -> in_flight (same as try_get_batch)
    pub fn debug_state(&self) -> (usize, usize, Vec<usize>) {
        // Lock in same order as try_get_batch to avoid deadlock
        let pending = self.pending_files.lock().unwrap();
        let batches = self.batches.lock().unwrap();
        let in_flight = self.in_flight.lock().unwrap();
        (
            batches.len(),
            pending.len(),
            in_flight.keys().copied().collect(),
        )
    }

    /// Check if there are batches waiting to be picked up (excluding in-flight)
    /// IMPORTANT: Lock order must be pending_files -> batches (same as try_get_batch)
    pub fn has_pending_batches(&self) -> bool {
        let pending = self.pending_files.lock().unwrap();
        let batches = self.batches.lock().unwrap();
        !pending.is_empty() || !batches.is_empty()
    }

    /// Returns number of tests remaining (in pool + in-flight)
    /// IMPORTANT: Lock order must be pending_files -> batches -> in_flight (same as try_get_batch)
    pub fn remaining(&self) -> usize {
        let pending = self.pending_files.lock().unwrap();
        let batches = self.batches.lock().unwrap();
        let in_flight = self.in_flight.lock().unwrap();
        let pool_count: usize = batches.iter().map(|b| b.tests.len()).sum::<usize>() + pending.len();
        let in_flight_count: usize = in_flight.values().map(|b| b.test_count).sum();
        pool_count + in_flight_count
    }

    /// Check if there's a GPU slot available
    pub fn can_take_gpu_batch(&self) -> bool {
        match self.gpu_jobs {
            None => true, // No GPU limiting
            Some(max) => self.gpu_in_flight.load(Ordering::SeqCst) < max,
        }
    }

    /// Get the next batch of tests to run.
    /// Returns (batch_id, tests, kind) where batch_id is used to mark the batch complete later.
    /// If there are pending files (retries/repooled), rebuilds batches first.
    ///
    /// When `has_gpu_slot` is false and gpu_jobs is set, GPU batches will be skipped.
    pub fn try_get_batch(&self, has_gpu_slot: bool) -> Option<(usize, Vec<String>, BatchKind)> {
        // Check if we need to rebuild batches due to pending files
        {
            let mut pending = self.pending_files.lock().unwrap();
            if !pending.is_empty() {
                let mut batches = self.batches.lock().unwrap();
                let mut pool_predicted = self.pool_predicted.lock().unwrap();

                // Collect all remaining tests: pending + already batched
                let mut all_files: Vec<String> = pending.drain(..).collect();
                for batch in batches.drain(..) {
                    all_files.extend(batch.tests);
                }

                // Recalculate pool_predicted from all_files
                *pool_predicted = all_files.iter()
                    .map(|f| self.predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
                    .sum();

                // Rebuild batches
                *batches = Self::build_batches(
                    all_files,
                    &self.predictions,
                    self.has_timing_data,
                    self.max_batch_size,
                    self.target_batch_duration,
                    self.gpu_jobs,
                );
            }
        }

        // Find and pop an appropriate batch
        let batch_with_kind = {
            let mut batches = self.batches.lock().unwrap();

            if self.gpu_jobs.is_none() || has_gpu_slot {
                // No GPU limiting or we have a slot - just pop the next batch
                batches.pop()
            } else {
                // GPU limiting is active and we don't have a slot - find a CPU batch
                // Search from the end (where we pop from) for efficiency
                let mut cpu_idx = None;
                for (i, batch) in batches.iter().enumerate().rev() {
                    if batch.kind == BatchKind::Cpu {
                        cpu_idx = Some(i);
                        break;
                    }
                }
                cpu_idx.map(|i| batches.remove(i))
            }
        }?;

        let batch = batch_with_kind.tests;
        let kind = batch_with_kind.kind;

        // Calculate predicted duration for this batch
        let predicted_duration: f64 = batch.iter()
            .map(|f| self.predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
            .sum();

        // Subtract from pool predicted
        *self.pool_predicted.lock().unwrap() -= predicted_duration;

        // Track GPU batch count
        if kind == BatchKind::Gpu {
            self.gpu_in_flight.fetch_add(1, Ordering::SeqCst);
        }

        // Track as in-flight
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::SeqCst);
        let test_count = batch.len();
        self.in_flight.lock().unwrap().insert(batch_id, InFlightBatch {
            start_time: Instant::now(),
            predicted_duration,
            test_count,
            kind,
        });

        Some((batch_id, batch, kind))
    }

    /// Mark a batch as complete (tests finished running)
    pub fn complete_batch(&self, batch_id: usize) {
        let removed = self.in_flight.lock().unwrap().remove(&batch_id);
        if let Some(batch) = removed {
            if batch.kind == BatchKind::Gpu {
                self.gpu_in_flight.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }

    /// Calculate estimated time remaining.
    /// Takes into account pool remaining, in-flight batch progress, and the "long pole" problem.
    pub fn calculate_eta(&self, num_workers: usize) -> f64 {
        let pool_remaining = *self.pool_predicted.lock().unwrap();
        let in_flight = self.in_flight.lock().unwrap();

        let mut in_flight_remaining = 0.0f64;
        let mut longest_remaining = 0.0f64;

        for batch in in_flight.values() {
            let elapsed = batch.start_time.elapsed().as_secs_f64();
            let remaining = (batch.predicted_duration - elapsed).max(0.0);
            in_flight_remaining += remaining;
            longest_remaining = longest_remaining.max(remaining);
        }

        let total_remaining = pool_remaining + in_flight_remaining;
        let parallel_eta = total_remaining / num_workers.max(1) as f64;

        // ETA is the max of parallel completion time and the longest single batch
        parallel_eta.max(longest_remaining)
    }

}

// ============================================================================
// Worker State (for per-worker progress display)
// ============================================================================

/// Sentinel value meaning "no test running" (worker is idle or between batches)
pub const WORKER_IDLE: usize = usize::MAX;

/// State for a single worker, used to track what test is currently running.
/// The worker writes to this, and the progress thread reads from it.
pub struct WorkerState {
    /// Index into the current batch's test list. WORKER_IDLE means not running.
    pub current_test_idx: AtomicUsize,
    /// The current batch of tests (set when batch starts, cleared when batch ends)
    pub current_batch: Mutex<Vec<String>>,
}

impl WorkerState {
    pub fn new() -> Self {
        Self {
            current_test_idx: AtomicUsize::new(WORKER_IDLE),
            current_batch: Mutex::new(Vec::new()),
        }
    }

    /// Called when a worker starts a new batch
    pub fn start_batch(&self, batch: &[String]) {
        *self.current_batch.lock().unwrap() = batch.to_vec();
        self.current_test_idx.store(0, Ordering::SeqCst);
    }

    /// Called when a test completes - advance to next test
    pub fn advance(&self) {
        self.current_test_idx.fetch_add(1, Ordering::SeqCst);
    }

    /// Called when batch completes or worker goes idle
    pub fn clear(&self) {
        self.current_test_idx.store(WORKER_IDLE, Ordering::SeqCst);
        self.current_batch.lock().unwrap().clear();
    }

    /// Get the currently running test name, if any
    pub fn current_test(&self) -> Option<String> {
        let idx = self.current_test_idx.load(Ordering::SeqCst);
        if idx == WORKER_IDLE {
            return None;
        }
        let batch = self.current_batch.lock().unwrap();
        batch.get(idx).cloned()
    }
}

impl Default for WorkerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Container for all worker states
pub struct WorkerStates {
    states: Vec<WorkerState>,
}

impl WorkerStates {
    pub fn new(num_workers: usize) -> Self {
        Self {
            states: (0..num_workers).map(|_| WorkerState::new()).collect(),
        }
    }

    pub fn get(&self, worker_id: usize) -> &WorkerState {
        &self.states[worker_id]
    }
}

// ============================================================================
// Progress Display
// ============================================================================

pub struct ProgressDisplay {
    total_files: usize,
    start_time: Instant,
    machine_output: bool,
    last_reported_files: AtomicUsize,
    /// MultiProgress container - must be kept alive for progress bars to render correctly
    #[allow(dead_code)]
    multi_progress: Option<MultiProgress>,
    main_progress_bar: Option<ProgressBar>,
    worker_bars: Vec<Option<ProgressBar>>,
    verbose: bool,
    /// Cached GPU load percentage (updated periodically)
    last_gpu_load: Option<u32>,
    /// Cached CPU load percentage (updated periodically)
    last_cpu_load: f32,
    /// System info for CPU queries
    sys: System,
    /// Counter for throttling system queries
    sys_query_counter: u32,
}

impl ProgressDisplay {
    pub fn new(total_files: usize, machine_output: bool, num_workers: usize, verbose: bool) -> Self {
        let (multi_progress, main_progress_bar, worker_bars) = if machine_output {
            (None, None, Vec::new())
        } else {
            let mp = MultiProgress::new();

            // Worker progress bars first (so they appear above the main bar)
            // Only create them in verbose mode
            let worker_bars: Vec<Option<ProgressBar>> = if verbose {
                (0..num_workers)
                    .map(|_| {
                        let pb = mp.add(ProgressBar::new_spinner());
                        pb.set_style(
                            ProgressStyle::default_spinner()
                                .template("{msg:.dim}")
                                .unwrap(),
                        );
                        pb.set_message(""); // Empty initially
                        Some(pb)
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // Main progress bar at the bottom
            let main_pb = mp.add(ProgressBar::new(total_files as u64));
            main_pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg}")
                    .unwrap(),
            );

            (Some(mp), Some(main_pb), worker_bars)
        };

        Self {
            total_files,
            start_time: Instant::now(),
            machine_output,
            last_reported_files: AtomicUsize::new(0),
            multi_progress,
            main_progress_bar,
            worker_bars,
            verbose,
            last_gpu_load: None,
            last_cpu_load: 0.0,
            sys: System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything())),
            sys_query_counter: 0,
        }
    }

    pub fn update(&mut self, stats: &TestStats, _files_completed: usize, batches_running: usize, _batches_remaining: usize, has_pending_batches: bool, eta_seconds: Option<f64>, worker_states: Option<&WorkerStates>) {
        // Update CPU/GPU load every ~30 updates (~500ms at 16ms refresh rate)
        self.sys_query_counter += 1;
        if self.sys_query_counter >= 30 {
            self.sys_query_counter = 0;
            self.last_cpu_load = get_cpu_usage(&mut self.sys);
            self.last_gpu_load = get_gpu_usage();
        }
        let passed = stats.passed.load(Ordering::SeqCst);
        let failed = stats.failed.load(Ordering::SeqCst);
        let ignored = stats.ignored.load(Ordering::SeqCst);
        let tests_done = passed + failed + ignored;
        let tests_remaining = self.total_files.saturating_sub(tests_done);
        let elapsed = self.start_time.elapsed().as_secs_f64();

        if self.machine_output {
            // Report at 0%, 10%, 20%, ... 90%, 99%, 100%
            // last_reported_files stores (last_pct + 1) so 0 means "haven't reported yet"
            // We use 0-10 for 0%-100% in 10% steps, and 99 as a special marker for 99%
            let current_pct = (tests_done * 10) / self.total_files.max(1);
            let at_99_pct = tests_done * 100 >= self.total_files * 99 && tests_done < self.total_files;
            let last_reported = self.last_reported_files.load(Ordering::SeqCst);
            let should_report = if last_reported == 0 {
                true  // First report (0%)
            } else if at_99_pct && last_reported < 99 {
                true  // Report at 99%
            } else {
                current_pct >= last_reported && last_reported <= 10  // Next 10% threshold
            };
            if should_report {
                let new_marker = if at_99_pct { 99 } else { current_pct + 1 };
                self.last_reported_files.store(new_marker, Ordering::SeqCst);
                let percent = (tests_done as f64 / self.total_files.max(1) as f64) * 100.0;
                let eta = match eta_seconds {
                    Some(secs) if secs > 1.0 => format!(" | ETA: {:.1}s", secs),
                    Some(_) => " | ETA: <1s".to_string(),
                    None => String::new(),
                };
                eprintln!(
                    "[{}/{}/{}] {:.1}% | {} passed, {} failed, {} ignored | Elapsed: {:.1}s{}",
                    batches_running, tests_remaining, self.total_files,
                    percent, passed, failed, ignored, elapsed, eta
                );
            }
        } else if let Some(ref pb) = self.main_progress_bar {
            // Percentage based on tests completed vs total
            let percent = (tests_done as f64 / self.total_files.max(1) as f64) * 100.0;

            // Format ETA string
            let eta = match eta_seconds {
                Some(secs) if secs > 1.0 && tests_remaining > 0 => {
                    format!(" | ETA: {:.1}s", secs)
                }
                Some(_) if tests_remaining > 0 => {
                    " | ETA: <1s".to_string()
                }
                _ => String::new(),
            };

            let compiling_info = if let Some(secs) = stats.get_compiling_time() {
                format!(" COMPILING({:.0}s)", secs)
            } else {
                String::new()
            };

            let stuck_info = if let Some(secs) = stats.seconds_since_last_output() {
                if secs > 1.0 {
                    format!(" \x1b[2m[waiting {:.0}s]\x1b[0m", secs)
                } else {
                    String::new()
                }
            } else {
                " \x1b[2m[no timer]\x1b[0m".to_string()
            };

            let load_info = match self.last_gpu_load {
                Some(gpu) => format!(" | CPU: {:.0}% GPU: {}%", self.last_cpu_load, gpu),
                None if self.last_cpu_load > 0.0 => format!(" | CPU: {:.0}%", self.last_cpu_load),
                None => String::new(),
            };

            let msg = format!(
                "[{}/{}/{}] {:.1}% | {} passed, {} failed, {} ignored | Elapsed: {:.1}s{}{}{}{}",
                batches_running, tests_remaining, self.total_files,
                percent, passed, failed, ignored, elapsed,
                eta, load_info, compiling_info, stuck_info
            );
            pb.set_message(msg);

            // Update per-worker progress bars (verbose mode only)
            if self.verbose {
                if let Some(states) = worker_states {
                    for (worker_id, worker_bar_opt) in self.worker_bars.iter_mut().enumerate() {
                        if let Some(worker_bar) = worker_bar_opt {
                            let state = states.get(worker_id);

                            if let Some(test_name) = state.current_test() {
                                // Show worker line with current test
                                worker_bar.set_message(format!("  worker {}: {}", worker_id, test_name));
                            } else if !has_pending_batches {
                                // Worker idle and no batches waiting - hide the bar
                                worker_bar.set_draw_target(ProgressDrawTarget::hidden());
                                worker_bar.finish_and_clear();
                                *worker_bar_opt = None;
                            } else {
                                // Worker is between tests
                                worker_bar.set_message(format!("  worker {}:", worker_id));
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn finish(&self, stats: &TestStats) {
        // Clear worker bars first
        for worker_bar_opt in &self.worker_bars {
            if let Some(worker_bar) = worker_bar_opt {
                worker_bar.finish_and_clear();
            }
        }

        if let Some(ref pb) = self.main_progress_bar {
            pb.finish_and_clear();
        } else if self.machine_output {
            // Print final 100% status in machine mode
            let passed = stats.passed.load(Ordering::SeqCst);
            let failed = stats.failed.load(Ordering::SeqCst);
            let ignored = stats.ignored.load(Ordering::SeqCst);
            let elapsed = self.start_time.elapsed().as_secs_f64();
            eprintln!(
                "[0/0/{}] 100.0% | {} passed, {} failed, {} ignored | Elapsed: {:.1}s |",
                self.total_files, passed, failed, ignored, elapsed
            );
        }
    }
}

// ============================================================================
// Batch Context
// ============================================================================

pub struct BatchContext<'a> {
    pub slang_test: &'a PathBuf,
    pub root_dir: &'a PathBuf,
    pub test_files: &'a [String],
    pub extra_args: &'a [String],
    pub timeout: Duration,
    pub stats: &'a TestStats,
    pub failures: &'a Mutex<Vec<FailureInfo>>,
    pub max_retries: usize,
    pub retried_tests: &'a Mutex<HashMap<String, usize>>,
    pub work_pool: &'a Arc<WorkPool>,
    pub running: &'a AtomicUsize,
    pub machine_output: bool,
    pub verbose: bool,
}

// ============================================================================
// System Stats
// ============================================================================

pub struct SystemStats {
    sys: System,
}

impl SystemStats {
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::new().with_cpu(CpuRefreshKind::everything())
        );
        Self { sys }
    }

    pub fn refresh_and_log(&mut self, running: usize, pool_remaining: usize) {
        self.sys.refresh_cpu_all();

        let load = System::load_average();
        let cpu_usage: f32 = self.sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
            / self.sys.cpus().len().max(1) as f32;

        let gpu_usage = get_gpu_usage();

        let gpu_str = gpu_usage.map(|g| format!(" gpu={}%", g)).unwrap_or_default();

        log_event("stats", &format!(
            "load_1m={:.2} load_5m={:.2} cpu_avg={:.1}%{} running={} pool={}",
            load.one, load.five, cpu_usage, gpu_str, running, pool_remaining
        ));
    }
}

impl Default for SystemStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "gpu-info")]
fn get_gpu_usage() -> Option<u32> {
    let gpu = gfxinfo::active_gpu().ok()?;
    let info = gpu.info();
    Some(info.load_pct())
}

#[cfg(not(feature = "gpu-info"))]
fn get_gpu_usage() -> Option<u32> {
    None
}

fn get_cpu_usage(sys: &mut System) -> f32 {
    sys.refresh_cpu_all();
    sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
        / sys.cpus().len().max(1) as f32
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gpu_test() {
        // GPU tests - various backends
        assert!(is_gpu_test("tests/compute/foo.slang.0 (vk)"));
        assert!(is_gpu_test("tests/compute/foo.slang.1 (cuda)"));
        assert!(is_gpu_test("tests/compute/foo.slang.2 (dx11)"));
        assert!(is_gpu_test("tests/compute/foo.slang.3 (dx12)"));
        assert!(is_gpu_test("tests/compute/foo.slang.4 (metal)"));
        assert!(is_gpu_test("tests/compute/foo.slang.5 (vulkan)"));
        
        // GPU tests - gfx-unit-test-tool
        assert!(is_gpu_test("gfx-unit-test-tool/someTest.internal"));
        assert!(is_gpu_test("gfx-unit-test-tool/anotherTest.internal.0"));
        
        // CPU tests
        assert!(!is_gpu_test("tests/compute/foo.slang.0 (cpu)"));
        assert!(!is_gpu_test("tests/compute/foo.slang.1 (llvm)"));
        assert!(!is_gpu_test("slang-unit-test-tool/modulePtr.internal"));
        assert!(!is_gpu_test("tests/compute/foo.slang")); // No API suffix
        
        // Synthesized tests
        assert!(is_gpu_test("tests/compute/foo.slang.0 syn (vk)"));
        assert!(!is_gpu_test("tests/compute/foo.slang.0 syn (cpu)"));
    }

    #[test]
    fn test_batch_kind_segmentation() {
        let files = vec![
            "tests/a.slang.0 (vk)".to_string(),
            "tests/b.slang.0 (cpu)".to_string(),
            "tests/c.slang.0 (cuda)".to_string(),
            "tests/d.slang.0 (llvm)".to_string(),
            "gfx-unit-test-tool/test.internal".to_string(),
        ];
        
        let predictions = HashMap::new();
        
        // With gpu_jobs set, batches should be segmented
        let batches = WorkPool::build_batches(
            files.clone(),
            &predictions,
            false,
            100,
            10.0,
            Some(2),
        );
        
        // Should have at least one GPU batch and one CPU batch
        let gpu_batches: Vec<_> = batches.iter().filter(|b| b.kind == BatchKind::Gpu).collect();
        let cpu_batches: Vec<_> = batches.iter().filter(|b| b.kind == BatchKind::Cpu).collect();
        
        assert!(!gpu_batches.is_empty(), "Should have GPU batches");
        assert!(!cpu_batches.is_empty(), "Should have CPU batches");
        
        // Verify GPU batches only contain GPU tests
        for batch in &gpu_batches {
            for test in &batch.tests {
                assert!(is_gpu_test(test), "GPU batch should only contain GPU tests: {}", test);
            }
        }
        
        // Verify CPU batches only contain CPU tests
        for batch in &cpu_batches {
            for test in &batch.tests {
                assert!(!is_gpu_test(test), "CPU batch should only contain CPU tests: {}", test);
            }
        }
    }

    #[test]
    fn test_no_segmentation_without_gpu_jobs() {
        let files = vec![
            "tests/a.slang.0 (vk)".to_string(),
            "tests/b.slang.0 (cpu)".to_string(),
        ];
        
        let predictions = HashMap::new();
        
        // Without gpu_jobs, all batches should be Mixed
        let batches = WorkPool::build_batches(
            files,
            &predictions,
            false,
            100,
            10.0,
            None,
        );
        
        for batch in &batches {
            assert_eq!(batch.kind, BatchKind::Mixed, "Without gpu_jobs, all batches should be Mixed");
        }
    }
}
