use rand::Rng;
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
pub const EMA_OLD_WEIGHT: f64 = 0.3;
pub const BATCH_TIMEOUT_SECS: u64 = 300;
pub const OUTPUT_TRUNCATE_LINES: usize = 30;

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

/// Timing cache stores per-test durations.
/// Keys are test identifiers like "tests/foo.slang" or "tests/foo.slang.4" (with variant suffix).
/// The variant suffix is included when there are multiple tests from the same file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingCache {
    pub version: u32,
    /// Map from test identifier (with optional variant suffix) to duration in seconds
    pub timings: HashMap<String, f64>,
}

impl TimingCache {
    pub fn load() -> Self {
        if let Some(state_dir) = get_state_dir() {
            let path = state_dir.join("timing.json");
            if let Ok(contents) = std::fs::read_to_string(&path) {
                // Try to parse as new format (version 2)
                if let Ok(cache) = serde_json::from_str::<TimingCache>(&contents) {
                    if cache.version >= 2 {
                        return cache;
                    }
                }
                // Old format or version 1 - just start fresh
            }
        }
        Self {
            version: 2,
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

    /// Record a test's duration. Uses EMA to smooth out variations.
    pub fn record(&mut self, test_id: &str, duration: f64) {
        let existing = self.timings.entry(test_id.to_string()).or_insert(0.0);
        if *existing == 0.0 {
            *existing = duration;
        } else {
            *existing = duration * EMA_NEW_WEIGHT + *existing * EMA_OLD_WEIGHT;
        }
    }

    /// Predict duration for a test. Returns DEFAULT_PREDICTED_DURATION if unknown.
    pub fn predict(&self, test_id: &str) -> f64 {
        self.timings.get(test_id).copied().unwrap_or(DEFAULT_PREDICTED_DURATION)
    }

    /// Merge observed timings into the cache
    pub fn merge(&mut self, observed: &HashMap<String, f64>) {
        for (test_id, duration) in observed {
            self.record(test_id, *duration);
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

pub struct WorkPool {
    files: Mutex<Vec<String>>,
    max_batch_size: usize,
    num_workers: usize,
    /// Predictions keyed by test identifier (e.g., "tests/foo.slang.4")
    pub predictions: HashMap<String, f64>,
    pub total_predicted: f64,
    pub has_timing_data: bool,
    target_batch_duration: f64,
}

impl WorkPool {
    pub fn new(
        files: Vec<String>,
        max_batch_size: usize,
        num_workers: usize,
        predictions: HashMap<String, f64>,
        has_timing_data: bool,
        target_batch_duration: f64,
    ) -> Self {
        let total_predicted: f64 = files.iter()
            .map(|f| predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
            .sum();
        Self {
            files: Mutex::new(files),
            max_batch_size,
            num_workers,
            predictions,
            total_predicted,
            has_timing_data,
            target_batch_duration,
        }
    }

    pub fn predicted_remaining(&self) -> f64 {
        let files = self.files.lock().unwrap();
        files.iter()
            .map(|f| self.predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION))
            .sum()
    }

    pub fn add_file(&self, file: String) {
        self.files.lock().unwrap().push(file);
    }

    pub fn is_empty(&self) -> bool {
        self.files.lock().unwrap().is_empty()
    }

    pub fn remaining(&self) -> usize {
        self.files.lock().unwrap().len()
    }

    fn select_slow_biased_index(&self, files: &[String]) -> usize {
        if files.len() <= 2 {
            return 0;
        }

        if self.has_timing_data {
            let mut indexed: Vec<(usize, f64)> = files.iter().enumerate()
                .map(|(i, f)| {
                    let dur = self.predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                    (i, dur)
                })
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let n = indexed.len();
            let decay = 3.0 / n as f64;
            let weights: Vec<f64> = (0..n)
                .map(|i| (-decay * i as f64).exp())
                .collect();

            let total_weight: f64 = weights.iter().sum();
            let mut rng = rand::thread_rng();
            let mut choice = rng.gen_range(0.0..total_weight);

            for (i, &weight) in weights.iter().enumerate() {
                choice -= weight;
                if choice <= 0.0 {
                    return indexed[i].0;
                }
            }
            indexed[n - 1].0
        } else {
            let mut rng = rand::thread_rng();
            rng.gen_range(0..files.len())
        }
    }

    pub fn try_get_batch(&self) -> Option<Vec<String>> {
        let mut files = self.files.lock().unwrap();
        if files.is_empty() {
            return None;
        }

        let remaining = files.len();

        if remaining <= self.num_workers * 2 {
            return Some(vec![files.pop().unwrap()]);
        }

        if self.has_timing_data {
            let mut batch = Vec::new();
            let mut batch_duration = 0.0;

            while !files.is_empty() {
                let idx = self.select_slow_biased_index(&files);
                let file = &files[idx];
                let file_duration = self.predictions.get(file).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);

                if batch.is_empty() {
                    batch.push(files.swap_remove(idx));
                    batch_duration += file_duration;
                } else if batch_duration + file_duration <= self.target_batch_duration
                          && batch.len() < self.max_batch_size {
                    batch.push(files.swap_remove(idx));
                    batch_duration += file_duration;
                } else {
                    break;
                }
            }

            if batch.is_empty() { None } else { Some(batch) }
        } else {
            let batch_size = self.max_batch_size.min(remaining);
            let mut rng = rand::thread_rng();

            let mut batch = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                if files.is_empty() {
                    break;
                }
                let idx = rng.gen_range(0..files.len());
                batch.push(files.swap_remove(idx));
            }

            if batch.is_empty() { None } else { Some(batch) }
        }
    }

    pub fn try_get_medium_batch(&self) -> Option<Vec<String>> {
        let mut files = self.files.lock().unwrap();
        if files.is_empty() {
            return None;
        }

        let idx = self.select_slow_biased_index(&files);
        Some(vec![files.swap_remove(idx)])
    }
}

// ============================================================================
// Progress Display
// ============================================================================

pub struct ProgressDisplay {
    total_files: usize,
    total_predicted_time: f64,
    start_time: Instant,
    machine_output: bool,
    last_reported_files: AtomicUsize,
}

impl ProgressDisplay {
    pub fn new(total_files: usize, total_predicted_time: f64, machine_output: bool) -> Self {
        Self {
            total_files,
            total_predicted_time,
            start_time: Instant::now(),
            machine_output,
            last_reported_files: AtomicUsize::new(0),
        }
    }

    pub fn update(&self, stats: &TestStats, files_completed: usize, batches_running: usize, batches_remaining: usize, predicted_remaining_time: Option<f64>, adaptive_running: usize) {
        let passed = stats.passed.load(Ordering::SeqCst);
        let failed = stats.failed.load(Ordering::SeqCst);
        let ignored = stats.ignored.load(Ordering::SeqCst);
        let _total_tests = passed + failed + ignored;
        let elapsed = self.start_time.elapsed().as_secs_f64();

        if self.machine_output {
            let last_files = self.last_reported_files.load(Ordering::SeqCst);
            let report_interval = (self.total_files / 10).max(1);
            if files_completed >= last_files + report_interval {
                self.last_reported_files.store(files_completed, Ordering::SeqCst);
                let turbo = if adaptive_running > 0 { format!(" +{} turbo", adaptive_running) } else { String::new() };
                eprintln!(
                    "[{}/{}] {} passed, {} failed, {} ignored ({:.1}s) [{}/{}]{}",
                    files_completed, self.total_files, passed, failed, ignored, elapsed,
                    batches_running, batches_remaining, turbo
                );
            }
        } else {
            let percent = if let Some(remaining) = predicted_remaining_time {
                if self.total_predicted_time > 0.0 {
                    let completed_time = self.total_predicted_time - remaining;
                    (completed_time / self.total_predicted_time * 100.0).clamp(0.0, 100.0)
                } else {
                    (files_completed as f64 / self.total_files.max(1) as f64) * 100.0
                }
            } else {
                (files_completed as f64 / self.total_files.max(1) as f64) * 100.0
            };

            let eta = if let Some(predicted) = predicted_remaining_time {
                if predicted > 1.0 && batches_running > 0 {
                    let parallel_eta = predicted / batches_running.max(1) as f64;
                    format!(" ETA: {:.0}s", parallel_eta)
                } else if files_completed < self.total_files {
                    format!(" ETA: <1s")
                } else {
                    String::new()
                }
            } else if files_completed > 5 && files_completed < self.total_files {
                let rate = files_completed as f64 / elapsed;
                let remaining = self.total_files - files_completed;
                let eta_secs = remaining as f64 / rate;
                format!(" ETA: {:.0}s", eta_secs)
            } else {
                String::new()
            };

            let compiling_info = if let Some(secs) = stats.get_compiling_time() {
                format!(" COMPILING({:.0}s)", secs)
            } else {
                String::new()
            };

            let stuck_info = if let Some(secs) = stats.seconds_since_last_output() {
                if secs > 5.0 && batches_running > 0 {
                    format!(" [no output {:.0}s]", secs)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let turbo_info = if adaptive_running > 0 {
                format!(" \x1b[33m+{} turbo\x1b[0m", adaptive_running)
            } else {
                String::new()
            };

            eprint!(
                "\r\x1b[K[{:>5}/{:<5}] {:>5.1}% | {} passed, {} failed, {} ignored ({:.1}s){} [{}/{}]{}{}{}",
                files_completed, self.total_files, percent, passed, failed, ignored, elapsed, eta,
                batches_running, batches_remaining, turbo_info, compiling_info, stuck_info
            );
            let _ = std::io::stderr().flush();
        }
    }
}

// ============================================================================
// Batch Context
// ============================================================================

pub struct BatchContext<'a> {
    pub slang_test: &'a PathBuf,
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

    pub fn refresh_and_log(&mut self, running: usize, adaptive: usize, pool_remaining: usize) {
        self.sys.refresh_cpu_all();

        let load = System::load_average();
        let cpu_usage: f32 = self.sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
            / self.sys.cpus().len().max(1) as f32;

        let gpu_usage = get_gpu_usage();

        let gpu_str = gpu_usage.map(|g| format!(" gpu={}%", g)).unwrap_or_default();

        log_event("stats", &format!(
            "load_1m={:.2} load_5m={:.2} cpu_avg={:.1}%{} running={} adaptive={} pool={}",
            load.one, load.five, cpu_usage, gpu_str, running, adaptive, pool_remaining
        ));
    }
}

impl Default for SystemStats {
    fn default() -> Self {
        Self::new()
    }
}

fn get_gpu_usage() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                if let Ok(s) = String::from_utf8(output.stdout) {
                    return s.lines()
                        .filter_map(|line| line.trim().parse::<u32>().ok())
                        .max();
                }
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Get instantaneous CPU usage as a load-like value.
/// Returns the number of "busy CPUs" (e.g., 50% usage on 8 cores = 4.0)
pub fn get_instantaneous_load() -> Option<f64> {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_cpu(CpuRefreshKind::everything())
    );
    // Need to refresh twice with a small delay to get accurate readings
    sys.refresh_cpu_all();
    std::thread::sleep(std::time::Duration::from_millis(200));
    sys.refresh_cpu_all();

    let num_cpus = sys.cpus().len();
    if num_cpus == 0 {
        return None;
    }

    // Total CPU usage as percentage (0-100 per core)
    let total_usage: f32 = sys.cpus().iter().map(|c| c.cpu_usage()).sum();
    // Convert to load-like value: 100% on all cores = num_cpus
    Some((total_usage / 100.0) as f64)
}

