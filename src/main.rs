use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use rand::Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sysinfo::{System, CpuRefreshKind, RefreshKind};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, IsTerminal, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use walkdir::WalkDir;

// Constants
const DEFAULT_PREDICTED_DURATION: f64 = 0.5;
const EMA_NEW_WEIGHT: f64 = 0.7;
const EMA_OLD_WEIGHT: f64 = 0.3;
const BATCH_TIMEOUT_SECS: u64 = 300;
const OUTPUT_TRUNCATE_LINES: usize = 30;

/// Event logger for performance debugging
#[derive(Debug)]
struct EventLog {
    writer: Mutex<BufWriter<File>>,
    start_time: Instant,
}

impl EventLog {
    fn new(path: &PathBuf) -> Result<Self> {
        let file = File::create(path).context("Failed to create event log file")?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "timestamp_ms,event,details")?;
        Ok(Self {
            writer: Mutex::new(writer),
            start_time: Instant::now(),
        })
    }

    fn log(&self, event: &str, details: &str) {
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

    fn flush(&self) {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.flush();
        }
    }
}

/// Global event log (set once at startup if --event-log is provided)
static EVENT_LOG: LazyLock<Mutex<Option<Arc<EventLog>>>> = LazyLock::new(|| Mutex::new(None));

fn log_event(event: &str, details: &str) {
    if let Ok(guard) = EVENT_LOG.lock() {
        if let Some(ref log) = *guard {
            log.log(event, details);
        }
    }
}

fn init_event_log(path: &PathBuf) -> Result<()> {
    let log = EventLog::new(path)?;
    if let Ok(mut guard) = EVENT_LOG.lock() {
        *guard = Some(Arc::new(log));
    }
    Ok(())
}

fn flush_event_log() {
    if let Ok(guard) = EVENT_LOG.lock() {
        if let Some(ref log) = *guard {
            log.flush();
        }
    }
}

/// Get the state directory path (~/.local/state/slang-test-runner or XDG equivalent)
fn get_state_dir() -> Option<PathBuf> {
    dirs::state_dir()
        .or_else(|| dirs::data_local_dir()) // Fallback for macOS/Windows
        .map(|p| p.join("slang-test-runner"))
}

/// Timing cache for test files, stored per (file, backend)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TimingCache {
    version: u32,
    /// Map from test file path to backend -> duration in seconds
    /// Backend can be "vk", "cpu", "llvm", "cuda", etc., or "_none" for tests without backend
    timings: HashMap<String, HashMap<String, f64>>,
}

impl TimingCache {
    fn load() -> Self {
        if let Some(state_dir) = get_state_dir() {
            let path = state_dir.join("timing.json");
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(cache) = serde_json::from_str(&contents) {
                    return cache;
                }
            }
        }
        Self {
            version: 1,
            timings: HashMap::new(),
        }
    }

    fn save(&self) {
        if let Some(state_dir) = get_state_dir() {
            // Ensure state directory exists
            let _ = std::fs::create_dir_all(&state_dir);
            let path = state_dir.join("timing.json");
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = std::fs::write(&path, json);
            }
        }
    }

    /// Record timing for a specific test (file + backend)
    fn record(&mut self, file: &str, backend: &str, duration: f64) {
        let file_entry = self.timings.entry(file.to_string()).or_default();
        // Use exponential moving average to smooth out variance
        let existing = file_entry.entry(backend.to_string()).or_insert(0.0);
        if *existing == 0.0 {
            *existing = duration;
        } else {
            // EMA: adapts quickly but smooths outliers
            *existing = duration * EMA_NEW_WEIGHT + *existing * EMA_OLD_WEIGHT;
        }
    }

    /// Predict total duration for a file given the API filter
    /// If include_startup is true, adds startup overhead (for ETA).
    /// If false, excludes startup (for scheduling decisions).
    fn predict(&self, file: &str, api_filter: Option<&str>, include_startup: bool) -> f64 {
        if let Some(backends) = self.timings.get(file) {
            let base_time = match api_filter {
                Some(api) => {
                    // Only count timing for the specified API
                    backends.get(api).copied().unwrap_or(DEFAULT_PREDICTED_DURATION)
                }
                None => {
                    // Prefer _total (wall-clock time) as it includes compilation and overhead
                    // Fall back to summing individual backends if _total not available
                    if let Some(&total_time) = backends.get("_total") {
                        total_time
                    } else {
                        let total: f64 = backends.iter()
                            .filter(|(k, _)| *k != "_startup")
                            .map(|(_, v)| *v)
                            .sum();
                        if total > 0.0 { total } else { DEFAULT_PREDICTED_DURATION }
                    }
                }
            };

            // Add startup overhead if requested (for ETA estimation)
            // Excluded for scheduling decisions (batching, LPT sort)
            if include_startup {
                let startup = backends.get("_startup").copied().unwrap_or(0.0);
                base_time + startup
            } else {
                base_time
            }
        } else {
            // Unknown file, use default estimate
            DEFAULT_PREDICTED_DURATION
        }
    }

    /// Merge observed timings into this cache
    fn merge(&mut self, observed: &HashMap<String, HashMap<String, f64>>) {
        for (file, backends) in observed {
            for (backend, duration) in backends {
                self.record(file, backend, *duration);
            }
        }
    }
}

/// Extract the backend from a test name like "tests/foo.slang.1 (vk)" -> Some("vk")
fn extract_backend(test_name: &str) -> Option<String> {
    if let Some(start) = test_name.rfind('(') {
        if let Some(end) = test_name.rfind(')') {
            if start < end {
                return Some(test_name[start + 1..end].to_string());
            }
        }
    }
    None
}

/// Extract API filter from extra_args (e.g., ["-api", "vk"] -> Some("vk"))
fn extract_api_filter(extra_args: &[String]) -> Option<String> {
    let mut iter = extra_args.iter();
    while let Some(arg) = iter.next() {
        if arg == "-api" {
            if let Some(api) = iter.next() {
                return Some(api.clone());
            }
        }
    }
    None
}

/// Get the 1-minute load average (cross-platform via sysinfo)
fn get_load_average() -> Option<f64> {
    let load = System::load_average();
    // load_average returns LoadAvg { one, five, fifteen }
    // Use the 1-minute average
    if load.one > 0.0 {
        Some(load.one)
    } else {
        None
    }
}

/// System stats for logging
struct SystemStats {
    sys: System,
}

impl SystemStats {
    fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::new().with_cpu(CpuRefreshKind::everything())
        );
        Self { sys }
    }

    fn refresh_and_log(&mut self, running: usize, adaptive: usize, pool_remaining: usize) {
        self.sys.refresh_cpu_all();

        let load = System::load_average();
        let cpu_usage: f32 = self.sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
            / self.sys.cpus().len().max(1) as f32;

        // Try to get GPU usage via nvidia-smi (Linux)
        let gpu_usage = get_gpu_usage();

        let gpu_str = gpu_usage.map(|g| format!(" gpu={}%", g)).unwrap_or_default();

        log_event("stats", &format!(
            "load_1m={:.2} load_5m={:.2} cpu_avg={:.1}%{} running={} adaptive={} pool={}",
            load.one, load.five, cpu_usage, gpu_str, running, adaptive, pool_remaining
        ));
    }
}

/// Try to get GPU usage percentage (nvidia-smi on Linux)
fn get_gpu_usage() -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        // Try nvidia-smi first
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                if let Ok(s) = String::from_utf8(output.stdout) {
                    // May have multiple GPUs, take max
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

/// Global flag to indicate if Ctrl-C was pressed
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Global flag to indicate core module is being compiled - other workers should wait
static COMPILING_CORE: AtomicBool = AtomicBool::new(false);

fn is_compiling_core() -> bool {
    COMPILING_CORE.load(Ordering::SeqCst)
}

fn set_compiling_core(compiling: bool) {
    COMPILING_CORE.store(compiling, Ordering::SeqCst);
}

fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

fn set_interrupted() {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

#[derive(Parser, Debug)]
#[command(name = "slang-test-runner")]
#[command(about = "A parallel test runner for slang-test with better output")]
struct Args {
    /// Root directory of the slang project (defaults to current directory)
    #[arg(short = 'C', long, default_value = ".")]
    root_dir: PathBuf,

    /// Path to slang-test executable (relative to root_dir or absolute)
    #[arg(long, default_value = "build/Debug/bin/slang-test")]
    slang_test: PathBuf,

    /// Test directory (relative to root_dir or absolute)
    #[arg(long, default_value = "tests")]
    test_dir: PathBuf,

    /// Number of parallel workers
    #[arg(short = 'j', long, default_value_t = num_cpus())]
    jobs: usize,

    /// Maximum files per batch (with timing data, batches target ~10s duration up to this limit)
    #[arg(long, default_value_t = 100)]
    batch_size: usize,

    /// Target batch duration in seconds (only used with timing cache)
    #[arg(long, default_value_t = 10.0)]
    batch_duration: f64,

    /// Number of retries for failed tests
    #[arg(long, default_value_t = 2)]
    retries: usize,

    /// Test prefixes to run (if empty, runs all tests)
    #[arg()]
    prefixes: Vec<String>,

    /// Hide ignored tests from output
    #[arg(long)]
    hide_ignored: bool,

    /// Test patterns to ignore (can be specified multiple times)
    #[arg(long = "ignore")]
    ignore_patterns: Vec<String>,

    /// Diff tool for showing expected/actual differences: none, diff, difft (default: diff)
    #[arg(long, default_value = "diff")]
    diff: String,

    /// Additional arguments to pass to slang-test
    #[arg(last = true)]
    extra_args: Vec<String>,

    /// Verbose output: show batch reproduction commands for slow batches
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Ignore timing cache (don't use cached timing for scheduling or ETA)
    #[arg(long)]
    no_timing_cache: bool,

    /// Adaptive load balancing: spawn extra small batches when CPU is underutilized
    #[arg(long)]
    adaptive: bool,

    /// Write event log to file for performance debugging
    #[arg(long)]
    event_log: Option<PathBuf>,
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Check if stderr is a TTY (interactive terminal)
fn is_stderr_tty() -> bool {
    std::io::stderr().is_terminal()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TestResult {
    Passed,
    Failed,
    Ignored,
}

#[derive(Debug, Clone)]
struct TestOutcome {
    name: String,
    result: TestResult,
    duration: Option<Duration>,
    failure_output: Vec<String>,
}

#[derive(Debug, Default)]
struct TestStats {
    passed: AtomicUsize,
    failed: AtomicUsize,
    ignored: AtomicUsize,
    retried_and_passed: AtomicUsize,
    files_seen: Mutex<HashSet<String>>,
    file_durations: Mutex<HashMap<String, f64>>,
    last_test_output: Mutex<Option<Instant>>,
    compiling_since: Mutex<Option<Instant>>,
    /// Observed timings per (file, backend) for cache updates
    observed_timings: Mutex<HashMap<String, HashMap<String, f64>>>,
}

impl TestStats {
    fn record_file(&self, file: &str) {
        self.files_seen.lock().unwrap().insert(file.to_string());
    }

    fn record_duration(&self, file: &str, duration: Duration) {
        let secs = duration.as_secs_f64();
        let mut durations = self.file_durations.lock().unwrap();
        *durations.entry(file.to_string()).or_insert(0.0) += secs;
    }

    fn files_completed(&self) -> usize {
        self.files_seen.lock().unwrap().len()
    }

    fn slowest_files(&self, n: usize) -> Vec<(String, f64)> {
        let durations = self.file_durations.lock().unwrap();
        let mut sorted: Vec<_> = durations.iter().map(|(k, v)| (k.clone(), *v)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    fn set_compiling(&self, is_compiling: bool) {
        let mut compiling = self.compiling_since.lock().unwrap();
        if is_compiling {
            if compiling.is_none() {
                *compiling = Some(Instant::now());
            }
        } else {
            *compiling = None;
        }
    }

    fn get_compiling_time(&self) -> Option<f64> {
        self.compiling_since.lock().unwrap().map(|t| t.elapsed().as_secs_f64())
    }

    fn record_test_output(&self) {
        *self.last_test_output.lock().unwrap() = Some(Instant::now());
    }

    fn seconds_since_last_output(&self) -> Option<f64> {
        self.last_test_output.lock().unwrap().map(|t| t.elapsed().as_secs_f64())
    }

    /// Record observed timing for a (file, backend) pair
    /// For individual backends, we replace (last measurement wins)
    /// For _total, we accumulate (sum of all test times in the file)
    fn record_observed_timing(&self, file: &str, backend: &str, duration: f64) {
        let mut observed = self.observed_timings.lock().unwrap();
        let file_entry = observed.entry(file.to_string()).or_default();
        if backend == "_total" {
            // _total accumulates (it's sent once per file transition)
            *file_entry.entry(backend.to_string()).or_insert(0.0) = duration;
        } else {
            // Individual backend: replace (handles retries correctly)
            file_entry.insert(backend.to_string(), duration);
        }
    }

    /// Get all observed timings for cache update
    fn get_observed_timings(&self) -> HashMap<String, HashMap<String, f64>> {
        self.observed_timings.lock().unwrap().clone()
    }
}

/// Stores detailed failure information for pretty printing at the end
#[derive(Debug, Clone)]
struct FailureInfo {
    test_name: String,
    output_lines: Vec<String>,
    expected: Option<String>,
    actual: Option<String>,
}

/// Dynamic work pool that adaptively sizes batches
struct WorkPool {
    files: Mutex<Vec<PathBuf>>,
    max_batch_size: usize,
    num_workers: usize,
    /// Predicted duration per file (for scheduling, excludes startup)
    predictions: HashMap<String, f64>,
    /// Startup overhead per file (for ETA calculation)
    startups: HashMap<String, f64>,
    /// Total predicted time for all files (including startup)
    total_predicted: f64,
    /// Whether we have real timing data (affects batching strategy)
    has_timing_data: bool,
    /// Target batch duration in seconds
    target_batch_duration: f64,
}

impl WorkPool {
    fn new(
        files: Vec<PathBuf>,
        max_batch_size: usize,
        num_workers: usize,
        predictions: HashMap<String, f64>,
        startups: HashMap<String, f64>,
        has_timing_data: bool,
        target_batch_duration: f64,
    ) -> Self {
        let total_predicted: f64 = files.iter()
            .map(|f| {
                let key = f.to_string_lossy().to_string();
                let base = predictions.get(&key).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                let startup = startups.get(&key).copied().unwrap_or(0.0);
                base + startup
            })
            .sum();
        Self {
            files: Mutex::new(files),
            max_batch_size,
            num_workers,
            predictions,
            startups,
            total_predicted,
            has_timing_data,
            target_batch_duration,
        }
    }

    /// Get predicted remaining time based on files in queue (includes startup for ETA)
    fn predicted_remaining(&self) -> f64 {
        let files = self.files.lock().unwrap();
        files.iter()
            .map(|f| {
                let key = f.to_string_lossy().to_string();
                let base = self.predictions.get(&key).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                let startup = self.startups.get(&key).copied().unwrap_or(0.0);
                base + startup
            })
            .sum()
    }

    /// Add a file back to the pool (for retries)
    fn add_file(&self, file: PathBuf) {
        self.files.lock().unwrap().push(file);
    }

    /// Check if there's any work left
    fn is_empty(&self) -> bool {
        self.files.lock().unwrap().is_empty()
    }

    /// Get count of remaining files
    fn remaining(&self) -> usize {
        self.files.lock().unwrap().len()
    }

    /// Select an index using slow-biased weighted random selection
    /// Favors slower tests (LPT-style) but with stochastic mixing to avoid
    /// having ALL slow tests bunch at the start
    fn select_slow_biased_index(&self, files: &[PathBuf]) -> usize {
        if files.len() <= 2 {
            return 0;
        }

        if self.has_timing_data {
            // Sort by predicted duration descending (slowest first)
            let mut indexed: Vec<(usize, f64)> = files.iter().enumerate()
                .map(|(i, f)| {
                    let key = f.to_string_lossy().to_string();
                    let dur = self.predictions.get(&key).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                    (i, dur)
                })
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // One-sided exponential decay: strongly favor slow tests (index 0)
            // but occasionally pick faster ones
            // Weight = e^(-index * decay_factor)
            let n = indexed.len();
            let decay = 3.0 / n as f64; // Decay such that last item has ~5% weight of first
            let weights: Vec<f64> = (0..n)
                .map(|i| (-decay * i as f64).exp())
                .collect();

            // Weighted random selection
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
            // Without timing data, use uniform random
            let mut rng = rand::thread_rng();
            rng.gen_range(0..files.len())
        }
    }

    /// Atomically check if pool is empty AND get a batch if not
    /// Returns None only if pool is truly empty (no race with add_file)
    fn try_get_batch(&self) -> Option<Vec<PathBuf>> {
        let mut files = self.files.lock().unwrap();
        if files.is_empty() {
            return None;
        }

        let remaining = files.len();

        // Final stretch: single files for max parallelism
        if remaining <= self.num_workers * 2 {
            return Some(vec![files.pop().unwrap()]);
        }

        if self.has_timing_data {
            // With timing data: use duration-based batching with middle-biased selection
            let mut batch = Vec::new();
            let mut batch_duration = 0.0;

            while !files.is_empty() {
                // Select next file with slow-bias (LPT with stochastic mixing)
                let idx = self.select_slow_biased_index(&files);
                let file = &files[idx];
                let file_key = file.to_string_lossy().to_string();
                let file_duration = self.predictions.get(&file_key).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);

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
            // Without timing data: random sampling to spread slow tests
            let batch_size = self.max_batch_size.min(remaining);
            let mut rng = rand::thread_rng();

            let mut batch = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                if files.is_empty() {
                    break;
                }
                // Pick a random index and swap_remove for O(1)
                let idx = rng.gen_range(0..files.len());
                batch.push(files.swap_remove(idx));
            }

            if batch.is_empty() { None } else { Some(batch) }
        }
    }

    /// Get a batch for adaptive load balancing (turbo mode)
    /// Uses the same slow-biased selection as regular batches
    fn try_get_medium_batch(&self) -> Option<Vec<PathBuf>> {
        let mut files = self.files.lock().unwrap();
        if files.is_empty() {
            return None;
        }

        let idx = self.select_slow_biased_index(&files);
        Some(vec![files.swap_remove(idx)])
    }
}

fn discover_test_files(
    test_dir: &PathBuf,
    prefixes: &[String],
    ignore_patterns: &[String],
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let extensions = ["slang", "hlsl", "glsl"];

    for entry in WalkDir::new(test_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if extensions.iter().any(|e| ext == *e) {
                    let path_str = path.to_string_lossy();

                    // Check ignore patterns
                    if ignore_patterns.iter().any(|p| path_str.contains(p)) {
                        continue;
                    }

                    if prefixes.is_empty() || prefixes.iter().any(|p| path_str.contains(p)) {
                        files.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

// Static regexes for test output parsing (compiled once)
static PASSED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^passed test: '([^']+)'(?: (\d+\.?\d*)s)?").unwrap()
});
static FAILED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^FAILED test: '([^']+)'").unwrap()
});
static IGNORED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^ignored test: '([^']+)'").unwrap()
});
static BASE_FILE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(.+\.(slang|hlsl|glsl))(\.\d+)?$").unwrap()
});

fn parse_test_output(line: &str) -> Option<TestOutcome> {
    if let Some(caps) = PASSED_RE.captures(line) {
        let name = caps.get(1).unwrap().as_str().to_string();
        let duration = caps.get(2).map(|m| {
            let secs: f64 = m.as_str().parse().unwrap_or(0.0);
            Duration::from_secs_f64(secs)
        });
        return Some(TestOutcome {
            name,
            result: TestResult::Passed,
            duration,
            failure_output: vec![],
        });
    }

    if let Some(caps) = FAILED_RE.captures(line) {
        let name = caps.get(1).unwrap().as_str().to_string();
        return Some(TestOutcome {
            name,
            result: TestResult::Failed,
            duration: None,
            failure_output: vec![],
        });
    }

    if let Some(caps) = IGNORED_RE.captures(line) {
        let name = caps.get(1).unwrap().as_str().to_string();
        return Some(TestOutcome {
            name,
            result: TestResult::Ignored,
            duration: None,
            failure_output: vec![],
        });
    }

    None
}

/// Parse failure output to extract expected/actual for diff display
fn parse_failure_info(test_name: &str, lines: &[String]) -> FailureInfo {
    let mut expected: Option<String> = None;
    let mut actual: Option<String> = None;
    let mut in_expected = false;
    let mut in_actual = false;
    let mut expected_lines = Vec::new();
    let mut actual_lines = Vec::new();

    for line in lines {
        // Look for expected/actual output markers (slang-test uses EXPECTED{{{ and ACTUAL{{{)
        if line.contains("EXPECTED{{{") || line.contains("expected output =") || line.contains("expected =") {
            in_expected = true;
            in_actual = false;
            continue;
        }
        if line.contains("ACTUAL{{{") || line.contains("actual output =") || line.contains("actual =") {
            in_actual = true;
            in_expected = false;
            continue;
        }

        // End markers
        if line.contains("}}}") {
            if in_expected {
                in_expected = false;
                expected = Some(expected_lines.join("\n"));
            } else if in_actual {
                in_actual = false;
                actual = Some(actual_lines.join("\n"));
            }
            continue;
        }

        if in_expected {
            expected_lines.push(line.clone());
        } else if in_actual {
            actual_lines.push(line.clone());
        }
    }

    FailureInfo {
        test_name: test_name.to_string(),
        output_lines: lines.to_vec(),
        expected,
        actual,
    }
}

fn extract_base_test_file(test_name: &str) -> Option<String> {
    let name = test_name.split(" (").next().unwrap_or(test_name);
    let name = name.trim_end_matches(" syn");

    if name.contains(".internal") {
        return Some(name.to_string());
    }

    if let Some(caps) = BASE_FILE_RE.captures(name) {
        return Some(caps.get(1).unwrap().as_str().to_string());
    }

    Some(name.to_string())
}

/// Progress display that updates in place
struct ProgressDisplay {
    total_files: usize,
    total_predicted_time: f64,
    start_time: Instant,
    machine_output: bool,
    last_reported_files: AtomicUsize,
}

impl ProgressDisplay {
    fn new(total_files: usize, total_predicted_time: f64, machine_output: bool) -> Self {
        Self {
            total_files,
            total_predicted_time,
            start_time: Instant::now(),
            machine_output,
            last_reported_files: AtomicUsize::new(0),
        }
    }

    fn update(&self, stats: &TestStats, files_completed: usize, batches_running: usize, batches_remaining: usize, predicted_remaining_time: Option<f64>, adaptive_running: usize) {
        let passed = stats.passed.load(Ordering::SeqCst);
        let failed = stats.failed.load(Ordering::SeqCst);
        let ignored = stats.ignored.load(Ordering::SeqCst);
        let _total_tests = passed + failed + ignored;
        let elapsed = self.start_time.elapsed().as_secs_f64();

        if self.machine_output {
            // In machine mode, report every 10% of files completed
            let last_files = self.last_reported_files.load(Ordering::SeqCst);
            let report_interval = (self.total_files / 10).max(1); // Report every 10%, at least every file
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
            // Calculate percentage: prefer time-based if we have timing data
            let percent = if let Some(remaining) = predicted_remaining_time {
                if self.total_predicted_time > 0.0 {
                    // Time-based: (total - remaining) / total * 100
                    let completed_time = self.total_predicted_time - remaining;
                    (completed_time / self.total_predicted_time * 100.0).clamp(0.0, 100.0)
                } else {
                    // Fall back to file-based
                    (files_completed as f64 / self.total_files.max(1) as f64) * 100.0
                }
            } else {
                // No timing data, use file-based
                (files_completed as f64 / self.total_files.max(1) as f64) * 100.0
            };

            // Use predicted remaining time if available, otherwise fall back to rate-based ETA
            let eta = if let Some(predicted) = predicted_remaining_time {
                if predicted > 1.0 && batches_running > 0 {
                    // Scale by parallelism (rough estimate)
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

            // Check if any batch is compiling
            let compiling_info = if let Some(secs) = stats.get_compiling_time() {
                format!(" COMPILING({:.0}s)", secs)
            } else {
                String::new()
            };

            // Show if we haven't seen output in a while (stuck?)
            let stuck_info = if let Some(secs) = stats.seconds_since_last_output() {
                if secs > 5.0 && batches_running > 0 {
                    format!(" [no output {:.0}s]", secs)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            // Show turbo mode indicator
            let turbo_info = if adaptive_running > 0 {
                format!(" \x1b[33m+{} turbo\x1b[0m", adaptive_running)  // Yellow
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

/// Context for batch execution - bundles related parameters
struct BatchContext<'a> {
    slang_test: &'a PathBuf,
    test_files: &'a [PathBuf],
    extra_args: &'a [String],
    timeout: Duration,
    stats: &'a TestStats,
    failures: &'a Mutex<Vec<FailureInfo>>,
    max_retries: usize,
    retried_tests: &'a Mutex<HashSet<String>>,
    work_pool: &'a Arc<WorkPool>,
    running: &'a AtomicUsize,
    machine_output: bool,
    verbose: bool,
}

/// Determines if and how a failed test should be retried
fn should_retry_test(
    outcome: &TestOutcome,
    max_retries: usize,
    retried_tests: &Mutex<HashSet<String>>,
    work_pool: &Arc<WorkPool>,
) -> bool {
    if max_retries == 0 {
        return false;
    }

    let should_retry = {
        let mut retried = retried_tests.lock().unwrap();
        if retried.contains(&outcome.name) {
            false // Already retried
        } else {
            retried.insert(outcome.name.clone());
            true
        }
    };

    if should_retry {
        if let Some(base_file) = extract_base_test_file(&outcome.name) {
            let file_path = PathBuf::from(&base_file);
            if file_path.exists() {
                work_pool.add_file(file_path);
                return true;
            }
        }
    }
    false
}

/// Process a single test outcome, updating stats and handling retries
/// Returns true if the outcome was a retry (and should be skipped for failure recording)
fn process_outcome(
    outcome: TestOutcome,
    ctx: &BatchContext,
    failed_outcomes: &mut Vec<TestOutcome>,
) -> bool {
    // Record the file as seen and its duration
    if let Some(base_file) = extract_base_test_file(&outcome.name) {
        ctx.stats.record_file(&base_file);
        if let Some(duration) = outcome.duration {
            ctx.stats.record_duration(&base_file, duration);
        }
    }

    match outcome.result {
        TestResult::Passed => {
            let was_retry = ctx.retried_tests.lock().unwrap().contains(&outcome.name);
            ctx.stats.passed.fetch_add(1, Ordering::SeqCst);
            if was_retry {
                ctx.stats.retried_and_passed.fetch_add(1, Ordering::SeqCst);
            }
            false
        }
        TestResult::Ignored => {
            ctx.stats.ignored.fetch_add(1, Ordering::SeqCst);
            false
        }
        TestResult::Failed => {
            if should_retry_test(&outcome, ctx.max_retries, ctx.retried_tests, ctx.work_pool) {
                return true; // Skip recording as failure, will be retried
            }
            ctx.stats.failed.fetch_add(1, Ordering::SeqCst);
            failed_outcomes.push(outcome);
            false
        }
    }
}

/// Collect failure output from stderr lines for failed tests
fn collect_failure_output(failed_outcomes: &mut [TestOutcome], stderr_lines: &[String]) {
    for outcome in failed_outcomes.iter_mut() {
        let test_prefix = format!("[{}]", outcome.name);
        let mut capture = false;
        let mut relevant_lines: Vec<String> = Vec::new();

        for line in stderr_lines {
            if line.starts_with(&test_prefix) {
                capture = true;
            }
            if capture {
                relevant_lines.push(line.clone());
                if line.contains("}}}") && relevant_lines.iter().filter(|l| l.contains("}}}")).count() >= 2 {
                    break;
                }
            }
        }
        outcome.failure_output = relevant_lines;
    }
}

/// Run a batch with work pool - sends retries back to the pool
fn run_batch_with_pool(
    slang_test: &PathBuf,
    test_files: &[PathBuf],
    extra_args: &[String],
    timeout: Duration,
    stats: &TestStats,
    failures: &Mutex<Vec<FailureInfo>>,
    max_retries: usize,
    retried_tests: &Mutex<HashSet<String>>,
    work_pool: &Arc<WorkPool>,
    running: &AtomicUsize,
    machine_output: bool,
    verbose: bool,
) {
    let ctx = BatchContext {
        slang_test,
        test_files,
        extra_args,
        timeout,
        stats,
        failures,
        max_retries,
        retried_tests,
        work_pool,
        running,
        machine_output,
        verbose,
    };
    // Track that this batch is now running
    ctx.running.fetch_add(1, Ordering::SeqCst);
    let batch_start = Instant::now();
    let batch_id = std::thread::current().id();
    let file_info: Vec<_> = ctx.test_files.iter()
        .map(|f| {
            let name = f.file_name().unwrap_or_default().to_string_lossy().to_string();
            let key = f.to_string_lossy().to_string();
            let pred = ctx.work_pool.predictions.get(&key).copied().unwrap_or(0.0);
            format!("{}({:.2}s)", name, pred)
        })
        .collect();
    let total_pred: f64 = ctx.test_files.iter()
        .map(|f| ctx.work_pool.predictions.get(&f.to_string_lossy().to_string()).copied().unwrap_or(0.0))
        .sum();
    log_event("batch_start", &format!("{:?} files={} pred={:.2}s items=[{}]",
        batch_id, ctx.test_files.len(), total_pred, file_info.join(" ")));

    // Track sum of individual test durations for overhead calculation
    let test_time_sum = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let test_count = Arc::new(AtomicUsize::new(0));

    // Ensure we decrement running count when done
    struct Guard<'a> {
        running: &'a AtomicUsize,
        batch_id: std::thread::ThreadId,
        start: Instant,
        pred_ms: u64,
        test_time_sum: Arc<std::sync::atomic::AtomicU64>,
        test_count: Arc<AtomicUsize>,
    }
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            self.running.fetch_sub(1, Ordering::SeqCst);
            let total_ms = self.start.elapsed().as_millis() as u64;
            let sum_ms = self.test_time_sum.load(Ordering::SeqCst);
            let count = self.test_count.load(Ordering::SeqCst);
            let overhead_ms = total_ms.saturating_sub(sum_ms);
            log_event("batch_end", &format!("{:?} total_ms={} test_sum_ms={} overhead_ms={} pred_ms={} tests={}",
                self.batch_id, total_ms, sum_ms, overhead_ms, self.pred_ms, count));
        }
    }
    let _guard = Guard {
        running: ctx.running,
        batch_id,
        start: batch_start,
        pred_ms: (total_pred * 1000.0) as u64,
        test_time_sum: test_time_sum.clone(),
        test_count: test_count.clone(),
    };

    if is_interrupted() {
        return;
    }

    let mut cmd = Command::new(ctx.slang_test);
    cmd.args(ctx.test_files)
        .args(ctx.extra_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("ERROR: Failed to spawn slang-test: {}", e);
            return;
        }
    };

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let (outcome_tx, outcome_rx) = crossbeam_channel::unbounded::<TestOutcome>();
    let (stderr_tx, stderr_rx) = crossbeam_channel::unbounded::<String>();
    // Channel for file timing: (file, backend, duration_secs)
    let (timing_tx, timing_rx) = crossbeam_channel::unbounded::<(String, String, f64)>();

    // Capture machine_output for thread closures
    let machine_output_for_stderr = ctx.machine_output;
    let num_files_in_batch = ctx.test_files.len();
    let batch_start_time = Instant::now();

    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut seen_tests: HashSet<String> = HashSet::new();

        // Track timing between test output lines
        let mut last_test_time: Option<Instant> = None;
        let mut current_file: Option<String> = None;
        let mut file_start = Instant::now();
        let mut startup_time_per_file: Option<f64> = None;

        for line in reader.lines() {
            if is_interrupted() {
                break;
            }
            if let Ok(line) = line {
                // Detect startup/preamble lines
                // When we see the last preamble, calculate startup time
                if line.starts_with("Supported backends:")
                    || line.starts_with("Check ")
                    || line.starts_with("Retrying ")
                {
                    let now = Instant::now();
                    // Calculate startup time from batch start to this preamble line
                    let startup_time = batch_start_time.elapsed().as_secs_f64();
                    if num_files_in_batch > 0 {
                        startup_time_per_file = Some(startup_time / num_files_in_batch as f64);
                    }
                    last_test_time = Some(now);
                    file_start = now;  // Reset file_start to exclude startup from _total
                    continue;
                }

                if let Some(mut outcome) = parse_test_output(&line) {
                    let now = Instant::now();

                    // Calculate actual test duration from time since last test output
                    // Ignored tests should reset the timer but not record timing
                    let should_record_timing = last_test_time.is_some()
                        && outcome.result != TestResult::Ignored;

                    let test_duration = if should_record_timing {
                        outcome.duration
                            .map(|d| d.as_secs_f64())
                            .unwrap_or_else(|| last_test_time.unwrap().elapsed().as_secs_f64())
                    } else {
                        0.0
                    };

                    // Update outcome duration with measured time if not already set
                    if outcome.duration.is_none() && should_record_timing {
                        outcome.duration = Some(Duration::from_secs_f64(test_duration));
                    }

                    // Always update last_test_time (even for ignored tests)
                    // This resets the timer for the next test
                    last_test_time = Some(now);

                    if let Some(base) = extract_base_test_file(&outcome.name) {
                        // Only record timing for passed/failed tests (not ignored)
                        if should_record_timing {
                            let backend = extract_backend(&outcome.name).unwrap_or_else(|| "_none".to_string());
                            let _ = timing_tx.send((base.clone(), backend, test_duration));
                        }

                        // Detect file transition for _total timing
                        if current_file.as_ref() != Some(&base) {
                            // Finalize timing for previous file
                            if let Some(ref prev_file) = current_file {
                                let duration = file_start.elapsed().as_secs_f64();
                                let _ = timing_tx.send((prev_file.clone(), "_total".to_string(), duration));
                            }
                            // Start tracking new file
                            current_file = Some(base.clone());
                            file_start = now;
                        }

                        seen_tests.insert(base);
                    }

                    let _ = outcome_tx.send(outcome);
                }
            }
        }

        // Finalize last file
        if let Some(ref prev_file) = current_file {
            let duration = file_start.elapsed().as_secs_f64();
            let _ = timing_tx.send((prev_file.clone(), "_total".to_string(), duration));
        }

        // Record startup time for all files we saw
        if let Some(startup) = startup_time_per_file {
            for file in &seen_tests {
                let _ = timing_tx.send((file.clone(), "_startup".to_string(), startup));
            }
        }

        seen_tests
    });

    let (compiling_tx, compiling_rx) = crossbeam_channel::unbounded::<bool>();
    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                if line.contains("Compiling core module") {
                    // Signal that core module compilation is in progress
                    // Other workers should kill their processes and wait
                    set_compiling_core(true);
                    let _ = compiling_tx.send(true);
                    if machine_output_for_stderr {
                        eprintln!("INFO: {}", line);
                    }
                } else if line.contains("Compiling") {
                    let _ = compiling_tx.send(true);
                }
                let _ = stderr_tx.send(line);
            }
        }
        // Compilation done for this batch
        let _ = compiling_tx.send(false);
    });

    let mut failed_outcomes: Vec<TestOutcome> = Vec::new();
    let start = Instant::now();
    let mut this_batch_is_compiling = false;
    let mut killed_for_compilation = false;

    loop {
        if is_interrupted() {
            let _ = child.kill();
            break;
        }

        // Check for compiling signals from THIS batch
        while let Ok(is_compiling) = compiling_rx.try_recv() {
            ctx.stats.set_compiling(is_compiling);
            if is_compiling {
                this_batch_is_compiling = true;
            }
        }

        // If another batch is compiling core module and we're not the one doing it,
        // kill our process and return files to the pool
        if is_compiling_core() && !this_batch_is_compiling {
            let _ = child.kill();
            killed_for_compilation = true;
            break;
        }

        // Process timing data
        while let Ok((file, backend, duration)) = timing_rx.try_recv() {
            ctx.stats.record_observed_timing(&file, &backend, duration);
        }

        while let Ok(outcome) = outcome_rx.try_recv() {
            // Test output means compilation is done and we're making progress
            ctx.stats.set_compiling(false);
            ctx.stats.record_test_output();
            // If we were compiling core module, signal that we're done
            if this_batch_is_compiling {
                set_compiling_core(false);
                this_batch_is_compiling = false;
            }
            // Log test result and track timing
            let result_str = match outcome.result {
                TestResult::Passed => "passed",
                TestResult::Failed => "failed",
                TestResult::Ignored => "ignored",
            };
            let duration_ms = outcome.duration.map(|d| d.as_millis() as u64).unwrap_or(0);
            test_time_sum.fetch_add(duration_ms, Ordering::SeqCst);
            test_count.fetch_add(1, Ordering::SeqCst);
            log_event("test", &format!("{:?} {} {} duration_ms={}", batch_id, result_str, outcome.name, duration_ms));

            process_outcome(outcome, &ctx, &mut failed_outcomes);
        }

        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() > ctx.timeout {
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => break,
        }
    }


    let loop_time = start.elapsed();
    // Warn if loop took much longer than expected (only show repro in verbose mode)
    if ctx.verbose && loop_time.as_secs() > 30 {
        let extra_args_str = if ctx.extra_args.is_empty() {
            String::new()
        } else {
            format!(" {}", ctx.extra_args.join(" "))
        };
        eprintln!("\nWARNING: Batch took {:.1}s", loop_time.as_secs_f64());
        eprintln!("  Reproduce: time {} {}{}",
            ctx.slang_test.display(),
            ctx.test_files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" "),
            extra_args_str
        );
    }

    // Process remaining timing data
    while let Ok((file, backend, duration)) = timing_rx.try_recv() {
        ctx.stats.record_observed_timing(&file, &backend, duration);
    }

    // Process remaining outcomes
    while let Ok(outcome) = outcome_rx.try_recv() {
        process_outcome(outcome, &ctx, &mut failed_outcomes);
    }

    // If we were killed because another batch is compiling, return our files to the pool
    if killed_for_compilation {
        for file in ctx.test_files {
            ctx.work_pool.add_file(file.clone());
        }
        // Wait for compilation to finish before returning
        while is_compiling_core() && !is_interrupted() {
            thread::sleep(Duration::from_millis(50));
        }
        // Don't process results - we killed the process
        return;
    }

    let join_start = Instant::now();
    let _seen_tests = stdout_handle.join().unwrap_or_default();
    let stdout_join_time = join_start.elapsed();

    let stderr_join_start = Instant::now();
    stderr_handle.join().ok();
    let stderr_join_time = stderr_join_start.elapsed();

    // Warn if joins took too long (suggests pipe/process issue)
    if stdout_join_time.as_secs() > 5 || stderr_join_time.as_secs() > 5 {
        eprintln!("\nWARNING: Slow thread joins - stdout: {:.1}s, stderr: {:.1}s for {:?}",
            stdout_join_time.as_secs_f64(),
            stderr_join_time.as_secs_f64(),
            ctx.test_files.iter().map(|p| p.file_name().unwrap_or_default().to_string_lossy()).collect::<Vec<_>>()
        );
    }

    let stderr_lines: Vec<String> = stderr_rx.try_iter().collect();

    collect_failure_output(&mut failed_outcomes, &stderr_lines);

    for outcome in failed_outcomes {
        let info = parse_failure_info(&outcome.name, &outcome.failure_output);
        ctx.failures.lock().unwrap().push(info);
    }
}

struct TestRunner {
    args: Args,
    stats: Arc<TestStats>,
    failures: Arc<Mutex<Vec<FailureInfo>>>,
    retried_tests: Arc<Mutex<HashSet<String>>>,
    machine_output: bool,
    timing_cache: Mutex<TimingCache>,
}

impl TestRunner {
    fn new(args: Args) -> Self {
        // Use machine-friendly output if stderr is not a TTY (e.g., piped to file)
        let machine_output = !is_stderr_tty();
        let timing_cache = TimingCache::load();
        Self {
            args,
            stats: Arc::new(TestStats::default()),
            failures: Arc::new(Mutex::new(Vec::new())),
            retried_tests: Arc::new(Mutex::new(HashSet::new())),
            machine_output,
            timing_cache: Mutex::new(timing_cache),
        }
    }

    /// Save timing cache after run
    fn save_timing(&self) {
        let observed = self.stats.get_observed_timings();
        if !observed.is_empty() {
            let mut cache = self.timing_cache.lock().unwrap();
            cache.merge(&observed);
            cache.save();
        }
    }

    /// Get predicted duration for a file based on cached timing
    /// Does NOT include startup overhead (used for scheduling decisions)
    fn predict_duration(&self, file: &PathBuf) -> f64 {
        let api_filter = extract_api_filter(&self.args.extra_args);
        let cache = self.timing_cache.lock().unwrap();
        cache.predict(&file.to_string_lossy(), api_filter.as_deref(), false)
    }

    fn run(&self) -> Result<bool> {
        let start_time = Instant::now();

        let all_prefixes_are_files = !self.args.prefixes.is_empty()
            && self.args.prefixes.iter().all(|p| {
                let path = PathBuf::from(p);
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|ext| ext == "slang" || ext == "hlsl" || ext == "glsl")
            });

        let has_internal_prefix = self
            .args
            .prefixes
            .iter()
            .any(|p| p.contains("slang-unit-test-tool") || p.ends_with('/'));

        if all_prefixes_are_files {
            let test_files: Vec<PathBuf> = self
                .args
                .prefixes
                .iter()
                .filter(|p| {
                    !self
                        .args
                        .ignore_patterns
                        .iter()
                        .any(|ignore| p.contains(ignore))
                })
                .map(PathBuf::from)
                .collect();
            self.run_file_tests(&test_files)?;
        } else if has_internal_prefix || !self.args.prefixes.is_empty() {
            self.run_with_prefixes()?;
        } else {
            let test_files =
                discover_test_files(&self.args.test_dir, &[], &self.args.ignore_patterns)?;
            if !test_files.is_empty() {
                self.run_file_tests(&test_files)?;
            } else {
                self.run_with_prefixes()?;
            }
        }

        let elapsed = start_time.elapsed();

        if is_interrupted() {
            eprintln!("\n{}", "Interrupted by Ctrl-C".red().bold());
        }

        self.print_summary(elapsed);

        Ok(self.stats.failed.load(Ordering::SeqCst) == 0 && !is_interrupted())
    }

    fn run_with_prefixes(&self) -> Result<()> {
        let mut cmd = Command::new(&self.args.slang_test);

        if self.args.hide_ignored {
            cmd.arg("-hide-ignored");
        }

        cmd.args(&self.args.prefixes)
            .args(&self.args.extra_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().context("Failed to spawn slang-test")?;
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let stats = self.stats.clone();
        let failures = self.failures.clone();
        let failures_for_thread = failures.clone();

        // Simple progress for prefix mode - we don't know total count
        let progress_stats = stats.clone();
        let machine_output = self.machine_output;
        let progress_shutdown = Arc::new(AtomicBool::new(false));
        let progress_shutdown_clone = progress_shutdown.clone();
        let progress_handle = thread::spawn(move || {
            let start = Instant::now();
            let mut last_total = 0;
            while !progress_shutdown_clone.load(Ordering::SeqCst) {
                let passed = progress_stats.passed.load(Ordering::SeqCst);
                let failed = progress_stats.failed.load(Ordering::SeqCst);
                let ignored = progress_stats.ignored.load(Ordering::SeqCst);
                let total = passed + failed + ignored;
                let elapsed = start.elapsed().as_secs_f64();

                if machine_output {
                    // In machine mode, report every 100 tests
                    if total >= last_total + 100 {
                        last_total = total;
                        eprintln!(
                            "[{}] {} passed, {} failed, {} ignored ({:.1}s)",
                            total, passed, failed, ignored, elapsed
                        );
                    }
                } else {
                    eprint!(
                        "\r\x1b[KRunning: {} passed, {} failed, {} ignored ({:.1}s)",
                        passed, failed, ignored, elapsed
                    );
                    let _ = std::io::stderr().flush();
                }

                thread::sleep(Duration::from_millis(100));
            }
        });

        let stdout_handle = thread::spawn(move || {
            let reader = BufReader::new(stdout);
            let mut pending_failure_lines: Vec<String> = Vec::new();

            for line in reader.lines() {
                if let Ok(line) = line {
                    if let Some(outcome) = parse_test_output(&line) {
                        match outcome.result {
                            TestResult::Passed => {
                                stats.passed.fetch_add(1, Ordering::SeqCst);
                                pending_failure_lines.clear();
                            }
                            TestResult::Failed => {
                                stats.failed.fetch_add(1, Ordering::SeqCst);
                                // Failure output comes BEFORE the "FAILED test:" line
                                let info = parse_failure_info(&outcome.name, &pending_failure_lines);
                                failures_for_thread.lock().unwrap().push(info);
                                pending_failure_lines.clear();
                            }
                            TestResult::Ignored => {
                                stats.ignored.fetch_add(1, Ordering::SeqCst);
                                pending_failure_lines.clear();
                            }
                        }
                    } else {
                        // Collect potential failure output (lines starting with [test-name])
                        if line.starts_with('[') || !pending_failure_lines.is_empty() {
                            pending_failure_lines.push(line);
                        }
                    }
                }
            }
        });

        // Capture stderr (failure output is here)
        let (stderr_tx, stderr_rx) = crossbeam_channel::unbounded::<String>();
        let stderr_handle = thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let _ = stderr_tx.send(line);
                }
            }
        });

        stdout_handle.join().unwrap();
        stderr_handle.join().unwrap();

        // Collect stderr lines
        let stderr_lines: Vec<String> = stderr_rx.try_iter().collect();

        // Update failures with stderr output
        let mut failures_guard = failures.lock().unwrap();
        for failure in failures_guard.iter_mut() {
            // Find stderr lines for this test
            let test_prefix = format!("[{}]", failure.test_name);
            let mut capture = false;
            let mut relevant_lines: Vec<String> = Vec::new();

            for line in &stderr_lines {
                if line.starts_with(&test_prefix) {
                    capture = true;
                }
                if capture {
                    relevant_lines.push(line.clone());
                    if line.contains("}}}") && relevant_lines.iter().filter(|l| l.contains("}}}")).count() >= 2 {
                        break;
                    }
                }
            }

            if !relevant_lines.is_empty() {
                // Re-parse with stderr output
                let info = parse_failure_info(&failure.test_name, &relevant_lines);
                failure.output_lines = info.output_lines;
                failure.expected = info.expected;
                failure.actual = info.actual;
            }
        }
        drop(failures_guard);

        // Stop progress thread
        progress_shutdown.store(true, Ordering::SeqCst);
        let _ = progress_handle.join();
        if !self.machine_output {
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }

        let _ = child.wait();
        Ok(())
    }

    fn run_file_tests(&self, test_files: &[PathBuf]) -> Result<()> {
        if test_files.is_empty() {
            return Ok(());
        }

        // Check if we have timing data (and should use it)
        let has_timing_data = !self.args.no_timing_cache && {
            let cache = self.timing_cache.lock().unwrap();
            !cache.timings.is_empty()
        };

        // Schedule files using interleaved LPT (Longest Processing Time First)
        // Deal slow tests like cards across worker positions, then fill with fast tests
        // This ensures each worker gets a mix of slow and fast tests
        let sorted_files: Vec<PathBuf> = if has_timing_data {
            let mut files_with_duration: Vec<_> = test_files.iter()
                .map(|f| (f.clone(), self.predict_duration(f)))
                .collect();
            // Sort by duration descending (slowest first)
            files_with_duration.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Deal like cards: position i gets tests i, i+N, i+2N, ...
            // This spreads slow tests across all batch positions
            let n = self.args.jobs;
            let mut slots: Vec<Vec<PathBuf>> = (0..n).map(|_| Vec::new()).collect();

            for (i, (file, _)) in files_with_duration.into_iter().enumerate() {
                slots[i % n].push(file);
            }

            // Flatten: first test from each slot, then second, etc.
            let mut result = Vec::with_capacity(test_files.len());
            let max_len = slots.iter().map(|s| s.len()).max().unwrap_or(0);
            for i in 0..max_len {
                for slot in &slots {
                    if i < slot.len() {
                        result.push(slot[i].clone());
                    }
                }
            }
            result
        } else {
            // No timing data: use original order
            test_files.to_vec()
        };

        // Build predictions map (without startup - used for scheduling)
        // and startups map (for ETA calculation)
        let (predictions, startups): (HashMap<String, f64>, HashMap<String, f64>) = {
            let api_filter = extract_api_filter(&self.args.extra_args);
            let cache = self.timing_cache.lock().unwrap();
            sorted_files.iter()
                .map(|f| {
                    let key = f.to_string_lossy().to_string();
                    let pred = cache.predict(&key, api_filter.as_deref(), false);
                    let startup = cache.timings.get(&key)
                        .and_then(|backends| backends.get("_startup").copied())
                        .unwrap_or(0.0);
                    ((key.clone(), pred), (key, startup))
                })
                .unzip()
        };

        if has_timing_data {
            let total_predicted: f64 = predictions.values().zip(startups.values())
                .map(|(pred, startup)| pred + startup)
                .sum();
            eprintln!(
                "Running {} test files with {} workers (predicted {:.0}s with cached timing)",
                test_files.len(),
                self.args.jobs,
                total_predicted / self.args.jobs as f64  // rough parallel estimate
            );
        } else {
            eprintln!(
                "Running {} test files with {} workers",
                test_files.len(),
                self.args.jobs
            );
        }

        // Create dynamic work pool with predictions
        let work_pool = Arc::new(WorkPool::new(
            sorted_files,
            self.args.batch_size,
            self.args.jobs,
            predictions,
            startups,
            has_timing_data,
            self.args.batch_duration,
        ));

        let stats = self.stats.clone();
        let failures = self.failures.clone();
        let retries = self.args.retries;
        let retried_tests = self.retried_tests.clone();

        // Track running batches (for display)
        let running = Arc::new(AtomicUsize::new(0));
        // Track adaptive/turbo batches separately
        let adaptive_running = Arc::new(AtomicUsize::new(0));

        // Progress display thread
        let progress_stats = stats.clone();
        let progress_running = running.clone();
        let progress_adaptive = adaptive_running.clone();
        let progress_pool = work_pool.clone();
        let progress_shutdown = Arc::new(AtomicBool::new(false));
        let progress_shutdown_clone = progress_shutdown.clone();
        let total_files = test_files.len();
        let total_predicted_time = if has_timing_data { work_pool.total_predicted } else { 0.0 };
        let machine_output = self.machine_output;
        let progress_handle = thread::spawn(move || {
            let display = ProgressDisplay::new(total_files, total_predicted_time, machine_output);
            let mut sys_stats = SystemStats::new();
            let mut stats_counter = 0u32;
            while !progress_shutdown_clone.load(Ordering::SeqCst) {
                let files_done = progress_stats.files_completed();
                let batches_running = progress_running.load(Ordering::SeqCst);
                let batches_remaining = progress_pool.remaining();
                let adaptive_count = progress_adaptive.load(Ordering::SeqCst);
                let predicted_remaining = if progress_pool.has_timing_data {
                    Some(progress_pool.predicted_remaining())
                } else {
                    None
                };
                display.update(&progress_stats, files_done, batches_running, batches_remaining, predicted_remaining, adaptive_count);

                // Log system stats every ~1 second (every 10 iterations)
                stats_counter += 1;
                if stats_counter >= 10 {
                    stats_counter = 0;
                    sys_stats.refresh_and_log(batches_running, adaptive_count, batches_remaining);
                }

                thread::sleep(Duration::from_millis(100));
            }
        });

        // Spawn worker threads that pull from the pool
        // All workers start immediately - no warmup delay
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        if !work_pool.is_empty() {
            for _ in 0..self.args.jobs {
                let slang_test = self.args.slang_test.clone();
                let extra_args = self.args.extra_args.clone();
                let stats = stats.clone();
                let failures = failures.clone();
                let retried_tests = retried_tests.clone();
                let pool = work_pool.clone();
                let running = running.clone();
                let shutdown = shutdown.clone();
                let machine_output = self.machine_output;
                let verbose = self.args.verbose;

                let handle = thread::spawn(move || {
                    loop {
                        if shutdown.load(Ordering::SeqCst) || is_interrupted() {
                            break;
                        }

                        // Get a batch from the pool (dynamically sized)
                        // Use try_get_batch which holds the lock during the empty check
                        // to avoid race with add_file from retries
                        if let Some(batch) = pool.try_get_batch() {
                            run_batch_with_pool(
                                &slang_test,
                                &batch,
                                &extra_args,
                                Duration::from_secs(BATCH_TIMEOUT_SECS),
                                &stats,
                                &failures,
                                retries,
                                &retried_tests,
                                &pool,
                                &running,
                                machine_output,
                                verbose,
                            );
                        } else {
                            // No work available right now, wait for shutdown signal
                            // or for retries to be added back to the pool
                            thread::sleep(Duration::from_millis(10));
                        }
                    }
                });
                handles.push(handle);
            }

            // Adaptive load balancing: spawn extra batches when CPU is underutilized
            let adaptive = self.args.adaptive;
            let num_cpus = self.args.jobs;
            let adaptive_handles: Arc<Mutex<Vec<thread::JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));

            // Wait until pool is empty and no batches running
            while !work_pool.is_empty() || running.load(Ordering::SeqCst) > 0 {
                if is_interrupted() {
                    break;
                }

                // Adaptive mode: if we have spare capacity, spawn extra batches
                // Use running batch count as primary metric (reacts instantly)
                // and load average as secondary check
                if adaptive && !work_pool.is_empty() {
                    let current_running = running.load(Ordering::SeqCst);
                    let current_adaptive = adaptive_running.load(Ordering::SeqCst);
                    let total_running = current_running + current_adaptive;

                    // Spawn more if we're below target parallelism
                    // Also check load average if available to avoid overloading
                    let should_spawn = if let Some(load) = get_load_average() {
                        // Have capacity AND system load is reasonable
                        total_running < num_cpus && load < (num_cpus as f64 * 1.5)
                    } else {
                        // No load info, just use running count
                        total_running < num_cpus
                    };

                    if should_spawn {
                        // Spawn enough to fill capacity
                        let extra_to_spawn = (num_cpus - total_running).min(4);
                        for _ in 0..extra_to_spawn {
                            if let Some(batch) = work_pool.try_get_medium_batch() {
                                let slang_test = self.args.slang_test.clone();
                                let extra_args = self.args.extra_args.clone();
                                let stats = stats.clone();
                                let failures = failures.clone();
                                let retried_tests = retried_tests.clone();
                                let pool = work_pool.clone();
                                let running = running.clone();
                                let adaptive_counter = adaptive_running.clone();
                                let machine_output = self.machine_output;
                                let verbose = self.args.verbose;

                                // Increment adaptive counter
                                adaptive_counter.fetch_add(1, Ordering::SeqCst);
                                log_event("turbo_spawn", &format!("total_running={} file={}",
                                    total_running, batch[0].file_name().unwrap_or_default().to_string_lossy()));

                                let handle = thread::spawn(move || {
                                    run_batch_with_pool(
                                        &slang_test,
                                        &batch,
                                        &extra_args,
                                        Duration::from_secs(BATCH_TIMEOUT_SECS),
                                        &stats,
                                        &failures,
                                        retries,
                                        &retried_tests,
                                        &pool,
                                        &running,
                                        machine_output,
                                        verbose,
                                    );
                                    // Decrement adaptive counter when done
                                    adaptive_counter.fetch_sub(1, Ordering::SeqCst);
                                });
                                adaptive_handles.lock().unwrap().push(handle);
                            }
                        }
                    }
                }

                thread::sleep(Duration::from_millis(20));
            }

            // Signal workers to stop
            shutdown.store(true, Ordering::SeqCst);

            // Wait for all workers
            for handle in handles {
                handle.join().unwrap();
            }

            // Wait for any adaptive workers
            let mut adaptive_guard = adaptive_handles.lock().unwrap();
            for handle in adaptive_guard.drain(..) {
                handle.join().unwrap();
            }

            // Stop progress display
            progress_shutdown.store(true, Ordering::SeqCst);
            let _ = progress_handle.join();
            if !self.machine_output {
                eprint!("\r\x1b[K");
                let _ = std::io::stderr().flush();
            }
        }

        Ok(())
    }

    fn show_diff(&self, expected: &str, actual: &str) {
        let indent = if self.machine_output { "" } else { "  " };
        let indent2 = if self.machine_output { "" } else { "    " };

        // Check if expected and actual are identical
        if expected.trim() == actual.trim() {
            if self.machine_output {
                println!("(expected and actual appear identical)");
            } else {
                println!(
                    "{}{}",
                    indent,
                    "(expected and actual appear identical - slang-test comparison may involve details beyond this output)"
                        .yellow()
                );
            }
            println!("{}Content:", indent);
            for line in expected.lines().take(10) {
                if self.machine_output {
                    println!("{}", line);
                } else {
                    println!("{}{}", indent2, line.dimmed());
                }
            }
            if expected.lines().count() > 10 {
                println!("{}(truncated) ...", indent2);
            }
            return;
        }

        match self.args.diff.as_str() {
            "none" => {
                // Show raw output without diff
                if self.machine_output {
                    println!("Expected:");
                } else {
                    println!("{}{}:", indent, "Expected".green());
                }
                for line in expected.lines().take(20) {
                    if self.machine_output {
                        println!("{}", line);
                    } else {
                        println!("{}{}", indent2, line.green());
                    }
                }
                if expected.lines().count() > 20 {
                    println!("{}(truncated) ...", indent2);
                }

                if self.machine_output {
                    println!("Actual:");
                } else {
                    println!("{}{}:", indent, "Actual".red());
                }
                for line in actual.lines().take(20) {
                    if self.machine_output {
                        println!("{}", line);
                    } else {
                        println!("{}{}", indent2, line.red());
                    }
                }
                if actual.lines().count() > 20 {
                    println!("{}(truncated) ...", indent2);
                }
            }
            "difft" => {
                // Use difft for side-by-side diff
                self.run_external_diff("difft", &["--color", "always"], expected, actual);
            }
            _ => {
                // Default: use diff with side-by-side
                self.run_external_diff("diff", &["-y", "--color=always", "-W", "160"], expected, actual);
            }
        }
    }

    fn run_external_diff(&self, cmd: &str, args: &[&str], expected: &str, actual: &str) {
        use std::io::Write as _;

        let indent = if self.machine_output { "" } else { "  " };

        // Write expected and actual to temp files
        let expected_file = std::env::temp_dir().join("slang-test-expected.txt");
        let actual_file = std::env::temp_dir().join("slang-test-actual.txt");

        if let (Ok(mut ef), Ok(mut af)) = (
            std::fs::File::create(&expected_file),
            std::fs::File::create(&actual_file),
        ) {
            let _ = ef.write_all(expected.as_bytes());
            let _ = af.write_all(actual.as_bytes());

            let output = Command::new(cmd)
                .args(args)
                .arg(&expected_file)
                .arg(&actual_file)
                .output();

            match output {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        println!("{}{}", indent, line);
                    }
                }
                Err(_) => {
                    // Fallback to raw output if diff command fails
                    println!("{}({} not available, showing raw output)", indent, cmd);
                    if self.machine_output {
                        println!("Expected:");
                        for line in expected.lines().take(20) {
                            println!("{}", line);
                        }
                        println!("Actual:");
                        for line in actual.lines().take(20) {
                            println!("{}", line);
                        }
                    } else {
                        println!("{}{}:", indent, "Expected".green());
                        for line in expected.lines().take(20) {
                            println!("    {}", line.green());
                        }
                        println!("{}{}:", indent, "Actual".red());
                        for line in actual.lines().take(20) {
                            println!("    {}", line.red());
                        }
                    }
                }
            }

            // Clean up temp files
            let _ = std::fs::remove_file(&expected_file);
            let _ = std::fs::remove_file(&actual_file);
        }
    }

    fn print_summary(&self, elapsed: Duration) {
        let passed = self.stats.passed.load(Ordering::SeqCst);
        let failed = self.stats.failed.load(Ordering::SeqCst);
        let ignored = self.stats.ignored.load(Ordering::SeqCst);
        let retried = self.stats.retried_and_passed.load(Ordering::SeqCst);

        let failures = self.failures.lock().unwrap();

        // Print failures with nice formatting
        if !failures.is_empty() {
            println!("\n{}", "=".repeat(70));
            println!("{}", "FAILURES".red().bold());
            println!("{}", "=".repeat(70));

            let mut sorted_failures: Vec<_> = failures.iter().collect();
            sorted_failures.sort_by(|a, b| a.test_name.cmp(&b.test_name));

            for failure in &sorted_failures {
                println!("\n{}", failure.test_name.red());

                // If we have expected/actual, show a diff
                if let (Some(expected), Some(actual)) = (&failure.expected, &failure.actual) {
                    self.show_diff(expected, actual);
                } else if !failure.output_lines.is_empty() {
                    // Show first few lines of failure output
                    let relevant_lines: Vec<_> = failure
                        .output_lines
                        .iter()
                        .filter(|l| {
                            !l.trim().is_empty()
                                && !l.contains("Supported backends:")
                                && !l.contains("Check ")
                        })
                        .take(OUTPUT_TRUNCATE_LINES)
                        .collect();

                    for line in relevant_lines {
                        if self.machine_output {
                            println!("{}", line);
                        } else {
                            println!("  {}", line.dimmed());
                        }
                    }
                    if failure.output_lines.len() > OUTPUT_TRUNCATE_LINES {
                        if self.machine_output {
                            println!("(truncated) ...");
                        } else {
                            println!("  {} ...", "(truncated)".dimmed());
                        }
                    }
                }
            }
        }

        // Summary
        println!("\n{}", "=".repeat(70));

        let total_run = passed + failed;
        let interrupted = is_interrupted();

        if total_run > 0 {
            if interrupted {
                // When interrupted, don't show percentage as it's misleading
                if failed == 0 {
                    println!(
                        "{}: {} passed, {} ignored in {:.1}s (incomplete - interrupted)",
                        "INTERRUPTED".yellow().bold(),
                        passed,
                        ignored,
                        elapsed.as_secs_f64()
                    );
                } else {
                    println!(
                        "{}: {} passed, {} failed, {} ignored in {:.1}s (incomplete - interrupted)",
                        "INTERRUPTED".yellow().bold(),
                        passed,
                        failed,
                        ignored,
                        elapsed.as_secs_f64()
                    );
                }
            } else {
                let pass_pct = (passed as f64 / total_run as f64) * 100.0;
                if failed == 0 {
                    println!(
                        "{}: {:.0}% passed ({} passed, {} ignored) in {:.1}s",
                        "OK".green().bold(),
                        pass_pct,
                        passed,
                        ignored,
                        elapsed.as_secs_f64()
                    );
                } else {
                    println!(
                        "{}: {:.0}% passed ({} passed, {} failed, {} ignored) in {:.1}s",
                        "FAILED".red().bold(),
                        pass_pct,
                        passed,
                        failed,
                        ignored,
                        elapsed.as_secs_f64()
                    );
                }
            }
        } else {
            println!("No tests run");
        }

        if retried > 0 {
            if self.machine_output {
                println!("({} tests passed after retry)", retried);
            } else {
                println!("  ({} tests passed after retry)", retried);
            }
        }

        // Show slowest files in verbose mode only
        if self.args.verbose {
            let num_slowest = 15;
            let slowest = self.stats.slowest_files(num_slowest);
            if !slowest.is_empty() && slowest[0].1 > 1.0 {
                println!("\n{}:", "Slowest files".yellow());
                for (file, secs) in &slowest {
                    if *secs > 0.5 {
                        println!("  {:>6.1}s  {}", secs, file);
                    }
                }

                // Show per-backend breakdown for slowest files
                let observed = self.stats.get_observed_timings();
                println!("\n{}:", "Per-backend timing (slowest files)".yellow());
                for (file, _total_secs) in slowest.iter().take(5) {
                    if let Some(backends) = observed.get(file) {
                        let mut backend_list: Vec<_> = backends.iter()
                            .filter(|(b, _)| *b != "_total")
                            .collect();
                        backend_list.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                        if !backend_list.is_empty() {
                            let backend_str: String = backend_list.iter()
                                .map(|(b, s)| format!("{}:{:.1}s", b, s))
                                .collect::<Vec<_>>()
                                .join(", ");
                            println!("  {}  {}", file, backend_str.dimmed());
                        }
                    }
                }
            }
        }

        // Rerun command
        if !failures.is_empty() {
            let mut test_files: Vec<String> = failures
                .iter()
                .filter_map(|f| extract_base_test_file(&f.test_name))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            test_files.sort();

            println!("\n{}", "To rerun failed tests:".yellow());

            // Use slang-test-runner if there are file-based tests
            let has_file_tests = test_files.iter().any(|f| f.ends_with(".slang") || f.ends_with(".hlsl") || f.ends_with(".glsl"));

            let exe = std::env::args().next().unwrap_or_else(|| "slang-test-runner".to_string());
            if has_file_tests {
                print!("{}", exe);
                if self.args.root_dir != PathBuf::from(".") {
                    print!(" -C {}", self.args.root_dir.display());
                }
                for file in &test_files {
                    print!(" {}", file);
                }
                println!();
            } else {
                print!("{}", self.args.slang_test.display());
                for file in &test_files {
                    print!(" \"{}\"", file);
                }
                println!();
            }
        }

        println!("{}", "=".repeat(70));
    }
}

fn main() -> Result<()> {
    // Set up Ctrl-C handler
    ctrlc::set_handler(|| {
        set_interrupted();
    })
    .expect("Error setting Ctrl-C handler");

    let mut args = Args::parse();

    // Initialize event log if requested
    if let Some(ref log_path) = args.event_log {
        init_event_log(log_path)?;
        log_event("start", &format!("jobs={} batch_size={} adaptive={}",
            args.jobs, args.batch_size, args.adaptive));
    }

    let root_dir = args.root_dir.canonicalize().unwrap_or(args.root_dir.clone());

    if args.slang_test.is_relative() {
        args.slang_test = root_dir.join(&args.slang_test);
    }

    std::env::set_current_dir(&root_dir)
        .with_context(|| format!("Failed to change to root directory: {}", root_dir.display()))?;

    let runner = TestRunner::new(args);
    let success = runner.run()?;

    // Save timing data for future runs
    runner.save_timing();

    log_event("end", &format!("success={}", success));
    flush_event_log();

    std::process::exit(if success { 0 } else { 1 });
}
