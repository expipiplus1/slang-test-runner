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

// ============================================================================
// Unsupported API Detection
// ============================================================================

/// Tracks which APIs are supported/unsupported on this system.
/// This is populated by running slang-test on a simple file at startup.
#[derive(Debug, Default, Clone)]
pub struct UnsupportedApis {
    /// Set of unsupported API names (lowercase): "dx11", "dx12", "vk", "cuda", "mtl", etc.
    pub unsupported: HashSet<String>,
    /// Set of supported API names (lowercase) - used to detect unknown APIs
    pub supported: HashSet<String>,
    /// Whether the API check completed successfully
    pub check_completed: bool,
    /// Error message if the check failed
    pub error: Option<String>,
}

impl UnsupportedApis {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add API aliases to a set
    fn add_with_aliases(set: &mut HashSet<String>, api: &str) {
        let api_lower = api.to_lowercase();
        set.insert(api_lower.clone());

        // Add common aliases
        match api_lower.as_str() {
            "vk" => { set.insert("vulkan".to_string()); }
            "vulkan" => { set.insert("vk".to_string()); }
            "dx11" => { set.insert("d3d11".to_string()); }
            "d3d11" => { set.insert("dx11".to_string()); }
            "dx12" => { set.insert("d3d12".to_string()); }
            "d3d12" => { set.insert("dx12".to_string()); }
            "mtl" => { set.insert("metal".to_string()); }
            "metal" => { set.insert("mtl".to_string()); }
            _ => {}
        }
    }

    /// Add an unsupported API (and its aliases)
    pub fn add_unsupported(&mut self, api: &str) {
        Self::add_with_aliases(&mut self.unsupported, api);
    }

    /// Add a supported API (and its aliases)
    pub fn add_supported(&mut self, api: &str) {
        Self::add_with_aliases(&mut self.supported, api);
    }

    /// Check if a test's API is unsupported
    pub fn is_test_unsupported(&self, test: &str) -> bool {
        let test_id = TestId::parse(test);
        if let Some(api) = test_id.api {
            self.unsupported.contains(&api.to_lowercase())
        } else {
            false
        }
    }

    /// Check if a test's API is unknown (not in supported or unsupported lists)
    /// Returns the API name if unknown, None otherwise
    pub fn get_unknown_api(&self, test: &str) -> Option<String> {
        let test_id = TestId::parse(test);
        if let Some(api) = test_id.api {
            let api_lower = api.to_lowercase();
            // Tests without API suffix (like internal tests) are fine
            // Only flag APIs that we haven't seen in Check lines
            if !self.supported.contains(&api_lower) && !self.unsupported.contains(&api_lower) {
                return Some(api_lower);
            }
        }
        None
    }

    /// Get platform-default unsupported APIs (before running actual check)
    pub fn platform_defaults() -> Self {
        let mut result = Self::new();

        // CPU is always supported
        result.add_supported("cpu");

        #[cfg(target_os = "macos")]
        {
            // On macOS, DirectX is not available
            result.add_unsupported("dx11");
            result.add_unsupported("dx12");
            result.add_unsupported("d3d11");
            result.add_unsupported("d3d12");
            // WGPU is not supported on macOS
            result.add_unsupported("wgpu");
        }

        #[cfg(target_os = "linux")]
        {
            // On Linux, Metal is not available
            result.add_unsupported("mtl");
            result.add_unsupported("metal");
            // WGPU is not supported on Linux
            result.add_unsupported("wgpu");
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, Metal is not available
            result.add_unsupported("mtl");
            result.add_unsupported("metal");
            // WGPU may be supported on Windows - don't mark as unsupported
        }

        result
    }
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
    /// Map from build type to (test identifier -> ETA fudge factor)
    /// Fudge factor = actual_elapsed / predicted_eta, used to correct displayed ETAs
    #[serde(default)]
    pub fudge_factors_by_build: HashMap<String, HashMap<String, f64>>,
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
                        cache.timings_by_build.insert("release".to_string(), cache.timings.clone());
                        cache.timings.clear();
                        cache.version = 4;
                    }
                    // Migrate from version 3 to 4 (add fudge factors)
                    if cache.version == 3 {
                        cache.version = 4;
                    }
                    if cache.version >= 4 {
                        return cache;
                    }
                }
                // Old format or version 1 - just start fresh
            }
        }
        Self {
            version: 4,
            timings_by_build: HashMap::new(),
            fudge_factors_by_build: HashMap::new(),
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

    /// Get the fudge factors map for a specific build type
    fn get_fudge_factors(&self, build_type: BuildType) -> Option<&HashMap<String, f64>> {
        self.fudge_factors_by_build.get(&build_type.to_string())
    }

    /// Get or create the fudge factors map for a specific build type
    fn get_fudge_factors_mut(&mut self, build_type: BuildType) -> &mut HashMap<String, f64> {
        self.fudge_factors_by_build
            .entry(build_type.to_string())
            .or_insert_with(HashMap::new)
    }

    /// Record a fudge factor for a test. Uses EMA to smooth out variations.
    pub fn record_fudge_factor(&mut self, build_type: BuildType, test_id: &str, fudge: f64) {
        let factors = self.get_fudge_factors_mut(build_type);
        let existing = factors.entry(test_id.to_string()).or_insert(1.0);
        // EMA: weight new measurement, but be conservative (slower to change)
        *existing = fudge * 0.3 + *existing * 0.7;
    }

    /// Get the average fudge factor for a set of tests.
    /// Returns 1.0 if no fudge data is available.
    pub fn average_fudge_factor(&self, build_type: BuildType, test_ids: &[String]) -> f64 {
        let Some(factors) = self.get_fudge_factors(build_type) else {
            return 1.0;
        };

        let mut sum = 0.0;
        let mut count = 0;
        for test_id in test_ids {
            let timing_key = test_to_timing_key(test_id);
            if let Some(&fudge) = factors.get(&timing_key) {
                sum += fudge;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            1.0
        }
    }

    /// Record fudge factors for all tests in a run.
    /// fudge = actual_elapsed / predicted_eta
    pub fn record_fudge_factors(&mut self, build_type: BuildType, test_ids: &[String], fudge: f64) {
        // Clamp fudge factor to reasonable range (0.5x to 3x)
        let clamped_fudge = fudge.clamp(0.5, 3.0);
        for test_id in test_ids {
            let timing_key = test_to_timing_key(test_id);
            self.record_fudge_factor(build_type, &timing_key, clamped_fudge);
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
    /// Batch size histogram: maps batch_size -> count
    pub batch_sizes: Mutex<HashMap<usize, usize>>,
    /// Initial predicted ETA (for fudge factor calculation at end of run)
    pub initial_predicted_eta: Mutex<Option<f64>>,
    /// Start time of test execution (for fudge factor calculation)
    pub execution_start_time: Mutex<Option<Instant>>,
    /// List of test files in this run (for fudge factor recording)
    pub test_files: Mutex<Vec<String>>,
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

    /// Record that a batch of a given size was executed
    pub fn record_batch_size(&self, size: usize) {
        let mut sizes = self.batch_sizes.lock().unwrap();
        *sizes.entry(size).or_insert(0) += 1;
    }

    /// Get the batch size histogram
    pub fn get_batch_sizes(&self) -> HashMap<usize, usize> {
        self.batch_sizes.lock().unwrap().clone()
    }

    /// Record the initial predicted ETA and test files for fudge factor calculation
    pub fn record_initial_prediction(&self, predicted_eta: f64, test_files: Vec<String>) {
        *self.initial_predicted_eta.lock().unwrap() = Some(predicted_eta);
        *self.execution_start_time.lock().unwrap() = Some(Instant::now());
        *self.test_files.lock().unwrap() = test_files;
    }

    /// Calculate the fudge factor (actual_elapsed / predicted_eta)
    /// Returns None if we don't have the necessary data
    pub fn calculate_fudge_factor(&self) -> Option<f64> {
        let predicted = (*self.initial_predicted_eta.lock().unwrap())?;
        let start_time = (*self.execution_start_time.lock().unwrap())?;

        // Only calculate if prediction was meaningful
        if predicted < 1.0 {
            return None;
        }

        let actual_elapsed = start_time.elapsed().as_secs_f64();
        Some(actual_elapsed / predicted)
    }

    /// Get the test files from this run
    pub fn get_test_files(&self) -> Vec<String> {
        self.test_files.lock().unwrap().clone()
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
// Scheduler - Message-Passing Architecture
// ============================================================================
//
// The scheduler owns all batch scheduling state and runs in a dedicated thread.
// Workers communicate via messages, eliminating lock ordering complexity.
//
// Architecture:
//   - Scheduler thread: owns batches, pending_files, in_flight, etc.
//   - SchedulerHandle: cloneable handle for workers to send messages
//   - All state mutations happen in the scheduler thread (no mutexes!)

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

        // Don't sort batches - rely on random input order to spread GPU tests
        // and tail prioritization (in try_get_batch) to handle slow tests at the end.
        // Sorting by duration front-loads slow GPU tests, causing contention.
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
                .partition(|t| !is_gpu_test(t));

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
    /// Fudge factor for ETA display (actual/predicted from historical runs)
    eta_fudge_factor: f64,
}

impl ProgressDisplay {
    pub fn new(total_files: usize, machine_output: bool, num_workers: usize, verbose: bool, eta_fudge_factor: f64) -> Self {
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
            eta_fudge_factor,
        }
    }

    pub fn update(&mut self, stats: &TestStats, _files_completed: usize, batches_running: usize, _batches_remaining: usize, has_pending_batches: bool, eta_seconds: Option<f64>, worker_states: Option<&WorkerStates>) {
        // Update CPU/GPU load every ~30 updates (~500ms at 16ms refresh rate), only in verbose mode
        if self.verbose {
            self.sys_query_counter += 1;
            if self.sys_query_counter >= 30 {
                self.sys_query_counter = 0;
                self.last_cpu_load = get_cpu_usage(&mut self.sys);
                self.last_gpu_load = get_gpu_usage();
            }
        }

        // Apply fudge factor to ETA (historical actual/predicted ratio)
        let adjusted_eta = eta_seconds.map(|eta| eta * self.eta_fudge_factor);

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
                let eta = match adjusted_eta {
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

            // Format ETA string (velocity-adjusted)
            let eta = match adjusted_eta {
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

            let load_info = if self.verbose {
                match self.last_gpu_load {
                    Some(gpu) => format!(" | CPU: {:.0}% GPU: {}%", self.last_cpu_load, gpu),
                    None if self.last_cpu_load > 0.0 => format!(" | CPU: {:.0}%", self.last_cpu_load),
                    None => String::new(),
                }
            } else {
                String::new()
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
    pub scheduler: &'a SchedulerHandle,
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
        let batches = Scheduler::build_batches(
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
        let batches = Scheduler::build_batches(
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
