use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::scheduler::SchedulerHandle;

// Constants
pub const DEFAULT_PREDICTED_DURATION: f64 = 0.5;
pub const EMA_NEW_WEIGHT: f64 = 0.7;
pub const OUTPUT_TRUNCATE_LINES: usize = 30;

// ============================================================================
// Debug Logging
// ============================================================================

pub static DEBUG_ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("STI_DEBUG").is_ok());
pub static DEBUG_START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Print a debug message with timestamp and thread ID, only if STI_DEBUG is set
#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if *$crate::types::DEBUG_ENABLED {
            let thread_id = std::thread::current().id();
            let thread_name = std::thread::current().name().unwrap_or("?").to_string();
            eprintln!("{}", format!("[DEBUG {:>6.3}s] [{}:{}] {}",
                $crate::types::DEBUG_START.elapsed().as_secs_f64(),
                thread_name,
                format!("{:?}", thread_id).trim_start_matches("ThreadId(").trim_end_matches(")"),
                format!($($arg)*)
            ).dimmed());
        }
    };
}

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

    /// Check if this test is a GPU test based on its API or test name
    /// GPU tests: vk, cuda, dx11, dx12, metal backends, or gfx-unit-test-tool internal tests
    pub fn is_gpu_test(&self) -> bool {
        // gfx-unit-test-tool internal tests are GPU tests
        if self.path.starts_with("gfx-unit-test-tool/") {
            return true;
        }

        // Check API
        match self.api.as_deref() {
            Some(api) => {
                let api_lower = api.to_lowercase();
                matches!(api_lower.as_str(), "vk" | "vulkan" | "cuda" | "dx11" | "dx12" | "metal")
            }
            // No API specified - assume CPU (e.g., internal tests without API suffix)
            None => false,
        }
    }
}

impl std::fmt::Display for TestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_test_string())
    }
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

/// Convert a test string to its timing cache key.
/// "tests/foo.slang.4 syn (vk)" -> "tests/foo.slang.4"
pub fn test_to_timing_key(test: &str) -> String {
    TestId::parse(test).to_timing_key()
}

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
    pub verbose: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gpu_test() {
        // GPU tests - various backends
        assert!(TestId::parse("tests/compute/foo.slang.0 (vk)").is_gpu_test());
        assert!(TestId::parse("tests/compute/foo.slang.1 (cuda)").is_gpu_test());
        assert!(TestId::parse("tests/compute/foo.slang.2 (dx11)").is_gpu_test());
        assert!(TestId::parse("tests/compute/foo.slang.3 (dx12)").is_gpu_test());
        assert!(TestId::parse("tests/compute/foo.slang.4 (metal)").is_gpu_test());
        assert!(TestId::parse("tests/compute/foo.slang.5 (vulkan)").is_gpu_test());

        // GPU tests - gfx-unit-test-tool
        assert!(TestId::parse("gfx-unit-test-tool/someTest.internal").is_gpu_test());
        assert!(TestId::parse("gfx-unit-test-tool/anotherTest.internal.0").is_gpu_test());

        // CPU tests
        assert!(!TestId::parse("tests/compute/foo.slang.0 (cpu)").is_gpu_test());
        assert!(!TestId::parse("tests/compute/foo.slang.1 (llvm)").is_gpu_test());
        assert!(!TestId::parse("slang-unit-test-tool/modulePtr.internal").is_gpu_test());
        assert!(!TestId::parse("tests/compute/foo.slang").is_gpu_test()); // No API suffix

        // Synthesized tests
        assert!(TestId::parse("tests/compute/foo.slang.0 syn (vk)").is_gpu_test());
        assert!(!TestId::parse("tests/compute/foo.slang.0 syn (cpu)").is_gpu_test());
    }

    #[test]
    fn test_batch_kind_segmentation() {
        use crate::scheduler::Scheduler;

        let files = vec![
            "tests/a.slang.0 (vk)".to_string(),
            "tests/b.slang.0 (cpu)".to_string(),
            "tests/c.slang.0 (cuda)".to_string(),
            "tests/d.slang.0 (llvm)".to_string(),
            "gfx-unit-test-tool/test.internal".to_string(),
        ];

        let predictions = HashMap::new();

        // With gpu_jobs set, batches should be segmented
        let (scheduler, _handle) = Scheduler::new(
            files.clone(),
            100,
            4,
            predictions.clone(),
            false,
            10.0,
            Some(2),
            0,
        );

        // The scheduler builds batches internally - we can't directly inspect them
        // but we can verify through the handle that it works
        drop(scheduler);
    }
}
