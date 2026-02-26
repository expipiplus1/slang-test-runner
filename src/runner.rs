use anyhow::Result;
use colored::Colorize;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use terminal_size::{terminal_size, Width};

use crate::api::UnsupportedApis;
use crate::debug_log;
use crate::discovery::DiscoveryResult;
use crate::event_log::log_event;
use crate::progress::{ProgressDisplay, WorkerState, WorkerStates, SystemStats, PROGRESS_UPDATE_INTERVAL_MS};
use crate::scheduler::{Scheduler, SchedulerHandle};
use crate::timing::{BuildType, TimingCache};
use crate::types::{
    BatchContext, FailureInfo, TestId, TestOutcome, TestResult, TestStats,
    DEBUG_START, DEFAULT_PREDICTED_DURATION, OUTPUT_TRUNCATE_LINES, test_to_timing_key,
};

// ============================================================================
// Diff Tool Resolution
// ============================================================================

/// Check if difft is available (cached)
static DIFFT_AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
    Command::new("difft")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
});

/// Get terminal width for diff output (cached at startup, None if not a TTY)
/// Subtracts 2 for the indent prefix added to diff output
static TERM_WIDTH: LazyLock<Option<String>> = LazyLock::new(|| {
    terminal_size().map(|(Width(w), _)| w.saturating_sub(2).to_string())
});

/// Resolve "auto" diff tool to actual tool name
fn resolve_diff_tool(tool: &str) -> &str {
    match tool {
        "auto" => {
            if *DIFFT_AVAILABLE {
                "difft"
            } else {
                "diff"
            }
        }
        other => other,
    }
}

// ============================================================================
// ETA Calculation and Display
// ============================================================================

/// Calculate initial ETA from predictions, accounting for GPU job limits.
/// When --gpu-jobs is set, GPU tests are limited to that many concurrent workers,
/// which can significantly increase ETA if most tests are GPU tests.
fn calculate_initial_eta(
    predictions: &HashMap<String, f64>,
    num_workers: usize,
    gpu_jobs: Option<usize>,
) -> f64 {
    let mut gpu_total = 0.0f64;
    let mut cpu_total = 0.0f64;
    let mut longest = 0.0f64;

    for (test, &pred) in predictions {
        longest = longest.max(pred);
        if TestId::parse(test).is_gpu_test() {
            gpu_total += pred;
        } else {
            cpu_total += pred;
        }
    }

    match gpu_jobs {
        Some(max_gpu) => {
            // GPU tests limited to max_gpu concurrent workers
            // CPU tests can use all workers
            let gpu_workers = max_gpu.min(num_workers);
            let gpu_eta = gpu_total / gpu_workers.max(1) as f64;
            let cpu_eta = cpu_total / num_workers.max(1) as f64;
            // GPU and CPU batches run in parallel, so take the longer path
            gpu_eta.max(cpu_eta).max(longest)
        }
        None => {
            // No GPU limiting - all workers can run anything
            let total = gpu_total + cpu_total;
            (total / num_workers.max(1) as f64).max(longest)
        }
    }
}

// ============================================================================
// Global Flags
// ============================================================================

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

// ============================================================================
// Process Reaper - cleans up finished processes without blocking
// ============================================================================

/// A process sent to the reaper, with an optional label for logging
struct ReaperItem {
    child: std::process::Child,
    label: Option<String>,
}

static REAPER_TX: LazyLock<crossbeam_channel::Sender<ReaperItem>> = LazyLock::new(|| {
    let (tx, rx) = crossbeam_channel::unbounded::<ReaperItem>();

    thread::spawn(move || {
        while let Ok(mut item) = rx.recv() {
            let _ = item.child.kill();
            let status = item.child.wait();
            if let Some(label) = item.label {
                let exit_info = match status {
                    Ok(s) => format_exit_status(Some(&s)),
                    Err(e) => format!("wait error: {}", e),
                };
                log_event("process_exit", &format!("{} {}", label, exit_info));
            }
        }
    });

    tx
});

/// Send a child process to the reaper for cleanup
pub fn reap_process(child: std::process::Child) {
    let _ = REAPER_TX.send(ReaperItem { child, label: None });
}

/// Send a child process to the reaper for cleanup, with a label for logging
pub fn reap_process_with_label(child: std::process::Child, label: String) {
    let _ = REAPER_TX.send(ReaperItem { child, label: Some(label) });
}

pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

pub fn set_interrupted() {
    if INTERRUPTED.swap(true, Ordering::SeqCst) {
        std::process::exit(130);
    }
    eprintln!("\nInterrupt received (Ctrl-C)");
}

/// Format exit status in a platform-appropriate way
pub fn format_exit_status(status: Option<&std::process::ExitStatus>) -> String {
    match status {
        None => "unknown exit status".to_string(),
        Some(status) => {
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                if let Some(signal) = status.signal() {
                    let signal_name = match signal {
                        6 => "SIGABRT",
                        9 => "SIGKILL",
                        11 => "SIGSEGV",
                        15 => "SIGTERM",
                        _ => "signal",
                    };
                    return format!("{} ({})", signal_name, signal);
                }
            }

            #[cfg(windows)]
            {
                if let Some(code) = status.code() {
                    // Windows crash codes are in the 0xC0000000+ range
                    if (code as u32) >= 0xC0000000 {
                        let name = match code as u32 {
                            0xC0000005 => "ACCESS_VIOLATION",
                            0xC00000FD => "STACK_OVERFLOW",
                            0xC0000409 => "STACK_BUFFER_OVERRUN",
                            _ => "crash",
                        };
                        return format!("{} (0x{:08X})", name, code as u32);
                    }
                }
            }

            if let Some(code) = status.code() {
                format!("exit code {}", code)
            } else {
                "terminated".to_string()
            }
        }
    }
}

/// Check if an exit status indicates a crash likely caused by a test (SIGSEGV, SIGABRT, etc.)
/// vs an external kill (SIGTERM, SIGKILL, taskkill) where we should requeue all tests.
/// Returns true if the crash was likely caused by a test.
fn is_test_caused_crash(status: Option<&std::process::ExitStatus>) -> bool {
    match status {
        Some(status) => {
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                if let Some(signal) = status.signal() {
                    // Signals that indicate a test caused the crash
                    return matches!(signal,
                        6 |   // SIGABRT - assertion failure, abort()
                        7 |   // SIGBUS - bus error
                        8 |   // SIGFPE - floating point exception
                        11 |  // SIGSEGV - segmentation fault
                        31    // SIGSYS - bad system call
                    );
                }
            }

            #[cfg(windows)]
            {
                if let Some(code) = status.code() {
                    // Windows crash codes in 0xC0000000+ range indicate test-caused crashes
                    // (ACCESS_VIOLATION, STACK_OVERFLOW, etc.)
                    // taskkill uses exit code 1, TerminateProcess uses the code you specify
                    if (code as u32) >= 0xC0000000 {
                        return true;
                    }
                }
            }

            // Non-zero exit code without a signal - could be test failure or external kill
            // Be conservative: treat as external kill so we don't lose tests
            false
        }
        None => false,
    }
}

// ============================================================================
// Static Regexes
// ============================================================================

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
    Regex::new(r"^(.+\.(slang|hlsl|glsl|c))(\.\d+)?$").unwrap()
});

// ============================================================================
// Parsing Functions
// ============================================================================

pub fn parse_test_output(line: &str) -> Option<TestOutcome> {
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

pub fn parse_failure_info(test_name: &str, lines: &[String]) -> FailureInfo {
    let mut expected: Option<String> = None;
    let mut actual: Option<String> = None;
    let mut in_expected = false;
    let mut in_actual = false;
    let mut expected_lines = Vec::new();
    let mut actual_lines = Vec::new();

    for line in lines {
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

pub fn extract_base_test_file(test_name: &str) -> Option<String> {
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

/// Minimize a list of failed test names into compact filter patterns.
/// Groups tests by base file and uses regex alternation for multiple variants.
/// Examples:
///   - Single variant: "tests/foo.slang.0"
///   - Multiple variants: "tests/foo.slang.(0|2|5)"
pub fn minimize_test_filters(test_names: &[&str]) -> Vec<String> {
    // Group by base file, tracking variant numbers
    let mut by_base_file: HashMap<String, Vec<Option<u32>>> = HashMap::new();
    for name in test_names {
        let test_id = TestId::parse(name);
        by_base_file
            .entry(test_id.path.clone())
            .or_default()
            .push(test_id.variant);
    }

    let mut result: Vec<String> = Vec::new();
    for (base_path, variants) in &by_base_file {
        // Deduplicate variants (same variant can fail multiple times with different APIs)
        let mut unique_variants: Vec<Option<u32>> = variants.iter().copied().collect();
        unique_variants.sort();
        unique_variants.dedup();

        if unique_variants.len() > 1 {
            // Multiple variants - build regex pattern: tests/foo.slang.(0|2|5)
            let variant_alts: Vec<String> = unique_variants
                .iter()
                .filter_map(|v| v.map(|n| n.to_string()))
                .collect();
            if !variant_alts.is_empty() {
                result.push(format!("{}.({})", base_path, variant_alts.join("|")));
            } else {
                result.push(base_path.clone());
            }
        } else if let Some(Some(variant)) = unique_variants.first() {
            // Single variant with number
            result.push(format!("{}.{}", base_path, variant));
        } else {
            // No variant number (e.g., internal test)
            result.push(base_path.clone());
        }
    }
    result.sort();
    result
}

// ============================================================================
// Retry and Outcome Processing
// ============================================================================

fn should_retry_test(
    outcome: &TestOutcome,
    max_retries: usize,
    retried_tests: &Mutex<HashMap<String, usize>>,
    scheduler: &SchedulerHandle,
) -> bool {
    if max_retries == 0 {
        return false;
    }

    let should_retry = {
        let mut retried = retried_tests.lock().unwrap();
        let retry_count = retried.entry(outcome.name.clone()).or_insert(0);
        if *retry_count < max_retries {
            *retry_count += 1;
            true
        } else {
            false
        }
    };

    if should_retry {
        if let Some(base_file) = extract_base_test_file(&outcome.name) {
            let file_path = PathBuf::from(&base_file);
            if file_path.exists() {
                // Re-queue the test by its full name (with variant info if present)
                scheduler.add_test(outcome.name.clone());
                return true;
            }
        }
    }
    false
}

fn process_outcome(
    outcome: TestOutcome,
    ctx: &BatchContext,
    failed_outcomes: &mut Vec<TestOutcome>,
) -> bool {
    if let Some(base_file) = extract_base_test_file(&outcome.name) {
        ctx.stats.record_file(&base_file);
    }

    match outcome.result {
        TestResult::Passed => {
            let was_retry = ctx.retried_tests.lock().unwrap().contains_key(&outcome.name);
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
            if should_retry_test(&outcome, ctx.max_retries, ctx.retried_tests, ctx.scheduler) {
                return true;
            }
            ctx.stats.failed.fetch_add(1, Ordering::SeqCst);
            failed_outcomes.push(outcome);
            false
        }
    }
}

// ============================================================================
// Batch Execution - Helper Functions
// ============================================================================

/// Spawn the slang-test process for a batch
fn spawn_batch_process(
    ctx: &BatchContext,
) -> Result<(std::process::Child, os_pipe::PipeReader), String> {
    // Convert test names to the format slang-test accepts (strip API suffix)
    let slang_test_args: Vec<String> = ctx.test_files
        .iter()
        .map(|f| TestId::parse(f).to_slang_test_arg())
        .collect();

    // Create a shared pipe for stdout and stderr so we get ordered output
    let (pipe_reader, pipe_writer) = match os_pipe::pipe() {
        Ok(p) => p,
        Err(e) => return Err(format!("Failed to create pipe: {}", e)),
    };
    let pipe_writer2 = match pipe_writer.try_clone() {
        Ok(p) => p,
        Err(e) => return Err(format!("Failed to clone pipe: {}", e)),
    };

    let mut cmd = Command::new(ctx.slang_test);
    cmd.current_dir(ctx.root_dir)
        .arg("-explicit-test-order")
        .arg("-disable-retries")
        .args(&slang_test_args)
        .args(ctx.extra_args)
        .stdout(pipe_writer)
        .stderr(pipe_writer2);

    debug_log!("slang-test invocation: {} -explicit-test-order -disable-retries {} {}",
        ctx.slang_test.display(),
        slang_test_args.join(" "),
        ctx.extra_args.join(" "));

    match cmd.spawn() {
        Ok(child) => {
            debug_log!("slang-test spawned successfully");
            // IMPORTANT: Drop cmd to close pipe writers in parent process
            drop(cmd);
            Ok((child, pipe_reader))
        }
        Err(e) => Err(format!("Failed to spawn slang-test: {}", e)),
    }
}

/// Handle incomplete batch (crash/timeout) - determine what tests to repool
fn handle_incomplete_batch(
    ctx: &BatchContext,
    seen_tests: &HashSet<String>,
    exit_status: Option<std::process::ExitStatus>,
    killed_for_timeout: bool,
    loop_time: Duration,
) {
    let exit_info = if killed_for_timeout {
        format!("timeout after {}s", ctx.timeout.as_secs())
    } else {
        format_exit_status(exit_status.as_ref())
    };

    // Collect all unaccounted tests
    let mut unaccounted_tests: Vec<String> = Vec::new();
    for file in ctx.test_files {
        let test_id = TestId::parse(file);
        let timing_key = test_id.to_timing_key();
        if !seen_tests.contains(&timing_key) {
            unaccounted_tests.push(file.clone());
        }
    }

    // Determine if this was a test-caused crash or an external kill
    let test_caused = killed_for_timeout || is_test_caused_crash(exit_status.as_ref());

    if test_caused {
        // Test caused the crash/timeout - blame the first unaccounted test, repool the rest
        let crashed_test = unaccounted_tests.first().cloned();
        let tests_to_repool: Vec<String> = unaccounted_tests.iter().skip(1).cloned().collect();

        debug_log!("test-caused crash, blaming {:?}, repooling {} tests",
            crashed_test, tests_to_repool.len());

        // Batch add tests back to scheduler
        ctx.scheduler.add_tests(tests_to_repool.clone());

        if let Some(test) = crashed_test {
            let error_type = if killed_for_timeout { "timed out" } else { "crashed" };
            eprintln!(
                "\n{}: slang-test {} ({}), skipping: {}",
                "ERROR".red(),
                error_type, exit_info, test
            );
            if !tests_to_repool.is_empty() {
                eprintln!("  Repooling {} subsequent tests", tests_to_repool.len());
            }
            ctx.stats.failed.fetch_add(1, Ordering::SeqCst);
            let failure_msg = if killed_for_timeout {
                format!("Test timed out ({})", exit_info)
            } else {
                format!("Test caused slang-test to crash ({})", exit_info)
            };
            ctx.failures.lock().unwrap().push(FailureInfo {
                test_name: test.clone(),
                output_lines: vec![failure_msg],
                expected: None,
                actual: None,
            });

            // Record timing for crashed/timed-out test using actual elapsed time
            let test_id = TestId::parse(&test);
            let timing_key = test_id.to_timing_key();
            ctx.stats.record_observed_timing(&timing_key, loop_time.as_secs_f64());
        }
    } else {
        // External kill (SIGTERM, SIGKILL, taskkill) - repool ALL unaccounted tests
        debug_log!("external kill detected, repooling all {} unaccounted tests", unaccounted_tests.len());
        // Clear current line (progress bar) before printing
        eprintln!(
            "\r\x1b[K{}",
            format!("slang-test killed externally ({}), repooling {} tests",
                exit_info, unaccounted_tests.len()).dimmed()
        );
        // Batch add all unaccounted tests back to scheduler
        ctx.scheduler.add_tests(unaccounted_tests);
    }
}

// ============================================================================
// Batch Execution - Output Reader
// ============================================================================

/// Channels for receiving data from the output reader thread
struct OutputReaderChannels {
    outcome_rx: crossbeam_channel::Receiver<TestOutcome>,
    timing_rx: crossbeam_channel::Receiver<(String, f64)>,
    done_rx: crossbeam_channel::Receiver<bool>,
    compiling_rx: crossbeam_channel::Receiver<bool>,
}

/// Result of spawning the output reader
struct OutputReaderHandle {
    channels: OutputReaderChannels,
    thread_handle: thread::JoinHandle<()>,
}

/// Spawn the output reader thread that parses slang-test output
fn spawn_output_reader(pipe_reader: os_pipe::PipeReader) -> OutputReaderHandle {
    let (outcome_tx, outcome_rx) = crossbeam_channel::unbounded::<TestOutcome>();
    let (timing_tx, timing_rx) = crossbeam_channel::unbounded::<(String, f64)>();
    let (done_tx, done_rx) = crossbeam_channel::bounded::<bool>(1);
    let (compiling_tx, compiling_rx) = crossbeam_channel::unbounded::<bool>();

    let thread_handle = thread::Builder::new()
        .name("output-reader".to_string())
        .spawn(move || {
            let reader = BufReader::new(pipe_reader);

            let mut last_test_time: Option<Instant> = None;
            let mut pending_output: Vec<String> = Vec::new();
            let mut past_initial_spew = false;
            let mut lines_read = 0;

            for line in reader.lines() {
                lines_read += 1;
                if is_interrupted() {
                    break;
                }
                if let Ok(line) = line {
                    // Detect the summary line that indicates normal completion
                    if line.starts_with("===") || line.contains("% of tests") || line == "no tests run" {
                        let _ = done_tx.try_send(true);
                        continue;
                    }

                    // Handle "Compiling" messages - notify progress display
                    if line.contains("Compiling") {
                        let _ = compiling_tx.send(true);
                        continue;
                    }

                    // Skip slang-test initial output
                    if line.starts_with("Supported backends:")
                        || line.starts_with("Check ")
                        || line.starts_with("Retrying ")
                    {
                        last_test_time = Some(Instant::now());
                        continue;
                    }

                    // Detect end of initial spew
                    if !past_initial_spew {
                        if line.starts_with('[') || line.contains("EXPECTED{{{") || line.contains("ACTUAL{{{") {
                            past_initial_spew = true;
                        }
                    }

                    // Try to parse as test result line
                    if let Some(mut outcome) = parse_test_output(&line) {
                        let now = Instant::now();

                        let test_duration = if last_test_time.is_some() {
                            outcome.duration
                                .map(|d| d.as_secs_f64())
                                .unwrap_or_else(|| last_test_time.unwrap().elapsed().as_secs_f64())
                        } else {
                            0.0
                        };

                        if outcome.duration.is_none() && last_test_time.is_some() {
                            outcome.duration = Some(Duration::from_secs_f64(test_duration));
                        }

                        last_test_time = Some(now);

                        if outcome.result == TestResult::Failed {
                            outcome.failure_output = std::mem::take(&mut pending_output);
                        } else {
                            pending_output.clear();
                        }

                        past_initial_spew = true;

                        let test_id = TestId::parse(&outcome.name);
                        let timing_key = test_id.to_timing_key();

                        if test_duration > 0.0 {
                            let _ = timing_tx.send((timing_key.clone(), test_duration));
                        }

                        let _ = outcome_tx.send(outcome);
                    } else if past_initial_spew {
                        pending_output.push(line);
                    }
                }
            }
            let _ = compiling_tx.send(false);
            debug_log!("output reader exiting after {} lines", lines_read);
        })
        .expect("failed to spawn output reader thread");

    OutputReaderHandle {
        channels: OutputReaderChannels {
            outcome_rx,
            timing_rx,
            done_rx,
            compiling_rx,
        },
        thread_handle,
    }
}

// ============================================================================
// Batch Execution - Monitoring
// ============================================================================

/// Result of monitoring a batch execution
struct BatchMonitorResult {
    need_kill: bool,
    killed_for_timeout: bool,
    exit_status: Option<std::process::ExitStatus>,
    seen_tests: HashSet<String>,
}

/// Monitor the batch execution, processing outcomes as they arrive
fn monitor_batch_execution(
    child: &mut std::process::Child,
    channels: &OutputReaderChannels,
    ctx: &BatchContext,
    expected_test_count: usize,
    batch_id: std::thread::ThreadId,
    test_time_sum: &std::sync::atomic::AtomicU64,
    test_count: &AtomicUsize,
    failed_outcomes: &mut Vec<TestOutcome>,
    worker_state: Option<&WorkerState>,
) -> BatchMonitorResult {
    let start = Instant::now();
    let mut killed_for_timeout = false;
    let mut exit_status: Option<std::process::ExitStatus> = None;
    let mut seen_tests: HashSet<String> = HashSet::new();
    let mut need_kill = false;

    debug_log!("entering batch monitor loop, expecting {} tests", expected_test_count);

    loop {
        if is_interrupted() {
            debug_log!("batch monitor: interrupted");
            need_kill = true;
            break;
        }

        // Check compilation status
        while let Ok(is_compiling) = channels.compiling_rx.try_recv() {
            ctx.stats.set_compiling(is_compiling);
        }

        // Process timing updates
        while let Ok((test_id, duration)) = channels.timing_rx.try_recv() {
            ctx.stats.record_observed_timing(&test_id, duration);
        }

        // Process test outcomes
        while let Ok(outcome) = channels.outcome_rx.try_recv() {
            ctx.stats.set_compiling(false);
            ctx.stats.record_test_output();
            let result_str = match outcome.result {
                TestResult::Passed => "passed",
                TestResult::Failed => "failed",
                TestResult::Ignored => "ignored",
            };
            let duration_ms = outcome.duration.map(|d| d.as_millis() as u64).unwrap_or(0);
            test_time_sum.fetch_add(duration_ms, Ordering::SeqCst);
            test_count.fetch_add(1, Ordering::SeqCst);
            log_event("test", &format!("{:?} {} {} duration_ms={}", batch_id, result_str, outcome.name, duration_ms));

            let test_id = TestId::parse(&outcome.name);
            seen_tests.insert(test_id.to_timing_key());

            process_outcome(outcome, ctx, failed_outcomes);

            if let Some(state) = worker_state {
                state.advance();
            }
        }

        // Check completion conditions
        if seen_tests.len() >= expected_test_count {
            debug_log!("batch monitor: all {} tests seen", expected_test_count);
            break;
        }

        if channels.done_rx.try_recv().is_ok() {
            debug_log!("batch monitor: done signal received");
            break;
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                debug_log!("batch monitor: child exited with {:?}, seen {}/{} tests",
                    status, seen_tests.len(), expected_test_count);
                exit_status = Some(status);
                break;
            }
            Ok(None) => {
                if start.elapsed() > ctx.timeout {
                    debug_log!("batch monitor: timeout after {}s", ctx.timeout.as_secs());
                    need_kill = true;
                    killed_for_timeout = true;
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                debug_log!("batch monitor: try_wait error: {}", e);
                break;
            }
        }
    }

    debug_log!("batch monitor loop exited: need_kill={} seen={}/{}", need_kill, seen_tests.len(), expected_test_count);

    BatchMonitorResult {
        need_kill,
        killed_for_timeout,
        exit_status,
        seen_tests,
    }
}

/// Drain any remaining outcomes after the output reader thread joins
fn drain_remaining_outcomes(
    channels: &OutputReaderChannels,
    ctx: &BatchContext,
    seen_tests: &mut HashSet<String>,
    failed_outcomes: &mut Vec<TestOutcome>,
) {
    debug_log!("drain: draining remaining outcomes");
    let mut drained_count = 0;
    while let Ok(outcome) = channels.outcome_rx.try_recv() {
        debug_log!("drain: got outcome {}", outcome.name);
        drained_count += 1;

        let test_id = TestId::parse(&outcome.name);
        seen_tests.insert(test_id.to_timing_key());

        process_outcome(outcome, ctx, failed_outcomes);
    }
    debug_log!("drain: drained {} outcomes", drained_count);

    while let Ok((test_id, duration)) = channels.timing_rx.try_recv() {
        ctx.stats.record_observed_timing(&test_id, duration);
    }
}

/// Print a warning for slow batches (verbose mode only)
fn warn_slow_batch(ctx: &BatchContext, loop_time: Duration, total_pred: f64) {
    if ctx.verbose && loop_time.as_secs() > 30 {
        let extra_args_str = if ctx.extra_args.is_empty() {
            String::new()
        } else {
            format!(" {}", ctx.extra_args.join(" "))
        };
        let repro_args: Vec<String> = ctx.test_files
            .iter()
            .map(|f| TestId::parse(f).to_slang_test_arg())
            .collect();
        eprintln!("{}", format!("\nWARNING: Batch took {:.1}s (predicted {:.1}s)", loop_time.as_secs_f64(), total_pred).dimmed());
        eprintln!("{}", format!("  Reproduce: time {} {}{}",
            ctx.slang_test.display(),
            repro_args.join(" "),
            extra_args_str
        ).dimmed());
    }
}

// ============================================================================
// Batch Execution - Main Function
// ============================================================================

pub fn run_batch_with_pool(
    slang_test: &PathBuf,
    root_dir: &PathBuf,
    test_files: &[String],
    extra_args: &[String],
    timeout: Duration,
    stats: &TestStats,
    failures: &Mutex<Vec<FailureInfo>>,
    max_retries: usize,
    retried_tests: &Mutex<HashMap<String, usize>>,
    scheduler: &SchedulerHandle,
    running: &AtomicUsize,
    verbose: bool,
    worker_state: Option<&WorkerState>,
) {
    let ctx = BatchContext {
        slang_test,
        root_dir,
        test_files,
        extra_args,
        timeout,
        stats,
        failures,
        max_retries,
        retried_tests,
        scheduler,
        running,
        verbose,
    };

    // Track current test for progress display
    if let Some(state) = worker_state {
        state.start_batch(test_files);
    }

    ctx.running.fetch_add(1, Ordering::SeqCst);
    ctx.stats.record_batch_size(ctx.test_files.len());
    let batch_start = Instant::now();
    let batch_id = std::thread::current().id();
    let file_info: Vec<_> = ctx.test_files.iter()
        .map(|f| {
            let pred = ctx.scheduler.predictions.get(f).copied().unwrap_or(0.0);
            format!("{}({:.2}s)", f, pred)
        })
        .collect();
    let total_pred: f64 = ctx.test_files.iter()
        .map(|f| ctx.scheduler.predictions.get(f).copied().unwrap_or(0.0))
        .sum();
    log_event("batch_start", &format!("{:?} files={} pred={:.2}s items=[{}]",
        batch_id, ctx.test_files.len(), total_pred, file_info.join(" ")));

    let test_time_sum = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let test_count = Arc::new(AtomicUsize::new(0));

    // RAII guard for cleanup
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

    // Spawn the slang-test process
    let (mut child, pipe_reader) = match spawn_batch_process(&ctx) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    };

    // Spawn output reader thread
    let output_reader = spawn_output_reader(pipe_reader);

    // Monitor batch execution
    let mut failed_outcomes: Vec<TestOutcome> = Vec::new();
    let expected_test_count = ctx.test_files.len();
    let monitor_start = Instant::now();

    let mut monitor_result = monitor_batch_execution(
        &mut child,
        &output_reader.channels,
        &ctx,
        expected_test_count,
        batch_id,
        &test_time_sum,
        &test_count,
        &mut failed_outcomes,
        worker_state,
    );

    // Clean up the child process
    if monitor_result.need_kill {
        debug_log!("killing child process");
        let _ = child.kill();
    }
    debug_log!("sending child to reaper");
    reap_process(child);

    let loop_time = monitor_start.elapsed();
    warn_slow_batch(&ctx, loop_time, total_pred);

    // Drain remaining outcomes after joining output thread
    debug_log!("drain: joining output thread");
    let join_result = output_reader.thread_handle.join();
    debug_log!("drain: output thread joined with result {:?}", join_result.is_ok());

    drain_remaining_outcomes(&output_reader.channels, &ctx, &mut monitor_result.seen_tests, &mut failed_outcomes);

    // Handle incomplete batches (crash/timeout)
    let all_tests_seen = monitor_result.seen_tests.len() >= expected_test_count;
    if !all_tests_seen && !is_interrupted() {
        handle_incomplete_batch(&ctx, &monitor_result.seen_tests, monitor_result.exit_status, monitor_result.killed_for_timeout, loop_time);
    }

    // Store failure info
    for outcome in failed_outcomes {
        let info = parse_failure_info(&outcome.name, &outcome.failure_output);
        ctx.failures.lock().unwrap().push(info);
    }

    // Clear worker state at end of batch
    if let Some(state) = worker_state {
        state.clear();
    }
}

// ============================================================================
// Test Runner - Helper Types
// ============================================================================

/// Effective parameters calculated for a test run
struct EffectiveParams {
    workers: usize,
    gpu_jobs: Option<usize>,
    batch_size: usize,
    batch_duration: f64,
    has_timing_data: bool,
    fudge_factor: f64,
}

/// Calculate effective parameters for a test run
fn calculate_effective_params(
    args: &crate::Args,
    timing_cache: &TimingCache,
    build_type: Option<BuildType>,
    test_files: &[String],
    predictions: &HashMap<String, f64>,
) -> EffectiveParams {
    let num_tests = test_files.len();
    let effective_workers = args.jobs.min(num_tests);

    let effective_gpu_jobs = args.gpu_jobs.map(|g| {
        let gpu_test_count = test_files.iter().filter(|t| TestId::parse(t).is_gpu_test()).count();
        g.min(gpu_test_count.max(1))
    });

    let has_timing_data = !args.no_timing_cache && build_type.is_some() &&
        build_type.map(|bt| timing_cache.has_timing_data(bt)).unwrap_or(false);

    let predicted_runtime = if has_timing_data {
        calculate_initial_eta(predictions, effective_workers, effective_gpu_jobs)
    } else {
        (test_files.len() as f64 * DEFAULT_PREDICTED_DURATION) / effective_workers as f64
    };

    let effective_batch_size = if args.batch_size == 0 {
        if effective_workers == 1 {
            usize::MAX
        } else if has_timing_data {
            ((test_files.len() / effective_workers) * 2).max(1)
        } else {
            50.min(test_files.len() / effective_workers).max(1)
        }
    } else {
        args.batch_size
    };

    let effective_batch_duration = if args.batch_duration == 0.0 {
        if effective_workers == 1 {
            f64::INFINITY
        } else if has_timing_data {
            (predicted_runtime / 2.0).max(1.0)
        } else {
            10.0
        }
    } else {
        args.batch_duration
    };

    let fudge_factor = if has_timing_data {
        build_type
            .map(|bt| timing_cache.average_fudge_factor(bt, test_files))
            .unwrap_or(1.0)
    } else {
        1.0
    };

    EffectiveParams {
        workers: effective_workers,
        gpu_jobs: effective_gpu_jobs,
        batch_size: effective_batch_size,
        batch_duration: effective_batch_duration,
        has_timing_data,
        fudge_factor,
    }
}

/// Build predictions for test durations
fn build_predictions(
    timing_cache: &TimingCache,
    build_type: Option<BuildType>,
    test_files: &[String],
) -> HashMap<String, f64> {
    test_files.iter()
        .map(|f| {
            let timing_key = test_to_timing_key(f);
            let pred = build_type
                .map(|bt| timing_cache.predict(bt, &timing_key))
                .unwrap_or(DEFAULT_PREDICTED_DURATION);
            (f.clone(), pred)
        })
        .collect()
}

/// Constrained random shuffle of tests based on timing predictions
fn shuffle_tests_by_timing(
    test_files: &[String],
    predictions: &HashMap<String, f64>,
    has_timing_data: bool,
) -> Vec<String> {
    use rand::Rng;
    use rand::seq::SliceRandom;

    if has_timing_data {
        let mut rng = rand::thread_rng();
        let total_predicted: f64 = predictions.values().sum();
        let n = test_files.len();

        if total_predicted > 0.0 && n > 0 {
            let mut files_with_pos: Vec<_> = test_files.iter()
                .map(|f| {
                    let dur = predictions.get(f).copied().unwrap_or(DEFAULT_PREDICTED_DURATION);
                    let latest = ((total_predicted - dur) / total_predicted).max(0.5);
                    let position = rng.gen_range(0.0..=latest);
                    (f.clone(), position)
                })
                .collect();

            files_with_pos.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            files_with_pos.into_iter().map(|(f, _)| f).collect()
        } else {
            let mut files = test_files.to_vec();
            files.shuffle(&mut rng);
            files
        }
    } else {
        let mut files = test_files.to_vec();
        files.shuffle(&mut rand::thread_rng());
        files
    }
}

/// Print warnings about GPU and unknown APIs
fn print_api_warnings(
    unsupported_apis: Option<&UnsupportedApis>,
    unknown_apis: &HashSet<String>,
    has_unknown_apis: bool,
) {
    // Don't print warnings if interrupted
    if is_interrupted() {
        return;
    }

    if let Some(unsupported) = unsupported_apis {
        if unsupported.gpu_disabled {
            let apis = UnsupportedApis::disabled_gpu_apis();
            eprintln!("{}", format!(
                "GPU tests skipped (-g 0): {}, gfx-unit-test-tool",
                apis.join(", ")
            ).dimmed());
        }
    }

    if has_unknown_apis {
        let mut apis_list: Vec<_> = unknown_apis.iter().collect();
        apis_list.sort();
        let apis_str = apis_list.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
        eprintln!("{}", format!(
            "Warning: Found tests for unknown APIs ({}) - will not skip API detection in batch runs",
            apis_str
        ).dimmed());
    }
}

/// Spawn the progress display thread
fn spawn_progress_thread(
    stats: Arc<TestStats>,
    running: Arc<AtomicUsize>,
    scheduler_handle: SchedulerHandle,
    shutdown: Arc<AtomicBool>,
    total_files: usize,
    machine_output: bool,
    num_workers: usize,
    verbose: bool,
    fudge_factor: f64,
    worker_states: Option<Arc<WorkerStates>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut display = ProgressDisplay::new(total_files, machine_output, num_workers, verbose, fudge_factor);
        let mut sys_stats: Option<SystemStats> = None;
        let mut stats_counter = 0u32;
        while !shutdown.load(Ordering::SeqCst) {
            let files_done = stats.files_completed();
            let batches_running = running.load(Ordering::SeqCst);

            let status = scheduler_handle.get_status();
            let eta = if scheduler_handle.has_timing_data {
                Some(status.eta)
            } else {
                None
            };
            display.update(&stats, files_done, batches_running, status.remaining, status.has_pending_batches, eta, worker_states.as_deref());

            stats_counter += 1;
            if stats_counter >= 10 {
                stats_counter = 0;
                let sys = sys_stats.get_or_insert_with(SystemStats::new);
                sys.refresh_and_log(batches_running, status.remaining);
            }

            thread::sleep(Duration::from_millis(PROGRESS_UPDATE_INTERVAL_MS));
        }
        display.finish(&stats);
    })
}

/// Spawn a single worker thread
fn spawn_worker_thread(
    worker_id: usize,
    slang_test: PathBuf,
    root_dir: PathBuf,
    extra_args: Vec<String>,
    timeout: Duration,
    stats: Arc<TestStats>,
    failures: Arc<Mutex<Vec<FailureInfo>>>,
    retries: usize,
    retried_tests: Arc<Mutex<HashMap<String, usize>>>,
    scheduler_handle: SchedulerHandle,
    running: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    verbose: bool,
    worker_states: Option<Arc<WorkerStates>>,
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name(format!("worker-{}", worker_id))
        .spawn(move || {
            debug_log!("started, getting first batch");
            let my_state = worker_states.as_ref().map(|ws| ws.get(worker_id));
            loop {
                if shutdown.load(Ordering::SeqCst) {
                    debug_log!("shutdown flag set, exiting");
                    break;
                }
                if is_interrupted() {
                    debug_log!("interrupted, exiting");
                    break;
                }

                let get_batch_start = Instant::now();
                if let Some(assignment) = scheduler_handle.get_batch() {
                    let get_batch_time = get_batch_start.elapsed();
                    if get_batch_time.as_millis() > 10 {
                        debug_log!("get_batch took {:.3}s", get_batch_time.as_secs_f64());
                    }
                    debug_log!("got batch {} of {} tests ({:?})", assignment.batch_id, assignment.tests.len(), assignment.kind);
                    run_batch_with_pool(
                        &slang_test,
                        &root_dir,
                        &assignment.tests,
                        &extra_args,
                        timeout,
                        &stats,
                        &failures,
                        retries,
                        &retried_tests,
                        &scheduler_handle,
                        &running,
                        verbose,
                        my_state,
                    );
                    debug_log!("batch {} completed, calling complete_batch", assignment.batch_id);
                    scheduler_handle.complete_batch(assignment.batch_id);
                    debug_log!("batch {} complete_batch done", assignment.batch_id);
                } else {
                    let status = scheduler_handle.get_status();
                    debug_log!("no batch available: batches={} pending={} in_flight={:?}",
                        status.debug_state.0, status.debug_state.1, status.debug_state.2);
                    thread::sleep(Duration::from_millis(10));
                }
            }
            debug_log!("worker loop exited");
        })
        .expect("failed to spawn worker thread")
}

/// Run the main monitoring loop until all work is complete
fn run_main_monitoring_loop(
    scheduler_handle: &SchedulerHandle,
    running: &AtomicUsize,
    shutdown: &AtomicBool,
) {
    debug_log!("entering main loop");

    loop {
        let status = scheduler_handle.get_status();
        let running_count = running.load(Ordering::SeqCst);

        if status.is_empty && running_count == 0 {
            debug_log!("main loop: scheduler empty and no batches running, exiting");
            break;
        }

        if is_interrupted() {
            debug_log!("main loop: interrupted, exiting");
            break;
        }

        static LAST_LOG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let now_ms = DEBUG_START.elapsed().as_millis() as u64;
        let last = LAST_LOG.load(Ordering::Relaxed);
        if now_ms - last > 1000 {
            LAST_LOG.store(now_ms, Ordering::Relaxed);
            debug_log!("main loop: running={} batches={} pending={} in_flight={:?}",
                running_count, status.debug_state.0, status.debug_state.1, status.debug_state.2);
        }

        thread::sleep(Duration::from_millis(20));
    }

    debug_log!("main loop exited, setting shutdown flag");
    shutdown.store(true, Ordering::SeqCst);
}

// ============================================================================
// Test Runner
// ============================================================================

pub struct TestRunner {
    pub args: crate::Args,
    pub stats: Arc<TestStats>,
    pub failures: Arc<Mutex<Vec<FailureInfo>>>,
    pub retried_tests: Arc<Mutex<HashMap<String, usize>>>,
    pub machine_output: bool,
    pub timing_cache: Mutex<TimingCache>,
    pub build_type: Option<BuildType>,
    /// Unsupported APIs detected at startup (or None if check disabled)
    pub unsupported_apis: Option<UnsupportedApis>,
    /// Count of tests ignored due to unsupported APIs
    pub api_ignored_count: AtomicUsize,
    /// Whether to pass -skip-api-detection to slang-test (when we've done the check)
    pub skip_api_detection: bool,
    /// APIs found in tests but not in Check output (unknown APIs)
    pub unknown_apis: HashSet<String>,
    /// Pre-discovered test files (from concurrent discovery)
    pub discovered_tests: Vec<String>,
}

impl TestRunner {
    /// Create a TestRunner with pre-discovered data from concurrent discovery
    pub fn new_with_discovery(args: crate::Args, discovery: DiscoveryResult) -> Self {
        let machine_output = !crate::is_stderr_tty();
        let build_type = args.slang_test.as_ref()
            .and_then(|p| BuildType::from_path(p));
        Self {
            args,
            stats: Arc::new(TestStats::default()),
            failures: Arc::new(Mutex::new(Vec::new())),
            retried_tests: Arc::new(Mutex::new(HashMap::new())),
            machine_output,
            timing_cache: Mutex::new(discovery.timing_cache),
            build_type,
            unsupported_apis: discovery.unsupported_apis,
            api_ignored_count: AtomicUsize::new(discovery.api_ignored_count),
            skip_api_detection: discovery.skip_api_detection,
            unknown_apis: discovery.unknown_apis,
            discovered_tests: discovery.tests,
        }
    }

    pub fn save_timing(&self) {
        if self.args.no_timing_cache {
            return;
        }
        let Some(build_type) = self.build_type else {
            return;
        };
        let observed = self.stats.get_observed_timings();
        let mut cache = self.timing_cache.lock().unwrap();

        if !observed.is_empty() {
            cache.merge(build_type, &observed);
        }

        if let Some(fudge) = self.stats.calculate_fudge_factor() {
            let test_files = self.stats.get_test_files();
            if !test_files.is_empty() {
                cache.record_fudge_factors(build_type, &test_files, fudge);
            }
        }

        cache.save();
    }

    pub fn run(&self) -> Result<bool> {
        let start_time = Instant::now();

        // Use pre-discovered tests from concurrent discovery
        let test_files = &self.discovered_tests;
        let api_ignored = self.api_ignored_count.load(Ordering::SeqCst);
        let has_unknown_apis = !self.unknown_apis.is_empty();

        if test_files.is_empty() {
            // Only print "no tests found" if not interrupted (otherwise it's expected)
            if !is_interrupted() {
                let msg = if api_ignored > 0 {
                    format!("No tests found matching the specified criteria (ignoring {} tests on unsupported APIs)", api_ignored)
                } else {
                    "No tests found matching the specified criteria".to_string()
                };
                eprintln!("{}", msg);
            }
        } else {
            // Note: dry_run is handled in main.rs before creating TestRunner
            self.run_file_tests(test_files, has_unknown_apis)?;
        }

        let elapsed = start_time.elapsed();

        if is_interrupted() {
            eprintln!("\n{}", "Interrupted by Ctrl-C".red().bold());
        }

        self.print_summary(elapsed);

        Ok(self.stats.failed.load(Ordering::SeqCst) == 0 && !is_interrupted())
    }

    fn run_file_tests(&self, test_files: &[String], has_unknown_apis: bool) -> Result<()> {
        if test_files.is_empty() {
            return Ok(());
        }

        // Build predictions and calculate effective parameters
        let predictions = {
            let cache = self.timing_cache.lock().unwrap();
            build_predictions(&cache, self.build_type, test_files)
        };

        let params = {
            let cache = self.timing_cache.lock().unwrap();
            calculate_effective_params(&self.args, &cache, self.build_type, test_files, &predictions)
        };

        // Shuffle tests for better parallelization
        let sorted_files = shuffle_tests_by_timing(test_files, &predictions, params.has_timing_data);

        // Print warnings
        print_api_warnings(self.unsupported_apis.as_ref(), &self.unknown_apis, has_unknown_apis);

        let skip_api_detection = self.skip_api_detection && !has_unknown_apis;

        // Record initial ETA prediction
        if params.has_timing_data {
            let raw_eta = calculate_initial_eta(&predictions, params.workers, params.gpu_jobs);
            self.stats.record_initial_prediction(raw_eta, test_files.to_vec());
        } else {
            *self.stats.test_files.lock().unwrap() = test_files.to_vec();
        }

        if self.args.verbose {
            eprintln!(
                "{}",
                format!(
                    "Batch config: max_size={}, max_duration={:.1}s",
                    params.batch_size, params.batch_duration
                ).dimmed()
            );
        }

        // Create scheduler
        let (mut scheduler, scheduler_handle) = Scheduler::new(
            sorted_files,
            params.batch_size,
            params.workers,
            predictions,
            params.has_timing_data,
            params.batch_duration,
            params.gpu_jobs,
            self.args.gpu_stagger,
        );

        // Shared state
        let running = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let timeout = Duration::from_secs(self.args.timeout);
        let worker_states = if !self.machine_output {
            Some(Arc::new(WorkerStates::new(params.workers)))
        } else {
            None
        };

        self.stats.mark_execution_started();

        // Spawn scheduler thread
        debug_log!("about to spawn scheduler thread");
        let scheduler_thread = thread::Builder::new()
            .name("scheduler".to_string())
            .spawn(move || {
                scheduler.run();
            })
            .expect("failed to spawn scheduler thread");

        // Spawn progress thread
        debug_log!("scheduler thread spawned, about to spawn progress thread");
        let progress_shutdown = Arc::new(AtomicBool::new(false));
        let progress_handle = spawn_progress_thread(
            self.stats.clone(),
            running.clone(),
            scheduler_handle.clone(),
            progress_shutdown.clone(),
            test_files.len(),
            self.machine_output,
            params.workers,
            self.args.verbose,
            params.fudge_factor,
            worker_states.clone(),
        );

        // Spawn workers and run main loop
        debug_log!("progress thread spawned, about to spawn workers");
        let initial_status = scheduler_handle.get_status();
        if !initial_status.is_empty {
            let mut handles = Vec::new();

            for i in 0..params.workers {
                debug_log!("spawning worker {}", i);
                let mut extra_args = self.args.extra_args.clone();
                if skip_api_detection {
                    extra_args.push("-skip-api-detection".to_string());
                }

                let handle = spawn_worker_thread(
                    i,
                    self.args.slang_test.as_ref().unwrap().clone(),
                    self.args.root_dir_effective.clone(),
                    extra_args,
                    timeout,
                    self.stats.clone(),
                    self.failures.clone(),
                    self.args.retries,
                    self.retried_tests.clone(),
                    scheduler_handle.clone(),
                    running.clone(),
                    shutdown.clone(),
                    self.args.verbose,
                    worker_states.clone(),
                );
                handles.push(handle);
            }

            debug_log!("all workers spawned, entering main loop");
            run_main_monitoring_loop(&scheduler_handle, &running, &shutdown);

            debug_log!("dropping worker handles");
            drop(handles);
        }

        // Cleanup
        scheduler_handle.shutdown();
        let _ = scheduler_thread.join();

        progress_shutdown.store(true, Ordering::SeqCst);
        let _ = progress_handle.join();

        Ok(())
    }

    fn show_diff(&self, expected: &str, actual: &str) {
        let indent = if self.machine_output { "" } else { "  " };
        let indent2 = if self.machine_output { "" } else { "    " };

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
            return;
        }

        match resolve_diff_tool(self.args.diff.as_str()) {
            "none" => {
                if self.machine_output {
                    println!("Expected:");
                } else {
                    println!("{}{}:", indent, "Expected".green());
                }
                for line in expected.lines() {
                    if self.machine_output {
                        println!("{}", line);
                    } else {
                        println!("{}{}", indent2, line.green());
                    }
                }
                if self.machine_output {
                    println!("Actual:");
                } else {
                    println!("{}{}:", indent, "Actual".red());
                }
                for line in actual.lines() {
                    if self.machine_output {
                        println!("{}", line);
                    } else {
                        println!("{}{}", indent2, line.red());
                    }
                }
            }
            "difft" => {
                self.run_external_diff("difft", &["--color", "always"], expected, actual);
            }
            _ => {
                if let Some(w) = TERM_WIDTH.as_ref().filter(|_| !self.machine_output) {
                    self.run_external_diff("diff", &["-y", "--color=always", "-t", "-W", w], expected, actual);
                } else {
                    self.run_external_diff("diff", &["-y", "--color=always", "-t"], expected, actual);
                }
            }
        }
    }

    fn run_external_diff(&self, cmd: &str, args: &[&str], expected: &str, actual: &str) {
        print!("{}", Self::compute_external_diff(cmd, args, expected, actual, self.machine_output));
    }

    fn compute_external_diff(cmd: &str, args: &[&str], expected: &str, actual: &str, machine_output: bool) -> String {
        use std::io::Write as _;

        let indent = if machine_output { "" } else { "  " };

        let id = std::process::id();
        let thread_id = format!("{:?}", std::thread::current().id());
        let expected_file = std::env::temp_dir().join(format!("slang-test-expected-{}-{}.txt", id, thread_id.replace(|c: char| !c.is_alphanumeric(), "")));
        let actual_file = std::env::temp_dir().join(format!("slang-test-actual-{}-{}.txt", id, thread_id.replace(|c: char| !c.is_alphanumeric(), "")));

        let mut result = String::new();

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
                        result.push_str(&format!("{}{}\n", indent, line));
                    }
                }
                Err(_) => {
                    result.push_str(&format!("{}({} not available, showing raw output)\n", indent, cmd));
                    result.push_str(&format!("{}Expected:\n", indent));
                    for line in expected.lines() {
                        result.push_str(&format!("{}  {}\n", indent, line));
                    }
                    result.push_str(&format!("{}Actual:\n", indent));
                    for line in actual.lines() {
                        result.push_str(&format!("{}  {}\n", indent, line));
                    }
                }
            }

            let _ = std::fs::remove_file(&expected_file);
            let _ = std::fs::remove_file(&actual_file);
        }

        result
    }

    fn print_summary(&self, elapsed: Duration) {
        let passed = self.stats.passed.load(Ordering::SeqCst);
        let failed = self.stats.failed.load(Ordering::SeqCst);
        let ignored = self.stats.ignored.load(Ordering::SeqCst);
        let retried = self.stats.retried_and_passed.load(Ordering::SeqCst);

        let failures = self.failures.lock().unwrap();

        if !failures.is_empty() {
            println!("\n{}", "=".repeat(70));
            println!("{}", "FAILURES".red().bold());
            println!("{}", "=".repeat(70));

            let mut sorted_failures: Vec<_> = failures.iter().collect();
            sorted_failures.sort_by(|a, b| a.test_name.cmp(&b.test_name));

            let diff_tool = resolve_diff_tool(self.args.diff.as_str());
            let machine_output = self.machine_output;
            let diff_results: Vec<Option<String>> = if diff_tool != "none" {
                let handles: Vec<_> = sorted_failures
                    .iter()
                    .map(|failure| {
                        if let (Some(expected), Some(actual)) = (&failure.expected, &failure.actual) {
                            let expected = expected.clone();
                            let actual = actual.clone();
                            let tool = diff_tool.to_string();
                            Some(thread::spawn(move || {
                                let (cmd, args): (&str, Vec<&str>) = if tool == "difft" {
                                    ("difft", vec!["--color", "always"])
                                } else if let Some(w) = TERM_WIDTH.as_ref().filter(|_| !machine_output) {
                                    ("diff", vec!["-y", "--color=always", "-t", "-W", w])
                                } else {
                                    ("diff", vec!["-y", "--color=always", "-t"])
                                };
                                Self::compute_external_diff(cmd, &args, &expected, &actual, machine_output)
                            }))
                        } else {
                            None
                        }
                    })
                    .collect();

                handles
                    .into_iter()
                    .map(|h| h.map(|handle| handle.join().unwrap_or_default()))
                    .collect()
            } else {
                vec![None; sorted_failures.len()]
            };

            for (failure, diff_output) in sorted_failures.iter().zip(diff_results.iter()) {
                println!("\n{}", failure.test_name.red().bold().underline());

                if let (Some(expected), Some(actual)) = (&failure.expected, &failure.actual) {
                    if let Some(diff) = diff_output {
                        print!("{}", diff);
                    } else {
                        self.show_diff(expected, actual);
                    }
                } else if !failure.output_lines.is_empty() {
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
                }
            }
        }

        println!("\n{}", "=".repeat(70));

        let total_run = passed + failed;
        let interrupted = is_interrupted();
        let total_tests = self.stats.get_test_files().len();
        let not_run = total_tests.saturating_sub(passed + failed + ignored);

        if total_run > 0 {
            if interrupted {
                if failed == 0 {
                    println!(
                        "{}: {} passed, {} ignored, {} not run in {:.1}s",
                        "Interrupted".yellow().bold(),
                        passed,
                        ignored,
                        not_run,
                        elapsed.as_secs_f64()
                    );
                } else {
                    println!(
                        "{}: {} passed, {} failed, {} ignored, {} not run in {:.1}s",
                        "Interrupted".yellow().bold(),
                        passed,
                        failed,
                        ignored,
                        not_run,
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

        if self.args.verbose {
            let observed = self.stats.get_observed_timings();
            if !observed.is_empty() {
                let mut slowest: Vec<_> = observed.iter().collect();
                slowest.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                slowest.truncate(15);

                if !slowest.is_empty() && *slowest[0].1 > 1.0 {
                    println!("\n{}:", "Slowest tests".yellow());
                    for (test_id, secs) in &slowest {
                        if **secs > 0.5 {
                            println!("  {:>6.1}s  {}", secs, test_id);
                        }
                    }
                }
            }

            let batch_sizes = self.stats.get_batch_sizes();
            if !batch_sizes.is_empty() {
                println!("\n{}:", "Batch size distribution".yellow());
                self.print_batch_histogram(&batch_sizes);
            }
        }

        if !failures.is_empty() {
            let failed_tests: Vec<&str> = failures.iter().map(|f| f.test_name.as_str()).collect();
            let test_args = minimize_test_filters(&failed_tests);

            println!("\n{}", "To rerun failed tests:".yellow());

            let has_file_tests = test_args.iter().any(|f| {
                f.ends_with(".slang") || f.ends_with(".hlsl") || f.ends_with(".glsl") || f.ends_with(".c")
                    || f.contains(".slang.") || f.contains(".hlsl.") || f.contains(".glsl.") || f.contains(".c.")
            });

            let exe = std::env::args().next().unwrap_or_else(|| "sti".to_string());
            if has_file_tests {
                print!("{}", exe);

                if let Some(ref root_dir) = self.args.root_dir_original {
                    print!(" -C {}", root_dir);
                }
                if let Some(ref build_type) = self.args.build_type {
                    print!(" --build-type {}", build_type);
                }
                if let Some(ref slang_test) = self.args.slang_test_original {
                    print!(" --slang-test {}", slang_test);
                }

                for arg in &test_args {
                    if arg.contains('|') || arg.contains('(') || arg.contains(')') {
                        print!(" '{}'", arg);
                    } else {
                        print!(" {}", arg);
                    }
                }

                if !self.args.extra_args.is_empty() {
                    print!(" --");
                    for arg in &self.args.extra_args {
                        print!(" {}", arg);
                    }
                }

                println!();
            } else {
                print!("{}", self.args.slang_test.as_ref().unwrap().display());
                for arg in &test_args {
                    print!(" \"{}\"", arg);
                }
                println!();
            }
        }

        println!("{}", "=".repeat(70));
    }

    fn print_batch_histogram(&self, batch_sizes: &HashMap<usize, usize>) {
        if batch_sizes.is_empty() {
            return;
        }

        let mut sizes: Vec<_> = batch_sizes.iter().collect();
        sizes.sort_by_key(|(size, _)| *size);

        let max_count = *sizes.iter().map(|(_, count)| *count).max().unwrap_or(&1);
        let total_batches: usize = sizes.iter().map(|(_, count)| **count).sum();

        const BRAILLE: [char; 9] = ['⠀', '⡀', '⡄', '⡆', '⡇', '⣇', '⣧', '⣷', '⣿'];
        const BAR_WIDTH: usize = 30;

        for (size, count) in &sizes {
            let ratio = **count as f64 / max_count as f64;
            let full_chars = (ratio * BAR_WIDTH as f64) as usize;
            let partial = ((ratio * BAR_WIDTH as f64 - full_chars as f64) * 8.0) as usize;

            let mut bar = String::new();
            for _ in 0..full_chars {
                bar.push('⣿');
            }
            if full_chars < BAR_WIDTH {
                bar.push(BRAILLE[partial.min(8)]);
                for _ in (full_chars + 1)..BAR_WIDTH {
                    bar.push('⠀');
                }
            }

            let pct = (**count as f64 / total_batches as f64) * 100.0;
            println!("  {:>3} tests │{}│ {:>4} ({:>4.1}%)", size, bar, count, pct);
        }

        println!("  {:>10} {:>32} batches", "", total_batches);
    }
}
