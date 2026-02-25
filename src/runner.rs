use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::types::{*, TestId, test_to_timing_key, WorkerState, WorkerStates};

// ============================================================================
// Debug Logging
// ============================================================================

static DEBUG_ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("STI_DEBUG").is_ok());
static DEBUG_START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Print a debug message with timestamp and thread ID, only if STI_DEBUG is set
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if *DEBUG_ENABLED {
            let thread_id = std::thread::current().id();
            let thread_name = std::thread::current().name().unwrap_or("?").to_string();
            eprintln!("{}", format!("[DEBUG {:>6.3}s] [{}:{}] {}",
                DEBUG_START.elapsed().as_secs_f64(),
                thread_name,
                format!("{:?}", thread_id).trim_start_matches("ThreadId(").trim_end_matches(")"),
                format!($($arg)*)
            ).dimmed());
        }
    };
}

// ============================================================================
// ETA Calculation and Display
// ============================================================================

/// Calculate initial ETA from predictions.
/// ETA = max(total_predicted / num_workers, longest_single_test)
/// This accounts for the "long pole" problem where one slow test dominates.
fn calculate_initial_eta(predictions: impl Iterator<Item = f64>, num_workers: usize) -> f64 {
    let mut total = 0.0f64;
    let mut longest = 0.0f64;
    for pred in predictions {
        total += pred;
        longest = longest.max(pred);
    }
    let parallel_eta = total / num_workers.max(1) as f64;
    parallel_eta.max(longest)
}

/// Format the "Running N tests with M workers" message.
fn format_running_message(num_tests: usize, num_workers: usize, predicted_eta: Option<f64>) -> String {
    match predicted_eta {
        Some(eta) => format!(
            "Running {} tests with {} workers {}",
            num_tests, num_workers, format!("(predicted {:.0}s)", eta).dimmed()
        ),
        None => format!(
            "Running {} tests with {} workers",
            num_tests, num_workers
        ),
    }
}

// ============================================================================
// Global Flags
// ============================================================================

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static COMPILING_CORE: AtomicBool = AtomicBool::new(false);

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

pub fn is_compiling_core() -> bool {
    COMPILING_CORE.load(Ordering::SeqCst)
}

pub fn set_compiling_core(compiling: bool) {
    COMPILING_CORE.store(compiling, Ordering::SeqCst);
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

// ============================================================================
// Retry and Outcome Processing
// ============================================================================

fn should_retry_test(
    outcome: &TestOutcome,
    max_retries: usize,
    retried_tests: &Mutex<HashMap<String, usize>>,
    work_pool: &Arc<WorkPool>,
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
                work_pool.add_file(outcome.name.clone());
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
            if should_retry_test(&outcome, ctx.max_retries, ctx.retried_tests, ctx.work_pool) {
                return true;
            }
            ctx.stats.failed.fetch_add(1, Ordering::SeqCst);
            failed_outcomes.push(outcome);
            false
        }
    }
}


// ============================================================================
// Batch Execution
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
    work_pool: &Arc<WorkPool>,
    running: &AtomicUsize,
    machine_output: bool,
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
        work_pool,
        running,
        machine_output,
        verbose,
    };

    // Track current test for progress display
    if let Some(state) = worker_state {
        state.start_batch(test_files);
    }

    ctx.running.fetch_add(1, Ordering::SeqCst);
    let batch_start = Instant::now();
    let batch_id = std::thread::current().id();
    let file_info: Vec<_> = ctx.test_files.iter()
        .map(|f| {
            let pred = ctx.work_pool.predictions.get(f).copied().unwrap_or(0.0);
            format!("{}({:.2}s)", f, pred)
        })
        .collect();
    let total_pred: f64 = ctx.test_files.iter()
        .map(|f| ctx.work_pool.predictions.get(f).copied().unwrap_or(0.0))
        .sum();
    log_event("batch_start", &format!("{:?} files={} pred={:.2}s items=[{}]",
        batch_id, ctx.test_files.len(), total_pred, file_info.join(" ")));

    let test_time_sum = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let test_count = Arc::new(AtomicUsize::new(0));

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

    // Convert test names to the format slang-test accepts (strip API suffix)
    let slang_test_args: Vec<String> = ctx.test_files
        .iter()
        .map(|f| TestId::parse(f).to_slang_test_arg())
        .collect();

    // Create a shared pipe for stdout and stderr so we get ordered output
    let (pipe_reader, pipe_writer) = match os_pipe::pipe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: Failed to create pipe: {}", e);
            return;
        }
    };
    let pipe_writer2 = match pipe_writer.try_clone() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: Failed to clone pipe: {}", e);
            return;
        }
    };

    let mut cmd = Command::new(ctx.slang_test);
    cmd.current_dir(ctx.root_dir)
        .arg("-explicit-test-order")
        .arg("-disable-retries")
        .args(&slang_test_args)
        .args(ctx.extra_args)
        .stdout(pipe_writer)
        .stderr(pipe_writer2);

    debug_log!("spawning slang-test for {} tests", ctx.test_files.len());
    let mut child = match cmd.spawn() {
        Ok(child) => {
            debug_log!("slang-test spawned successfully");
            child
        }
        Err(e) => {
            eprintln!("ERROR: Failed to spawn slang-test: {}", e);
            return;
        }
    };

    // IMPORTANT: Drop the Command to close the pipe writers in the parent process.
    // Otherwise the pipe reader will never get EOF when the child dies.
    drop(cmd);

    let (outcome_tx, outcome_rx) = crossbeam_channel::unbounded::<TestOutcome>();
    let (timing_tx, timing_rx) = crossbeam_channel::unbounded::<(String, f64)>(); // (test_id, duration)
    let (done_tx, done_rx) = crossbeam_channel::bounded::<bool>(1); // Signal when output is complete
    let (compiling_tx, compiling_rx) = crossbeam_channel::unbounded::<bool>();

    let machine_output_for_stderr = ctx.machine_output;

    // Single reader thread for combined stdout/stderr
    let output_handle = thread::Builder::new()
        .name("output-reader".to_string())
        .spawn(move || {
        let reader = BufReader::new(pipe_reader);

        let mut last_test_time: Option<Instant> = None;
        // Accumulate lines between test results - these are failure details
        let mut pending_output: Vec<String> = Vec::new();
        // Track if we've finished the initial slang-test spew (Supported backends, Check lines)
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
                    // Signal early that we're done - don't wait for process exit
                    let _ = done_tx.try_send(true);
                    continue;
                }

                // Handle "Compiling" messages (originally from stderr)
                if line.contains("Compiling core module") {
                    set_compiling_core(true);
                    let _ = compiling_tx.send(true);
                    if machine_output_for_stderr {
                        eprintln!("{}", format!("INFO: {}", line).dimmed());
                    }
                    continue;
                } else if line.contains("Compiling") {
                    let _ = compiling_tx.send(true);
                    continue;
                }

                // Skip slang-test initial output (before any tests run)
                if line.starts_with("Supported backends:")
                    || line.starts_with("Check ")
                    || line.starts_with("Retrying ")
                {
                    last_test_time = Some(Instant::now());
                    continue;
                }

                // Detect end of initial spew - test output starts with [test_name] or error markers
                if !past_initial_spew {
                    if line.starts_with('[') || line.contains("EXPECTED{{{") || line.contains("ACTUAL{{{") {
                        past_initial_spew = true;
                    }
                }

                // Try to parse as test result line
                if let Some(mut outcome) = parse_test_output(&line) {
                    let now = Instant::now();

                    // Record timing for all test results (passed, failed, ignored)
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

                    // Attach accumulated output to this outcome (failure details come before the FAILED line)
                    if outcome.result == TestResult::Failed {
                        outcome.failure_output = std::mem::take(&mut pending_output);
                    } else {
                        // Clear pending output for passed/ignored tests
                        pending_output.clear();
                    }

                    past_initial_spew = true;

                    // Extract the timing key
                    let test_id = TestId::parse(&outcome.name);
                    let timing_key = test_id.to_timing_key();

                    if test_duration > 0.0 {
                        let _ = timing_tx.send((timing_key.clone(), test_duration));
                    }

                    let _ = outcome_tx.send(outcome);
                } else if past_initial_spew {
                    // Accumulate non-result lines (failure details, error messages)
                    // Only after we've passed the initial slang-test spew
                    pending_output.push(line);
                }
            }
        }
        let _ = compiling_tx.send(false);
        debug_log!("output reader exiting after {} lines", lines_read);
    }).expect("failed to spawn output reader thread");

    let mut failed_outcomes: Vec<TestOutcome> = Vec::new();
    let start = Instant::now();
    let mut this_batch_is_compiling = false;
    let mut killed_for_compilation = false;
    let mut killed_for_timeout = false;
    let mut exit_status: Option<std::process::ExitStatus> = None;
    let mut seen_tests: HashSet<String> = HashSet::new();
    let expected_test_count = ctx.test_files.len();

    let mut need_kill = false;
    debug_log!("entering batch monitor loop, expecting {} tests", expected_test_count);
    loop {
        if is_interrupted() {
            debug_log!("batch monitor: interrupted");
            need_kill = true;
            break;
        }

        while let Ok(is_compiling) = compiling_rx.try_recv() {
            ctx.stats.set_compiling(is_compiling);
            if is_compiling {
                this_batch_is_compiling = true;
            }
        }

        if is_compiling_core() && !this_batch_is_compiling {
            debug_log!("batch monitor: killing for compilation");
            need_kill = true;
            killed_for_compilation = true;
            break;
        }

        while let Ok((test_id, duration)) = timing_rx.try_recv() {
            ctx.stats.record_observed_timing(&test_id, duration);
        }

        while let Ok(outcome) = outcome_rx.try_recv() {
            ctx.stats.set_compiling(false);
            ctx.stats.record_test_output();
            if this_batch_is_compiling {
                set_compiling_core(false);
                this_batch_is_compiling = false;
            }
            let result_str = match outcome.result {
                TestResult::Passed => "passed",
                TestResult::Failed => "failed",
                TestResult::Ignored => "ignored",
            };
            let duration_ms = outcome.duration.map(|d| d.as_millis() as u64).unwrap_or(0);
            test_time_sum.fetch_add(duration_ms, Ordering::SeqCst);
            test_count.fetch_add(1, Ordering::SeqCst);
            log_event("test", &format!("{:?} {} {} duration_ms={}", batch_id, result_str, outcome.name, duration_ms));

            // Track seen tests for crash detection
            let test_id = TestId::parse(&outcome.name);
            seen_tests.insert(test_id.to_timing_key());

            process_outcome(outcome, &ctx, &mut failed_outcomes);

            // Advance to next test in batch for progress display
            if let Some(state) = worker_state {
                state.advance();
            }
        }

        // Check if we've seen all expected tests - no need to wait for slang-test
        // With shared stdout/stderr pipe, failure output comes before next test result
        if seen_tests.len() >= expected_test_count {
            debug_log!("batch monitor: all {} tests seen", expected_test_count);
            break;
        }

        // Check if output is complete (saw summary or "no tests run")
        if done_rx.try_recv().is_ok() {
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

    // Clean up the child process
    if need_kill {
        debug_log!("killing child process");
        let _ = child.kill();
    }
    debug_log!("sending child to reaper");
    reap_process(child);

    let loop_time = start.elapsed();
    if ctx.verbose && loop_time.as_secs() > 30 {
        let extra_args_str = if ctx.extra_args.is_empty() {
            String::new()
        } else {
            format!(" {}", ctx.extra_args.join(" "))
        };
        eprintln!("{}", format!("\nWARNING: Batch took {:.1}s", loop_time.as_secs_f64()).dimmed());
        eprintln!("{}", format!("  Reproduce: time {} {}{}",
            ctx.slang_test.display(),
            ctx.test_files.join(" "),
            extra_args_str
        ).dimmed());
    }

    if killed_for_compilation {
        debug_log!("killed for compilation, repooling {} tests", ctx.test_files.len());
        // Clear worker state before returning
        if let Some(state) = worker_state {
            state.clear();
        }
        for file in ctx.test_files {
            ctx.work_pool.add_file(file.to_string());
        }
        debug_log!("waiting for core compilation to complete");
        while is_compiling_core() && !is_interrupted() {
            thread::sleep(Duration::from_millis(50));
        }
        debug_log!("core compilation wait finished");
        return;
    }

    // Drain remaining outcomes from the channel.
    // The output thread will close the channel when it exits (after EOF on pipe).
    // We join the output thread to ensure we get all outcomes before proceeding.
    debug_log!("drain: joining output thread");
    let join_result = output_handle.join();
    debug_log!("drain: output thread joined with result {:?}", join_result.is_ok());

    // Now drain any remaining items from the channels (they're closed, so this is non-blocking)
    debug_log!("drain: draining remaining outcomes");
    let mut drained_count = 0;
    while let Ok(outcome) = outcome_rx.try_recv() {
        debug_log!("drain: got outcome {}", outcome.name);
        drained_count += 1;

        // Track seen tests for crash detection
        let test_id = TestId::parse(&outcome.name);
        seen_tests.insert(test_id.to_timing_key());

        process_outcome(outcome, &ctx, &mut failed_outcomes);
    }
    debug_log!("drain: drained {} outcomes", drained_count);
    // Final drain of timing channel
    while let Ok((test_id, duration)) = timing_rx.try_recv() {
        ctx.stats.record_observed_timing(&test_id, duration);
    }

    let all_tests_seen = seen_tests.len() >= expected_test_count;
    if !all_tests_seen && !is_interrupted() && !killed_for_compilation {
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

            for test in &tests_to_repool {
                ctx.work_pool.add_file(test.clone());
            }

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
            for test in &unaccounted_tests {
                ctx.work_pool.add_file(test.clone());
            }
        }
    }

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
}

impl TestRunner {
    pub fn new(args: crate::Args) -> Self {
        let machine_output = !crate::is_stderr_tty();
        // Detect build type from slang-test path
        let build_type = args.slang_test.as_ref()
            .and_then(|p| BuildType::from_path(p));
        // Don't load timing cache yet - will be loaded concurrently with discovery
        Self {
            args,
            stats: Arc::new(TestStats::default()),
            failures: Arc::new(Mutex::new(Vec::new())),
            retried_tests: Arc::new(Mutex::new(HashMap::new())),
            machine_output,
            timing_cache: Mutex::new(TimingCache::default()),
            build_type,
        }
    }

    pub fn save_timing(&self) {
        // Only save if we have a known build type
        let Some(build_type) = self.build_type else {
            return;
        };
        let observed = self.stats.get_observed_timings();
        if !observed.is_empty() {
            let mut cache = self.timing_cache.lock().unwrap();
            cache.merge(build_type, &observed);
            cache.save();
        }
    }

    pub fn run(&self) -> Result<bool> {
        let start_time = Instant::now();

        // Load timing cache concurrently with test discovery (via channel)
        let load_cache = !self.args.no_timing_cache && self.build_type.is_some();
        let build_type = self.build_type;
        let (cache_tx, cache_rx) = crossbeam_channel::bounded::<TimingCache>(1);
        if load_cache {
            thread::spawn(move || {
                let cache = TimingCache::load();
                let _ = cache_tx.send(cache);
            });
        }

        // Discover tests via slang-test -dry-run (streaming)
        let (rx, error_rx, compiling_rx) = crate::discover_tests_streaming(
            self.args.slang_test.as_ref().unwrap(),
            &self.args.root_dir,
            &self.args.filters,
            &self.args.ignore_patterns,
            &self.args.apis,
            &self.args.ignore_apis,
        )?;

        // Collect tests while showing streaming progress (in TTY mode)
        let mut test_files: Vec<String> = Vec::new();
        let mut cache_for_display: Option<TimingCache> = None;
        let mut shown_compiling = false;
        // Track running totals for ETA calculation (avoids O(n²) iteration)
        let mut total_predicted: f64 = 0.0;
        let mut longest_test: f64 = 0.0;

        // Create discovery progress bar (TTY mode only)
        let discovery_pb = if self.machine_output {
            None
        } else {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{msg}")
                    .unwrap(),
            );
            Some(pb)
        };

        loop {
            // Check for errors from the discovery thread
            if let Ok(error_msg) = error_rx.try_recv() {
                if let Some(ref pb) = discovery_pb {
                    pb.finish_and_clear();
                }
                anyhow::bail!("{}", error_msg);
            }

            // Check for compiling signal
            if !shown_compiling && compiling_rx.try_recv().is_ok() {
                if let Some(ref pb) = discovery_pb {
                    pb.set_message("\x1b[2mCompiling core module...\x1b[0m".to_string());
                }
                shown_compiling = true;
            }

            // Try to receive with a timeout so we can check compiling signal
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(test) => {
                    // Check if cache finished loading (non-blocking)
                    if cache_for_display.is_none() {
                        if let Ok(cache) = cache_rx.try_recv() {
                            // Cache just loaded - compute totals for tests collected so far
                            if let Some(bt) = build_type {
                                for f in &test_files {
                                    let pred = cache.predict(bt, &test_to_timing_key(f));
                                    total_predicted += pred;
                                    longest_test = longest_test.max(pred);
                                }
                            }
                            cache_for_display = Some(cache);
                        }
                    }

                    // Update running totals incrementally
                    if let (Some(ref cache), Some(bt)) = (&cache_for_display, build_type) {
                        let pred = cache.predict(bt, &test_to_timing_key(&test));
                        total_predicted += pred;
                        longest_test = longest_test.max(pred);
                    }

                    test_files.push(test);

                    // Update progress display (only in TTY mode)
                    if let Some(ref pb) = discovery_pb {
                        let eta = if cache_for_display.is_some() {
                            let parallel_eta = total_predicted / self.args.jobs as f64;
                            Some(parallel_eta.max(longest_test))
                        } else {
                            None
                        };
                        pb.set_message(format_running_message(test_files.len(), self.args.jobs, eta));
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Still waiting - continue loop to check compiling signal
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Channel closed - check for errors one more time
                    if let Ok(error_msg) = error_rx.try_recv() {
                        if let Some(ref pb) = discovery_pb {
                            pb.finish_and_clear();
                        }
                        anyhow::bail!("{}", error_msg);
                    }
                    break;
                }
            }
        }

        // Check for errors after discovery completes
        if let Ok(error_msg) = error_rx.try_recv() {
            if let Some(pb) = discovery_pb {
                pb.finish_and_clear();
            }
            anyhow::bail!("{}", error_msg);
        }

        // Wait for cache loading to complete if not already done
        if let Some(cache) = cache_for_display {
            *self.timing_cache.lock().unwrap() = cache;
        } else if load_cache {
            // Cache wasn't received during discovery, wait for it now
            if let Ok(cache) = cache_rx.recv() {
                *self.timing_cache.lock().unwrap() = cache;
            }
        }

        // Sort tests for deterministic ordering
        test_files.sort();

        if test_files.is_empty() {
            if let Some(pb) = discovery_pb {
                pb.finish_and_clear();
            }
            eprintln!("No tests found matching the specified criteria");
        } else if self.args.dry_run {
            if let Some(pb) = discovery_pb {
                pb.finish_and_clear();
            }
            // Just print the tests that would be run
            for test in &test_files {
                println!("{}", test);
            }
            eprintln!("{} tests would be run", test_files.len());
            return Ok(true);
        } else {
            self.run_file_tests(&test_files, discovery_pb)?;
        }

        let elapsed = start_time.elapsed();

        if is_interrupted() {
            eprintln!("\n{}", "Interrupted by Ctrl-C".red().bold());
        }

        self.print_summary(elapsed);

        Ok(self.stats.failed.load(Ordering::SeqCst) == 0 && !is_interrupted())
    }

    fn run_file_tests(&self, test_files: &[String], discovery_pb: Option<ProgressBar>) -> Result<()> {
        if test_files.is_empty() {
            if let Some(pb) = discovery_pb {
                pb.finish_and_clear();
            }
            return Ok(());
        }

        let has_timing_data = !self.args.no_timing_cache && self.build_type.is_some() && {
            let cache = self.timing_cache.lock().unwrap();
            self.build_type.map(|bt| cache.has_timing_data(bt)).unwrap_or(false)
        };

        // Build predictions map: test string -> predicted duration
        // We use the timing key (path + variant) for lookups but store by full test string
        let predictions: HashMap<String, f64> = {
            let cache = self.timing_cache.lock().unwrap();
            let build_type = self.build_type;
            test_files.iter()
                .map(|f| {
                    let timing_key = test_to_timing_key(f);
                    let pred = build_type
                        .map(|bt| cache.predict(bt, &timing_key))
                        .unwrap_or(DEFAULT_PREDICTED_DURATION);
                    (f.clone(), pred)
                })
                .collect()
        };

        let sorted_files: Vec<String> = if has_timing_data {
            let mut files_with_duration: Vec<_> = test_files.iter()
                .map(|f| (f.clone(), *predictions.get(f).unwrap_or(&DEFAULT_PREDICTED_DURATION)))
                .collect();
            files_with_duration.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let n = self.args.jobs;
            let mut slots: Vec<Vec<String>> = (0..n).map(|_| Vec::new()).collect();

            for (i, (file, _)) in files_with_duration.into_iter().enumerate() {
                slots[i % n].push(file);
            }

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
            test_files.to_vec()
        };

        let eta = if has_timing_data {
            Some(calculate_initial_eta(predictions.values().copied(), self.args.jobs))
        } else {
            None
        };
        let running_msg = format_running_message(test_files.len(), self.args.jobs, eta);
        if let Some(pb) = discovery_pb {
            pb.finish_with_message(running_msg);
        } else {
            // Machine output mode
            eprintln!("{}", running_msg);
        }

        let work_pool = Arc::new(WorkPool::new(
            sorted_files,
            self.args.batch_size,
            self.args.jobs,
            predictions,
            has_timing_data,
            self.args.batch_duration,
            self.args.gpu_jobs,
        ));

        let stats = self.stats.clone();
        let failures = self.failures.clone();
        let retries = self.args.retries;
        let retried_tests = self.retried_tests.clone();

        let running = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        let timeout = Duration::from_secs(self.args.timeout);

        // Create worker states for per-worker progress display (TTY mode only)
        let worker_states = if !self.machine_output {
            Some(Arc::new(WorkerStates::new(self.args.jobs)))
        } else {
            None
        };

        // Mark execution started so progress display can show "waiting for output"
        stats.mark_execution_started();

        debug_log!("about to spawn progress thread");

        // Spawn progress thread first
        let progress_stats = stats.clone();
        let progress_running = running.clone();
        let progress_pool = work_pool.clone();
        let progress_shutdown = Arc::new(AtomicBool::new(false));
        let progress_shutdown_clone = progress_shutdown.clone();
        let total_files = test_files.len();
        let machine_output = self.machine_output;
        let num_workers = self.args.jobs;
        let progress_worker_states = worker_states.clone();
        let verbose_for_progress = self.args.verbose;
        let progress_handle = thread::spawn(move || {
            let mut display = ProgressDisplay::new(total_files, machine_output, num_workers, verbose_for_progress);
            // Initialize SystemStats lazily to avoid blocking the first progress update
            let mut sys_stats: Option<SystemStats> = None;
            let mut stats_counter = 0u32;
            while !progress_shutdown_clone.load(Ordering::SeqCst) {
                let files_done = progress_stats.files_completed();
                let batches_running = progress_running.load(Ordering::SeqCst);
                let batches_remaining = progress_pool.remaining();
                let has_pending_batches = progress_pool.has_pending_batches();
                let eta = if progress_pool.has_timing_data {
                    Some(progress_pool.calculate_eta(num_workers))
                } else {
                    None
                };
                display.update(&progress_stats, files_done, batches_running, batches_remaining, has_pending_batches, eta, progress_worker_states.as_deref());

                stats_counter += 1;
                if stats_counter >= 10 {
                    stats_counter = 0;
                    let sys = sys_stats.get_or_insert_with(SystemStats::new);
                    sys.refresh_and_log(batches_running, batches_remaining);
                }

                thread::sleep(Duration::from_millis(16));
            }
            display.finish(&progress_stats);
        });

        debug_log!("progress thread spawned, about to spawn workers");

        // Spawn worker threads
        if !work_pool.is_empty() {
            for i in 0..self.args.jobs {
                debug_log!("spawning worker {}", i);
                let slang_test = self.args.slang_test.as_ref().unwrap().clone();
                let root_dir = self.args.root_dir.clone();
                let extra_args = self.args.extra_args.clone();
                let stats = stats.clone();
                let failures = failures.clone();
                let retried_tests = retried_tests.clone();
                let pool = work_pool.clone();
                let running = running.clone();
                let shutdown = shutdown.clone();
                let machine_output = self.machine_output;
                let verbose = self.args.verbose;
                let worker_states_clone = worker_states.clone();

                let worker_id = i;
                let handle = thread::Builder::new()
                    .name(format!("worker-{}", worker_id))
                    .spawn(move || {
                    debug_log!("started, getting first batch");
                    let my_state = worker_states_clone.as_ref().map(|ws| ws.get(worker_id));
                    loop {
                        if shutdown.load(Ordering::SeqCst) {
                            debug_log!("shutdown flag set, exiting");
                            break;
                        }
                        if is_interrupted() {
                            debug_log!("interrupted, exiting");
                            break;
                        }

                        // Check if we can take a GPU batch
                        let has_gpu_slot = pool.can_take_gpu_batch();

                        let get_batch_start = Instant::now();
                        if let Some((batch_id, batch, kind)) = pool.try_get_batch(has_gpu_slot) {
                            let get_batch_time = get_batch_start.elapsed();
                            if get_batch_time.as_millis() > 10 {
                                debug_log!("try_get_batch took {:.3}s", get_batch_time.as_secs_f64());
                            }
                            debug_log!("got batch {} of {} tests ({:?})", batch_id, batch.len(), kind);
                            run_batch_with_pool(
                                &slang_test,
                                &root_dir,
                                &batch,
                                &extra_args,
                                timeout,
                                &stats,
                                &failures,
                                retries,
                                &retried_tests,
                                &pool,
                                &running,
                                machine_output,
                                verbose,
                                my_state,
                            );
                            debug_log!("batch {} completed, calling complete_batch", batch_id);
                            pool.complete_batch(batch_id);
                            debug_log!("batch {} complete_batch done", batch_id);
                        } else {
                            // No batch available - log why and wait
                            let (batches_count, pending_count, in_flight_ids) = pool.debug_state();
                            debug_log!("no batch available: batches={} pending={} in_flight={:?} has_gpu_slot={}",
                                batches_count, pending_count, in_flight_ids, has_gpu_slot);
                            thread::sleep(Duration::from_millis(10));
                        }
                    }
                    debug_log!("worker loop exited");
                }).expect("failed to spawn worker thread");
                handles.push(handle);
            }

            debug_log!("all workers spawned, entering main loop");

            loop {
                let pool_empty = work_pool.is_empty();
                let running_count = running.load(Ordering::SeqCst);

                if pool_empty && running_count == 0 {
                    debug_log!("main loop: pool empty and no batches running, exiting");
                    break;
                }

                if is_interrupted() {
                    debug_log!("main loop: interrupted, exiting");
                    break;
                }

                // Log state periodically (every ~1 second when DEBUG is on)
                static LAST_LOG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let now_ms = DEBUG_START.elapsed().as_millis() as u64;
                let last = LAST_LOG.load(Ordering::Relaxed);
                if now_ms - last > 1000 {
                    LAST_LOG.store(now_ms, Ordering::Relaxed);
                    let (batches_count, pending_count, in_flight_ids) = work_pool.debug_state();
                    debug_log!("main loop: running={} batches={} pending={} in_flight={:?}",
                        running_count, batches_count, pending_count, in_flight_ids);
                }

                thread::sleep(Duration::from_millis(20));
            }

            debug_log!("main loop exited, setting shutdown flag");
            shutdown.store(true, Ordering::SeqCst);

            // Don't wait for workers - they'll exit on their own when they see shutdown
            debug_log!("dropping worker handles");
            drop(handles);

            progress_shutdown.store(true, Ordering::SeqCst);
            // Must join progress thread to ensure it stops writing before we print failures
            let _ = progress_handle.join();
        }

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

        match self.args.diff.as_str() {
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
                self.run_external_diff("diff", &["-y", "--color=always", "-W", "160"], expected, actual);
            }
        }
    }

    fn run_external_diff(&self, cmd: &str, args: &[&str], expected: &str, actual: &str) {
        print!("{}", Self::compute_external_diff(cmd, args, expected, actual, self.machine_output));
    }

    /// Compute diff output as a string (can be run in parallel)
    fn compute_external_diff(cmd: &str, args: &[&str], expected: &str, actual: &str, machine_output: bool) -> String {
        use std::io::Write as _;

        let indent = if machine_output { "" } else { "  " };

        // Use unique temp files for parallel safety
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
                    if machine_output {
                        result.push_str("Expected:\n");
                        for line in expected.lines() {
                            result.push_str(&format!("{}\n", line));
                        }
                        result.push_str("Actual:\n");
                        for line in actual.lines() {
                            result.push_str(&format!("{}\n", line));
                        }
                    } else {
                        result.push_str(&format!("{}Expected:\n", indent));
                        for line in expected.lines() {
                            result.push_str(&format!("    {}\n", line));
                        }
                        result.push_str(&format!("{}Actual:\n", indent));
                        for line in actual.lines() {
                            result.push_str(&format!("    {}\n", line));
                        }
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

            // Pre-compute diffs in parallel for failures that have expected/actual
            let diff_tool = self.args.diff.as_str();
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
                                let (cmd, args): (&str, &[&str]) = if tool == "difft" {
                                    ("difft", &["--color", "always"])
                                } else {
                                    ("diff", &["-y", "--color=always", "-W", "160"])
                                };
                                Self::compute_external_diff(cmd, args, &expected, &actual, machine_output)
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

            // Print results sequentially
            for (failure, diff_output) in sorted_failures.iter().zip(diff_results.iter()) {
                println!("\n{}", failure.test_name.red());

                if let (Some(expected), Some(actual)) = (&failure.expected, &failure.actual) {
                    if let Some(diff) = diff_output {
                        // Pre-computed diff from parallel execution
                        print!("{}", diff);
                    } else {
                        // diff_tool == "none", show inline
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

        if total_run > 0 {
            if interrupted {
                if failed == 0 {
                    println!(
                        "{}: {} passed, {} ignored in {:.1}s",
                        "Interrupted".yellow().bold(),
                        passed,
                        ignored,
                        elapsed.as_secs_f64()
                    );
                } else {
                    println!(
                        "{}: {} passed, {} failed, {} ignored in {:.1}s",
                        "Interrupted".yellow().bold(),
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
        }

        if !failures.is_empty() {
            let mut test_files: Vec<String> = failures
                .iter()
                .filter_map(|f| extract_base_test_file(&f.test_name))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            test_files.sort();

            println!("\n{}", "To rerun failed tests:".yellow());

            let has_file_tests = test_files.iter().any(|f| f.ends_with(".slang") || f.ends_with(".hlsl") || f.ends_with(".glsl") || f.ends_with(".c"));

            let exe = std::env::args().next().unwrap_or_else(|| "sti".to_string());
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
                print!("{}", self.args.slang_test.as_ref().unwrap().display());
                for file in &test_files {
                    print!(" \"{}\"", file);
                }
                println!();
            }
        }

        println!("{}", "=".repeat(70));
    }
}
