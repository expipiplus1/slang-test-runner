use anyhow::Result;
use colored::Colorize;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::types::{*, TestId, test_to_timing_key};

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
    INTERRUPTED.store(true, Ordering::SeqCst);
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

    let mut cmd = Command::new(ctx.slang_test);
    cmd.current_dir(ctx.root_dir)
        .arg("-explicit-test-order")
        .arg("-disable-retries")
        .args(&slang_test_args)
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
    let (timing_tx, timing_rx) = crossbeam_channel::unbounded::<(String, f64)>(); // (test_id, duration)
    let (done_tx, done_rx) = crossbeam_channel::bounded::<bool>(1); // Signal when output is complete

    let machine_output_for_stderr = ctx.machine_output;

    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut seen_tests: HashSet<String> = HashSet::new();
        let mut saw_summary = false;

        let mut last_test_time: Option<Instant> = None;

        for line in reader.lines() {
            if is_interrupted() {
                break;
            }
            if let Ok(line) = line {
                // Detect the summary line that indicates normal completion
                if line.starts_with("===") || line.contains("% of tests") || line == "no tests run" {
                    saw_summary = true;
                    // Signal early that we're done - don't wait for process exit
                    let _ = done_tx.try_send(true);
                    continue;
                }

                if line.starts_with("Supported backends:")
                    || line.starts_with("Check ")
                    || line.starts_with("Retrying ")
                {
                    last_test_time = Some(Instant::now());
                    continue;
                }

                if let Some(mut outcome) = parse_test_output(&line) {
                    let now = Instant::now();

                    // Record timing for all test results (passed, failed, ignored)
                    // since they all contribute to overall runtime
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

                    // Extract the timing key: base file path + variant suffix (no backend/syn suffix)
                    // e.g., "tests/foo.slang.4 syn (vk)" -> "tests/foo.slang.4"
                    let test_id = TestId::parse(&outcome.name);
                    let timing_key = test_id.to_timing_key();

                    if test_duration > 0.0 {
                        let _ = timing_tx.send((timing_key.clone(), test_duration));
                    }

                    seen_tests.insert(timing_key);

                    let _ = outcome_tx.send(outcome);
                }
            }
        }

        (seen_tests, saw_summary)
    });

    let (compiling_tx, compiling_rx) = crossbeam_channel::unbounded::<bool>();
    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                if line.contains("Compiling core module") {
                    set_compiling_core(true);
                    let _ = compiling_tx.send(true);
                    if machine_output_for_stderr {
                        eprintln!("{}", format!("INFO: {}", line).dimmed());
                    }
                } else if line.contains("Compiling") {
                    let _ = compiling_tx.send(true);
                }
                let _ = stderr_tx.send(line);
            }
        }
        let _ = compiling_tx.send(false);
    });

    let mut failed_outcomes: Vec<TestOutcome> = Vec::new();
    let start = Instant::now();
    let mut this_batch_is_compiling = false;
    let mut killed_for_compilation = false;
    let mut killed_for_timeout = false;
    let mut exit_status: Option<std::process::ExitStatus> = None;

    loop {
        if is_interrupted() {
            let _ = child.kill();
            break;
        }

        while let Ok(is_compiling) = compiling_rx.try_recv() {
            ctx.stats.set_compiling(is_compiling);
            if is_compiling {
                this_batch_is_compiling = true;
            }
        }

        if is_compiling_core() && !this_batch_is_compiling {
            let _ = child.kill();
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

            process_outcome(outcome, &ctx, &mut failed_outcomes);
        }

        // Check if output is complete (saw summary or "no tests run")
        // Don't wait for process exit - send to reaper and continue immediately
        if done_rx.try_recv().is_ok() {
            reap_process(child);
            break;
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                exit_status = Some(status);
                break;
            }
            Ok(None) => {
                if start.elapsed() > ctx.timeout {
                    let _ = child.kill();
                    killed_for_timeout = true;
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => break,
        }
    }

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

    while let Ok((test_id, duration)) = timing_rx.try_recv() {
        ctx.stats.record_observed_timing(&test_id, duration);
    }

    while let Ok(outcome) = outcome_rx.try_recv() {
        process_outcome(outcome, &ctx, &mut failed_outcomes);
    }

    if killed_for_compilation {
        for file in ctx.test_files {
            ctx.work_pool.add_file(file.to_string());
        }
        while is_compiling_core() && !is_interrupted() {
            thread::sleep(Duration::from_millis(50));
        }
        return;
    }

    let join_start = Instant::now();
    let (seen_tests, saw_summary) = stdout_handle.join().unwrap_or_default();
    let stdout_join_time = join_start.elapsed();

    let stderr_join_start = Instant::now();
    stderr_handle.join().ok();
    let stderr_join_time = stderr_join_start.elapsed();

    if stdout_join_time.as_secs() > 5 || stderr_join_time.as_secs() > 5 {
        eprintln!("{}", format!("\nWARNING: Slow thread joins - stdout: {:.1}s, stderr: {:.1}s for {:?}",
            stdout_join_time.as_secs_f64(),
            stderr_join_time.as_secs_f64(),
            ctx.test_files
        ).dimmed());
    }

    let stderr_lines: Vec<String> = stderr_rx.try_iter().collect();

    // Report any tests that weren't accounted for in the output
    if !is_interrupted() && !killed_for_compilation {
        let mut unaccounted: Vec<&String> = Vec::new();
        for file in ctx.test_files {
            let test_id = TestId::parse(file);
            let timing_key = test_id.to_timing_key();
            if !seen_tests.contains(&timing_key) {
                unaccounted.push(file);
            }
        }
        if !unaccounted.is_empty() {
            eprintln!("{}", format!("\nWARNING: {} unaccounted tests in batch:", unaccounted.len()).yellow());
            for test in &unaccounted {
                eprintln!("  {}", test);
            }
            if ctx.verbose {
                eprintln!("  seen_tests: {:?}", seen_tests);
            }
        }
    }

    // Detect crash or timeout: if we didn't see the summary and weren't interrupted/killed for compilation
    if !saw_summary && !is_interrupted() && !killed_for_compilation {
        let exit_info = if killed_for_timeout {
            format!("timeout after {}s", ctx.timeout.as_secs())
        } else {
            format_exit_status(exit_status.as_ref())
        };

        // Since tests are now individual (with variant numbers), the crash/timeout must be
        // caused by the first test we didn't see output for
        let mut crashed_test: Option<String> = None;
        let mut tests_to_repool: Vec<String> = Vec::new();
        let mut found_crashed = false;

        for file in ctx.test_files {
            let test_id = TestId::parse(file);
            let timing_key = test_id.to_timing_key();
            if !seen_tests.contains(&timing_key) {
                if !found_crashed {
                    crashed_test = Some(file.clone());
                    found_crashed = true;
                } else {
                    // All tests after the crashed/timed-out one should be repooled
                    tests_to_repool.push(file.clone());
                }
            }
        }

        // Repool tests that didn't get a chance to run
        for test in &tests_to_repool {
            ctx.work_pool.add_file(test.clone());
        }

        if let Some(test) = crashed_test {
            let error_type = if killed_for_timeout { "timed out" } else { "crashed" };
            eprintln!(
                "\nERROR: slang-test {} ({}), skipping: {}",
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
            // This helps scheduling know this test is slow/problematic
            let test_id = TestId::parse(&test);
            let timing_key = test_id.to_timing_key();
            ctx.stats.record_observed_timing(&timing_key, loop_time.as_secs_f64());
        }
    }

    collect_failure_output(&mut failed_outcomes, &stderr_lines);

    for outcome in failed_outcomes {
        let info = parse_failure_info(&outcome.name, &outcome.failure_output);
        ctx.failures.lock().unwrap().push(info);
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

        loop {
            // Check for errors from the discovery thread
            if let Ok(error_msg) = error_rx.try_recv() {
                anyhow::bail!("{}", error_msg);
            }

            // Check for compiling signal
            if !shown_compiling && compiling_rx.try_recv().is_ok() {
                if !self.machine_output {
                    eprint!("\x1b[2mCompiling core module...\x1b[0m");
                    let _ = std::io::stderr().flush();
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
                    if !self.machine_output {
                        let eta = if cache_for_display.is_some() {
                            let parallel_eta = total_predicted / self.args.jobs as f64;
                            Some(parallel_eta.max(longest_test))
                        } else {
                            None
                        };
                        eprint!("\r\x1b[K{}", format_running_message(test_files.len(), self.args.jobs, eta));
                        let _ = std::io::stderr().flush();
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Still waiting - continue loop to check compiling signal
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Channel closed - check for errors one more time
                    if let Ok(error_msg) = error_rx.try_recv() {
                        anyhow::bail!("{}", error_msg);
                    }
                    break;
                }
            }
        }

        // Clear the discovery line
        if !self.machine_output && !test_files.is_empty() {
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }

        // Check for errors after discovery completes
        if let Ok(error_msg) = error_rx.try_recv() {
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
            eprintln!("No tests found matching the specified criteria");
        } else if self.args.dry_run {
            // Just print the tests that would be run
            for test in &test_files {
                println!("{}", test);
            }
            eprintln!("{} tests would be run", test_files.len());
            return Ok(true);
        } else {
            self.run_file_tests(&test_files)?;
        }

        let elapsed = start_time.elapsed();

        if is_interrupted() {
            eprintln!("\n{}", "Interrupted by Ctrl-C".red().bold());
        }

        self.print_summary(elapsed);

        Ok(self.stats.failed.load(Ordering::SeqCst) == 0 && !is_interrupted())
    }

    fn run_file_tests(&self, test_files: &[String]) -> Result<()> {
        if test_files.is_empty() {
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
        eprintln!("{}", format_running_message(test_files.len(), self.args.jobs, eta));

        let work_pool = Arc::new(WorkPool::new(
            sorted_files,
            self.args.batch_size,
            self.args.jobs,
            predictions,
            has_timing_data,
            self.args.batch_duration,
        ));

        let stats = self.stats.clone();
        let failures = self.failures.clone();
        let retries = self.args.retries;
        let retried_tests = self.retried_tests.clone();

        let running = Arc::new(AtomicUsize::new(0));
        let adaptive_running = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        let timeout = Duration::from_secs(self.args.timeout);

        // Mark execution started so progress display can show "waiting for output"
        stats.mark_execution_started();

        let debug_start = Instant::now();
        eprintln!("[DEBUG {:>6.3}s] about to spawn progress thread", debug_start.elapsed().as_secs_f64());

        // Spawn progress thread first
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
            // Initialize SystemStats lazily to avoid blocking the first progress update
            let mut sys_stats: Option<SystemStats> = None;
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

                stats_counter += 1;
                if stats_counter >= 10 {
                    stats_counter = 0;
                    let sys = sys_stats.get_or_insert_with(SystemStats::new);
                    sys.refresh_and_log(batches_running, adaptive_count, batches_remaining);
                }

                thread::sleep(Duration::from_millis(100));
            }
        });

        eprintln!("[DEBUG {:>6.3}s] progress thread spawned, about to spawn workers", debug_start.elapsed().as_secs_f64());

        // Spawn worker threads
        if !work_pool.is_empty() {
            for i in 0..self.args.jobs {
                eprintln!("[DEBUG {:>6.3}s] spawning worker {}", debug_start.elapsed().as_secs_f64(), i);
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

                let worker_id = i;
                let worker_debug_start = debug_start;
                let handle = thread::spawn(move || {
                    eprintln!("[DEBUG {:>6.3}s] worker {} started, getting first batch", worker_debug_start.elapsed().as_secs_f64(), worker_id);
                    loop {
                        if shutdown.load(Ordering::SeqCst) || is_interrupted() {
                            break;
                        }

                        let get_batch_start = Instant::now();
                        if let Some(batch) = pool.try_get_batch() {
                            let get_batch_time = get_batch_start.elapsed();
                            if get_batch_time.as_millis() > 10 {
                                eprintln!("[DEBUG {:>6.3}s] worker {} try_get_batch took {:.3}s", worker_debug_start.elapsed().as_secs_f64(), worker_id, get_batch_time.as_secs_f64());
                            }
                            eprintln!("[DEBUG {:>6.3}s] worker {} got batch of {} tests", worker_debug_start.elapsed().as_secs_f64(), worker_id, batch.len());
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
                            );
                        } else {
                            thread::sleep(Duration::from_millis(10));
                        }
                    }
                });
                handles.push(handle);
            }

            eprintln!("[DEBUG {:>6.3}s] all workers spawned, entering main loop", debug_start.elapsed().as_secs_f64());

            let adaptive = self.args.adaptive;
            let num_cpus = self.args.jobs;
            let adaptive_handles: Arc<Mutex<Vec<thread::JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));
            let mut last_adaptive_check = Instant::now();

            while !work_pool.is_empty() || running.load(Ordering::SeqCst) > 0 {
                if is_interrupted() {
                    break;
                }

                // Check adaptive spawning every 5 seconds
                if adaptive && !work_pool.is_empty() && last_adaptive_check.elapsed() >= Duration::from_secs(5) {
                    last_adaptive_check = Instant::now();

                    let current_running = running.load(Ordering::SeqCst);
                    let current_adaptive = adaptive_running.load(Ordering::SeqCst);
                    let total_running = current_running + current_adaptive;

                    let should_spawn = if let Some(load) = get_instantaneous_load() {
                        total_running < num_cpus && load < (num_cpus as f64 * 1.5)
                    } else {
                        total_running < num_cpus
                    };

                    if should_spawn {
                        let extra_to_spawn = (num_cpus - total_running).min(4);
                        for _ in 0..extra_to_spawn {
                            if let Some(batch) = work_pool.try_get_medium_batch() {
                                let slang_test = self.args.slang_test.as_ref().unwrap().clone();
                                let root_dir = self.args.root_dir.clone();
                                let extra_args = self.args.extra_args.clone();
                                let stats = stats.clone();
                                let failures = failures.clone();
                                let retried_tests = retried_tests.clone();
                                let pool = work_pool.clone();
                                let running = running.clone();
                                let adaptive_counter = adaptive_running.clone();
                                let machine_output = self.machine_output;
                                let verbose = self.args.verbose;

                                adaptive_counter.fetch_add(1, Ordering::SeqCst);
                                log_event("turbo_spawn", &format!("total_running={} file={}",
                                    total_running, &batch[0]));

                                let handle = thread::spawn(move || {
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
                                    );
                                    adaptive_counter.fetch_sub(1, Ordering::SeqCst);
                                });
                                adaptive_handles.lock().unwrap().push(handle);
                            }
                        }
                    }
                }

                thread::sleep(Duration::from_millis(20));
            }

            shutdown.store(true, Ordering::SeqCst);

            for handle in handles {
                handle.join().unwrap();
            }

            let mut adaptive_guard = adaptive_handles.lock().unwrap();
            for handle in adaptive_guard.drain(..) {
                handle.join().unwrap();
            }

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
                self.run_external_diff("difft", &["--color", "always"], expected, actual);
            }
            _ => {
                self.run_external_diff("diff", &["-y", "--color=always", "-W", "160"], expected, actual);
            }
        }
    }

    fn run_external_diff(&self, cmd: &str, args: &[&str], expected: &str, actual: &str) {
        use std::io::Write as _;

        let indent = if self.machine_output { "" } else { "  " };

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

        if !failures.is_empty() {
            println!("\n{}", "=".repeat(70));
            println!("{}", "FAILURES".red().bold());
            println!("{}", "=".repeat(70));

            let mut sorted_failures: Vec<_> = failures.iter().collect();
            sorted_failures.sort_by(|a, b| a.test_name.cmp(&b.test_name));

            for failure in &sorted_failures {
                println!("\n{}", failure.test_name.red());

                if let (Some(expected), Some(actual)) = (&failure.expected, &failure.actual) {
                    self.show_diff(expected, actual);
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

        println!("\n{}", "=".repeat(70));

        let total_run = passed + failed;
        let interrupted = is_interrupted();

        if total_run > 0 {
            if interrupted {
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
