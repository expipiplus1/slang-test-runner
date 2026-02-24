use anyhow::{Context, Result};
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

use crate::types::*;

// ============================================================================
// Global Flags
// ============================================================================

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static COMPILING_CORE: AtomicBool = AtomicBool::new(false);

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
    retried_tests: &Mutex<HashSet<String>>,
    work_pool: &Arc<WorkPool>,
) -> bool {
    if max_retries == 0 {
        return false;
    }

    let should_retry = {
        let mut retried = retried_tests.lock().unwrap();
        if retried.contains(&outcome.name) {
            false
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

fn process_outcome(
    outcome: TestOutcome,
    ctx: &BatchContext,
    failed_outcomes: &mut Vec<TestOutcome>,
) -> bool {
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
    let (timing_tx, timing_rx) = crossbeam_channel::unbounded::<(String, String, f64)>();

    let machine_output_for_stderr = ctx.machine_output;
    let num_files_in_batch = ctx.test_files.len();
    let batch_start_time = Instant::now();

    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut seen_tests: HashSet<String> = HashSet::new();

        let mut last_test_time: Option<Instant> = None;
        let mut current_file: Option<String> = None;
        let mut file_start = Instant::now();
        let mut startup_time_per_file: Option<f64> = None;

        for line in reader.lines() {
            if is_interrupted() {
                break;
            }
            if let Ok(line) = line {
                if line.starts_with("Supported backends:")
                    || line.starts_with("Check ")
                    || line.starts_with("Retrying ")
                {
                    let now = Instant::now();
                    let startup_time = batch_start_time.elapsed().as_secs_f64();
                    if num_files_in_batch > 0 {
                        startup_time_per_file = Some(startup_time / num_files_in_batch as f64);
                    }
                    last_test_time = Some(now);
                    file_start = now;
                    continue;
                }

                if let Some(mut outcome) = parse_test_output(&line) {
                    let now = Instant::now();

                    let should_record_timing = last_test_time.is_some()
                        && outcome.result != TestResult::Ignored;

                    let test_duration = if should_record_timing {
                        outcome.duration
                            .map(|d| d.as_secs_f64())
                            .unwrap_or_else(|| last_test_time.unwrap().elapsed().as_secs_f64())
                    } else {
                        0.0
                    };

                    if outcome.duration.is_none() && should_record_timing {
                        outcome.duration = Some(Duration::from_secs_f64(test_duration));
                    }

                    last_test_time = Some(now);

                    if let Some(base) = extract_base_test_file(&outcome.name) {
                        if should_record_timing {
                            let backend = extract_backend(&outcome.name).unwrap_or_else(|| "_none".to_string());
                            let _ = timing_tx.send((base.clone(), backend, test_duration));
                        }

                        if current_file.as_ref() != Some(&base) {
                            if let Some(ref prev_file) = current_file {
                                let duration = file_start.elapsed().as_secs_f64();
                                let _ = timing_tx.send((prev_file.clone(), "_total".to_string(), duration));
                            }
                            current_file = Some(base.clone());
                            file_start = now;
                        }

                        seen_tests.insert(base);
                    }

                    let _ = outcome_tx.send(outcome);
                }
            }
        }

        if let Some(ref prev_file) = current_file {
            let duration = file_start.elapsed().as_secs_f64();
            let _ = timing_tx.send((prev_file.clone(), "_total".to_string(), duration));
        }

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

        while let Ok((file, backend, duration)) = timing_rx.try_recv() {
            ctx.stats.record_observed_timing(&file, &backend, duration);
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

    while let Ok((file, backend, duration)) = timing_rx.try_recv() {
        ctx.stats.record_observed_timing(&file, &backend, duration);
    }

    while let Ok(outcome) = outcome_rx.try_recv() {
        process_outcome(outcome, &ctx, &mut failed_outcomes);
    }

    if killed_for_compilation {
        for file in ctx.test_files {
            ctx.work_pool.add_file(file.clone());
        }
        while is_compiling_core() && !is_interrupted() {
            thread::sleep(Duration::from_millis(50));
        }
        return;
    }

    let join_start = Instant::now();
    let _seen_tests = stdout_handle.join().unwrap_or_default();
    let stdout_join_time = join_start.elapsed();

    let stderr_join_start = Instant::now();
    stderr_handle.join().ok();
    let stderr_join_time = stderr_join_start.elapsed();

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

// ============================================================================
// Test Runner
// ============================================================================

pub struct TestRunner {
    pub args: crate::Args,
    pub stats: Arc<TestStats>,
    pub failures: Arc<Mutex<Vec<FailureInfo>>>,
    pub retried_tests: Arc<Mutex<HashSet<String>>>,
    pub machine_output: bool,
    pub timing_cache: Mutex<TimingCache>,
}

impl TestRunner {
    pub fn new(args: crate::Args) -> Self {
        let machine_output = !crate::is_stderr_tty();
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

    pub fn save_timing(&self) {
        let observed = self.stats.get_observed_timings();
        if !observed.is_empty() {
            let mut cache = self.timing_cache.lock().unwrap();
            cache.merge(&observed);
            cache.save();
        }
    }

    fn predict_duration(&self, file: &PathBuf) -> f64 {
        let api_filter = extract_api_filter(&self.args.extra_args);
        let cache = self.timing_cache.lock().unwrap();
        cache.predict(&file.to_string_lossy(), api_filter.as_deref(), false)
    }

    pub fn run(&self) -> Result<bool> {
        let start_time = Instant::now();

        let all_prefixes_are_files = !self.args.prefixes.is_empty()
            && self.args.prefixes.iter().all(|p| {
                let path = PathBuf::from(p);
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|ext| ext == "slang" || ext == "hlsl" || ext == "glsl" || ext == "c")
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
                crate::discover_test_files(&self.args.test_dir, &[], &self.args.ignore_patterns)?;
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
        let mut cmd = Command::new(self.args.slang_test.as_ref().unwrap());

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
                        if line.starts_with('[') || !pending_failure_lines.is_empty() {
                            pending_failure_lines.push(line);
                        }
                    }
                }
            }
        });

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

        let stderr_lines: Vec<String> = stderr_rx.try_iter().collect();

        let mut failures_guard = failures.lock().unwrap();
        for failure in failures_guard.iter_mut() {
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
                let info = parse_failure_info(&failure.test_name, &relevant_lines);
                failure.output_lines = info.output_lines;
                failure.expected = info.expected;
                failure.actual = info.actual;
            }
        }
        drop(failures_guard);

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

        let has_timing_data = !self.args.no_timing_cache && {
            let cache = self.timing_cache.lock().unwrap();
            !cache.timings.is_empty()
        };

        let sorted_files: Vec<PathBuf> = if has_timing_data {
            let mut files_with_duration: Vec<_> = test_files.iter()
                .map(|f| (f.clone(), self.predict_duration(f)))
                .collect();
            files_with_duration.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let n = self.args.jobs;
            let mut slots: Vec<Vec<PathBuf>> = (0..n).map(|_| Vec::new()).collect();

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
                total_predicted / self.args.jobs as f64
            );
        } else {
            eprintln!(
                "Running {} test files with {} workers",
                test_files.len(),
                self.args.jobs
            );
        }

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

        let running = Arc::new(AtomicUsize::new(0));
        let adaptive_running = Arc::new(AtomicUsize::new(0));

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

                stats_counter += 1;
                if stats_counter >= 10 {
                    stats_counter = 0;
                    sys_stats.refresh_and_log(batches_running, adaptive_count, batches_remaining);
                }

                thread::sleep(Duration::from_millis(100));
            }
        });

        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        if !work_pool.is_empty() {
            for _ in 0..self.args.jobs {
                let slang_test = self.args.slang_test.as_ref().unwrap().clone();
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
                            thread::sleep(Duration::from_millis(10));
                        }
                    }
                });
                handles.push(handle);
            }

            let adaptive = self.args.adaptive;
            let num_cpus = self.args.jobs;
            let adaptive_handles: Arc<Mutex<Vec<thread::JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));

            while !work_pool.is_empty() || running.load(Ordering::SeqCst) > 0 {
                if is_interrupted() {
                    break;
                }

                if adaptive && !work_pool.is_empty() {
                    let current_running = running.load(Ordering::SeqCst);
                    let current_adaptive = adaptive_running.load(Ordering::SeqCst);
                    let total_running = current_running + current_adaptive;

                    let should_spawn = if let Some(load) = get_load_average() {
                        total_running < num_cpus && load < (num_cpus as f64 * 1.5)
                    } else {
                        total_running < num_cpus
                    };

                    if should_spawn {
                        let extra_to_spawn = (num_cpus - total_running).min(4);
                        for _ in 0..extra_to_spawn {
                            if let Some(batch) = work_pool.try_get_medium_batch() {
                                let slang_test = self.args.slang_test.as_ref().unwrap().clone();
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
            let num_slowest = 15;
            let slowest = self.stats.slowest_files(num_slowest);
            if !slowest.is_empty() && slowest[0].1 > 1.0 {
                println!("\n{}:", "Slowest files".yellow());
                for (file, secs) in &slowest {
                    if *secs > 0.5 {
                        println!("  {:>6.1}s  {}", secs, file);
                    }
                }

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
