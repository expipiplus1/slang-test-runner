//! Concurrent discovery module
//!
//! Handles the three discovery phases that run concurrently:
//! 1. API detection - runs slang-test to detect supported APIs
//! 2. Timing cache loading - loads cached timing data from disk
//! 3. Test discovery via -dry-run - enumerates all tests
//!
//! All three phases stream their data back via separate channels to a unified
//! loop that uses select! to integrate results and handle interrupts.

use anyhow::{Context, Result};
use colored::Colorize;
use crossbeam_channel::{bounded, select, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::api::UnsupportedApis;
use crate::debug_log;
use crate::event_log::log_event;
use crate::runner::{is_interrupted, reap_process, reap_process_with_label};
use crate::timing::{BuildType, TimingCache};
use crate::types::{test_to_timing_key, TestId, DEBUG_ENABLED};

/// Result of the concurrent discovery phase
pub struct DiscoveryResult {
    /// All discovered tests (sorted)
    pub tests: Vec<String>,
    /// Unsupported APIs (or None if check disabled/failed)
    pub unsupported_apis: Option<UnsupportedApis>,
    /// Loaded timing cache
    pub timing_cache: TimingCache,
    /// Count of tests ignored due to unsupported APIs
    pub api_ignored_count: usize,
    /// APIs found in tests but not in the Check output
    pub unknown_apis: HashSet<String>,
    /// Whether API check completed successfully (can skip per-batch detection)
    pub skip_api_detection: bool,
}

/// Configuration for discovery
pub struct DiscoveryConfig<'a> {
    pub slang_test: &'a PathBuf,
    pub root_dir: &'a PathBuf,
    pub filters: &'a [String],
    pub ignore_patterns: &'a [String],
    pub apis: &'a [String],
    pub ignore_apis: &'a [String],
    pub no_early_api_check: bool,
    pub no_timing_cache: bool,
    pub build_type: Option<BuildType>,
    pub gpu_jobs: Option<usize>,
    pub machine_output: bool,
    pub num_workers: usize,
}

/// Run all three discovery phases concurrently and collect results.
/// Returns when all phases are complete or interrupted.
pub fn run_concurrent_discovery(config: &DiscoveryConfig) -> Result<DiscoveryResult> {
    // Create separate channels for each discovery source
    let (test_tx, test_rx) = bounded::<String>(1000);
    let (test_err_tx, test_err_rx) = bounded::<String>(1);
    let (api_tx, api_rx) = bounded::<UnsupportedApis>(1);
    let (timing_tx, timing_rx) = bounded::<TimingCache>(1);
    // Unbounded since both dry-run and api-detection threads can send compiling signals
    let (compiling_tx, compiling_rx) = crossbeam_channel::unbounded::<()>();

    // Interrupt signal channel - wakes up select! on Ctrl-C
    let (sig_tx, sig_rx) = bounded::<()>(1);
    // Shutdown channel for the interrupt-poll thread - closed when main loop exits
    let (shutdown_tx, shutdown_rx) = bounded::<()>(1);

    // Hook into the existing interrupt system
    let interrupt_poll_handle = thread::Builder::new()
        .name("interrupt-poll".to_string())
        .spawn(move || {
            // Wait for either: interrupt detected, or shutdown signal (main loop done)
            loop {
                if is_interrupted() {
                    let _ = sig_tx.send(());
                    break;
                }
                // Check if main loop has finished (shutdown channel closed or received)
                match shutdown_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                }
            }
        })
        .expect("failed to spawn interrupt-poll thread");

    // Spawn all three discovery threads
    spawn_test_discovery(
        test_tx,
        test_err_tx,
        compiling_tx.clone(),
        config.slang_test.clone(),
        config.root_dir.clone(),
        config.filters.to_vec(),
        config.ignore_patterns.to_vec(),
        config.apis.to_vec(),
        config.ignore_apis.to_vec(),
    )?;

    if !config.no_early_api_check {
        spawn_api_detection(api_tx, compiling_tx, config.slang_test.clone(), config.root_dir.clone());
    }

    if !config.no_timing_cache {
        if let Some(_build_type) = config.build_type {
            spawn_timing_cache_loader(timing_tx);
        }
    }

    // Results
    let mut tests: Vec<String> = Vec::new();
    let mut unsupported_apis: Option<UnsupportedApis> = None;
    let mut timing_cache = TimingCache::default();
    let mut error: Option<String> = None;

    // For progress display with ETA
    let mut total_predicted: f64 = 0.0;
    let mut longest_test: f64 = 0.0;
    let mut fudge_factor: f64 = 1.0;

    // For conservative display count during discovery.
    // Starts with platform defaults, updated to real API results when check completes.
    // This avoids showing inflated counts that drop once we know which APIs are unavailable.
    let mut display_apis = UnsupportedApis::platform_defaults();
    let mut display_count: usize = 0;
    let mut display_ignored: usize = 0;

    // Progress bar for TTY mode
    let discovery_pb = if config.machine_output {
        None
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner().template("{msg}").unwrap());
        pb.set_message("Discovering tests...");
        Some(pb)
    };

    let mut is_compiling = false;
    let mut compiling_started: Option<Instant> = None;
    let mut dirty = false;
    let mut has_tests = false;

    // Channel state tracking - set to true when channel is disconnected
    let mut test_channel_closed = false;
    let mut api_channel_closed = config.no_early_api_check;
    let mut timing_channel_closed = config.no_timing_cache || config.build_type.is_none();

    // Helper: add a test's predicted duration to totals
    let accumulate_prediction = |test: &str,
                                  timing_cache: &TimingCache,
                                  build_type: Option<BuildType>,
                                  total_predicted: &mut f64,
                                  longest_test: &mut f64| {
        if !timing_cache.timings_by_build.is_empty() {
            if let Some(bt) = build_type {
                let pred = timing_cache.predict(bt, &test_to_timing_key(test));
                *total_predicted += pred;
                *longest_test = longest_test.max(pred);
            }
        }
    };

    // Helper: recalculate all predictions when timing cache arrives
    let recalculate_predictions = |tests: &[String],
                                    timing_cache: &TimingCache,
                                    build_type: Option<BuildType>| -> (f64, f64) {
        let mut total: f64 = 0.0;
        let mut longest: f64 = 0.0;
        if let Some(bt) = build_type {
            for test in tests {
                let pred = timing_cache.predict(bt, &test_to_timing_key(test));
                total += pred;
                longest = longest.max(pred);
            }
        }
        (total, longest)
    };

    // Helper: compute fudge-adjusted ETA for display
    let compute_display_eta = |total_predicted: f64,
                                longest_test: f64,
                                num_workers: usize,
                                fudge_factor: f64,
                                has_timing: bool| -> Option<f64> {
        if has_timing && total_predicted > 0.0 {
            let parallel_eta = total_predicted / num_workers.max(1) as f64;
            Some(parallel_eta.max(longest_test) * fudge_factor)
        } else {
            None
        }
    };

    // Main discovery loop using select!
    loop {
        // 1. Priority drain - fast non-blocking receive of all pending messages
        loop {
            match test_rx.try_recv() {
                Ok(test) => {
                    accumulate_prediction(&test, &timing_cache, config.build_type, &mut total_predicted, &mut longest_test);
                    // Update display count (filtered by current display_apis)
                    if display_apis.is_test_unsupported(&test) {
                        display_ignored += 1;
                    } else {
                        display_count += 1;
                    }
                    tests.push(test);
                    dirty = true;
                    has_tests = true;
                    // Tests arriving means compilation finished
                    if is_compiling {
                        is_compiling = false;
                        compiling_started = None;
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    test_channel_closed = true;
                    break;
                }
            }
        }

        if let Ok(err) = test_err_rx.try_recv() {
            error = Some(err);
        }

        match api_rx.try_recv() {
            Ok(result) => {
                debug_log!("api check completed: {} unsupported", result.unsupported.len());
                // API check completed - recalculate display counts with real API info
                display_apis = result.clone();
                display_count = 0;
                display_ignored = 0;
                for test in &tests {
                    if display_apis.is_test_unsupported(test) {
                        display_ignored += 1;
                    } else {
                        display_count += 1;
                    }
                }
                unsupported_apis = Some(result);
                api_channel_closed = true;
                dirty = true;
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                if !api_channel_closed {
                    debug_log!("api channel closed (no result)");
                }
                api_channel_closed = true;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }

        match timing_rx.try_recv() {
            Ok(cache) => {
                debug_log!("timing cache received: {} build types", cache.timings_by_build.len());
                // Recalculate predictions for already-collected tests
                (total_predicted, longest_test) = recalculate_predictions(&tests, &cache, config.build_type);
                // Compute fudge factor from timing cache
                if let Some(bt) = config.build_type {
                    fudge_factor = cache.average_fudge_factor(bt, &tests);
                }
                timing_cache = cache;
                timing_channel_closed = true;
                dirty = true;
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                if !timing_channel_closed {
                    debug_log!("timing channel closed (no cache)");
                }
                timing_channel_closed = true;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {}
        }

        // Check for compiling signals (from either dry-run or api-detection)
        while compiling_rx.try_recv().is_ok() {
            if !is_compiling {
                is_compiling = true;
                compiling_started = Some(Instant::now());
                dirty = true;
            }
        }

        // 2. Check for errors
        if error.is_some() {
            debug_log!("exiting loop: error");
            break;
        }

        // 3. Check if all channels are done
        if test_channel_closed && api_channel_closed && timing_channel_closed {
            debug_log!("exiting loop: all channels closed");
            break;
        }

        // 4. Update progress display
        // Show "Running X tests" when we have tests, or "Compiling..." when compiling before tests arrive
        if dirty && (has_tests || is_compiling) {
            if has_tests {
                let displayed_workers = config.num_workers.min(display_count.max(1));
                let has_timing = !timing_cache.timings_by_build.is_empty();
                let eta = compute_display_eta(total_predicted, longest_test, displayed_workers, fudge_factor, has_timing);
                let compiling_secs = compiling_started.map(|t| t.elapsed().as_secs_f64());
                let msg = format_running_message(display_count, displayed_workers, eta, display_ignored, compiling_secs);
                debug_log!("progress: {} tests, {} ignored, {} total raw, compiling={}", display_count, display_ignored, tests.len(), is_compiling);
                if *DEBUG_ENABLED {
                    eprintln!("{}", msg);
                } else if let Some(ref pb) = discovery_pb {
                    pb.set_message(msg);
                }
            } else if is_compiling {
                // Compiling before any tests arrived - show compiling message
                let compiling_secs = compiling_started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
                let msg = format!("{}", format!("Compiling core module... ({:.0}s)", compiling_secs).dimmed());
                debug_log!("compiling (no tests yet): {:.0}s", compiling_secs);
                if *DEBUG_ENABLED {
                    eprintln!("{}", msg);
                } else if let Some(ref pb) = discovery_pb {
                    pb.set_message(msg);
                }
            }
            dirty = false;
        }

        // 5. Park until data arrives, disconnect, or interrupt
        // Use select! with timeout when compiling to keep timer updated
        if is_compiling {
            select! {
                recv(test_rx) -> msg => {
                    match msg {
                        Ok(test) => {
                            accumulate_prediction(&test, &timing_cache, config.build_type, &mut total_predicted, &mut longest_test);
                            if display_apis.is_test_unsupported(&test) {
                                display_ignored += 1;
                            } else {
                                display_count += 1;
                            }
                            tests.push(test);
                            dirty = true;
                            has_tests = true;
                            is_compiling = false;
                            compiling_started = None;
                        }
                        Err(_) => {
                            test_channel_closed = true;
                        }
                    }
                }
                recv(compiling_rx) -> _ => {
                    // Already compiling, ignore additional signals
                }
                recv(sig_rx) -> _ => {
                    break;
                }
                default(Duration::from_millis(16)) => {
                    // Timeout to update compiling timer at same rate as progress display
                    dirty = true;
                }
            }
            continue;
        }

        // Use select! to wait on all channels simultaneously
        select! {
            recv(test_rx) -> msg => {
                match msg {
                    Ok(test) => {
                        accumulate_prediction(&test, &timing_cache, config.build_type, &mut total_predicted, &mut longest_test);
                        // Update display count (filtered by current display_apis)
                        if display_apis.is_test_unsupported(&test) {
                            display_ignored += 1;
                        } else {
                            display_count += 1;
                        }
                        tests.push(test);
                        dirty = true;
                        has_tests = true;
                        // Tests arriving means compilation finished
                        if is_compiling {
                            is_compiling = false;
                            compiling_started = None;
                        }
                    }
                    Err(_) => {
                        if !test_channel_closed {
                            debug_log!("test channel closed, total tests={}", tests.len());
                        }
                        test_channel_closed = true;
                    }
                }
            }
            recv(test_err_rx) -> msg => {
                if let Ok(err) = msg {
                    debug_log!("test error received");
                    error = Some(err);
                }
            }
            recv(api_rx) -> msg => {
                match msg {
                    Ok(result) => {
                        debug_log!("api check completed: {} unsupported", result.unsupported.len());
                        // API check completed - recalculate display counts with real API info
                        display_apis = result.clone();
                        display_count = 0;
                        display_ignored = 0;
                        for test in &tests {
                            if display_apis.is_test_unsupported(test) {
                                display_ignored += 1;
                            } else {
                                display_count += 1;
                            }
                        }
                        unsupported_apis = Some(result);
                        api_channel_closed = true;
                        dirty = true;
                    }
                    Err(_) => {
                        if !api_channel_closed {
                            debug_log!("api channel closed (no result)");
                        }
                        api_channel_closed = true;
                    }
                }
            }
            recv(timing_rx) -> msg => {
                match msg {
                    Ok(cache) => {
                        debug_log!("timing cache received: {} build types", cache.timings_by_build.len());
                        (total_predicted, longest_test) = recalculate_predictions(&tests, &cache, config.build_type);
                        if let Some(bt) = config.build_type {
                            fudge_factor = cache.average_fudge_factor(bt, &tests);
                        }
                        timing_cache = cache;
                        timing_channel_closed = true;
                        dirty = true;
                    }
                    Err(_) => {
                        if !timing_channel_closed {
                            debug_log!("timing channel closed (no cache)");
                        }
                        timing_channel_closed = true;
                    }
                }
            }
            recv(compiling_rx) -> msg => {
                // Only update compiling state if we actually received a signal (not channel close)
                if msg.is_ok() && !is_compiling {
                    debug_log!("compiling signal received");
                    is_compiling = true;
                    compiling_started = Some(Instant::now());
                    dirty = true;
                }
            }
            recv(sig_rx) -> _ => {
                debug_log!("interrupt signal received");
                // Interrupt signal received, exit loop
                break;
            }
        }
    }

    debug_log!("discovery loop exited, tests={}", tests.len());

    // Signal the interrupt-poll thread to exit and wait for it
    debug_log!("signaling interrupt-poll thread to exit");
    drop(shutdown_tx);
    debug_log!("waiting for interrupt-poll thread to join");
    let _ = interrupt_poll_handle.join();
    debug_log!("interrupt-poll thread joined");

    // Check for errors before doing anything else
    if let Some(err) = error {
        if let Some(pb) = discovery_pb {
            pb.finish_and_clear();
        }
        anyhow::bail!("{}", err);
    }

    // Handle -g 0: mark all GPU APIs as unsupported
    if config.gpu_jobs == Some(0) {
        let mut apis = unsupported_apis.unwrap_or_else(UnsupportedApis::platform_defaults);
        apis.disable_all_gpu_apis();
        unsupported_apis = Some(apis);
    }

    debug_log!("filtering tests by API support");

    // Now apply API filtering to the collected tests
    let mut api_ignored_count = 0;
    let mut unknown_apis: HashSet<String> = HashSet::new();
    let mut filtered_tests: Vec<String> = Vec::new();

    for test in tests {
        if let Some(ref apis) = unsupported_apis {
            if apis.is_test_unsupported(&test) {
                api_ignored_count += 1;
                continue;
            }
            if let Some(unknown_api) = apis.get_unknown_api(&test) {
                unknown_apis.insert(unknown_api);
            }
        }
        filtered_tests.push(test);
    }

    debug_log!("filtered {} tests, {} ignored", filtered_tests.len(), api_ignored_count);

    // Sort tests for deterministic ordering
    debug_log!("sorting tests");
    filtered_tests.sort();
    debug_log!("sorting done");

    // Recalculate fudge factor for filtered tests
    if let Some(bt) = config.build_type {
        if !timing_cache.timings_by_build.is_empty() {
            fudge_factor = timing_cache.average_fudge_factor(bt, &filtered_tests);
        }
    }

    // Recalculate predictions for filtered tests (api filtering may have removed some)
    let (total_predicted, longest_test) = recalculate_predictions(&filtered_tests, &timing_cache, config.build_type);

    debug_log!("finishing progress bar");

    // Finish progress bar and print final summary with newline
    if let Some(pb) = discovery_pb {
        pb.finish_and_clear();
        // Only print final "Running X tests" message if not interrupted
        if !is_interrupted() {
            let displayed_workers = config.num_workers.min(filtered_tests.len().max(1));
            let has_timing = !timing_cache.timings_by_build.is_empty();
            let eta = compute_display_eta(total_predicted, longest_test, displayed_workers, fudge_factor, has_timing);
            // Final message never shows compiling (discovery is done)
            eprintln!("{}", format_running_message(filtered_tests.len(), displayed_workers, eta, api_ignored_count, None));
        }
    }

    debug_log!("discovery complete");

    // Warn if API detection had errors (we'll have to detect APIs per-batch)
    if let Some(ref apis) = unsupported_apis {
        if let Some(ref err) = apis.error {
            eprintln!(
                "{}",
                format!("Warning: API detection: {}", err).dimmed()
            );
        }
    }

    // Determine if we can skip per-batch API detection
    // Only skip if API check completed successfully (no error) and no unknown APIs found
    let skip_api_detection = unsupported_apis
        .as_ref()
        .map(|u| u.check_completed && u.error.is_none() && unknown_apis.is_empty())
        .unwrap_or(false);

    Ok(DiscoveryResult {
        tests: filtered_tests,
        unsupported_apis,
        timing_cache,
        api_ignored_count,
        unknown_apis,
        skip_api_detection,
    })
}

/// Spawn the test discovery thread (-dry-run)
/// Just returns raw test names - no API filtering here
fn spawn_test_discovery(
    tx: Sender<String>,
    err_tx: Sender<String>,
    compiling_tx: Sender<()>,
    slang_test: PathBuf,
    root_dir: PathBuf,
    filters: Vec<String>,
    ignore_patterns: Vec<String>,
    apis: Vec<String>,
    ignore_apis: Vec<String>,
) -> Result<()> {
    // Compile filter regexes upfront
    let filter_regexes: Vec<Regex> = filters
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("Invalid filter regex: {}", p)))
        .collect::<Result<Vec<_>>>()?;

    let ignore_regexes: Vec<Regex> = ignore_patterns
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("Invalid ignore regex: {}", p)))
        .collect::<Result<Vec<_>>>()?;

    log_event(
        "dry_run",
        &format!("{} -dry-run -skip-api-detection", slang_test.display()),
    );

    let mut child = Command::new(&slang_test)
        .arg("-dry-run")
        .arg("-skip-api-detection")
        .current_dir(&root_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to run {} -dry-run", slang_test.display()))?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let err_tx_for_stderr = err_tx.clone();
    let reaper_label = format!("dry_run:{}", slang_test.display());

    // Stderr reader thread - check for errors and compiling
    thread::Builder::new()
        .name("discovery-stderr".to_string())
        .spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if line.contains("unknown option") && line.contains("-dry-run") {
                        let _ = err_tx_for_stderr.send(
                            "Your slang-test is too old and does not support the -dry-run option. \
                             Please update to a newer version of slang."
                                .to_string(),
                        );
                        return;
                    }
                    if line.contains("Compiling core module") {
                        debug_log!("stderr triggered compiling: {:?}", line);
                        let _ = compiling_tx.send(());
                    }
                }
            }
        })
        .expect("failed to spawn discovery-stderr thread");

    // Stdout reader thread - parse test names (no API filtering)
    thread::Builder::new()
        .name("discovery-stdout".to_string())
        .spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };

            // "no tests run" means we're done - close channel by dropping sender
            if line == "no tests run" {
                drop(tx);
                reap_process_with_label(child, reaper_label);
                return;
            }

            // Skip header lines
            if line.starts_with("Supported backends:") || line.starts_with("Check ") {
                continue;
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Apply ignore patterns (regex) - these are user-specified ignores
            if ignore_regexes.iter().any(|re| re.is_match(line)) {
                continue;
            }

            // Apply filter patterns (regex) - test must match at least one filter
            if !filter_regexes.is_empty() && !filter_regexes.iter().any(|re| re.is_match(line)) {
                continue;
            }

            // Apply --api and --ignore-api filters (these are different from API support detection)
            let test_id = TestId::parse(line);
            let test_api = test_id.api.as_deref();

            if !apis.is_empty() {
                match test_api {
                    Some(api) if apis.iter().any(|a| a.eq_ignore_ascii_case(api)) => {}
                    _ => continue,
                }
            }

            if !ignore_apis.is_empty() {
                if let Some(api) = test_api {
                    if ignore_apis.iter().any(|a| a.eq_ignore_ascii_case(api)) {
                        continue;
                    }
                }
            }

            // Send test name - if channel closed, exit
            if tx.send(line.to_string()).is_err() {
                break;
            }
        }

        // Channel closes when tx is dropped here
        reap_process_with_label(child, reaper_label);
    })
    .expect("failed to spawn discovery-stdout thread");

    Ok(())
}

/// Spawn the API detection thread
fn spawn_api_detection(
    tx: Sender<UnsupportedApis>,
    compiling_tx: Sender<()>,
    slang_test: PathBuf,
    root_dir: PathBuf,
) {
    thread::Builder::new()
        .name("api-detection".to_string())
        .spawn(move || {
        log_event(
            "api_check_start",
            &format!("{} -only-api-detection", slang_test.display()),
        );

        // Start fresh - no platform defaults, we get actual results from slang-test
        let mut unsupported = UnsupportedApis::new();

        let child = Command::new(&slang_test)
            .arg("-only-api-detection")
            .current_dir(&root_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                unsupported.error =
                    Some(format!("Failed to spawn slang-test for API check: {}", e));
                let _ = tx.send(unsupported);
                return;
            }
        };

        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                unsupported.error = Some("Failed to capture stdout for API check".to_string());
                let _ = child.kill();
                reap_process(child);
                let _ = tx.send(unsupported);
                return;
            }
        };

        // Also capture stderr for compiling messages
        let stderr = child.stderr.take();
        if let Some(stderr) = stderr {
            let compiling_tx_clone = compiling_tx.clone();
            thread::Builder::new()
                .name("api-detection-stderr".to_string())
                .spawn(move || {
                    let reader = BufReader::new(stderr);
                    for line in reader.lines() {
                        if let Ok(line) = line {
                            if line.contains("Compiling core module") {
                                debug_log!("api-detection stderr triggered compiling: {:?}", line);
                                let _ = compiling_tx_clone.send(());
                            }
                        }
                    }
                })
                .expect("failed to spawn api-detection-stderr thread");
        }

        let reader = BufReader::new(stdout);
        let mut saw_any_check = false;
        let mut saw_not_checked = false;

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };

            // Parse "Check vk,vulkan: Supported" or "Check dx12,d3d12: Not Supported"
            if line.starts_with("Check ") {
                saw_any_check = true;

                if let Some(colon_pos) = line.find(':') {
                    let api_part = &line[6..colon_pos];
                    let status_part = line[colon_pos + 1..].trim();

                    for api in api_part.split(',') {
                        let api = api.trim();
                        if !api.is_empty() {
                            if status_part == "Not Supported" {
                                unsupported.add_unsupported(api);
                            } else if status_part == "Supported" {
                                unsupported.add_supported(api);
                            }
                        }
                    }
                }
                continue;
            }

            // Parse "Not checked: mtl wgpu" - mark these as unsupported and exit
            if line.starts_with("Not checked:") {
                let apis_part = &line[12..].trim();
                for api in apis_part.split_whitespace() {
                    unsupported.add_unsupported(api);
                }
                saw_not_checked = true;
                // This is our sentinel - we're done
                break;
            }
        }

        // Send to reaper (process should exit quickly after "Not checked" line)
        reap_process_with_label(child, "api_check".to_string());

        if saw_any_check && saw_not_checked {
            // Modern slang-test with full API detection
            unsupported.check_completed = true;
        } else if !saw_any_check {
            // Old slang-test or error - just warn and continue without API info
            unsupported.check_completed = false;
            // Only warn if not interrupted (otherwise it's expected to have no output)
            if !is_interrupted() {
                eprintln!("{}", "Warning: slang-test does not support -only-api-detection, API detection skipped".dimmed());
            }
        } else {
            // Saw some Check lines but no "Not checked" sentinel - partial result
            unsupported.check_completed = false;
        }

        log_event(
            "api_check_end",
            &format!(
                "unsupported={:?} completed={}",
                unsupported.unsupported.iter().collect::<Vec<_>>(),
                unsupported.check_completed
            ),
        );

        // Channel closes when tx is dropped
        let _ = tx.send(unsupported);
    })
    .expect("failed to spawn api-detection thread");
}

/// Spawn the timing cache loader thread
fn spawn_timing_cache_loader(tx: Sender<TimingCache>) {
    thread::Builder::new()
        .name("timing-cache".to_string())
        .spawn(move || {
            let cache = TimingCache::load();
            // Channel closes when tx is dropped
            let _ = tx.send(cache);
        })
        .expect("failed to spawn timing-cache thread");
}

/// Format the "Running N tests with M workers" message
pub(crate) fn format_running_message(
    num_tests: usize,
    num_workers: usize,
    predicted_eta: Option<f64>,
    api_ignored: usize,
    compiling_secs: Option<f64>,
) -> String {
    let ignored_part = if api_ignored > 0 {
        format!(
            " {}",
            format!("(ignoring {} tests on unsupported APIs)", api_ignored).dimmed()
        )
    } else {
        String::new()
    };

    let compiling_part = match compiling_secs {
        Some(secs) => format!(" {}", format!("Compiling core module... ({:.0}s)", secs).dimmed()),
        None => String::new(),
    };

    match predicted_eta {
        Some(eta) => format!(
            "Running {} tests with {} workers{} {}{}",
            num_tests,
            num_workers,
            ignored_part,
            format!("(predicted ETA {:.0}s)", eta).dimmed(),
            compiling_part
        ),
        None => format!(
            "Running {} tests with {} workers{}{}",
            num_tests, num_workers, ignored_part, compiling_part
        ),
    }
}
