mod runner;
mod types;

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use crossbeam_channel;
use std::io::IsTerminal;
use std::path::PathBuf;

use runner::{set_interrupted, TestRunner, run_early_api_check};
use types::{flush_event_log, init_event_log, log_event, UnsupportedApis};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "sti")]
#[command(about = "A parallel test runner for slang-test with better output")]
pub struct Args {
    /// Root directory of the slang project (defaults to current directory)
    #[arg(short = 'C', long, default_value = ".")]
    pub root_dir: PathBuf,

    /// Path to slang-test executable (relative to root_dir or absolute)
    /// If not specified, auto-detects the newest build
    #[arg(long)]
    pub slang_test: Option<PathBuf>,

    /// Build type to use: debug, release, or relwithdebinfo
    /// If not specified, uses the newest available build
    #[arg(long)]
    pub build_type: Option<String>,

    /// Number of parallel workers
    #[arg(short = 'j', long, default_value_t = num_cpus())]
    pub jobs: usize,

    /// Maximum number of concurrent GPU jobs (vk/cuda/dx11/dx12/metal tests and gfx-unit-test-tool)
    /// When set, batches are segmented into GPU-only and CPU-only batches
    #[arg(short = 'g', long)]
    pub gpu_jobs: Option<usize>,

    /// Maximum files per batch (default: (num_tests/jobs)*3; with timing data, batches target batch_duration up to this limit)
    #[arg(long, default_value_t = 0)]
    pub batch_size: usize,

    /// Target batch duration in seconds (default: predicted_runtime/2; only used with timing cache)
    #[arg(long, default_value_t = 0.0)]
    pub batch_duration: f64,

    /// Number of retries for failed tests
    #[arg(long, default_value_t = 2)]
    pub retries: usize,

    /// Test filter regexes (union: test runs if it matches ANY filter; if empty, runs all tests)
    /// Examples: "^tests/compute" (prefix), "diagnostic" (infix), "\.slang$" (suffix)
    #[arg()]
    pub filters: Vec<String>,

    /// Hide ignored tests from output
    #[arg(long)]
    pub hide_ignored: bool,

    /// Ignore patterns as regexes (union: test is ignored if it matches ANY pattern)
    #[arg(long = "ignore")]
    pub ignore_patterns: Vec<String>,

    /// Only run tests for specific APIs (union: test runs if it matches ANY specified API)
    /// Examples: --api vk --api cuda
    #[arg(long = "api")]
    pub apis: Vec<String>,

    /// Exclude tests for specific APIs (union: test is excluded if it matches ANY specified API)
    /// Examples: --ignore-api vk --ignore-api metal
    #[arg(long = "ignore-api")]
    pub ignore_apis: Vec<String>,

    /// Diff tool for showing expected/actual differences: none, diff, difft (default: diff)
    #[arg(long, default_value = "diff")]
    pub diff: String,

    /// Additional arguments to pass to slang-test
    #[arg(last = true)]
    pub extra_args: Vec<String>,

    /// Verbose output: show batch reproduction commands for slow batches
    #[arg(short = 'v', long)]
    pub verbose: bool,

    /// Ignore timing cache (don't use cached timing for scheduling or ETA)
    #[arg(long)]
    pub no_timing_cache: bool,

    /// Write event log to file for performance debugging
    #[arg(long)]
    pub event_log: Option<PathBuf>,

    /// List tests that would be run without actually running them
    #[arg(long)]
    pub dry_run: bool,

    /// Timeout per test batch in seconds (default: 600 = 10 minutes)
    #[arg(long, default_value_t = 600)]
    pub timeout: u64,

    /// GPU stagger increment in milliseconds. The first N batches (N = jobs) will have
    /// increasing amounts of CPU work at the start to stagger GPU test launches.
    /// Batch 0 gets 1x this value, batch 1 gets 2x, etc. Set to 0 to disable.
    #[arg(long, default_value_t = 100)]
    pub gpu_stagger: u64,

    /// Disable early API detection. By default, we run a quick check to detect which
    /// APIs are supported before discovering tests, allowing us to skip tests for
    /// unsupported APIs early.
    #[arg(long)]
    pub no_early_api_check: bool,
}

// ============================================================================
// Utility Functions
// ============================================================================

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

pub fn is_stderr_tty() -> bool {
    std::io::stderr().is_terminal()
}

/// Discover all tests using slang-test -dry-run (blocking version)
/// Returns a list of test identifiers which can be:
/// - Simple: "tests/path/file.slang"
/// - With variant: "tests/path/file.slang.0 (vk)"
/// - Synthesized: "tests/path/file.slang.1 syn (llvm)"
/// - Internal: "slang-unit-test-tool/modulePtr.internal"
/// Filters support regex patterns (e.g., "^tests/compute" for prefix, "diagnostic" for infix)
pub fn discover_tests_via_dry_run(
    slang_test: &PathBuf,
    root_dir: &PathBuf,
    filters: &[String],
    ignore_patterns: &[String],
    apis: &[String],
    ignore_apis: &[String],
) -> Result<Vec<String>> {
    let (rx, error_rx, _compiling_rx) = discover_tests_streaming(slang_test, root_dir, filters, ignore_patterns, apis, ignore_apis)?;
    let mut tests: Vec<String> = rx.iter().collect();
    // Check for errors after iteration completes
    if let Ok(error_msg) = error_rx.try_recv() {
        anyhow::bail!("{}", error_msg);
    }
    tests.sort();
    Ok(tests)
}

/// Discover tests using slang-test -dry-run, streaming results via channel
/// Tests are sent as they are discovered, unsorted
/// Returns (test_receiver, error_receiver, compiling_receiver) - check error_receiver after iteration
/// The compiling_receiver signals when "Compiling core module" is detected on stderr
pub fn discover_tests_streaming(
    slang_test: &PathBuf,
    root_dir: &PathBuf,
    filters: &[String],
    ignore_patterns: &[String],
    apis: &[String],
    ignore_apis: &[String],
) -> Result<(crossbeam_channel::Receiver<String>, crossbeam_channel::Receiver<String>, crossbeam_channel::Receiver<()>)> {
    use regex::Regex;
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    use types::TestId;

    // Compile filter regexes upfront
    let filter_regexes: Vec<Regex> = filters
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("Invalid filter regex: {}", p)))
        .collect::<Result<Vec<_>>>()?;

    let ignore_regexes: Vec<Regex> = ignore_patterns
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("Invalid ignore regex: {}", p)))
        .collect::<Result<Vec<_>>>()?;

    // Clone API filters for the thread
    let apis: Vec<String> = apis.to_vec();
    let ignore_apis: Vec<String> = ignore_apis.to_vec();

    // Log the dry-run invocation
    log_event("dry_run", &format!("{} -dry-run -skip-api-detection", slang_test.display()));

    let mut child = Command::new(slang_test)
        .arg("-dry-run")
        .arg("-skip-api-detection")
        .current_dir(root_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to run {} -dry-run", slang_test.display()))?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let (error_tx, error_rx) = crossbeam_channel::bounded::<String>(1);
    let (compiling_tx, compiling_rx) = crossbeam_channel::bounded::<()>(1);

    // Spawn a thread to check stderr for "unknown option" error (old slang-test)
    // and "Compiling core module" message
    let stderr_error_tx = error_tx.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                if line.contains("unknown option") && line.contains("-dry-run") {
                    let _ = stderr_error_tx.send(
                        "Your slang-test is too old and does not support the -dry-run option. \
                         Please update to a newer version of slang.".to_string()
                    );
                    return;
                }
                if line.contains("Compiling core module") {
                    let _ = compiling_tx.try_send(());
                }
            }
        }
    });

    // Label for reaper logging
    let reaper_label = format!("dry_run:{}", slang_test.display());

    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };

            // "no tests run" means we're done - send to reaper and return immediately
            if line == "no tests run" {
                runner::reap_process_with_label(child, reaper_label);
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

            // Apply ignore patterns (regex)
            if ignore_regexes.iter().any(|re| re.is_match(line)) {
                continue;
            }

            // Apply filter patterns (regex) - test must match at least one filter
            if !filter_regexes.is_empty() && !filter_regexes.iter().any(|re| re.is_match(line)) {
                continue;
            }

            // Apply API filters
            let test_id = TestId::parse(line);
            let test_api = test_id.api.as_deref();

            // If --api is specified, only include tests matching one of the APIs
            if !apis.is_empty() {
                match test_api {
                    Some(api) if apis.iter().any(|a| a.eq_ignore_ascii_case(api)) => {}
                    _ => continue, // Skip tests without API or with non-matching API
                }
            }

            // If --ignore-api is specified, exclude tests matching any of the APIs
            if !ignore_apis.is_empty() {
                if let Some(api) = test_api {
                    if ignore_apis.iter().any(|a| a.eq_ignore_ascii_case(api)) {
                        continue;
                    }
                }
            }

            if tx.send(line.to_string()).is_err() {
                break;
            }
        }

        // Send to reaper for async cleanup instead of blocking on wait
        runner::reap_process_with_label(child, reaper_label);
    });

    // Check immediately if there's an error (give it a moment to detect)
    std::thread::sleep(std::time::Duration::from_millis(100));
    if let Ok(error_msg) = error_rx.try_recv() {
        anyhow::bail!("{}", error_msg);
    }

    Ok((rx, error_rx, compiling_rx))
}

fn detect_slang_test_build(
    root_dir: &PathBuf,
    preferred_type: Option<&str>,
) -> Result<(PathBuf, String, Vec<(String, PathBuf, std::time::SystemTime)>)> {
    let build_types = [
        ("debug", "build/Debug/bin/slang-test"),
        ("release", "build/Release/bin/slang-test"),
        ("relwithdebinfo", "build/RelWithDebInfo/bin/slang-test"),
    ];

    let mut available: Vec<(String, PathBuf, std::time::SystemTime)> = Vec::new();

    for (name, rel_path) in &build_types {
        let path = root_dir.join(rel_path);
        if path.exists() {
            if let Ok(metadata) = std::fs::metadata(&path) {
                if let Ok(modified) = metadata.modified() {
                    available.push((name.to_string(), path, modified));
                }
            }
        }
    }

    if available.is_empty() {
        anyhow::bail!(
            "No slang-test executable found in build/{{Debug,Release,RelWithDebInfo}}/bin/"
        );
    }

    if let Some(preferred) = preferred_type {
        if let Some((_, path, _)) = available.iter().find(|(name, _, _)| name == preferred) {
            return Ok((path.clone(), preferred.to_string(), available));
        } else {
            anyhow::bail!(
                "Build type '{}' not found. Available: {}",
                preferred,
                available
                    .iter()
                    .map(|(n, _, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    available.sort_by(|a, b| b.2.cmp(&a.2));

    let (build_type, path, _) = available[0].clone();
    Ok((path, build_type, available))
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> Result<()> {
    ctrlc::set_handler(|| {
        set_interrupted();
    })
    .expect("Error setting Ctrl-C handler");

    let mut args = Args::parse();

    // Validate job counts
    if args.jobs == 0 {
        anyhow::bail!("-j/--jobs must be at least 1");
    }
    if args.gpu_jobs == Some(0) {
        anyhow::bail!("-g/--gpu-jobs must be at least 1");
    }

    // Validate API filter combinations
    if !args.apis.is_empty() && !args.ignore_apis.is_empty() {
        // Check for overlap - it's nonsense to both include and exclude the same API
        for api in &args.apis {
            if args.ignore_apis.iter().any(|a| a.eq_ignore_ascii_case(api)) {
                anyhow::bail!(
                    "Conflicting API filters: '{}' appears in both --api and --ignore-api",
                    api
                );
            }
        }
    }

    if let Some(ref log_path) = args.event_log {
        init_event_log(log_path)?;
        log_event(
            "start",
            &format!(
                "jobs={} batch_size={}",
                args.jobs, args.batch_size
            ),
        );
    }

    let root_dir = args.root_dir.canonicalize().unwrap_or(args.root_dir.clone());

    let slang_test_path = if let Some(path) = args.slang_test {
        if path.is_relative() {
            root_dir.join(&path)
        } else {
            path
        }
    } else {
        let (path, build_type, available) =
            detect_slang_test_build(&root_dir, args.build_type.as_deref())?;

        // Only show build selection info if user didn't explicitly choose and multiple are available
        if available.len() > 1 && args.build_type.is_none() {
            eprintln!(
                "{}",
                format!(
                    "Using {} build: {}",
                    build_type,
                    path.display()
                ).dimmed()
            );
            let others: Vec<_> = available
                .iter()
                .filter(|(name, _, _)| name != &build_type)
                .map(|(name, _, _)| name.as_str())
                .collect();
            if !others.is_empty() {
                eprintln!("{}", format!("  {} is also available", others.join(", ")).dimmed());
            }
        }

        path
    };

    args.slang_test = Some(slang_test_path.clone());
    args.root_dir = root_dir.clone();

    // Start early API detection (unless disabled)
    let api_check_rx = if !args.no_early_api_check {
        Some(run_early_api_check(&slang_test_path, &root_dir))
    } else {
        None
    };

    // Fast path for dry-run: skip TestRunner creation, stream output
    if args.dry_run {
        let is_tty = is_stderr_tty();

        // Wait for API check to complete (with timeout) for filtering
        let unsupported_apis = api_check_rx.and_then(|rx| {
            rx.recv_timeout(std::time::Duration::from_secs(5)).ok()
        });

        let (rx, error_rx, compiling_rx) = discover_tests_streaming(
            &slang_test_path,
            &args.root_dir,
            &args.filters,
            &args.ignore_patterns,
            &args.apis,
            &args.ignore_apis,
        )?;

        let mut count = 0;
        let mut api_ignored = 0;
        let mut shown_compiling = false;

        loop {
            // Check for errors from the discovery thread
            if let Ok(error_msg) = error_rx.try_recv() {
                anyhow::bail!("{}", error_msg);
            }

            // Check for compiling signal
            if !shown_compiling && compiling_rx.try_recv().is_ok() {
                if is_tty {
                    eprint!("\x1b[2mCompiling core module...\x1b[0m");
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
                shown_compiling = true;
            }

            // Try to receive with a timeout so we can check compiling signal
            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(test) => {
                    // Filter tests for unsupported APIs
                    if let Some(ref unsupported) = unsupported_apis {
                        if unsupported.is_test_unsupported(&test) {
                            api_ignored += 1;
                            continue;
                        }
                    }

                    // On first test output, clear any compiling message
                    if count == 0 && shown_compiling && is_tty {
                        eprint!("\r\x1b[K");
                    }
                    println!("{}", test);
                    count += 1;
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Still waiting - continue loop to check compiling signal
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Channel closed - check for errors one more time
                    if let Ok(error_msg) = error_rx.try_recv() {
                        anyhow::bail!("{}", error_msg);
                    }
                    if shown_compiling && count == 0 && is_tty {
                        eprint!("\r\x1b[K");
                    }
                    break;
                }
            }
        }

        let ignored_msg = if api_ignored > 0 {
            format!(" (ignored {} tests on unsupported APIs)", api_ignored)
        } else {
            String::new()
        };
        eprintln!("{} tests would be run{}", count, ignored_msg);
        std::process::exit(0);
    }

    // Wait for API check to complete before creating TestRunner
    let unsupported_apis: Option<UnsupportedApis> = if let Some(rx) = api_check_rx {
        match rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(result) => {
                // Warn if API check had errors
                if let Some(ref error) = result.error {
                    eprintln!("{}", format!("Warning: API detection: {}", error).dimmed());
                }
                Some(result)
            }
            Err(_) => {
                eprintln!("{}", "Warning: API detection timed out, will detect APIs per-batch".dimmed());
                None
            }
        }
    } else {
        None
    };

    let runner = TestRunner::new(args, unsupported_apis);
    let success = runner.run()?;

    runner.save_timing();

    log_event("end", &format!("success={}", success));
    flush_event_log();

    std::process::exit(if success { 0 } else { 1 });
}
