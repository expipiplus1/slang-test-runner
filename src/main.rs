mod runner;
mod types;

use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel;
use std::io::IsTerminal;
use std::path::PathBuf;

use runner::{set_interrupted, TestRunner};
use types::{flush_event_log, init_event_log, log_event};

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

    /// Maximum files per batch (with timing data, batches target ~10s duration up to this limit)
    #[arg(long, default_value_t = 100)]
    pub batch_size: usize,

    /// Target batch duration in seconds (only used with timing cache)
    #[arg(long, default_value_t = 10.0)]
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

    /// Adaptive load balancing: spawn extra small batches when CPU is underutilized
    #[arg(long)]
    pub adaptive: bool,

    /// Write event log to file for performance debugging
    #[arg(long)]
    pub event_log: Option<PathBuf>,

    /// List tests that would be run without actually running them
    #[arg(long)]
    pub dry_run: bool,
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
) -> Result<Vec<String>> {
    let rx = discover_tests_streaming(slang_test, root_dir, filters, ignore_patterns)?;
    let mut tests: Vec<String> = rx.iter().collect();
    tests.sort();
    Ok(tests)
}

/// Discover tests using slang-test -dry-run, streaming results via channel
/// Tests are sent as they are discovered, unsorted
pub fn discover_tests_streaming(
    slang_test: &PathBuf,
    root_dir: &PathBuf,
    filters: &[String],
    ignore_patterns: &[String],
) -> Result<crossbeam_channel::Receiver<String>> {
    use regex::Regex;
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    // Compile filter regexes upfront
    let filter_regexes: Vec<Regex> = filters
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("Invalid filter regex: {}", p)))
        .collect::<Result<Vec<_>>>()?;

    let ignore_regexes: Vec<Regex> = ignore_patterns
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("Invalid ignore regex: {}", p)))
        .collect::<Result<Vec<_>>>()?;

    // Try to use stdbuf to force line-buffering (avoids delay waiting for "no tests run")
    // Falls back to running slang-test directly if stdbuf isn't available
    let (mut child, using_stdbuf) = {
        #[cfg(unix)]
        {
            let stdbuf_result = Command::new("stdbuf")
                .arg("-oL")
                .arg(slang_test)
                .arg("-dry-run")
                .arg("-skip-api-detection")
                .current_dir(root_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn();

            match stdbuf_result {
                Ok(child) => (child, true),
                Err(_) => {
                    // stdbuf not available, fall back to direct execution
                    let child = Command::new(slang_test)
                        .arg("-dry-run")
                        .arg("-skip-api-detection")
                        .current_dir(root_dir)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::null())
                        .spawn()
                        .with_context(|| format!("Failed to run {} -dry-run", slang_test.display()))?;
                    (child, false)
                }
            }
        }
        #[cfg(not(unix))]
        {
            let child = Command::new(slang_test)
                .arg("-dry-run")
                .arg("-skip-api-detection")
                .current_dir(root_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .with_context(|| format!("Failed to run {} -dry-run", slang_test.display()))?;
            (child, false)
        }
    };
    let _ = using_stdbuf; // suppress unused warning

    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();

    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };

            // "no tests run" means we're done - send to reaper and return immediately
            if line == "no tests run" {
                runner::reap_process(child);
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

            if tx.send(line.to_string()).is_err() {
                break;
            }
        }

        // Send to reaper for async cleanup instead of blocking on wait
        runner::reap_process(child);
    });

    Ok(rx)
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

    if let Some(ref log_path) = args.event_log {
        init_event_log(log_path)?;
        log_event(
            "start",
            &format!(
                "jobs={} batch_size={} adaptive={}",
                args.jobs, args.batch_size, args.adaptive
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

        if available.len() > 1 {
            eprintln!(
                "Auto-detected {} build (newest): {}",
                build_type,
                path.display()
            );
            let others: Vec<_> = available
                .iter()
                .filter(|(name, _, _)| name != &build_type)
                .map(|(name, _, _)| name.as_str())
                .collect();
            if !others.is_empty() {
                eprintln!("  Other available: {}", others.join(", "));
                eprintln!(
                    "  Use --build-type <type> or --slang-test <path> to choose a specific build"
                );
            }
        }

        path
    };

    args.slang_test = Some(slang_test_path.clone());

    std::env::set_current_dir(&root_dir)
        .with_context(|| format!("Failed to change to root directory: {}", root_dir.display()))?;

    // Fast path for dry-run: skip TestRunner creation, stream output
    if args.dry_run {
        let rx = discover_tests_streaming(
            &slang_test_path,
            &root_dir,
            &args.filters,
            &args.ignore_patterns,
        )?;

        let mut count = 0;
        for test in rx {
            println!("{}", test);
            count += 1;
        }
        eprintln!("{} tests would be run", count);
        std::process::exit(0);
    }

    let runner = TestRunner::new(args);
    let success = runner.run()?;

    runner.save_timing();

    log_event("end", &format!("success={}", success));
    flush_event_log();

    std::process::exit(if success { 0 } else { 1 });
}
