mod api;
mod discovery;
mod event_log;
mod progress;
mod runner;
mod scheduler;
mod timing;
mod types;

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use std::io::IsTerminal;
use std::path::PathBuf;

use crate::types::DEBUG_START;
use discovery::{run_concurrent_discovery, DiscoveryConfig};
use runner::{set_interrupted, TestRunner};
use event_log::{flush_event_log, init_event_log, log_event};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "sti")]
#[command(about = "A parallel test runner for slang-test with better output")]
pub struct Args {
    /// Root directory of the slang project (defaults to current directory)
    #[arg(short = 'C', long)]
    pub root_dir: Option<PathBuf>,

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

    /// Diff tool for showing expected/actual differences: difft, git, diff, none, auto
    /// (default: auto, fallback chain: difft → git → diff → none)
    #[arg(long, default_value = "auto")]
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

    // ---- Internal fields (not CLI args) ----
    /// The original -C argument as provided (for rerun command)
    #[arg(skip)]
    pub root_dir_original: Option<String>,

    /// The original --slang-test argument as provided (for rerun command)
    #[arg(skip)]
    pub slang_test_original: Option<String>,

    /// The effective root directory (resolved from root_dir or default ".")
    #[arg(skip)]
    pub root_dir_effective: PathBuf,
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
            // Find the expected path for the requested build type
            let expected_path = build_types
                .iter()
                .find(|(name, _)| *name == preferred)
                .map(|(_, path)| *path);

            let path_hint = match expected_path {
                Some(path) => format!("\n  Expected: {}", root_dir.join(path).display()),
                None => format!(
                    "\n  Valid build types: {}",
                    build_types.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
                ),
            };

            anyhow::bail!(
                "Build type '{}' not found.{}\n  Available: {}",
                preferred,
                path_hint,
                if available.is_empty() {
                    "none (no slang-test executables found)".to_string()
                } else {
                    available
                        .iter()
                        .map(|(n, p, _)| format!("{} ({})", n, p.display()))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
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
    // Initialize debug timer at program start
    let _ = *DEBUG_START;
    debug_log!("program start");

    ctrlc::set_handler(|| {
        set_interrupted();
    })
    .expect("Error setting Ctrl-C handler");

    let mut args = Args::parse();

    // Validate job counts
    if args.jobs == 0 {
        anyhow::bail!("-j/--jobs must be at least 1");
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

    // Handle root_dir: save original for rerun command, then resolve effective path
    args.root_dir_original = args.root_dir.as_ref().map(|p| p.to_string_lossy().to_string());
    let root_dir_input = args.root_dir.clone().unwrap_or_else(|| PathBuf::from("."));
    let root_dir = root_dir_input.canonicalize().unwrap_or(root_dir_input);
    args.root_dir_effective = root_dir.clone();

    // Save original --slang-test for rerun command
    args.slang_test_original = args.slang_test.as_ref().map(|p| p.to_string_lossy().to_string());

    let slang_test_path = if let Some(path) = args.slang_test.take() {
        // User explicitly specified a path - use it as-is
        // (let Command::new handle PATH lookup for bare names like "slang-test")
        path
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

    // Detect build type for timing cache
    let build_type = timing::BuildType::from_path(&slang_test_path);

    // Run concurrent discovery: API check, timing cache load, and -dry-run all at once
    let discovery_config = DiscoveryConfig {
        slang_test: &slang_test_path,
        root_dir: &root_dir,
        filters: &args.filters,
        ignore_patterns: &args.ignore_patterns,
        apis: &args.apis,
        ignore_apis: &args.ignore_apis,
        no_early_api_check: args.no_early_api_check,
        no_timing_cache: args.no_timing_cache,
        build_type,
        gpu_jobs: args.gpu_jobs,
        machine_output: !is_stderr_tty(),
        num_workers: args.jobs,
    };

    let discovery_result = run_concurrent_discovery(&discovery_config)?;

    // Handle dry-run: just print tests and exit
    if args.dry_run {
        for test in &discovery_result.tests {
            println!("{}", test);
        }
        let ignored_msg = if discovery_result.api_ignored_count > 0 {
            format!(
                " (ignoring {} tests on unsupported APIs)",
                discovery_result.api_ignored_count
            )
        } else {
            String::new()
        };
        eprintln!("{} tests would be run{}", discovery_result.tests.len(), ignored_msg);
        std::process::exit(0);
    }

    // Create TestRunner with pre-discovered data
    let runner = TestRunner::new_with_discovery(args, discovery_result);
    let success = runner.run()?;

    runner.save_timing();

    log_event("end", &format!("success={}", success));
    flush_event_log();

    std::process::exit(if success { 0 } else { 1 });
}
