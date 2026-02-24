mod runner;
mod types;

use anyhow::{Context, Result};
use clap::Parser;
use std::io::IsTerminal;
use std::path::PathBuf;
use walkdir::WalkDir;

use runner::{set_interrupted, TestRunner};
use types::{flush_event_log, init_event_log, log_event};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "slang-test-runner")]
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

    /// Test directory (relative to root_dir or absolute)
    #[arg(long, default_value = "tests")]
    pub test_dir: PathBuf,

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

    /// Test prefixes to run (if empty, runs all tests)
    #[arg()]
    pub prefixes: Vec<String>,

    /// Hide ignored tests from output
    #[arg(long)]
    pub hide_ignored: bool,

    /// Test patterns to ignore (can be specified multiple times)
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

pub fn discover_test_files(
    test_dir: &PathBuf,
    prefixes: &[String],
    ignore_patterns: &[String],
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let extensions = ["slang", "hlsl", "glsl", "c"];

    for entry in WalkDir::new(test_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if extensions.iter().any(|e| ext == *e) {
                    let path_str = path.to_string_lossy();

                    if ignore_patterns.iter().any(|p| path_str.contains(p)) {
                        continue;
                    }

                    if prefixes.is_empty() || prefixes.iter().any(|p| path_str.contains(p)) {
                        files.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    files.sort();
    Ok(files)
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

    args.slang_test = Some(slang_test_path);

    std::env::set_current_dir(&root_dir)
        .with_context(|| format!("Failed to change to root directory: {}", root_dir.display()))?;

    let runner = TestRunner::new(args);
    let success = runner.run()?;

    runner.save_timing();

    log_event("end", &format!("success={}", success));
    flush_event_log();

    std::process::exit(if success { 0 } else { 1 });
}
