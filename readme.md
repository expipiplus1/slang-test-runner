# Slang Test Interceptor

A wrapper for slang-test which implements several enhancements:

- Parallelized test running
- More user friendly output, during test running and as a final report
- Robust handling of segfaults or unexpected crashes
- Automatic retries of transiently failing tests

## Usage

```bash
# Run all tests from the slang directory
sti -C /path/to/slang

# Run tests matching a regex filter (infix match)
sti -C /path/to/slang diagnostic

# Run tests matching a prefix
sti -C /path/to/slang '^tests/compute'

# Run multiple filter patterns (union - matches ANY)
sti -C /path/to/slang '^tests/compute' '^tests/autodiff'

# Run internal unit tests
sti -C /path/to/slang slang-unit-test-tool

# List tests that would be run without running them
sti -C /path/to/slang --dry-run diagnostic

# Customize parallelism
sti -C /path/to/slang -j 8 '^tests/compute'

# Use difft for side-by-side diffs
sti -C /path/to/slang --diff difft '^tests/compute'

# Ignore tests matching a pattern
sti -C /path/to/slang --ignore 'cuda' --ignore 'metal'

# Pass extra arguments to slang-test
sti -C /path/to/slang '^tests/compute' -- -api vk
```

## Options

### Common options

- `<FILTERS>` - Regex patterns to filter tests (union: test runs if it matches ANY filter). Examples: `diagnostic` (infix), `^tests/compute` (prefix), `\.slang$` (suffix). If empty, runs all tests.
- `-C, --root-dir <PATH>` - Root directory of the slang project (default: current directory)
- `-j, --jobs <N>` - Number of parallel workers (default: number of CPUs)
- `--dry-run` - List tests that would be run without actually running them
- `--ignore <PATTERN>` - Ignore tests matching regex pattern (can be specified multiple times; union: ignored if matches ANY)
- `--diff <TOOL>` - Diff tool for expected/actual differences: `none`, `diff`, `difft` (default: `diff`)
- `-v, --verbose` - Verbose output: show batch reproduction commands for slow batches, extended slow-test report with per-backend timing
- `-- <ARGS>` - Additional arguments to pass directly to slang-test (e.g., `-- -api vk`)

### Build selection

- `--slang-test <PATH>` - Path to slang-test executable (default: auto-detects newest build)
- `--build-type <TYPE>` - Build type to use: debug, release, or relwithdebinfo (default: newest available)

### Advanced options

- `--retries <N>` - Number of retries for failed tests (default: 2)
- `--hide-ignored` - Hide ignored tests from output
- `--batch-size <N>` - Maximum tests per slang-test invocation (default: 100)
- `--batch-duration <SECS>` - Target batch duration in seconds when timing data is available (default: 10.0)
- `--no-timing-cache` - Ignore cached timing data for scheduling and ETA
- `--adaptive` - Adaptive load balancing: spawn extra workers when CPU is underutilized
- `--event-log <PATH>` - Write CSV event log for performance debugging

When stderr is not a TTY (e.g., in CI or when piped), output automatically switches to machine-readable format: no carriage returns, no terminal clearing, sparse progress updates.

## Output format

During execution, a progress line updates in place showing:
```
[  467/3112 ]  15.0% | 131 passed, 0 failed, 336 ignored (7.5s) ETA: 42s [24/98]
```

- Files completed / total files with percentage
- Test counts (passed, failed, ignored)
- Elapsed time and ETA
- `[running/remaining]` batches - helps identify parallelism issues

When stderr is not a TTY (CI, piped output), progress is printed on separate lines:
```
[467/3112] 131 passed, 0 failed, 336 ignored (7.5s) [24/98]
```

At completion, a summary shows:
- Failed tests with details and diff output
- Overall statistics
- Slowest files (if timing data available)
- Command to rerun failures

## Crash/Segfault Handling

When slang-test crashes (segfault, timeout, or abnormal exit), the runner uses the following strategy:

### How crashes are detected

1. **Exit code analysis**: Normal completion returns exit code 0 (all pass) or 1 (some failures). Any other exit code (e.g., 139 for segfault) indicates a crash.

2. **Timeout detection**: If a batch exceeds the 5-minute timeout, the process is killed and treated as a crash.

### How the crashing test is identified

Since slang-test only prints test results *after* a test completes, a crash means the crashing test's name was never printed. The runner identifies it by:

1. **Tracking completed tests**: All test names that were printed before the crash are collected.
2. **Computing remaining tests**: The runner compares the list of test files sent to the batch against the tests that completed. Any test file not accounted for in the output is considered "remaining".

### Recovery process

1. **Process completed results**: Tests that passed/failed/ignored before the crash are counted normally.

2. **Isolate the crash**: Each remaining test file is run individually in its own slang-test process.

3. **Identify the culprit**: If a single-test run also crashes, that specific test is marked as failed with "Test caused a crash/segfault" and skipped. If it passes, it was likely affected by test interaction or timing.

4. **Continue execution**: Other batches and workers continue running unaffected.

### Example scenario

Batch contains: `[test-a.slang, test-b.slang, test-c.slang, test-d.slang]`

slang-test output before crash:
```
passed test: 'test-a.slang (cpu)'
ignored test: 'test-b.slang (cuda)'
```
Then segfault occurs.

Recovery:
1. `test-a` counted as passed, `test-b` counted as ignored
2. `test-c.slang` and `test-d.slang` are run individually
3. If `test-c.slang` crashes again, it's marked as failed (crash)
4. If `test-d.slang` passes, it's counted normally

## Retry Logic

Failed tests are automatically retried to handle transient failures:

1. **When retries happen**: Failed tests are immediately re-queued for retry concurrently with other running batches. Retries don't wait for the full run to complete.
2. **Retry tracking**: Each test name is tracked to prevent infinite retry loops.
3. **Success on retry**: If the specific test passes on retry, it's counted as passed and noted in the summary.
4. **Persistent failure**: If retry also fails, the test is marked failed with full failure output.

Note: Retries only work for file-based tests where the test file exists on disk. Internal tests (`.internal`) are not retried.

## slang-test output formats

The runner parses the following output patterns from slang-test:

```
passed test: 'tests/compute/array-param.slang.1 (cpu)'
passed test: 'tests/autodiff/compileBenchmark.internal' 7.51838s
FAILED test: 'tests/compute/foo.slang (vk)'
ignored test: 'tests/compute/bar.slang (cuda)'
```

Test names have the format:
- `<test-file>.<variant> (<backend>)` - e.g., `tests/compute/array-param.slang.1 (cpu)`
- `<test-file>.<variant> syn (<backend>)` - synthesized tests, e.g., `tests/compute/array-param.slang.5 syn (llvm)`
- `<category>/<test-name>.internal` - internal unit tests, e.g., `slang-unit-test-tool/modulePtr.internal`

Failure details are printed on lines starting with `[test-name]` **before** the `FAILED test:` line:
```
[slang-unit-test-tool/RecordReplay_cpu_hello_world.internal] Failed to launch process of 'cpu-hello-world'
FAILED test: 'slang-unit-test-tool/RecordReplay_cpu_hello_world.internal'
```

Exit codes:
- `0` - All tests passed (or only ignored)
- `1` - Some tests failed

## Core Module Compilation

On debug builds of slang, the first invocation of slang-test compiles the core module which can take 10-20 seconds. The runner handles this automatically:

1. All workers start immediately for maximum parallelism
2. When "Compiling core module" is detected on stderr, the runner tracks which batch is compiling
3. Other batches that aren't doing the compilation are killed and their files re-queued
4. Once compilation completes, all workers resume normal execution

This ensures only one process compiles the core module while minimizing wasted work.

## Ctrl-C Handling

Pressing Ctrl-C gracefully interrupts the test run:

1. All running batches are signaled to stop
2. Current progress is preserved
3. A summary is printed with all results collected so far
4. The runner exits with a non-zero status

## Dynamic Work Pool

Instead of pre-creating batches, the runner uses a dynamic work pool that workers pull from:

1. **Duration-based batching**: When timing data is available, batches target a duration (default 10s via `--batch-duration`) rather than a fixed file count. This balances startup overhead against batch size. Without timing data, random batches up to `--batch-size` are used.

2. **Slow-first scheduling**: Files are sorted by predicted duration (longest first) and workers preferentially pick slower files. This prevents the "long tail" problem where one worker is stuck on slow tests at the end.

3. **End-game single files**: When few files remain (less than 2x worker count), files are dispatched individually for maximum parallelism.

4. **Retry integration**: Failed tests go back into the same pool, automatically getting picked up by available workers.

5. **Progress display**: Shows `[running/queued]` so you can see parallelism level:
   - `[32/100]` = 32 batches running, 100 files still in queue
   - `[8/0]` = 8 batches running, queue empty (finishing up)

## Timing-Based Scheduling

The runner maintains a cache of test execution times to optimize scheduling:

### How it works

1. **Per-test timing**: During execution, the runner tracks how long each test takes, broken down by backend (vk, cpu, llvm, etc.)

2. **Cache storage**: After each run, timing data is saved to the state directory (see below)

3. **LPT scheduling**: On subsequent runs, files are sorted by predicted duration (longest first). This ensures slow tests start early and run concurrently with faster tests, preventing the "long tail" problem where all workers finish except one stuck on slow tests.

4. **API-aware predictions**: When running with `-- -api vk`, only Vulkan backend timings are used for predictions. This prevents filtered runs from skewing estimates for full runs.

### State location

- Linux: `~/.local/state/slang-test-interceptor/timing.json`
- macOS: `~/Library/Application Support/slang-test-interceptor/timing.json`
- Windows: `%LOCALAPPDATA%\slang-test-interceptor\timing.json`

The state is automatically created and updated. Delete it to reset timing estimates.

## How it works

Test discovery uses `slang-test -dry-run` to enumerate all available tests (both file-based and internal tests). This ensures the runner knows exactly which tests exist without needing to scan the filesystem.

`slang-test` can be run over specific test cases in a single threaded mode like `slang-test tests/foo/bar.slang tests/baz/qux.slang`

There is some overhead in starting `slang-test` so we don't just want to spin up slang-test for every test, we want to run N batches of tests at once where N is the degree of parallelism we want, but not have batches so large that we might be waiting for the slow one at the end of the run.

There might be several tests per test file, some of which are synthesized and some are ignored.

There are also internal tests not represented by files, these are called `slang-unit-test-tool/modulePtr.internal` or similar.

At the end it will output a command which will run all the non-passing and non-ignored tests.

## Building

```bash
cargo build --release
```

The binary will be at `target/release/sti`.
