# Slang test runner

A wrapper for slang-test which implements several enhancements:

- Parallelized test running
- More user friendly output, during test running and as a final report
- Robust handling of segfaults or unexpected crashes
- Automatic retries of transiently failing tests

## Usage

```bash
# Run all tests from the slang directory
slang-test-runner -C /path/to/slang

# Run specific test files
slang-test-runner -C /path/to/slang tests/compute/array-param.slang

# Run tests matching a prefix
slang-test-runner -C /path/to/slang tests/compute/

# Run internal unit tests
slang-test-runner -C /path/to/slang slang-unit-test-tool/

# Customize parallelism
slang-test-runner -C /path/to/slang -j 8 tests/compute/

# Use difft for side-by-side diffs
slang-test-runner -C /path/to/slang --diff difft tests/compute/

# Pass extra arguments to slang-test
slang-test-runner -C /path/to/slang tests/compute/ -- -api vk
```

## Options

- `-C, --root-dir <PATH>` - Root directory of the slang project (default: current directory)
- `--slang-test <PATH>` - Path to slang-test executable (default: build/Debug/bin/slang-test)
- `--test-dir <PATH>` - Test directory for file discovery (default: tests)
- `-j, --jobs <N>` - Number of parallel workers (default: number of CPUs)
- `--batch-size <N>` - Tests per slang-test invocation (default: 20)
- `--retries <N>` - Number of retries for failed tests (default: 2)
- `--hide-ignored` - Hide ignored tests from output
- `--ignore <PATTERN>` - Ignore tests matching pattern (can be specified multiple times)
- `--diff <TOOL>` - Diff tool for showing expected/actual differences: `none`, `diff`, `difft` (default: `diff`)
- `--machine-output` - Machine-readable output: no carriage returns, no terminal clearing, sparse progress updates (every 100 tests)
- `-v, --verbose` - Verbose output: show batch reproduction commands for slow batches, extended slow-test report with per-backend timing
- `-- <ARGS>` - Additional arguments to pass directly to slang-test

## Output format

During execution, a progress line updates in place showing:
```
[  467/3112 ]  15.0% | 131 passed, 0 failed, 336 ignored (7.5s) ETA: 42s [24/98]
```

- Files completed / total files with percentage
- Test counts (passed, failed, ignored)
- Elapsed time and ETA
- `[running/remaining]` batches - helps identify parallelism issues

With `--machine-output`, progress is printed on separate lines every 100 tests without carriage returns:
```
[467/3112] 131 passed, 0 failed, 336 ignored (7.5s) [24/98 batches]
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

On debug builds of slang, the first invocation of slang-test compiles the core module which can take 10-20 seconds. To avoid having all parallel workers trying to compile simultaneously, the runner executes the first batch before starting workers:

1. Runs first batch of tests
2. If "Compiling" message is detected on stderr, displays it and waits for completion
3. Once first batch completes, starts parallel workers for remaining batches

This ensures the core module is cached before parallel execution begins, without wasting a separate warmup run.

## Ctrl-C Handling

Pressing Ctrl-C gracefully interrupts the test run:

1. All running batches are signaled to stop
2. Current progress is preserved
3. A summary is printed with all results collected so far
4. The runner exits with a non-zero status

## Dynamic Work Pool

Instead of pre-creating batches, the runner uses a dynamic work pool that workers pull from:

1. **Adaptive batch sizing**: Batch size depends on remaining work:
   - Many files remaining: full batch size (e.g., 20 files)
   - Getting low: half size
   - Nearly done: quarter size
   - Final stretch: single files for maximum parallelism

2. **Retry integration**: Failed tests go back into the same pool, automatically getting picked up by available workers

3. **Progress display**: Shows `[running/queued]` so you can see parallelism level:
   - `[32/100]` = 32 batches running, 100 files still in queue
   - `[8/0]` = 8 batches running, queue empty (finishing up)

## Timing-Based Scheduling

The runner maintains a cache of test execution times to optimize scheduling:

### How it works

1. **Per-test timing**: During execution, the runner tracks how long each test takes, broken down by backend (vk, cpu, llvm, etc.)

2. **Cache storage**: After each run, timing data is saved to `~/.cache/slang-test-runner/timing.json`

3. **LPT scheduling**: On subsequent runs, files are sorted by predicted duration (longest first). This ensures slow tests start early and run concurrently with faster tests, preventing the "long tail" problem where all workers finish except one stuck on slow tests.

4. **API-aware predictions**: When running with `-- -api vk`, only Vulkan backend timings are used for predictions. This prevents filtered runs from skewing estimates for full runs.

### State location

- Linux: `~/.local/state/slang-test-runner/timing.json`
- macOS: `~/Library/Application Support/slang-test-runner/timing.json`
- Windows: `%LOCALAPPDATA%\slang-test-runner\timing.json`

The state is automatically created and updated. Delete it to reset timing estimates.

## How it works

`slang-test` can be run over specific test cases in a single threaded mode like `slang-test tests/foo/bar.slang tests/baz/qux.slang`

There is some overhead in starting `slang-test` so we don't just want to spin up slang-test for every test, we want to run N batches of tests at once where N is the degree of parallelism we want, but not have batches so large that we might be waiting for the slow one at the end of the run.

There might be several tests per test file, some of which are synthesized and some are ignored.

There are also internal tests not represented by files, these are called `slang-unit-test-tool/modulePtr.internal` or similar.

Tests might be in .slang files, or .hlsl or .glsl files in the tests directory.

See `slang-test -h` for the full list of options.

At the end it will output a command which will run all the non-passing and non-ignored tests.

## Building

```bash
cargo build --release
```

The binary will be at `target/release/slang-test-runner`.
