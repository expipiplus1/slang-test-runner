# Slang test runner

A wrapper for slang-test which implements several enhancements:

- Parallelized test running
- More user friendly output, during test running and as a final report
- Robust handling of segfaults or unexpected crashes
- Automatic retries of transiently failing tests

## Examples of better test output

- Doesn't flood the terminal with tests which succeeded
- Doesn't flood the terminal with tests which failed and then later succeeded (unless in verbose mode)

## How it works

`slang-test` can be run over specific test cases in a single threaded mode like `slang-test tests/foo/bar.slang tests/baz/qux.slang`

There is some overhead in starting `slang-test` so we don't just want to spin up slang-test for every test, we want to run N batches of tests at once where N is the degree of parallelism we want, but not have batches so large that we might be waiting for the slow one at the end of the run. there is also the possibility that a batch might segfault half way through. This program will remember which tests passed/failed/ignored before that and resume from an appropriate position, and skip the segfaulting tests to keep making progress.

There might be several tests per test file, some of which are synthesized and some are ignored.

There are also internal tests not represented by files, these are called `slang-unit-test-tool/modulePtr.internal` or similar.

Tests might be in .slang files, or .hlsl or .glsl files in the tests directory.

See `slang-test -h` for the full list of options

At the end it will output a command which will tests all the non-passing and non-ignored tests.
