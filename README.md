# Octave reference results for Dynare 7.2's test suite

This branch holds no code. It records the results of running Dynare 7.2's own test suite under Octave once, so that pydyn and JDyn can compare against them without running Octave again.

**Contents**
- **The results themselves.** These are in `accept_ref.tar.gz`, attached to the GitHub release `accept-ref-octave-2026-11-24` (825 MB). It is a release asset rather than a file in git, so ordinary clones of the repository stay small.
  - sha256: `d9e9842f0670c22133f267060072c03c819b2dbef113d26d29d55c6119bf4868`
  - It unpacks to `accept_ref/`.
- **`index.csv`.** One row per test file, with these columns:
  - `test`: the path relative to Dynare's `tests/`, or `examples/...`;
  - `status`;
  - `elapsed_s`: wall-clock seconds;
  - `mat_bytes` and `mat_sha256` for the saved results;
  - Dynare's error message, where there was one.
- **`tools/`.** The runner as used: `accept_manifest.py` builds the test list, `accept_octave.py` runs it, and `accept_save.m` saves the results. These are snapshots; the maintained copies are in `tools/` on pydyn's main line.

**What each test produced** (as `<test path>.mat`, `.json` and `.log`, plus any files the test itself wrote)
- **`.mat`:** Dynare's `oo_`, the `M_` fields needed for comparison, and the `options_` fields used. The `.mat` files are saved with Octave's `-mat7-binary`. Objects that cannot be saved (dseries, dates, function handles) are removed recursively.
- **`.json`:** the run's status, message and elapsed time, whether a `.mat` exists, whether the test is expected to fail, and the arguments passed to `dynare`.
- **`.log`:** Octave's console output.

**Statuses** (846 entries from 708 manifest entries, since multi-file tests produce one entry per file)

| Status | Entries |
|---|---|
| ok | 732 |
| dynare-error | 36 |
| xfail (expected to fail) | 34 |
| timeout (over 900 s) | 30 |
| m-runner (companion `.m` test, not run) | 12 |
| no-result | 2 |

## Provenance

- **Dynare source:** the Debian upstream tarball `dynare_7.2.orig.tar.xz` (Dynare 7.2, announced 2026-09-11).
  - sha256 `88204354739547117b315e8fa3f8780a7570c576efd12f6a292f2a2468bb7a3c`
  - It is a release tarball, not a git checkout, so there is no commit hash.
  - Two local changes:
    - the datatypes check in `matlab/dynare_config.m` is turned into a warning, because Octave 8.4 lacks some of what it checks;
    - figure functions are stubbed during runs, because there is no graphics toolkit.
- **Preprocessor:** built from that tarball with Meson 1.3.2 (`buildtype=release`, `build_for=octave`) and g++ 13.3.0 on Ubuntu 24.04.5. Its path was `build-octave/preprocessor/src/dynare-preprocessor`.
- **Octave:** GNU Octave 8.4.0 (Ubuntu package), run headless as `GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet`, with `addpath <src>/matlab`.
- **How tests were run:** each test was copied with its whole folder into a fresh scratch directory and run with `nograph`, mimicking `tests/run_test_octave.m`. Multi-file entries ran in order, so later files see earlier files' outputs. Two tests ran in parallel at a time.

## Elapsed times: use with care for benchmarking

`elapsed_s` is wall-clock time for the whole `dynare` call: Octave start-up, preprocessing, the run, and saving.
- The machine had 2 cores and two tests ran at once.
- Other jobs, including pydyn's test suite and other agents' Octave runs, were often running at the same time.
- The times therefore show which tests are heavy (median 5.2 s; 10.4 h in total over the ok runs), but they are not benchmark-grade. For the benchmark table, re-time the chosen tests under the agreed protocol: one machine, one thread, warm and cold runs.

Generated on 2026-11-24 from pydyn's acceptance tooling.
