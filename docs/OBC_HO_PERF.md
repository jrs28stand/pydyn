# Performance of the order-2 / cubature OBC prototype (pydyn.obc_ho)

Harness: `bench/obc_ho_bench.py` (`time`, `baseline`, `compare`, `profile`); frozen original in
`bench/obc_ho_before.py` (commit f173780) so before/after run interleaved in one process (protocol: docs/OBC_BENCH_PROTOCOL.md;
warm-up, 5 repeats, min and median). Models: nk (reference/dynareobc_octave/nk_occ.mod, 300 shocks, horizon 32) and
nkm (reference/occest_nkm.mod, 300 N(0,1) draws, horizon 40). Single-threaded BLAS, two shared cores: noisy.
Correctness: tests/data/obc_ho_baseline.npz holds paths from the unmodified code; `tests/test_obc_ho.py::test_matches_stored_baseline`
asserts 1e-13.

## Baseline (before any change; per-period in microseconds, construction in ms, min of 5)
| model | case | construction | per period |
|---|---|---|---|
| nk | order 1 | 2.4 | 410 |
| nk | order 2 | 4.3 | 704 |
| nk | cubature level 9 | 7.0 | 1038 |
| nk | cubature default (10) | 7.2 | 1587 |
| nk | NewsOBC.simulate (ref) | 0.5 | 38 |
| nkm | order 1 | 3.3 | 574 |
| nkm | order 2 | 8.5 | 993 |
| nkm | cubature level 9 | 9.8 | 1264 |
| nkm | cubature default | 12.8 | 1718 |
| nkm | NewsOBC.simulate (ref) | 0.55 | 14 |

Profile (nkm, order 2): `expected_path` 80% of the time (a Python loop of N+1 steps with N x ns^2 kron-free but
many small matmuls, per period); kron 10%. With cubature (default): expected_path 50%, `news_shocks` 38% (forming
q + R xi for 2047 points and the relevance test).

## Changes
1. Precomputed affine map from state to gap path (structural). See below. 

## State / next step
