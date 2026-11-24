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

## Final (same process, interleaved, 9 repeats; min; construction ms / per period us; machine loaded by other agents, medians are noisier)
| model | case | construction before | after | per period before | after | speed-up |
|---|---|---|---|---|---|---|
| nk | order 1 | 2.1 | 2.6 | 484 | 24 | 20x |
| nk | order 2 | 3.9 | 5.7 | 1073 | 38 | 28x |
| nk | cubature 9 | 5.5 | 6.1 | 1028 | 375 | 2.7x |
| nk | cubature default | 5.9 | 6.1 | 1384 | 721 | 1.9x |
| nkm | order 1 | 2.9 | 3.7 | 872 | 19 | 45x |
| nkm | order 2 | 8.1 | 11.1 | 1372 | 36 | 38x |
| nkm | cubature 9 | 9.8 | 11.7 | 1927 | 199 | 9.7x |
| nkm | cubature default | 9.7 | 11.3 | 1853 | 271 | 6.8x |
| ref NewsOBC.simulate | | 0.5 | | 28 (nk), 14 (nkm) | | |

(The "before" per-period figures vary 410-1070 us between runs for nk order 1/2 because of load; compare only within a row.)
Order 2 is now about 1.5-2x the cost of NewsOBC itself per period; order 1 is within 1.2-1.4x of it.
Construction rose by 1-3 ms (the gap maps), of which k_order is ~60%; it pays back after ~3 simulated periods.

Maximum difference of the simulated paths (with_bound, without_bound, wedges; 2 models x 4 cases) from the stored baseline:
**3.6e-15** (largest over runs; 1.8e-15 in the last), exactly equal in several (order 1 nk, nk cubature 9, all base paths of nk).
Cubature bookkeeping (points, relevant points, LCP count) is identical to the original on all four cubature runs.

## Changes (what each bought; nkm, order 2, per period)
1. **Precomputed affine map from state to gap path (structural; 1372 -> ~36 us, the whole win at order 2).**
   The profile showed `expected_path` (a Python loop of N+1 steps, one covariance recursion each) at 80% of the time.
   The expected path is linear in (y1+y2, vec(x1 x1'), o_pre): W_k = F^k W_0 F'^k + S_k with F = ghx[states], S_k
   state-independent; y1 and y2 enter through the same coefficients F^(k-1) for k >= 1; the o part is P^k o. The gap q is
   linear in the path, so q = q0 + Gz z (N x (n + ns^2 + n)) is one matvec per period, plus aC.prev + aJ.u.
   Built once per parameter set by propagating only the three gap vectors (aB, aA, aC) backwards through the recursion (adjoint form):
   per k one ns x ns^2 update Z <- F Z + 0.5 C[ist], C_j <- F' C_j F (never forming kron(F,F)), so nothing of size (N+1) n ns^2 is stored.
   Order 1 has only the y and o blocks. `expected_path`/`gap_path` are kept unchanged as the executable spec and are tested against the map.
2. kron -> np.multiply.outer(...).ravel() in the base step (same products; ~5% at order 2 before change 1).
3. **Cubature step (structural for the rule, 1850 -> 270 us for nkm default level).**
   - Exact relevance skipping. The original forms q + R xi for all 2^(level+1)-1 points, tests every column. Now: (a) Cauchy-Schwarz
     p_h >= q_h - |R_h||xi_j|, so if q >= 0 every point with |xi_j| <= min_h q_h/|R_h| (1-1e-9) is non-relevant (points pre-sorted by norm, prefix
     skipped); if even the largest point is, the period returns zeros without touching the points. (b) min_h p <= ... : a column with min p >= -1e-8
     cannot be relevant whatever max|p| is, so |p| is only evaluated for the few candidates; max|p| = max(max p, -min p). Same set of relevant points as
     the original (verified: identical counts).
   - One augmented gemm [R q] @ [xi; 1] instead of gemm + broadcast add (the add was 2/3 of the gemm time).
   - With one region, the mean of q + R xi_j over relevant j is q + R mean(xi_j): an r-vector instead of an N x k array.
   - Sobol points cached per (dim, level) (6 ms per construction at level 10 saved when dim repeats, as in estimation); read-only array.
   Remaining cost in nk (71% of points relevant, near-binding model): the O(N x npts) min/max reductions (~0.3 ms per period).
4. Not done / not needed: skipping the LCP when non-binding (`_solve` already tests it: LCP is <5% of time everywhere);
   touching pydyn/obc.py (solve_active_set is under 2% in every profile; the amendment allows it only for a clear win).

## Structural (copy into the Julia port) vs Python-specific
Structural: (1) the state-to-gap affine map with adjoint construction, using the linearity of the covariance recursion in W_0 and the
state-independence of the shock part; y1 and y2 sharing coefficients; (3) the exact skipping rules with the Cauchy-Schwarz norm bound and the
min-before-abs relevance test; the mean-point identity q + R mean(xi); caching the Sobol set by (dim, level); only propagating the three gap vectors.
Python-specific: outer instead of kron (Julia has no such cost), the augmented gemm trick (allocation/broadcast overhead of NumPy; in Julia use in-place mul! and a fused loop),
sorting points by norm is useful in either language but the min/max reduction costs are lower in compiled code.

## State / next step
Done and pushed (wip/obc-ho-perf): all of the above; tests: tests/test_obc_ho.py (7), test_obc_dynareobc.py, test_obc.py: 45 pass.
Equality test `test_matches_stored_baseline` (1e-13) uses tests/data/obc_ho_baseline.npz made by bench/obc_ho_bench.py baseline with commit f173780 code (bench/obc_ho_before.py).
Possible next: exploit symmetry of vec(x x') (ns(ns+1)/2 columns, changes rounding only); cut the min/max reductions in the cubature step for near-binding models
(e.g. lower-envelope row pruning in q + R xi, only valid per period); `k_order` is now 60% of construction. Not measured: larger models (sw), horizon scaling, regions="all".
Run: `. env; python3 bench/obc_ho_bench.py time --reps 9`, `... compare tests/data/obc_ho_baseline.npz`.
