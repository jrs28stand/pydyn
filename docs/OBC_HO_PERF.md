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

## Part 2: regions="all" and regions=k (after merging wip/obc 4605e4d)
Cases: `bench/obc_ho_bench.py rtime` (L9 = qmc_level 9, 1,023 points, 100 periods; L12 = 8,191 points, 30 periods; order 2, cubature).
Per period, microseconds, min of 3, before = merged code (bench/obc_ho_before.py), same process, interleaved. Baseline paths:
tests/data/obc_ho_regions_baseline.npz (`baseline --regions`, merged code before any change).

| model | level | regions | before | after | speed-up |
|---|---|---|---|---|---|
| nk | 9 | all | 37,790 | 4,270 | 8.9x |
| nk | 9 | 2 | 8,242 | 2,493 | 3.3x |
| nk | 9 | 4 | 9,407 | 3,126 | 3.0x |
| nk | 9 | 16 | 16,312 | 6,750 | 2.4x |
| nk | 12 | all | 302,860 | 30,549 | 9.9x |
| nk | 12 | 2 | 84,915 | 12,118 | 7.0x |
| nk | 12 | 4 | 91,988 | 15,279 | 6.0x |
| nk | 12 | 16 | 96,189 | 18,193 | 5.3x |
| nkm | 9 | all | 8,433 | 1,958 | 4.3x |
| nkm | 9 | 2 / 4 / 16 | 2,456 / 3,374 / 6,687 | 2,354 / 2,636 / 3,610 | 1.0x / 1.3x / 1.9x |
| nkm | 12 | all | 15,538 | 3,111 | 5.0x |
| nkm | 12 | 2 / 4 / 16 | 5,032 / 7,735 / 9,739 | 3,530 / 4,429 / 4,741 | 1.4x / 1.7x / 2.1x |
Reference, regions=1 (same code, current): nk 392 us (level 9), nkm 204 us; order 2 without cubature 44 / 41 us.
nk has ~70% of the 1,023 points relevant (a model where the bound is nearly always in play), nkm ~16% (L9) and ~4% (L12).
Per solved point the batch solver costs about 6 us on nk (scalar: ~50 us).

Maximum difference of the simulated paths (with_bound, wedges) from the stored baseline, all 16 cases: **3.6e-15** (nk and nkm,
levels 9 and 12, regions all/2/4/16). test_regions_match_stored_baseline asserts 1e-12 (level 9 cases).

### What changed (structural)
1. **`solve_active_set_batch` (pydyn/obc.py, new; the scalar solver is untouched).** The scalar guess-and-verify iteration run in
   lockstep over all columns: start set S0 = (q < -tol); solve on S; drop negative wedges, add violated periods; stop on a repeated
   set or after maxit; the pivot test of `_solve_checked` (pivot <= 1e-10 * max|M|) and the acceptance test of `verify_lcp` per column.
   *Equivalence by construction:* each column follows exactly the scalar sequence of sets, the same rule decides every move,
   and the same tests accept or reject; the only changes are (a) several columns with the same guessed set share one factorisation, and
   (b) the linear solve is applied as a stored operator -M_AA^{-1} when cond_1(M_AA) <= 1e4 (error <= ~1e-12 relative to the scalar
   LU solve; more ill-conditioned sets use the LU solve, as the scalar code). Columns for which the iteration fails (batch_fail) go down the scalar
   route (`_solve`: scalar active set again, then the MILP), so the MILP fallback and its selection are unchanged. When <= 8 columns are left
   the scalar routine continues from their current set (the cycle memory restarts; a cycle is a failure either way).
   With several LCP solutions the batch returns the one the scalar iteration returns, because it is the same iteration.
   Fallback rate observed: 0 of 73,225 (nk L9), 168,072 (nk L12), 16,759 (nkm L9), 9,029 (nkm L12) solved points; 0 singular blocks; 0 sets
   above the conditioning threshold.
2. **Cache of solution operators by guessed set**, across periods (M is fixed per parameter set): 2,827 sets after 100 periods on nk L9 against
   ~73,000 solves. Operators are stored compact (|A| x |A|, padded to a power of two) so applying them to all columns of a size class is
   one batched matmul over small matrices. Gather/scatter through an extra zero row absorbs the padding.
3. Sets are keyed by one uint64 (packed bits) for n_s <= 64: np.unique and dict lookups on integers.
4. Verification reuses the n_s rows of q + M y already formed in the iteration; only the check-ahead rows are computed at the end.
5. `regions=k`: principal components from the eigendecomposition of the 2N x 2N Gram matrix instead of the SVD of the tall
   (m x 2N) matrix (SVD was 70% of the time); region sums of squares kept and updated only for the two regions a split changes;
   region mean points formed as q + R mean(xi) rather than from the N x m matrix. Component signs are arbitrary in both versions and the split is
   invariant to them (the Otsu criterion on a centred projection is symmetric in j <-> m - j), so the partition is the same unless a
   point lies within rounding of a threshold; all 16 cases above reproduce the baseline to 3.6e-15.

### Profile after (nk, L9, all): per period 4.3 ms = first two heavy iterations of the lockstep loop (the q<0 guess is wrong for ~90% of
columns, so two passes of ~700 columns are needed): grouping/keys 0.7, cache lookups 0.7, batched operator application 1.3, verification 0.6,
rest 0.9 (numpy call overhead on small arrays dominates). nkm: 2 ms (1.2 ms of it outside the solver).

### Which are structural (Julia) and which are Python-specific
Structural: lockstep scalar iteration over the points of a period with identical decision rules (equivalence argument above); grouping by guessed
set and caching M_AA factorisations (or inverses) across periods keyed by the set; applying the cached operator, with a conditioning guard, instead of
refactorising; reusing the n_s rows of q + M y for verification; the Gram-matrix PCA and incremental region SSE; region mean point as q + R mean(xi).
In Julia the scalar solve is already 8-14 us, so the win of batching is smaller: the useful part is (cache of factorisations/inverses by set across
periods and points) and, if points are many, a multi-RHS solve per set. Python-specific: packing sets into uint64 for np.unique, padding to power-of-two
operator classes with a zero row to turn many small solves into a few batched matmuls, the scalar tail (<= 8 columns), integer cache keys.

## State / next step
Done and pushed (wip/obc-ho-perf): everything above; tests: tests/test_obc_ho.py (9), test_obc_dynareobc.py, test_obc.py, test_obc_perf.py pass.
Possible next: (a) the first lockstep pass is mostly wasted (90% of columns change set); a better start cannot be used without changing the
accepted solution, but the whole q-dependence is low rank (q + R xi, rank r+1), so the first guess could be shared among neighbouring points
only if equivalence is re-argued; (b) ~1 ms per period of numpy overhead in regions=k (eigh 128 x 128 for the Gram matrix) could drop with a
low-rank update; (c) not measured: regions="all" on sw-size models, horizons other than 32 / 40, level 14.
Run: `python3 bench/obc_ho_bench.py rtime --reps 3`, `... compare tests/data/obc_ho_regions_baseline.npz --regions`.
