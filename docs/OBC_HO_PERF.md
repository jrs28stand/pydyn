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

## Scaling: large and multi-constraint models
Harness: `bench/obc_ho_scale.py` (`probe`: which models are accepted; `run MODEL [--horizon H] [--levels 9,12]`). Same process, minimum of 3 repeats
(construction, order 1/2 per period) or mean over up to 100 base-path gap vectors (cubature: time of `news_shocks(q)` per period, on q from the order-2
base path; a failing call is timed separately). Shock scale per model: largest of {0.1 .. 32} x sd at which order 2 solves 300 periods, stopping at >= 30% binding.
The machine was shared with two other workers (load ~2): single numbers vary +-30%; only orders of magnitude are claimed.

### Which models are accepted (exactly)
| model | n | verdict |
|---|---|---|
| nk (dynareobc_octave/nk_occ.mod), nkm (Dynare tests/occbin/filter), occest_nkm, occest_nkms, zlb, borrcon, borrcon_sim | 5-24 | accepted, one constraint |
| zlb2 (reference/obc_zlb2.mod) | 7 | accepted, **two constraints** (was a crash, see below) |
| **sw** (reference/obc_sw_zlb.mod; also Dynare ppf/bf_ppf.mod) | 32 | **refused** at construction: `OBCClassError: constraint 'zlb' is multi_equation: 2 equations change between the regimes` (the observed-rate equation *and* the auxiliary `rlag` equation switch: a tied wedge; the news-shock method puts one wedge on one equation per constraint) |
| Dynare ppf/swpkfppf.mod | 32 | cannot be loaded: MATLAB `set_param_value('ctrend', mean(dy(74:end)))` (ModelError) |
| occ_rbc1, occest_rbc | 5-8 | refused: `constraint 'INEG' is inconsistent: the bind condition is not equivalent to the alternative equation's residual changing sign` |
| Dynare model_irrcap_twoconstraints/dynrbc.mod | | cannot be loaded: `occbin_solver` after MATLAB code that changes `oo_` |
Largest real model accepted is therefore nkm (24 variables, 7 states). Two stand-ins for large models were built:
* **sw_cp**: obc_sw_zlb.mod with `rlag = rnot` (the policy rule smooths on the shadow rate; the second switching equation removed), so the bound
  switches one equation. 32 variables, 12 states. A different economy from SW (no lag on the bound rate), the same size and structure; the bind condition is the same.
* **padN**: occest_nkm.mod plus N blocks of an AR(1) driven by one of 5 extra shocks and 3 lags of it (4 variables, 3 states per block), independent of the bound:
  pad5 n=44, pad19 n=100, pad44 n=200 (ns = 22, 64, 139). Padding is neutral for the LCP (same M, same points), so it isolates the cost of model size.

### Several constraints: what was wrong and what was fixed
`HigherOrderNews` built its gap map from `constraints[0]` only and returned a wedge vector of length T; with `n_c = 2` order 1 and 2 raised
`ValueError: matmul ... size 20 is different from 40` (a crash, not a wrong answer) and cubature raised `NotImplementedError`. Now: gap functionals per constraint,
the gap maps stacked and reordered like `NewsOBC` (complementarity rows first), wedge vector `n_c T`, and the conditional root built per constraint.
Tests (tests/test_obc_ho.py): order 1 equals `NewsOBC.simulate` on zlb2 (5e-16 at small shocks, 1.8e-14 at 5x shocks where both constraints bind in >100 of 200 periods);
order-2 base equals the pruned simulation; `gap_from_state` equals `gap_path(expected_path)`; cubature with vanishing variance equals no cubature;
`regions="all"` (batch solver, two constraints) equals the scalar sum of `_solve` over the relevant points, and both raise on the same perturbed points.
`solve_active_set_batch` needed no change for `n_c > 1` (it only sees the stacked M); the batch-vs-scalar test above covers it.

### Construction and per-period cost against model size
(ms for construction, us per period; order 2 unless stated)
| model | n | ns | n_c | nz = len(z) | Gz MB | constr o1 | constr o2 | of which k_order | of which gap maps | constr cubature (1st / repeat) | R dim | binding | o1 us | o2 us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zlb | 5 | 1 | 1 | 11 | 0.01 | 1.9 | 4.6 | 1.5 | 1.4 | 16.6 / 4.8 | 12 | 30% | 28 | 40 |
| nk | 5 | 1 | 1 | 11 | 0.01 | 2.1 | 5.2 | 2.1 | 1.5 | 21.8 / 5.5 | 12 | 41% | 31 | 44 |
| zlb2 | 7 | 1 | 2 | 15 | 0.02 | 3.3 | 5.5 | 2.0 | 1.5 | 23.2 / 6.2 | 12 | 31% | 31 | 45 |
| borrcon | 8 | 2 | 1 | 19 | 0.01 | 2.2 | 5.2 | 2.1 | 1.6 | 19.1 / 7.3 | 15 | 68% | 30 | 62 |
| nkm | 24 | 7 | 1 | 76 | 0.04 | 4.1 | 12.5 | 8.7 | 1.6 | 26.9 / 13.9 | 5 | 43% | 42 | 83 |
| sw_cp | 32 | 12 | 1 | 142 | 0.07 | 3.6 | 10.3 | 5.7 | 1.9 | 24.8 / 10.2 | 7 | 8% | 24 | 41 |
| pad5 | 44 | 22 | 1 | 341 | 0.17 | 3.7 | 25.8 | 19.0 | 3.3 | 38.5 / 25.9 | 5 | 35% | 57 | 73 |
| pad19 | 100 | 64 | 1 | 2,280 | 1.2 | 8.6 | 223 | 175 | 35 | 222 / 218 | 5 | 35% | 38 | 251 |
| pad44 | 200 | 139 | 1 | 10,130 | 5.2 | 29 | 2,536 | 2,277 | 388 | 2,672 / 2,861 | 5 | 35% | 54 | 1,122 |
* Order 1 is flat in n (30-57 us). **Order 2 per period grows like n x ns^2** (73 -> 251 -> 1,122 us for n = 44, 100, 200): each period applies ghxx (n x ns^2)
  and the folded gap map Gz (N x (ns(ns+1)/2 + 2n)), both memory-bound (31 MB and 10 MB at n = 200).
* Construction is dominated by `k_order` (the C++ library, not this code): 19 -> 175 -> 2,277 ms, about n^2.9. The gap maps were the second term and were
  made much cheaper in this task (below).
* Not a cubature effect: the Sobol cache makes the second construction cheap; the first costs 10-20 ms more (R and 1,023 points) except at n >= 100 where k_order dominates anyway.
* Relevant points per period and everything else on the LCP side do not depend on n: pad5, pad19 and pad44 have identical point counts, sets and pass counts.

### Cubature cost, level 9 and 12 (us per period of `news_shocks`, base-path q; "raise" = the call raised for a point without solution)
| model | L9 regions=1 | L9 regions=4 | L9 all | L12 regions=1 | L12 regions=4 | L12 all |
|---|---|---|---|---|---|---|
| zlb | 330 | 3,078 | 4,539 | 2,633 | 17,983 | 37,784 |
| nk | 350 | 3,612 | 5,239 | 2,404 | 12,850 | 27,146 |
| **zlb2 (2 constraints)** | 590 | 8,032 | 13,453 | 3,999 | 37,223 | **194,288** |
| borrcon | 422 | 4,656 | **raises 100/100 (43,001)** | 2,872 | 22,207 | **raises 40/40 (344,029)** |
| nkm | 221 | 2,111 | 2,831 | 1,581 | 12,767 | 27,878 |
| sw_cp | 398 | **raises 100/100 (23,205)** | **raises 100/100 (32,659)** | 2,532 | **raises 40/40 (34,890)** | **raises 40/40 (61,288)** |
| pad5 / pad19 / pad44 | 244 / 178 / 218 | 2,268 / 1,625 / 1,719 | 2,423 / 2,223 / 2,580 | 1,030 / 1,386 / 1,286 | 9,652 / 8,791 / 8,818 | 31,047 / 25,865 / 49,586 |
Add the order-2 per-period cost (table above) for the full period.

LCP-side counters for regions="all" (per call; level 9 | level 12):
| model | relevant points | lockstep passes | distinct sets per call (groups) | sets in cache at the end | cache MB | scalar tail | active-set failures |
|---|---|---|---|---|---|---|---|
| nk | 747 \| 5,532 | 3.4 \| 4.3 | 519 \| 2,032 | 3,132 \| 8,338 | 4 \| 12 | 0.3% | 0 |
| nkm | 638 \| 5,400 | 2.5 \| 3.6 | 46 \| 168 | 538 \| 1,107 | 0.8 \| 1.7 | 0.4% | 0 |
| zlb2 (n_c=2) | 840 \| 7,201 | 3.9 \| 4.8 | 1,119 \| 6,338 | 14,258 \| 17,698 | 55 \| 64 | 0.4% | 0 |
| borrcon | 1,021 \| 8,178 | 20 \| 22 | 3,209 \| 17,723 | 5,811 \| 15,249 | 13 \| 29 | 0.3% | **13.4% \| 12.3%** |
| sw_cp | 669 \| 4,736 | 9.1 \| 12.0 | 705 \| 2,640 | 2,775 \| 4,131 | 6 \| 9 | 0.9% | **56.8% \| 58.9%** |
| pad (any) | 602 \| 3,938 | 2.5 \| 3.6 | 44 \| 167 | 530 \| 1,229 | 0.8 \| 1.7 | 0.3% | 0 |

### Failures under cubature (the SW-type caveat)
What the code does today. `_solve(q)`: all gaps non-negative -> zero; else the active-set iteration; if that returns None, `solve_news` (a MILP, horizons T/4, T/2, T in turn,
**no time limit**); if that finds nothing, `OBCError("no solution within the news horizon", "no_solution")`. `regions=1` and `regions=k` call it on the region mean points;
`regions="all"` runs the batch solver and sends every failed column to `_solve`, so a point without solution aborts the whole period with `OBCError` (nothing is dropped
silently, and `simulate` attaches the period number). Measured:
* **sw_cp** (base path solves; shock scale 0.25, 8% binding): regions=1 solves in 100/100 periods (its mean point is close to the base path), but regions=4 and "all" raise in
  every period. At level 9, 56.8% of the relevant perturbed points fail the active-set iteration, and in a census of 5 periods (3,131 points) **all 1,835 such points have no solution at all**
  (the MILP proves infeasibility; none is found) -- M is not a P-matrix there. A proof of infeasibility costs ~21 ms per point (census: 40.7 s for 1,835), but because the call raises
  at the first failing point a period costs about one batch pass (all columns, ~30 ms) plus one MILP (~1-25 ms): 23-65 ms.
* **borrcon**: 13% of perturbed points have no solution; regions=1 fine, "all" raises in every period (43 ms at level 9, 344 ms at level 12). The active-set iteration runs 20+ lockstep passes (cycling states).
* nk, zlb, zlb2, nkm and all pad models: 0 failures at levels 9 and 12 (zlb2: 57 of 6,220 relevant points failed in the random-draw check of the test, i.e. ~1%, enough to raise most periods
  when the points are drawn from perturbed *random* states; none on the realistic base path above).
Proposed rule (since implemented: see "Cubature defaults and failure handling"): keep `raise` as the default, as DynareOBC does unless `IgnoreBoundFailures`. Add `on_failure="raise"|"drop"`: `drop` leaves failing points
out of the average with the weight of the remaining points renormalised, counts them (`stats["dropped"]`) and raises anyway if more than a stated share (say 5%) of a period's relevant points fail;
and give the MILP in the cubature path a time limit (1 s) because it only needs to say "no solution". The regions=k mean points need the same rule.

### Other things looked for
* **Operator-cache blow-up.** Distinct sets grow without saturating when there are two constraints (zlb2: 14,258 sets after 100 periods at level 9, 17,698 after 40 at level 12, ~440 new sets
  per period; 55-64 MB) and in borrcon (15k); with one constraint they saturate (nkm 538, 1,107; pad the same). The cache is cleared at 20,000 sets; **a byte cap (256 MB) was added** (the padded operators of a 40-row problem
  are up to 32 KB each: 20,000 sets would be 650 MB) with a test that a tiny cap changes nothing. With two constraints almost every point has its own set (groups/call 6,338 of 7,201 columns),
  so the cache and the grouping buy little: regions="all" costs 194 ms at level 12, 7x nk, about 27 us per solved point against ~55 us scalar, i.e. only ~2x over the scalar loop.
* **`_build_gap_maps`** was the quadratic-in-states cost: the direct recursion carried an ns x ns^2 array (Z) and n x ns x ns (C), O(ns^4 + n ns^3) per path period. Rewritten (adjoint form, only 3 n_c matrices of ns x ns:
  D_(k+1) = H(B_(k-1)) + twist(D_k)), O(n_c ns^3): pad19 230 -> 35 ms, pad44 0.39 s now (whole construction 8.6 s -> 2.9 s, before/after in one process). Same results to rounding (all baselines 3.6e-15).
* **vec(x1 x1')** is now folded onto ns(ns+1)/2 distinct entries in the order-2 step and in Gz: order-2 per period pad44 2,021 -> 1,183 us, pad19 373 -> 275 us (same process, interleaved), results to 3e-14 on those models.
* **Horizon** (nkm; 100 base-path q; load ~2): N = 40, 64, 80, 120 gives regions=1 ~190-600 us (flat within noise), regions=4 2.0, 4.0, 4.3, 6.8 ms and "all" 4.0, 3.8, 5.5, 5.4 ms,
  order 2 per period 95, 72, 57, 54 us (flat); gap maps 1.1, 1.7, 2.8, 3.0 ms; passes per call 2.5 at every horizon. pad19 behaves the same. Regions=k grows about linearly in N (the Gram matrix is 2N x 2N); nothing super-linear up to N = 120.
* **Cliffs**: (1) a perturbed point without solution -> MILP proof (16-25 ms each) and an exception, on the models above whose M is not a P-matrix; (2) two constraints: set explosion (~2x over scalar instead of ~10x); (3) order-2 per period at n >= 100 (n x ns^2, memory-bound) and `k_order` at n >= 200 (outside this code);
  (4) level 12 with regions=k (12-37 ms). No cliff in the horizon, in the number of relevant points (linear), or in `news_shocks` itself.

### For the Julia port
Fold vec(x x') from the start (distinct entries only); build the gap maps with the adjoint recursion (never carry ns x ns^2 arrays); give the MILP a time limit and decide the failure rule up front (the failures are not
rare: 13% on borrcon and 57% on a 32-variable SW-type model at level 9); bound the factorisation cache by memory and expect it to be useless with more than one constraint (distinct sets ~ number of points); keep per-constraint
gap functionals stacked with the complementarity rows first (the `NewsOBC` order) so the wedge vector is n_c T long; cost of the order-2 step is n x ns^2 per period -- for n = 100+ avoid computing the full y2 every period
if only the gap and the states are needed.

## Cubature defaults and failure handling
**Superseded (25 Nov) for the failure rule and the time limit:** the strict default below (`cubature_failure_tolerance=0.0`: raise on any failed point; `cubature_milp_time_limit=1.0`) is replaced by the rule in
[OBC_CUBATURE_FAILURES.md](OBC_CUBATURE_FAILURES.md): class (a) (proven no solution) still raises by default, classes (b) and (c) are tolerated up to `cubature_undecided_tolerance` = 0.5% of a period's
relevant points (never fewer than 3 points) with a warning and counts, and `cubature_milp_time_limit` is 0.1 s. `cubature_undecided_tolerance=0` restores the strict behaviour described here. The rest of this section
(defaults `regions="all"`, `qmc_level=11`, the M diagnostic, the classification of failed points) is unchanged.

Specification for `HigherOrderNews` with `cubature=True` (decisions of the project owner; implemented in pydyn/obc_ho.py; tests in tests/test_obc_ho.py).

### Defaults
* `regions="all"`: the full quasi-Monte Carlo average over every relevant point. `regions=1` (DynareOBC's default: the solution at the single mean of the
  relevant perturbed paths) and `regions=k` remain options. **One region does not price tail risk**: it solves one LCP at the mean path, so the wedge it
  returns is the wedge of the average path, not the average of the wedges; the full average changes results a great deal (NKM, full-average mean shift:
  -0.107 sd at level 9, -0.162 at level 11, -0.163 at level 13).
* `qmc_level=11`: the first 2^12 - 1 = 4,095 Sobol points after the origin (symmetric set, mean exactly zero).
* ~~`cubature_milp_time_limit=1.0`, `cubature_failure_tolerance=0.0` (raise on any failed point)~~ (superseded, see above: now 0.1 s and the split rule), `m_diagnostic="s"`.
Explicit old settings (`regions=1`, `qmc_level=9` / 10) reproduce the stored baselines and the DynareOBC references (tests unchanged apart from passing them).

### Up-front M diagnostic
When cubature is on, at construction `m_diagnostics` (pydyn/obc.py, reused unchanged) is run on the complementarity block `Ms` of M and stored as `h.m_diag`
(dict: `is_s`, `is_p`, `largest_p`, ..., `seconds`, `level`). A `UserWarning` "M is not an S-matrix: some paths can have no solution; cubature points may fail" is issued if
`is_s` is False. Levels: `"s"` (default) runs the S-matrix LP and a single leading P-block only (`is_p` is then None = not decided); `"full"` also runs the exact P-matrix tests;
`False` skips it (`m_diag` is None; no information, no warning). Cached per parameter set in a module-level cache keyed by a digest of M. Cost (this machine, load ~1.3):
| model | H = 20 | 32 | 40 | 60 | cached | full (T = 32) |
|---|---|---|---|---|---|---|
| nkm | 4.1 ms | 3.7 | 5.9 | 6.5 | 0.01-0.03 ms | 449 ms |
| zlb2 (n_c = 2) | 4.7 | 6.5 | 11.1 | 15.3 | 0.02-0.09 ms | 266 ms |
| sw_cp | 2.7 | 2.6 | 2.9 | 4.6 | 0.01 ms | 9 ms (not P: found early) |
The S test is only informative one way: a non-S M means some q have no solution; an S-matrix M can still lack solutions for some q (borrcon is S and still has failing points: it is not a P-matrix).

### What a failed point is
A *unit* is a perturbed path (`regions="all"`) or a region's mean path (`regions=1`, `k`; a failed region counts all its points). The scalar active-set iteration (unchanged) is run first
(batched for "all"); the unit is **failed** if no solution is found by it, by the MILP, or the MILP is undecided. Classification of a unit on which the iteration failed
(`_classify`; LP first, tolerance 1e-6 max(1, |p|)):
1. LP_head: min s s.t. y >= 0, p + M y + s >= 0 on the first n_s rows (the complementarity rows). s* > tol => **(a) proved no solution at this horizon** (necessary condition violated; ~2 ms).
2. else LP_full on all N rows (check-ahead rows included). s* > tol => **(b)**: the wedges inside the news horizon cannot meet the check-ahead rows although they can meet the first n_s rows.
3. else (both LPs feasible) the MILP (`solve_news`, horizons T/4, T/2, T in turn) with the shared per-point time limit: a solution => the unit is solved; alpha = 0 at every horizon => the MILP on the first n_s rows only decides
   **(b)** (a solution exists there) or **(a)**; the time limit reached => **(c) undecided** (counts as a failure).
What (b) establishes and what it does not: it establishes that no y >= 0 satisfies the gap conditions on all N rows (LP infeasibility is a proof of no LCP solution) while the head rows alone are satisfiable by some y >= 0 (an LP, not the
LCP: complementarity on the head rows is not checked in the LP step, but is in the MILP step). It does **not** establish that a longer horizon has a solution: that needs M and the gap path for a larger T and N, which are not built for perturbed points
(testing it means constructing a second `HigherOrderNews`). Empirically it is the right advice on borrcon (all failing points are (b); with horizon 32/48/64/96 the dropped share at level 7 is 13.4% / 2.2% / 0.4% / 0%).
(a) is exact for what it claims (infeasibility of a relaxation, up to the LP tolerance); a point whose LP is feasible by less than the tolerance goes to the MILP rather than being called (a). Points where HiGHS returns no status are treated as not proved.
In the MILP step the time limit is shared by all MILP calls for the point (not per horizon). It is rarely reached: on borrcon and sw_cp every failing point is decided by the LP (the MILP is never called).

### Raise (default, tolerance 0)
The first failed unit stops the search and `CubatureFailure(OBCError)` is raised, with: `period` (set by `simulate`, 1-based), `n_failed` (**a lower bound**, `lower_bound=True`, when units were left unexamined; it is the
count so far), `n_relevant`, `n_points`, `counts={'a','b','c'}` (of the examined units, in points), `n_active_set_failed` (for "all": the number of points on which the batch active-set solver failed over the whole period, an upper bound on `n_failed`;
None for regions modes), `tolerance`, `advice` (list of strings; (b): longer `horizon`; (c): raise `cubature_milp_time_limit`; (a): the model has reachable states without equilibrium) and `status` ('no_solution' if any (a), else
'horizon_too_short' if any (b), else 'not_converged'). Cost of reaching the raise at level 11 (4,095 points): borrcon 359 ms (of which 1 batch pass), sw_cp 42 ms.

### Tolerance mode (`cubature_failure_tolerance = tau > 0`, opt-in)
All failed units are found (class counts exact); the search stops early only to raise, as soon as the failed points exceed tau x (relevant points). With F failed points out of P points of which R are relevant:
* if F > tau R: raise `CubatureFailure` (`n_failed` is then a lower bound if units were left);
* else the failed points are dropped and the period's result is **(sum over solved relevant units of size x y) / (P - F)**. F = 0 gives the plain average sum / P.
Why P - F (redistribute over all non-failed points, not only over solved relevant points): non-relevant points are legitimate outcomes with y = 0 (the bound is not reached on that path); the failed points are the only ones
about which nothing is known, so they are removed from the sample and the rest keep equal weights 1/(P - F). Spreading their mass over the solved relevant points alone would load it all on the nonzero wedges and bias the average upwards in exactly the periods where the bound matters.
Undecided points (c) count as failed. Recording: `stats["dropped_points"]`, `stats["dropped_periods"]`; `simulate` returns `dropped` and `relevant` per period; one `UserWarning` per simulation with the totals. Weights check in the test: the period result equals
the scalar sum divided by (P - F).

### Measured
Per period of `simulate`, order 2, cubature (2 repeats of 60 periods; machine load ~1.3; us):
| model | old default (level 10, 1 region) | **new default (level 11, all)** | level 11, 4 regions | relevant share |
|---|---|---|---|---|
| nk | 711 | 14,044 | 10,496 | 69% |
| nkm | 561 | 14,606 | 9,085 | 65% |
| zlb2 (two constraints) | 1,275 | 99,824 | 32,056 | 86% |
Failure handling at the default level 11, 8 periods:
| model | mode | result | cost per period |
|---|---|---|---|
| borrcon (13% of points fail, class b) | raise | raised in period 1 after 359 ms (lower bound 1; 515 active-set failures) | 359 ms |
| | tolerance 0.2 / 0.7 | 13.1% of relevant points dropped (4,275 of 32,749) | 2.5 s |
| sw_cp (58% fail, class a) | raise | raised in period 1 after 42 ms | 42 ms |
| | tolerance 0.2 | raised in period 1 after 1.14 s (507 failed points counted before exceeding 20% of 2,533 relevant) | 1.1 s |
| | tolerance 0.7 | 58.2% dropped (11,960 of 20,564) | 3.5 s |
Cost of tolerance mode is ~2.3-5 ms per failed point (the two LPs; scipy/HiGHS call overhead dominates a 64 x 33 LP): it must classify every failed point, so it is the expensive mode on models that fail often. No MILP was needed in any of these runs.

## State / next step
Done and pushed (wip/obc-ho-perf): Parts 1-2, Scaling, and "Cubature defaults and failure handling". Tests: 76 pass (tests/test_obc_ho.py 18, test_obc_dynareobc.py, test_obc.py, test_obc_perf.py).
Open: the LP classification calls scipy `linprog` (~2.3 ms per point); a direct HiGHS call or a batched/dual-warm-started LP would cut tolerance mode several-fold; whether (b) failures are cured by a longer horizon is only
evidenced on borrcon (a second `HigherOrderNews` with a longer horizon would test it per point); not measured: level 13, cubature with more than two constraints, real SW (refused: tied wedge).
Run: `python3 bench/obc_ho_bench.py compare tests/data/obc_ho_baseline.npz`, `... compare tests/data/obc_ho_regions_baseline.npz --regions`, `python3 bench/obc_ho_scale.py run MODEL`.

### Port to Julia: checklist
1. Defaults: full average, 4,095 points; keep one region / k regions as options (document "does not price tail risk").
2. Up-front: S-matrix LP on the complementarity block of M, cached per parameter set; warn; store the diagnostic; make it skippable.
3. Per point: active set (batched, unchanged semantics) -> LP_head -> LP_full -> MILP with a time limit shared per point -> classes a/b/c exactly as above (note what (b) does not prove).
4. Raise by default with the full error payload (period, failed lower bound, relevant, total, counts, active-set failures, advice); tolerance mode: count everything, stop only to raise, result sum / (P - F), per-period and total counts, one warning per run.
5. Keep the LP tolerance 1e-6 max(1, |p|) and the rule that a feasible-within-tolerance LP goes to the MILP, not to (a).
6. Make the failure test cheap: the LP dominates tolerance mode; use a native HiGHS/JuMP model rebuilt per point from M (fixed) and p (changing right-hand side only) so the basis can be warm-started.

