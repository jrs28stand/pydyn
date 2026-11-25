# Order 2 and cubature for the regime formulation (pydyn, branch `obc-regime-ho`)

Goal: `HigherOrderNews` (pydyn/obc_ho.py) runs on `NewsOBC` (the LCP) and refuses constraints that are not one complementary pair. The regime formulation
(`RegimeOBC`, pydyn/obc.py) handles approx-class and multi-equation / tied-wedge models (NKM, bf, `obc_sw_zlb`), so this adds order 2 and cubature for it.
This is new design: there is no external reference for the non-LCP classes; the validation is internal consistency (docs below, tests/test_obc_regime_ho.py).

## 1. Design (written before the code; deviations are listed at the end of the section)

**What `HigherOrderNews` does.** State per period: `y1` (first-order pruned), `y2` (second-order pruned incl. risk term), `o` (bound offset, first-order effect
of the applied wedges). Per period: base step (y1, y2 from the pruned decision rules and the shock), `o_pre = P o`; the *expected path* `E_t[y_{t+k}]`, k = 0..N,
of the model without the constraint, with future shocks at their distribution: affine in `z = (y1+y2, distinct entries of x1 x1', o_pre)`, so the gap path is
`q = q0 + Gz z` (+ `aC.prev_total + aJ.u` in period 0), precomputed per parameter set in adjoint form; solve `q + M y >= 0 ⟂ y >= 0` (same first-order `M`);
`o = o_pre + V0 y`. Cubature: `q + R xi_j`, `R R'` = covariance of the first-order gap-path effect of the shocks of the next `pou` periods (cosine taper,
eigenvalue pruning); points with `q + R xi` violating the bound ("relevant") are solved, the rest contribute 0; average of the wedges over all points.

**What `RegimeOBC` does.** One wedge per switched equation (`cols`, `nb` blocks, `T` periods each). Given the regime guess `z` (N x n_c): wedges from the linear system
`R[S,S] w_S = -a[S]` (`a` = alternative residuals on the no-wedge path, `R` = their first-order response to wedges); the path of the variables read by the
bind/relax conditions is `Ycv + Wcv w`; OccBin's test (`bind` false where `z=0`, `relax` false where `z=1`, evaluated in LEVELS on that path) accepts or flips the
violating periods (guess-and-verify from the reference regime); on cycling: the tied-wedge LCP (`_find_tie`, `g = q + M y + G z`, with an LP infeasibility proof) and then a
big-M MILP; horizon_too_short when the iteration wants a regime change beyond `T`. Everything the search needs from the state is the pair `(a, Ycv)` (the state enters
only through them) and `V0` (next-state response to wedges).

**Construction at order 2 (confirmed).** Higher order changes only the unconstrained expected path; the response to wedges stays first order (`R`, `Wcv`, `V0`, `FE`, `G` unchanged,
as in DynareOBC where M is first order). Hence per period, with `E` = order-2 expected path of the model without the constraint (from `y1, y2, o_pre`):
1. `a_b(t) = a0_b + aC_b E(t-1) + aB_b E(t) + aA_b E(t+1) (+ aC_b prev_total + aJ_b u at t = 0)`, t = 0..T-1, per switched equation b (linear functional of the expected path,
   the same functional as at order 1; the alternative equation is used linearised, as the LCP uses the linear gap);
2. `Ycv(t) = E(t)[cv]`, t = 0..N-1 (E contains the offset path `P^k o_pre`);
3. both are AFFINE in `z`: `[a; Ycv] = c0 + Gz z` precomputed per parameter set by the adjoint recursion of `_build_gap_maps` generalised to an arbitrary list of
   functionals (one block per switched equation over the horizon, one per condition variable). The slow reference (direct `expected_path`, then the functionals) is kept for the test.
4. The regime search is `RegimeOBC`'s own `_iterate / tie / _milp / spells` on `(a, Ycv)`: unchanged code, so selection rules and infeasibility proofs carry over.
   **What enters OccBin's test at order 2:** the LEVELS `ys + E_t[y_{t+k}]^{order 2, no constraint} + (first-order response to the wedges)` of the condition variables, k = 0..N-1; the conditions
   are evaluated on this mean path (not as expectations of the conditions: the test is the certainty-equivalent one on the expected path, consistent with how the LCP gap uses it).
   Not included: second-order terms of the *alternative* equations and second-order effects of the wedges (they would also be absent in DynareOBC's treatment of the bound).
5. `o = o_pre + V0 w`, with `V0` having one column block per switched equation (so one wedge per switched equation is carried forward, as the LCP carries `y`).
6. Output: levels `ys + y1 + y2 + o`, wedges `w` (nb*T), regime indicators `z`.

**Cubature.** Perturbed paths `p_j = (a, Ycv) + Rall xi_j`. *Relevant point* := the reference regime everywhere (`z = 0`, `w = 0`) fails OccBin's bind test somewhere
on the perturbed path (this is exactly "the unconstrained path would violate the bound", i.e. `not all(p >= 0)` of the LCP when bind/gap are mirror images, and the right
notion when acceptance is by the bind condition and not by the sign of a gap). Non-relevant points contribute zero wedges. Which quantities are perturbed: ALL quantities that enter
the acceptance (`a`, `Ycv`), but along the SAME directions as the LCP version: the root is built from the covariance of the gap-like path of each constraint (the
primary alternative-equation residual, sign-normalised, over N periods; for an LCP-class model identically `NewsOBC`'s gap), pruned as in `conditional_root`; the other quantities
move with the retained principal components through their covariance with the gap path (`R_other = Cov(other, gap) U_k S_k^-1`). Reasons: (i) for exact-class models this reproduces
the LCP cubature points exactly (the equality check); (ii) a joint root over `a`, `Ycv` would change the pruning for every model. Failure handling and the tolerance mode as in
`HigherOrderNews`: a point fails when `RegimeOBC`'s search raises; status mapping: `no_solution` with a proof (LP infeasible, or MILP infeasible on the first `T` rows) = class (a);
`horizon_too_short` (iteration needs a change beyond `T`, or a solution exists only on the first T rows) = class (b); MILP time limit / no convergence = class (c).
Region options (`regions=1` / k) average over region mean points as now.

**Open questions (left open, not decided silently).**
1. Cubature root: gap-path directions (above) vs a joint root over all acceptance quantities. 2. The expected-path (certainty-equivalent) test on levels vs a test on the
expectation of the conditions. 3. Second-order terms of the alternative equations (nonlinear alt equation, e.g. `R = Rlb` is linear, but `lb = 0` type are too; a nonlinear alt equation would
need the second-order response). 4. Selection other than `occbin` for several constraints is refused as at order 1.

(Departures from the outline are recorded under "Results" below.)

## 2. Construction as implemented (pydyn/obc_regime_ho.py, `HigherOrderRegime`)

Entry point: `HigherOrderNews(model, ..., obc_formulation='auto'|'lcp'|'regime')` returns a `HigherOrderRegime` when the regime formulation applies (`auto`: any approx-class or
non-complementarity constraint; `regime`: always; `lcp`: ValueError unless every constraint is a complementary pair; `hybrid` is not offered at higher order). Exact-class models: `auto` is the
unchanged LCP class (same class and code path; two builds of the same model differ by ~1e-20 in `ghxx` (`k_order` nondeterminism that predates this work), so equality is tested to 1e-15: `test_exact_class_auto_is_bit_identical_to_lcp`; the stored baselines `tests/data/obc_ho_*baseline.npz` and the DynareOBC references pass).
`HigherOrderRegime` reuses `RegimeOBC` unchanged for everything that follows the state-dependent inputs. To make that possible `RegimeOBC._step` was split (`_search`,
`lcp_from`, `_solve_lcp_core`; order-1 behaviour bit-identical: the 267 existing OBC/OccBin tests pass).

*Per period*: the pruned base step (as `HigherOrderNews`); `inputs_from_state` -> `(a, Ycv, A)` by one matrix-vector product with the precomputed `Gall`
(rows: alternative residual of each switched equation over N periods, the condition variables over N periods, the gap-like functional of each constraint); `RegimeOBC._search` (guess-and-verify,
tied-wedge LCP with its LP proof, MILP, `spells` for `obc_selection`); `o = o_pre + V0 w`. The affine maps are the adjoint recursion of `HigherOrderNews._build_gap_maps` generalised to any list of
functionals (`_affine_maps`); the slow reference (`inputs_direct`) goes through `expected_path`.

*Cubature* (`news_wedges`): root `R` of the gap-like path covariance, built from the same rows in the same order as `HigherOrderNews.conditional_root` (for an LCP-class model `R` is
bit-identical to the LCP's: `np.abs(a.R - b.R).max() == 0.0` on nk); the other acceptance quantities move along the retained directions via `Ro = Cov(other, gap) pinv(R')`, recomputed when
`R` is replaced (so DynareOBC's own root can be used); relevance = reference regime fails OccBin's bind test on the perturbed path (vectorised over points: one evaluation of the compiled condition on
P*N rows); `_guess_verify_many` runs OccBin's iteration on all relevant points at once (points sharing a regime guess share one linear solve with many right-hand sides; the first iteration is the
relevance test itself); points it cannot settle (cycling, regime change beyond T, singular block) go through the complete search one at a time; failure classes: horizon_too_short -> (b),
not_converged (MILP time limit, or the MILP's regime sequence fails the levels test) -> (c), no_solution (LP-infeasibility proof or MILP infeasible) -> (a); `cubature_failure_tolerance`,
`CubatureFailure`, `regions='all'|1|k` (the LCP's region split on the gap-like paths, one full search per region mean point).

Departures from the outline: (1) the cubature perturbation directions are the gap-like path's (open question 1 of section 1 resolved this way, because it is what makes the exact-class equality
possible); (2) `m_diagnostic` has no counterpart (M's S-matrix theory applies to the LCP only), `m_diag` is None; (3) the point-wise tied-LCP shortcut of `RegimeOBC` is used only through `_search`.

## 3. Checks (all in tests/test_obc_regime_ho.py, 50 tests; numbers from `soak/hoapprox/ho_study.py`, results in `soak/hoapprox/results/`)

| check | models | result |
|---|---|---|
| (a) order 1 = `RegimeOBC.simulate` | nkm, zlb, zlb2, nk, bf, sw (150 periods) | max abs diff 3.9e-14, 2.7e-15, 2.9e-15, 2.2e-16, 7.5e-16, 9.5e-13 (sw: values of size 1e2-1e3) |
| (b) order 2 regime = order 2 LCP | zlb, zlb2 (2 constraints), nk, borrcon, nkm (approx, no knife edge here) | max diff 0, 0, 1.1e-16, 2.8e-16, 1.8e-15; wedges equal; and the DynareOBC reference `ref_o2.mat` (nk, 83 binding periods of 300): 3.3e-16 |
| (c) never binding | nkm, zlb2, bf, sw | `with_bound` = `without_bound` exactly = the pruned simulation (1e-11) |
| (d) affine maps = direct recursion | nkm, zlb2, bf, sw, orders 1 and 2 | 1e-12 relative |
| (e) approx class, order 2, regime vs LCP (300 periods, shock scale 1 and 1.6 sd) | nkm; zlbm d=0, 1e-3, 1e-2, 5e-2; zlb2m; nkocc (3 sizes) | nkm 0 periods differ (>1e-8); nkocc 0; zlbm d=0/1e-3: 0, d=1e-2: 2 periods (max 4e-6), d=5e-2: 1 period (1.5e-4); zlb2m d=1e-3, 1e-2: 14-40 periods at 2e-7 / 2e-5 (the mismatch term the LCP ignores); regime fails with no_solution / not_converged where OccBin itself would (zlbm 5e-2 at 1.6 sd: period 43; zlb2m 1e-2 at 1.6 sd: period 290; zlb2m 5e-2 at 1.0 sd: period 10). Order-2 minus order-1 regime path: 0.045 (nkm, 1 sd) to 0.34 (borrcon). Second-order terms set to zero reproduce order 1 to 1e-11 (test) |
| (f) levels test | nkm, zlb2, bf, sw, borrcon, every period | accepted `z` passes OccBin's compiled bind/relax functions on the independently recomputed level path (order-2 expected path + `Yw w`) |
| linear models | bf, sw | order 2 = order 1 exactly (their solutions have no second-order terms), so these two do not test the order-2 maps; the maps are tested on nkm, zlb2 (nonlinear) |

Cubature (step 3):
| check | result |
|---|---|
| exact class equals the LCP cubature: nk, zlb all points (level 7), one region (level 9), 4 regions; zlb2 1 and 4 regions | `R` equal to 1e-13, same relevant counts, `with_bound` to 1e-12 (observed 2e-15) |
| nk against DynareOBC (`ref_o2cub.mat`, its root and points, one region) | 3.3e-16; `R R'` vs DynareOBC's 4.8e-19; 4 regions: 294 of 300 periods < 1e-12 (as for the LCP; max 6e-3 in the 6 tie periods) |
| nk, level 11, all 4,095 points, 100 periods | max diff vs LCP 1.9e-10 in 2 periods: one of the 4,095 points has wedges differing by 6.3e-7 (size 1.2): the LCP's 1e-8 complementarity tolerance against OccBin's strict test at a knife edge |
| zlb2 failures and tolerance mode | same period (1), same class (b), same dropped points as the LCP; paths equal to 1e-12 |
| variance -> 0 | cubature = no cubature to 1e-9 (nkm, bf, zlb) |
| NKM, cubature effect (level 9, 100 periods, max over periods of |cub - no cub|) | regime 0.0783, LCP 0.0783; regime vs LCP with cubature: 2.4e-5 |
| other approx models, level 9 | nkocc_5e-3: effect 0.068 / 0.068, regime-LCP 5e-5; zlbm_1e-2: 9.7e-4 / 9.8e-4, 6e-6; zlb2m_1e-3: 3.6e-3 / 3.6e-3, 3.5e-6 |

Failures of perturbed points (level 9, tolerance 0.9 to count them; defaults raise at the first):
| model | relevant points | failed | classes | default (tolerance 0) |
|---|---|---|---|---|
| nkm | 18,445 | 8 (0.04%) | all (c): the MILP's regime sequence fails OccBin's levels test, or the 1 s limit | raises at period 4 of 60 (level 11) |
| nkocc_5e-3, zlbm_1e-2, zlb2m_1e-3 | 47,810 / 6,115 / 23,619 | 16 / 3 / 3 (0.03-0.05%, 0.01%) | (c) | raises within a few periods |
| nk (exact) | 66,749 | 0 | | runs |
| zlb2 (exact; the LCP drops the same) | 84,086 | 276 (0.33%) | (b) | raises at period 1 (LCP too) |
| bf | 37,857 | 145 (0.4%) | (c) | raises at period 1 |
| sw (obc_sw_zlb) | 39,109 | 22,525 (58%) | (a) 20,094, (b) 2,159, (c) 272 | raises at period 1 with class (a): no equilibrium at reachable perturbed states |
An LCP-class model never fails at these points on NKM (the LCP's path need not pass OccBin's test); for a regime-formulation user the failures on approx-class models are points where OccBin's own iteration does not settle (as in
the estimation study): 0.01-0.05% of relevant points, enough to make the default (raise) unusable for NKM-type models at level 11 -- see the open questions. The MILP rescued none of them on nkm and about 1% on bf (batch failures 237,
dropped 235).

## 4. Cost (seconds per period, one core, `soak/hoapprox/results/COST*.jsonl`; 40 periods, `cubature_failure_tolerance=0.9`)
| model | o1 regime / LCP | o2 regime / LCP | cub 9 regime / LCP | cub 11 regime / LCP | cub 11 regime, MILP limit 0.02 s |
|---|---|---|---|---|---|
| nkm (n_c=1, approx) | 4e-5 / 2e-5 | 5e-5 / 3e-5 | 6.1e-3 / 1.1e-3 | 65e-3 / 2.8e-3 | 16.6e-3 |
| zlb2m_1e-3 (2 constraints) | 5e-5 / 2e-5 | 6e-5 / 3e-5 | 59e-3 / 3.6e-3 | 131e-3 / 12.5e-3 | 33e-3 |
| nk (exact) | 5e-5 / 2e-5 | 5e-5 / 3e-5 | 12e-3 / 5.0e-3 | 36e-3 / 15e-3 | |
| bf (multi_equation) | | 1e-4 (O2 table) / refused | 1.6 (MILP 1 s) / refused | | 0.095 (cub 9), 0.37 (cub 11) |
| sw | 5e-5 / refused | 5e-5 / refused | 1.1 (58% points fail) | 4.6 | |
Without cubature the regime formulation costs 1.5-2.5x the LCP per period (order 1 and 2 alike; both are 20-70 microseconds). With cubature the cost of the failing points dominates: every point the
batched iteration cannot settle goes through the complete search, and the integer programme runs up to `cubature_milp_time_limit` (1 s) on it. On nkm 7 of 18,445 points cost 4/5 of the time at level 11.
Optimisations taken (all bit-identical in results): the first iteration of the batched guess-and-verify is the relevance test (reused), the condition evaluation reuses one steady-state-filled buffer
(`np.tile` was a third of the time), regime guesses are grouped so that each distinct sequence is one solve with many right-hand sides. Before/after on nkm level 11: 22.7 ms -> 14.1 ms per period
(MILP limit 0.01 s). Not taken: batching the LP/MILP of failing points; evaluating the conditions through their affine form (they may be nonlinear in levels, the levels test is the point).
Structural for the Julia port: (i) the affine maps `Gall` (one adjoint recursion for all functionals), (ii) relevance = bind test at z=0, vectorised over points, and reused as iteration 0, (iii) grouping of points by guess,
(iv) a small MILP limit for cubature points, (v) the shared condition buffer.

## 5. Open questions
1. Default for `cubature_failure_tolerance` on approx-class models: the strict default (raise on one failed point) makes regime cubature on NKM unusable at level 11 (raises within a few periods); the LCP never fails there. Options for the user:
tolerance of about 0.5% for regime cubature, or leave it explicit (nothing changed here).
2. The cubature perturbation of `Ycv` and `a` is confined to the retained gap-path directions (pruned gap covariance). A joint root over all acceptance quantities would add directions the gap path does not span.
3. The levels test is on the expected path (certainty equivalent); the alternative equations are used linearised. A nonlinear alternative equation would need its second-order response.
4. bf and sw are linear in levels (log-linear) as written, so order 2 changes nothing for them; the order-2 maps are validated on the nonlinear models only.
5. `obc_selection != 'occbin'` is available through `RegimeOBC.spells` for one constraint (as at order 1) but untested at order 2.

## 6. Not established
No external reference for the non-LCP classes (nkm, bf, sw regime results at order 2 and with cubature are validated only by internal consistency: order 1, exact-class equality, levels test, maps vs recursion, limits).
Whether the 0.01-0.05% failing points on approx models have an OccBin solution at all (the MILP time limit hides it for some). sw at order 2 only tested without binding stress beyond 0.25 sd (it has states with no solution at larger shocks, and cubature raises).

## 7. Checklist for the Julia port
Port `HigherOrderRegime` after the order-1 regime solver: generalised `_affine_maps` (functional list in, `Gall`/`c0` out); the split of the regime step into `search(Ycv, a, tie_fn, base_fn)`; `lcp_from(a, base)` and `solve_lcp_core`; root on the gap-like rows in the LCP's row order (permutation `news_order`) plus `Ro = Cov(other, gap) pinv(R')`;
relevance = bind test at z=0; batched guess-and-verify grouped by regime sequence; failure classes via the statuses; the equality tests against the LCP cubature and DynareOBC's `ref_o2*.mat`; tolerance semantics unchanged.

## State / next step
- Steps 1 to 4 done and tested (50 tests; 298 + 19 OBC/OccBin tests). Report (step 5) pending the final commit.
