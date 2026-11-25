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

## State / next step
- Step 1 (this note): done. Step 2: next.
