# OBC methods in pydyn: progress log

Contract: project doc `claude/obc-shared-contract.md`. Branch `obc` (from `initial-import` fd14d30).

## 2026-11-24: `obc_method='news'`, order 1

Done
- `pydyn/obc.py`: post-processor (`constraint_objects`), `M` and `q` (`NewsOBC`), solvers (`solve_active_set`,
  `solve_news`), diagnostics on `M` (`m_diagnostics`), simulation with OccBin's timing.
- Switch: `Occbin.simulate(..., obc_method='news', obc_horizon=40)`, also through `run()` and `irf_occbin()`; in the
  .mod file `options_.occbin.obc_method = 'news';` and `options_.occbin.obc_horizon = 30;` on their own lines.
- Reference from Dynare 7.2 under Octave (`reference/r_obc_zlb.m` -> `obc_zlb.mat`): `M` and `q` from Dynare's
  first-order solution of the model with news shocks written into the .mod file by hand; the LCP solved by GLPK in
  Holden's formulation; Dynare's own OccBin path.
- `tests/test_obc.py`: 20 tests.

Results (all run 2026-11-24)
- `M` and `q` against Dynare 7.2: 1e-14 and 5e-17. `M` against a stacked-system computation: 1e-15.
- Wedges against the GLPK reference: within GLPK's tolerance (1e-9). Path against Dynare's OccBin: 1e-17.
- Against pydyn's OccBin, identical regime histories and paths to 1e-14 or better on: `obc_zlb` (own model),
  `obc_zlb2` (own model, two interacting constraints), Dynare's `borrcon` (reference regime is the binding one),
  Dynare's `NKM_0_std_shocks` (approximate class, see below); file shocks and 200 random shock periods each.
- Timing, 1,000 shock periods, about half of them binding: OccBin 1.2 s, news 0.22 s (one constraint) and
  1.3 s vs 0.24 s (two). The integer programme was never needed: the active-set fast path solved every period.
  Forced through the integer programme: 3.2 s (one constraint), 33 s (two).

Classification of Dynare 7.2's OccBin files

| File | Constraint | Class | Reason |
|---|---|---|---|
| tests/occbin/model_borrcon/borrcon*.mod | borrcon | complementarity | |
| tests/occbin/filter/NKM_*.mod | zlb | complementarity (approximate, 0.6%) | `inom = inomlb*exp(mp)` vs `bind inomnot <= inomlb` |
| examples/occbin/rbc_occbin.mod | IRR | complementarity | |
| examples/occbin/rbc_occbin.mod | INEG | inconsistent | switches adjustment costs on; not a bound on one equation |
| tests/occbin/ppf/bf_ppf.mod, tests/estimation/slice/bf.mod | zlb | multi_equation | the rule's lag term also switches (`rlag = rnot` / `rlag = r`) |
| tests/occbin/model_irrcap_twoconstraints/dynrbc*.mod | | not loaded | pydyn refuses the file's MATLAB code (existing limit); same INEG constraint as rbc_occbin |
| tests/occbin/ppf/swpkfppf.mod | | not loaded | pydyn refuses `set_param_value(..., mean(dy(74:end)))` (existing limit) |

The classes decide which formulation `obc_method='news'` uses, not whether it runs (see the next section): the LCP needs
every constraint to be `complementarity`; anything else goes to the general regime formulation.

Contract amendments (2026-11-24)
1. Exit test when no `relax` condition is given: the "leave" function is `d_b` itself (OccBin leaves when the bind
   condition fails), not `-d_b`. And `s_y = -sign(c)`. The contract's first version had both wrong; a one-constraint
   model with an explicit `relax` hides the error, a constraint without `relax` gives a negative diagonal of `M`.
2. Approximate class: accept entry/exit fits with relative residual up to 0.05 and record it (`approx`). Needed for
   lower-bound models written as `inom = inomlb*exp(mp)`.
3. `q` is affine in (state, shock): precompute the map with `M`. This is what makes the method faster than OccBin.

Lessons for the other port
- Verify every solver answer against the LCP conditions. GLPK under Octave, with its LP presolver on (the default),
  returned an "optimal" point that violated a constraint by 0.7; with the presolver off it is correct.
- After the integer programme, re-solve the square system on the binding set it reports. That removes solver
  tolerance from the wedges and makes paths agree with OccBin to rounding.
- An OccBin-style active-set iteration on `M` (start from `q < 0`, solve, drop negative wedges, add violated
  periods) solves almost every period; keep the integer programme as the fallback and for the guarantee.
- A test model with two constraints, one of them without `relax`, catches sign errors that the standard
  lower-bound model does not.
- HiGHS prints `HighsMipSolverData::transformNewIntegerFeasibleSolution` lines to stdout on some problems.

## 2026-11-24 (later): general regime formulation, `RegimeOBC`

For constraints that switch several equations, or whose conditions are not the mirror image of the switched
equation. Each switched equation gets its own wedge on its reference version; in a period spent in the alternative
regime the wedges make the alternative versions hold. A regime sequence is accepted by OccBin's own test (conditions
in levels). So the answer is OccBin's piecewise-linear solution, found differently: wedge responses computed once,
one small linear solve per guess, guess updated as OccBin does. `Occbin.simulate(obc_method='news')` picks `NewsOBC`
(LCP) when every constraint is a complementary pair and `RegimeOBC` otherwise; `res.news['formulation']` says which.

Results (run 2026-11-24; paths and regime histories against pydyn's OccBin)
- `bf_ppf.mod` and `estimation/slice/bf.mod` (lower bound with the lag term of the rule switching too): 1.7e-14,
  200/200 regime histories, 200 random shock periods; 0.03-0.05 s against 0.17 s for OccBin.
- `rbc_occbin.mod` with the file's 169 shock periods (two constraints, one not a bound, conditions with logs):
  8.9e-16, 169/169. Needs `obc_horizon=130` because INEG stays in its alternative regime for up to 110 periods;
  with 40 it stops with `horizon_too_short`. Here OccBin is faster (0.07 s against 0.24 s with that horizon).
- On the complementarity models `RegimeOBC` equals both `NewsOBC` and OccBin to 1e-11 or better.

Limits of the general formulation
- No existence or uniqueness theory (that belongs to the LCP), and no guarantee: if the guess-and-verify iteration
  cycles, the fallback is a big-M integer programme over regime indicators on linearised conditions. It works for one
  constraint (0.02 s per period) but is slow for two (280 s for five periods of `obc_zlb2` at horizon 40), so it has a
  30 s time limit and then reports `not_converged`. No model in the tests needed it.
- An equation whose version depends on both constraints jointly is refused.
- The persistent-regime case shows the cost of a fixed news horizon: OccBin needs no horizon choice.

Diagnostics on M: the exact P-matrix test now uses the Tsatsomeros-Li recursion (16 rows in 0.3 s, 20 in 5 s; checked
against enumeration of minors). For the full 40-period M of the lower-bound models the answer is still "undecided":
`largest_p` reports the longest leading block that is a P-matrix.

Not done
- `minnorm` (being built in JDyn first). Orders 2 and 3, cubature, estimation.
- Diagnostics: Holden's fuller sufficient conditions are not ported; beyond 16-20 periods uniqueness is undecided
  unless the symmetric part of M is positive definite.
- A faster guaranteed fallback for the general formulation.
- Uniqueness is not reported per period; when `M` is not a P-matrix the returned solution is one of possibly several
  (the shortest news horizon that works).
- No reference yet from DynareOBC itself (it needs MATLAB and Dynare 4.5).

## 2026-11-24 (third entry): why Smets-Wouters with a lower bound fails, and a proof instead of a cycle

Benchmark `reference/obc_sw_zlb.mod` (Dynare's swpkfppf.mod without data and estimation lines; 32 variables; the
constraint switches the rule and its lag term).

Finding. OccBin cycles (error 313) on a -6 shock to `eb`, and at period 36 of a simulation at one standard
deviation. The cause is not the algorithm: those states have no solution. Others have several.

How it was established. In the alternative regime the second switched equation (`rlag = rnot`) ties its wedge to
the first one's: `w2 = -w1 - ms`. With that tie the model is a complementarity problem in one wedge per period,
`g = q + M y + G z` (`RegimeOBC.lcp`; G holds the `ms` offsets and vanishes when `ms = 0`). `_find_tie` detects the
structure from the wedge responses: all but one switched equation respond to wedges only in the same period, with
constant coefficients; and the bind/relax conditions are exact mirror images of the primary equation.
- M is not an S-matrix and not a P-matrix (largest leading P block: 10 periods). So existence fails for some
  states and uniqueness for others; this is the situation Holden's theory paper describes for this class of model.
- `eb = -4`: exactly two single-spell solutions, binding for 9 or 13 periods, with paths up to 29 apart (output on
  impact -18.8 or -26.1). OccBin and the news method both return the 9-period one. `eb = -2`: two as well.
- `eb <= -5`: the linear programme `y >= 0, q + M y >= 0` is infeasible at horizons 40 and 80. That is a proof that no
  path returns to the reference regime within the horizon. It takes 0.06 s, where OccBin cycles and the big-M
  integer programme timed out.
- 400 random periods (all shocks): at half a standard deviation 55 binding periods solved, 5 proved to have no
  solution, 1 horizon too short, 1 time-out; at one standard deviation 78 solved, 27 proved to have no solution,
  9 horizon too short, 1 unproved. Every binding state checked (55 and 60) had more than one single-spell solution.

Added
- `RegimeOBC.lcp`, `diagnostics`, `single_spell_solutions`; `_solve_lcp` is now the first fallback when the
  guess-and-verify iteration cycles: Holden's homogenised integer programme when G = 0, a one-binary-per-period
  programme otherwise, and in both cases an infeasible linear relaxation is reported as a proof.
- `pydyn/obc_lk.py` (`LKNews`): cached single-spell fast path after Boehl's (l, k) idea, with the complete LCP test.
  2,000 periods: 26 microseconds per period on `obc_zlb` (news 245, OccBin 1,057), 52 on NKM (news 219, OccBin 894).
  One constraint, single-equation only, not yet wired into `obc_method`.
- 32 tests in tests/test_obc.py.

What pydsge does (read from its source, engine.py): precomputes transition matrices and five verification rows per
(l, k) up to l_max = 3, k_max = 16 by default in estimation; checks five periods rather than the whole path; when
no (l, k) passes it returns an approximation with a flag instead of failing. Compiled with numba.

Caveats
- The benchmark is Dynare's test calibration with its shock sizes, not an estimated model.
- "No solution" means within the news horizon plus check-ahead; the infeasibility proof is for that horizon.
- Which equilibrium is returned when there are several is decided by the iteration's starting guess, as in OccBin.
- The tie structure is detected for one constraint; two constraints with several equations each still use the
  slow general integer programme as the last resort.

Not done: single-spell cache for the tied (several-equation) form; an equilibrium-selection option; `minnorm`.

## 2026-11-24 (fourth entry): Dynare's OccBin estimation examples; likelihood with the news solver

Decisions (Jacob): the test bed for estimation is Dynare's own OccBin estimation examples, not his model; equilibrium
selection is to be an option whose default is OccBin's choice.

Diagnostics on the examples (1,500 random periods at 1, 2 and 3 standard deviations, horizon 40)
- `tests/occbin/filter/NKM.mod` (24 variables, complementarity): M is an S-matrix; P undecided beyond 14 periods;
  no failures; none of the binding states checked (78, 150, 150) has a second single-spell solution.
- `tests/estimation/slice/bf.mod` (5 variables, two equations switch, tie found): M has a positive definite symmetric
  part, hence a P-matrix (unique solution); one unproved `no_solution` in 1,500 periods at 2 and at 3 standard
  deviations (not investigated); no multiplicity.
So the Smets-Wouters problems (no solution, several solutions) are specific to that model, not to these.

Inversion-filter likelihood with the news solver (`pydyn/obc_filter.py`, `NewsFilterSolver`; prototype switch
`est.obc_method = 'news'` on `OccbinEstimation`; `inversion_filter(..., solver=)`)
- NKM (9 parameters, 120 observations, 28-31 binding periods) and its small version (3 parameters): log posterior
  equal to OccBin's and to Dynare's stored value to 1e-11 at the 7 stored parameter vectors; same regimes; filtered
  shocks within 1e-12.
- Time per likelihood evaluation: OccBin 260-390 ms, news 42-47 ms (6 to 8 times faster) at 6 of the 7 points.
  At one point the root finder tries a shock for which the active-set iteration fails and the linear programme is
  feasible; the integer programme then runs into its 1 s limit and the evaluation takes 1.1 s against 0.36 s.
- 40 parameter vectors perturbed by 2% around the first: 23.6 s against 3.9 s; 39 agree to 1e-11; one differs by 0.015
  in the log posterior: in one period OccBin's condition says the bound binds for one period and the linearised
  residual says it does not (the paths and shocks agree to 1e-12, but the likelihood's Jacobian term depends on the
  regime). This is the "approximate" class of this model showing at a knife edge.

Not done: the same for several-equation constraints (`bf`), the piecewise Kalman filter, a proper option in place of
the attribute switch, mode finding and MCMC runs with the news solver, the selection option.

## 2026-11-24 (fifth entry): order 2 and cubature, prototype as executable spec

Design: project doc `claude/obc-higher-order-design.md`. Decisions (Jacob): the full DynareOBC machinery is wanted;
cubature validation against DynareOBC itself is left aside; higher order and cubature are restricted, initially, to
constraints that are one complementary pair (and later the tied-wedge class); production implementation starts in JDyn.

`pydyn/obc_ho.py` (`HigherOrderNews`) is a prototype written to pin the algebra down before the Julia implementation:
- state = (y1, y2, o): first- and second-order components of the pruned simulation that ignores the bound, and the
  bound offset (first-order accumulated effect of the news shocks, propagated by the first-order transition);
- expected path with future shocks at their distribution: `m1' = ghx m1s`, `W' = ghxs W ghxs' + ghus Sigma ghus'`,
  `m2' = ghx m2s + (ghxx vec(W) + ghuu vec(Sigma) + ghs2)/2`, `o' = P o`, starting from `W = y1s y1s'`;
- q = the (linearised) gap along that path; the LCP, M, solvers are those of order 1;
- cubature: `R` with `R R'` = covariance of the gap path from shocks arriving in the next `periods_of_uncertainty`
  periods (first order, DynareOBC's cosine taper), scrambled Sobol points mapped to normals, points where nothing
  binds dropped, then either one region (solution at the mean of the remaining points times their share: DynareOBC's
  default) or the plain average over all points.

Checks on Dynare's NKM estimation example (24 variables; `tests/test_obc_ho.py`, 5 tests)
- order 1 equals `NewsOBC` (4e-14);
- the base simulation at order 2 equals pydyn's pruned simulation, which is validated against Dynare 7.2 (2e-14);
  with shocks too small to reach the bound, with and without bound coincide exactly;
- the expected path equals the Monte Carlo mean of 20,000 pruned simulations from the same state over 12 periods for
  all variables (largest z-score 1.4);
- cubature with the shock variance scaled to 1e-12 equals no cubature (9e-12); `R R'` equals the exact covariance.

First numbers (2,000 periods, shocks of one standard deviation): news shocks are non-zero in 104 periods at order 1,
112 at order 2, and 1,245 with cubature (agents react to the risk before the bound is reached). The one-region
default and the average over all points differ by up to 1.8e-3 in the policy rate over 300 periods, which is as large
as the effect of cubature itself, so the number of regions matters. Prototype speed: 1.3 to 5 ms per period (pure
Python loops; not optimised, not the target).

Not done: order 3; several regions and clustering; second-order conditional covariance; linearisation around the
risky steady state; wiring into `obc_method` options; the tied-wedge class; any comparison with DynareOBC.

## Real DynareOBC under Octave: validation of orders 1-2 and cubature (24 Nov)

DynareOBC itself now runs here (Octave 8.4, Dynare 4.5.4, four small patches; recipe, patches and reference
output in `reference/dynareobc_octave/`). `tests/test_obc_dynareobc.py` compares `HigherOrderNews` with it on a
small NK model with a lower bound (300 periods, 83 at the bound):

| check | max abs difference |
|---|---|
| M matrix | 9e-16 |
| simulation, order 1 and order 2 | 2e-16 |
| conditional covariance of the gap path (taper, 1% pruning) | 5e-19 |
| order-2 cubature, DynareOBC's root and Sobol points | 4e-16 |
| order-2 cubature, pydyn's own (SciPy) Sobol points | 1.5e-6 (cubature effect: 4.8e-5) |

Changes to `obc_ho.py` from the comparison: cubature points are now DynareOBC's rule (2^(level+1)-1 unscrambled
Sobol points, exactly mean zero) instead of scrambled ones; the covariance root is pruned as in DynareOBC
(`pruning_cutoff=0.01`, `max_cubature_dimension`).

Findings: (1) with one region, cubature equals no cubature whenever the bound binds today (every point is
relevant, their mean is the expected path); it only acts when the bound binds in expectation later.
(2) Under Octave DynareOBC's root truncation dropped the largest eigenvalue (unordered `schur`); patched.
(3) Dynare's Sobol direction numbers differ from SciPy's beyond dimension 2, so exact agreement needs the
points exported. Not yet compared: several regions (needs DynareOBC's clustering), order 3, CATCH rules.

## Several cubature regions against DynareOBC (24 Nov, evening)

`HigherOrderNews(regions=k)` now implements DynareOBC's region splitting (principal components of the relevant
paths and their negative parts, repeated Otsu splits). Against DynareOBC with `CubatureRegions=4`, same root and
Sobol points: 294 of 300 periods agree to 3e-16; six differ by up to 0.006 (a near-tie in a split).

The number of regions matters far more than expected. Small NK model, order 2, level 9, mean over 300 periods:

| regions | mean c | mean pi | max change from no cubature | pydyn ms/period |
|---|---|---|---|---|
| none | 0.99609 | 0.99902 | 0 | 0.04 |
| 1 (DynareOBC default) | 0.99608 | 0.99902 | 5e-05 | 0.4 |
| 2 | 0.97082 | 0.98962 | 0.056 | 8.7 |
| 4 | 0.95777 | 0.98480 | 0.095 | 10.0 |
| 8 | 0.95199 | 0.98268 | 0.096 | 12.6 |
| 16 | 0.94557 | 0.98027 | 0.111 | 14.9 |
| 32 | 0.94128 | 0.97864 | 0.121 | 26.0 |
| every point (plain quasi-Monte Carlo) | 0.92191 | 0.97134 | 0.158 | 37.3 |

One region is close to "no cubature"; the full integral lowers mean consumption by 7%, and convergence in the
number of regions is slow. The size here owes much to the model (no endogenous state, news 32 periods ahead in a
purely forward-looking NK model, i.e. the forward-guidance puzzle), but the ordering is general: the mean of the
relevant points is usually a path that does not bind, so one region prices almost no risk.
Cost in Python is the per-period SVD for the split, not the LCP solves.
