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
