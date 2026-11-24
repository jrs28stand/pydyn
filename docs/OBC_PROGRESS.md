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

So of Dynare's own files only borrcon and NKM run under `news` today. A model needs every constraint to qualify.

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

Not done
- `minnorm` (being built in JDyn first). Orders 2 and 3, cubature, estimation, `multi_equation` constraints.
- Diagnostics: exact P-matrix test only up to 12 rows; Holden's fuller sufficient conditions are not ported.
- Uniqueness is not reported per period; when `M` is not a P-matrix the returned solution is one of possibly several
  (the shortest news horizon that works).
- No reference yet from DynareOBC itself (it needs MATLAB and Dynare 4.5).
