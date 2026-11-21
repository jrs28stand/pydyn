# pydyn

Python front end for Dynare models with no MATLAB/Octave at run time. It runs the Dynare
preprocessor (C++) on a `.mod` file, reads the JSON it writes, and does the numerical work in Python.

## Status

| Piece | State |
|---|---|
| Preprocessor runner, JSON model loader | done (Dynare 7.2 JSON layout; 6.0 differs and is not supported) |
| Parameters, `initval`, `shocks`, `steady_state_model` | done |
| Changing parameters and re-solving without reloading (`set_param`, `set_params`) | done, validated |
| Steady state: analytical block, numerical, MATLAB steady-state files and `.m` helpers (interpreted or translated), Python functions; Dynare's random restarts, `maxit`, and ports of `solve_algo` 4, 9, 13, 14 | done, validated |
| MATLAB code in `.mod` files (control flow, matrices, `M_.params`, `set_param_value`, ...) | done, validated on Dynare's test suite |
| First-order solution (`ghx`, `ghu`, IRFs, simulation) | done, validated |
| OccBin piecewise-linear simulation, one or two constraints, `curb_retrench` | done, validated |
| OccBin impulse responses (`Occbin.irf`) | done, validated |
| Orders 2 and 3 through Dynare's C++ k-order library | done, validated |
| Simulation and IRFs at orders 2 and 3, with and without pruning | done, validated |
| Model-local (`#`) and auxiliary variables (leads/lags beyond one, on endogenous and exogenous) | done, validated |
| Perfect foresight: `initval`/`endval`/`histval` with or without `steady`, `shocks`, `mshocks`, homotopy, `linear_approximation`, `endogenous_terminal_period`, purely backward/forward/static and linear models | done, validated |
| Perfect foresight with complementarity conditions (`lmmcp`, `[mcp=...]` tags and the perpendicular syntax; port of Dynare's LMMCP solver) | done, validated (lower bound binding 11 periods; two-sided bound) |
| Perfect foresight with expectation errors (`learnt_in`), and a helper for sequences of unanticipated shocks | done, validated |
| Perfect foresight: controlled paths, `initval_file`/`histval_file`/`datafile` (CSV, .mat, .m), vector shock values, `robust_lin_solve`; MATLAB code reading or writing `oo_.steady_state` | done, validated (47 of Dynare's 66 deterministic tests agree to 2e-14; the rest differ for documented reasons) |
| OccBin options: periodic solutions, regime guesses, check-ahead and reset settings, Dynare's error codes, Dynare-style IRFs (`irf_occbin`) | done, validated |
| `loglinear` in `stoch_simul` (rules, IRFs, simulation and moments in logs; order 1 only, as in Dynare) | done, validated |
| Moments: theoretical (incl. HP/band-pass filtered and pruned order 2), simulated, variance decompositions | done, validated |
| Kalman smoother, shock decomposition, initial-condition decomposition (stationary and diffuse, missing data, observation trends) | done, validated |
| Smoother and decompositions at the posterior mode/mean/median, ML mode, prior mean/mode or any parameter vector | done, validated at Dynare's own vectors |
| Shock groups, `init2shocks`, real-time decomposition, squeeze; decomposition figures (bars, detail panels, year-on-year, flip, steady state) | done, validated incl. the plotted bars against Dynare's drawing calls |
| Estimation at first order: Kalman likelihood (stationary, diffuse, missing data, univariate), Dynare's priors, posterior mode, Laplace approximation, random-walk Metropolis-Hastings | done, validated |
| Posterior analysis (`pydyn.posterior`, `pydyn.posterior_plots`): posterior statistics and HPD intervals, modified harmonic mean, model comparison, convergence diagnostics (inefficiency factors, Geweke, Brooks-Gelman, Raftery-Lewis), jscale tuning, jumping covariances, posterior IRFs, mode_check, prior and posterior densities; csminwel ported (default optimiser when the .mod file gives no `mode_compute`; Dynare's default newrat is not ported) | done, validated (sampler reproduces Dynare's chains exactly given Dynare's random numbers) |
| OccBin likelihood (piecewise Kalman filter, inversion filter), OccBin smoother and shock decomposition, estimation (`OccbinEstimation`: mode, Laplace, MH) | done, validated on Dynare's OccBin filter tests |

## Validation

`tests/` compares against Dynare 7.2 run under Octave 8.4 (`reference/*.mat`, produced by the
`reference/*.m` scripts):

- First order: `collard_2001_simulated_moments`, `rbc_occbin`, `schorfheide_2000_sim`.
  Decision rules agree to about 1e-13 and 28 IRFs to about 1e-15 (measured; the tests assert 1e-9).
- OccBin, two constraints: `rbc_occbin` (200 periods). Piecewise and linear paths agree to 1e-15
  and the regime history is identical in all 169 shock periods.
- OccBin, one constraint: Dynare's `borrcon` test model, default options and `curb_retrench`.
  Paths agree to 1e-16, regime history identical.
- Order 2 rules against Dynare's default second-order solver, order 3 against `k_order_solver`:
  about 1e-14.
- Simulation at orders 1, 2 and 3 against `simult_` with the same shock draws, pruned and
  unpruned: about 1e-15. IRFs at orders 2 and 3 against the `irf.m` algorithm with saved draws, and
  against Dynare's own `irf.m` run with a fixed seed whose draws are regenerated.
- Parameter loop: six (alp, rho) combinations on the Schorfheide model against
  `set_param_value` + `resol` in Dynare: steady state, rules and IRFs agree.
- Steady-state function: `nk_baseline` with a Python port of `nk_baseline_steadystate.m`
  (`examples/nk_baseline_steadystate.py`), including the parameter it overwrites (`Rbar`) after
  changing `betta`.
- Helper function inside `steady_state_model`: `collard_2001_analytical_steady_state`.
- Numerical steady states agree to Dynare's own solver tolerance (Dynare's steady state for the
  Collard model carries a residual of roughly 1e-6; ours is solved to 1e-14).

Run: `PYDYN_REQUIRE_KORDER=1 DYNARE_PREPROCESSOR=/path/to/dynare-preprocessor python -m pytest tests`
(without `PYDYN_REQUIRE_KORDER=1` the order-2 and order-3 tests are skipped when the compiled k-order
library has not been built; the test header says which applies)
(set `PYDYN_EXAMPLES` to Dynare's `examples` folder).

## Working after the model is loaded

The model is loaded once; everything afterwards is ordinary Python, so an external loop needs no
re-run of the preprocessor:

```python
m = pydyn.Model(pydyn.run_preprocessor("model.mod"))
for rho in (0.5, 0.7, 0.9):
    m.set_param("rho", rho)                    # like set_param_value
    sol = pydyn.first_order(m)                 # steady state recomputed, then decision rules
    resp = pydyn.irf(sol, shock, periods=20)
```

`sol.params` holds the parameters actually used, including any the steady state updated.

### MATLAB code in `.mod` files and steady-state files

`pydyn.matlab` is an interpreter for the subset of MATLAB that `.mod` files use to set up a
calibration. It runs the MATLAB lines of a `.mod` file in order with the Dynare statements, so
control flow around parameter assignments (`if`, `for`, `switch`, ...) behaves as in Dynare.

- A `<model>_steadystate.m` file next to the `.mod` file is used automatically. Where possible it
  is translated to Python once (`pydyn.matlab.translate_steady_state(path)` returns the source)
  and reused, which costs about the same per call as a hand-written Python function.
- `.m` helper functions called from a `steady_state_model` block are found and run the same way.
- A Python `steady_state_fn` or `functions={...}` entry passed to `Model` takes precedence.
- Display, plotting and post-processing lines are skipped with one summarised warning
  (`model.matlab_skipped` lists them); `options_.x = ...` assignments are stored in
  `model.matlab_options` and not acted on.
- Code that could change the calibration in a way the interpreter cannot follow raises
  `ModelError` naming the line. It is deliberately strict: refusing is preferred to guessing.
- MATLAB text is parsed and evaluated by the interpreter itself and never passed to Python's
  `eval`/`exec`; file access is limited to reading `.m` and `.mat` files next to the `.mod` file.

On 212 files from Dynare's own test suite that contain MATLAB code and that Dynare runs under
Octave: 163 load with a calibration identical to Dynare's (one more to solver tolerance), 33 are refused with a clear error, 15
are outside this package for unrelated reasons, none load with a different calibration
(`tools/sweep_matlab.py`; the 163 are re-checked by `tests/test_matlab.py`). An independent
adversarial review (617 mixed Dynare/MATLAB cases run in both systems) found eleven ways to get a
different calibration without an error; each is fixed or refused and has a regression test.
References are from Octave; MATLAB-only behaviour rests on documentation.

Two consequences for models without any MATLAB code: `model.params` after loading includes the
parameters that a `steady_state_model` block or steady-state file sets when the file has a `steady`
or `stoch_simul` command (as `M_.params` does in Dynare), and `initval`/`endval` expressions that
refer to other variables are evaluated as Dynare does (values set so far, zero if unset, the
computed steady state after `steady`).

## Dynare's shipped examples

`tests/test_examples.py` loads Dynare 7.2's example `.mod` files unedited and compares with Dynare's
own results (stored in `reference/examples`). Thirteen match (decision rules, second order where the
example asks for it, and the perfect-foresight path); `rbc_irf_matching` loads and solves but has
no reference; the expectation-errors example matches too; nine are outside the scope of this package: the six
heterogeneous-agent examples, the two Ramsey examples and the PAC model. `nk_baseline`
and `collard_2001_analytical_steady_state` use their MATLAB steady-state code directly.

## The C++ k-order library

`korder/build.sh /path/to/dynare-7.2` compiles Dynare's `mex/sources/libkorder` unchanged into
`korder/libpydyn_korder.so`. Two small files make that possible:

- `korder/stub/mex.h`: a stand-in for the MATLAB header (the library only touches it in code
  paths we never call);
- `korder/korder_capi.cc`: replaces the MEX gateway. Dynare's gateway calls back into MATLAB for
  the model derivatives; this one receives them precomputed from Python.

`pydyn.k_order(model, order=2)` returns `ghx`, `ghu`, `ghxx`, `ghxu`, `ghuu`, `ghs2` (and the
third-order arrays) laid out as in `oo_.dr`. Third order needs third derivatives in the JSON:
set `order=3` in the `.mod` file or run the preprocessor with `output=third`.

## Use

```python
import pydyn
jd = pydyn.run_preprocessor("model.mod")      # needs DYNARE_PREPROCESSOR or the binary on PATH
m = pydyn.Model(jd)
sol = pydyn.first_order(m)
sol.ghx, sol.ghu                               # same layout as oo_.dr.ghx / oo_.dr.ghu
sol.irf(shock_vector, periods=40)

ob = pydyn.Occbin(m)                           # model with an occbin_constraints block
res = ob.simulate(ob.surprise_shocks(), periods=200)
res.piecewise, res.linear, res.regime_history
ob.irf([-0.05], periods=40)                    # piecewise IRF, relative to a no-shock baseline

dr2 = pydyn.k_order(m, order=2)                # needs korder/libpydyn_korder.so
path = pydyn.simulate(dr2, shocks, pruning=True)
girf = pydyn.irf(dr2, impulse, m.Sigma_e, periods=40, pruning=True, replic=50, drop=100)
```

## Options from the .mod file

Loading a model does not run its commands; the Python functions do the work and take their
defaults from the `.mod` file where that is meaningful: `order`, `pruning`, `qz_criterium` and
`loglinear` (`k_order`, `first_order`, `simulate`, `irf`), `steady(tolf, nocheck)`, the `stoch_simul` options
used by `moments`, the `simul_*` options of OccBin, the estimation and smoother options listed in
their modules, and `periods`, `tolf`, `maxit` for perfect foresight. Options a function does not
act on are reported once in an `UnsupportedOptionWarning`; a few that would change the problem
(for example OccBin constraints in linear estimation) raise an error instead.
`docs/GAPS.md` is a command-by-command and option-by-option audit against the Dynare 7.2 manual.

## Performance

- The first-order solver is a port of Dynare's reduced algorithm (static variables removed by QR, QZ
  on the dynamic block). `PYDYN_FIRST_ORDER=full` selects the original full-size solver.
- Generated model functions run on Python floats and fall back to numpy where plain floats would
  raise or turn complex, so results are numpy's in every case (`PYDYN_CODEGEN=numpy` disables this).
- A `steady_state_model` block is compiled once per model (`PYDYN_SS_COMPILE=0` disables this).
- Compiled model functions are cached in `~/.cache/pydyn` (or `$PYDYN_CACHE_DIR`), keyed by a hash
  of the preprocessor output; `PYDYN_CACHE=0` or `Model(..., cache=False)` turns the cache off. The
  cache is never read from the model's own folder.

Time for one `set_param` + steady state + first-order solve (CPU, one thread): about 0.3 ms for a
15-variable model and 0.8-1.1 ms at 28-40 variables.

## Known limits

- MATLAB inside `.mod` files: complex numbers, N-d arrays, struct arrays, classes (`dseries`,
  dates), random numbers and file-system functions are not supported; see the section above.
- Not handled: external
  functions inside the model block, deterministic exogenous variables, Ramsey/optimal policy,
  heterogeneous-agent models, measurement-error covariances, `steady_state()` of an expression.
- Linear estimation and the smoother refuse models with `occbin_constraints`; use
  `pydyn.OccbinEstimation` and `pydyn.occbin_filter.occbin_smoother`. Not implemented for OccBin:
  particle filters, the diffuse filter, OccBin forecasts; missing data in the OccBin filters is
  not validated.
- `shocks(surprise, overwrite)`: the preprocessor's JSON does not record `overwrite` for surprise
  blocks, so several surprise blocks are merged (with a warning).
- OccBin regime matrices are evaluated at the exogenous steady state; Dynare uses zero. The two
  differ only for models with a non-zero exogenous steady state (not tested).
- The k-order library keeps process-wide state. Calls are serialised with a lock, and a call in a
  forked child process after the parent used the library raises an error: use the `spawn` start
  method for multiprocessing.
- The 7.2 preprocessor writes invalid JSON for some statements. The loader repairs `matched_irfs`
  and `estimated_params_bounds`; files using `shock_paths`, `model_comparison` or `colormap=`
  still fail to load.

## Independent review

`tests/test_review.py` holds regression tests for the bugs an independent review found, with
Dynare's results as expected values: comparisons used as numbers, chained comparisons, derivatives
of `max`/`min`, the order in which `shocks` blocks apply variances and correlations, models with
no leads or no lags, numerical steady states with auxiliary variables, `initval` entries that
refer to each other, linear models at second order, and name clashes with temporary terms.

## Licence

GPL-3.0-or-later (see `LICENSE`). Parts are ported from Dynare; `NOTICE.md` lists which files and
their upstream terms. This project is not affiliated with the Dynare Team.
