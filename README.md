# pydyn

Python package name: `dynpy`.

Python front end for Dynare models with no MATLAB/Octave at run time. It runs the Dynare
preprocessor (C++) on a `.mod` file, reads the JSON it writes, and does the numerical work in Python.

## Status

| Piece | State |
|---|---|
| Preprocessor runner, JSON model loader | done (Dynare 7.2 JSON layout; 6.0 differs and is not supported) |
| Parameters, `initval`, `shocks`, `steady_state_model` | done |
| Changing parameters and re-solving without reloading (`set_param`, `set_params`) | done, validated |
| Steady state: analytical block, numerical, Python steady-state function, helper functions | done, validated |
| First-order solution (`ghx`, `ghu`, IRFs, simulation) | done, validated |
| OccBin piecewise-linear simulation, one or two constraints, `curb_retrench` | done, validated |
| OccBin impulse responses (`Occbin.irf`) | done; no Dynare reference yet (built on the validated simulation) |
| Orders 2 and 3 through Dynare's C++ k-order library | done, validated |
| Simulation and IRFs at orders 2 and 3, with and without pruning | done, validated |
| Model-local (`#`) and auxiliary variables (leads/lags beyond one, on endogenous and exogenous) | done, validated |
| Perfect foresight (stacked Newton; `initval`, `endval`, deterministic `shocks`) | done, validated on one example |
| OccBin: periodic solutions, initial regime guesses | not ported |
| Estimation, piecewise Kalman filter | not started |

## Validation

`tests/` compares against Dynare 7.2 run under Octave 8.4 (`reference/*.mat`, produced by the
`reference/*.m` scripts):

- First order: `collard_2001_simulated_moments`, `rbc_occbin`, `schorfheide_2000_sim`.
  Decision rules agree to about 1e-13; 28 IRFs agree to about 1e-15.
- OccBin, two constraints: `rbc_occbin` (200 periods). Piecewise and linear paths agree to 1e-15
  and the regime history is identical in all 169 shock periods.
- OccBin, one constraint: Dynare's `borrcon` test model, default options and `curb_retrench`.
  Paths agree to 1e-16, regime history identical.
- Order 2 rules against Dynare's default second-order solver, order 3 against `k_order_solver`:
  about 1e-14.
- Simulation at orders 1, 2 and 3 against `simult_` with the same shock draws, pruned and
  unpruned: about 1e-15. Order-2 IRFs against the `irf.m` algorithm with the same random draws.
- Parameter loop: six (alp, rho) combinations on the Schorfheide model against
  `set_param_value` + `resol` in Dynare: steady state, rules and IRFs agree.
- Steady-state function: `nk_baseline` with a Python port of `nk_baseline_steadystate.m`
  (`examples/nk_baseline_steadystate.py`), including the parameter it overwrites (`Rbar`) after
  changing `betta`.
- Helper function inside `steady_state_model`: `collard_2001_analytical_steady_state`.
- Numerical steady states agree to Dynare's own solver tolerance (Dynare's steady state for the
  Collard model carries a residual of roughly 1e-6; ours is solved to 1e-14).

Run: `DYNARE_PREPROCESSOR=/path/to/dynare-preprocessor python -m pytest tests`
(set `DYNPY_EXAMPLES` to Dynare's `examples` folder).

## Working after the model is loaded

The model is loaded once; everything afterwards is ordinary Python, so an external loop needs no
re-run of the preprocessor:

```python
m = dynpy.Model(dynpy.run_preprocessor("model.mod"))
for rho in (0.5, 0.7, 0.9):
    m.set_param("rho", rho)                    # like set_param_value
    sol = dynpy.first_order(m)                 # steady state recomputed, then decision rules
    resp = dynpy.irf(sol, shock, periods=20)
```

`sol.params` holds the parameters actually used, including any the steady state updated.

### Steady-state files

MATLAB `_steadystate.m` files cannot be run. The options are:

- a `steady_state_model` block in the `.mod` file (works as is, including parameter updates);
- a Python function passed as `Model(..., steady_state_fn=fn)` with signature
  `fn(model, params, exo) -> ys` or `(ys, params)`; see `examples/nk_baseline_steadystate.py`;
- for `.m` helper functions called inside `steady_state_model`, a Python function of the same
  name passed as `Model(..., functions={"name": fn})`.

Plain scalar assignments in the `.mod` file (`phi = 0.1;`) are understood; any other MATLAB code
in the file is not executed and is reported in a warning.

## Dynare's shipped examples

`tools/sweep_examples.py` loads each of the 23 example `.mod` files in Dynare 7.2 and compares with
Dynare's own results. Twelve match (decision rules, second order where the example asks for it,
and the perfect-foresight path); `rbc_irf_matching` loads and solves but has no reference; ten
are outside the scope of this package: the six heterogeneous-agent examples, the two Ramsey
examples, perfect foresight with expectation errors, and the PAC model. No `.mod` file needed
editing; `nk_baseline` and `collard_2001_analytical_steady_state` need their MATLAB steady-state
code supplied as Python functions.

## The C++ k-order library

`korder/build.sh /path/to/dynare-7.2` compiles Dynare's `mex/sources/libkorder` unchanged into
`korder/libdynpy_korder.so`. Two small files make that possible:

- `korder/stub/mex.h`: a stand-in for the MATLAB header (the library only touches it in code
  paths we never call);
- `korder/korder_capi.cc`: replaces the MEX gateway. Dynare's gateway calls back into MATLAB for
  the model derivatives; this one receives them precomputed from Python.

`dynpy.k_order(model, order=2)` returns `ghx`, `ghu`, `ghxx`, `ghxu`, `ghuu`, `ghs2` (and the
third-order arrays) laid out as in `oo_.dr`. Third order needs third derivatives in the JSON:
set `order=3` in the `.mod` file or run the preprocessor with `output=third`.

## Use

```python
import dynpy
jd = dynpy.run_preprocessor("model.mod")      # needs DYNARE_PREPROCESSOR or the binary on PATH
m = dynpy.Model(jd)
sol = dynpy.first_order(m)
sol.ghx, sol.ghu                               # same layout as oo_.dr.ghx / oo_.dr.ghu
sol.irf(shock_vector, periods=40)

ob = dynpy.Occbin(m)                           # model with an occbin_constraints block
res = ob.simulate(ob.surprise_shocks(), periods=200)
res.piecewise, res.linear, res.regime_history
ob.irf([-0.05], periods=40)                    # piecewise IRF, relative to a no-shock baseline

dr2 = dynpy.k_order(m, order=2)                # needs korder/libdynpy_korder.so
path = dynpy.simulate(dr2, shocks, pruning=True)
girf = dynpy.irf(dr2, impulse, m.Sigma_e, periods=40, pruning=True, replic=50, drop=100)
```

## Known limits

- The first-order solver uses a full-size QZ decomposition (2n x 2n). Correct, but slower than
  Dynare's reduced algorithm for large models.
- MATLAB inside `.mod` files: scalar assignments, `set_param_value('name', value)` and
  `M_.params(i) = value` are executed. MATLAB control flow (`if`, `for`, ...) raises an error,
  because the parameter assignments inside it would otherwise be applied unconditionally. Use the
  macro language (`@#if`) or set parameters from Python.
- Not handled: `loglinear`, external functions inside the model block, deterministic exogenous
  variables, Ramsey/optimal policy, heterogeneous-agent models, measurement-error covariances,
  steady-state assignments with several outputs (`[a, b] = f(...)`), `steady_state()` of an
  expression.
- `shocks(surprise, overwrite)`: the preprocessor's JSON does not record `overwrite` for surprise
  blocks, so several surprise blocks are merged (with a warning).
- OccBin regime matrices are evaluated at the exogenous steady state; Dynare uses zero. The two
  differ only for models with a non-zero exogenous steady state (not tested).
- The k-order library keeps process-wide state. Calls are serialised with a lock, and a call in a
  forked child process after the parent used the library raises an error: use the `spawn` start
  method for multiprocessing.
- `examples/estimation/rbc_irf_matching.mod`: the 7.2 preprocessor writes invalid JSON for this
  file, so it cannot be loaded.

## Independent review

`tests/test_review.py` holds regression tests for the bugs an independent review found, with
Dynare's results as expected values: comparisons used as numbers, chained comparisons, derivatives
of `max`/`min`, the order in which `shocks` blocks apply variances and correlations, models with
no leads or no lags, numerical steady states with auxiliary variables, `initval` entries that
refer to each other, linear models at second order, and name clashes with temporary terms.

## Licence

GPL-3.0-or-later (see `LICENSE`). Parts are ported from Dynare; `NOTICE.md` lists which files and
their upstream terms. This project is not affiliated with the Dynare Team.
