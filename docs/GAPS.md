# pydyn vs Dynare 7.2: what is still missing

Status at the `results-layer` branch, 2026-11-24. This replaces the option-by-option audit of
2026-11-20, which is in git history: `git show 723a94a:docs/GAPS.md`.

Where to look for the rest:
- What *is* supported, and how each part is validated, is in README.md.
- How Dynare's own test suite fares is in docs/ACCEPTANCE.md.
- The cross-port capability table shared with the Julia port (JDyn) is kept in the claude.ai project "Dynare Conversion" (`claude/capabilities-and-profiling.md`).

pydyn's rule is to refuse rather than guess. Everything below either raises a clear error
(`NotImplementedError`, `Unsupported`, `ResultsError`) or, for options, warns once with
`UnsupportedOptionWarning`. Any option, or `options_` field set from MATLAB code, that pydyn does not act on is reported; none is ignored silently.

How options are kept honest (`pydyn/option_policy.py`): for every statement `pydyn.dynare()` runs, each option the Dynare 7.2
preprocessor can emit is classified as *honoured* (pydyn acts on it), *harmless* (it cannot change a number pydyn computes: printing,
figures, the name of an output folder, the choice of a numerical algorithm that gives the same solution; the reason is in
`HARMLESS_REASONS`) or *refused* (Dynare acts on it, pydyn does not: the statement raises a `ResultsError` naming the option).
Some are value-aware: `dr_cycle_reduction_tol` and the other solver tolerances are harmless at Dynare's default or tighter and refused when looser (the solution would differ from QZ), `fsolve_options` is harmless only for Display/AutoScaling-type settings, and `lyapunov=fixed_point` is refused (1.6e-8 off at its default tolerance). An option the table does not know is refused too, so an option added by a later Dynare is refused until it is classified.
`tools/options_audit.py` rebuilds the ground truth from `DynareBison.yy` and `tests/test_option_policy.py` fails if a table and the grammar
disagree; `tools/options_probe.py` runs a statement with and without an option and reports options whose results do not change
(the check used to confirm "honoured").  The plain API (`pydyn.Model`, the functions of the package) warns at load time about the
refused options of the statements in the file instead of raising.  Options written by MATLAB code (`options_.x = ...`) are covered
by `pydyn.options.HONOURED_MATLAB_OPTIONS` as before.

## 1. Features not implemented

| Area | Missing | Notes |
|---|---|---|
| Solution | Orders above 3; `k_order_welfare` (welfare above order 2) | Order 3 uses Dynare's C++ k-order library |
| Solution | Spectral density, `partial_information` | |
| Estimation | Order-3 (particle) estimation; the `gmf` filter | Order 2 supports `sis`, `apf`, `cpf`, `gf` and `nlkf` |
| Estimation | `mode_compute` 2, 6, 10 and above | 7 is Octave's `fminsearch`, ported; 1, 3 and 9 are scipy equivalents and warn |
| Estimation | Samplers `dime_mcmc` and `online`; TaRB whole-chain validation | TaRB matches block by block only |
| Estimation | Posterior distributions of `filter_covariance` and `smoothed_state_uncertainty` | |
| OccBin | Particle-filter options (`filter_particle_*`) | OccBin forecasts work (random or Sobol bands) |
| Forecasts | `forecast` at order > 1 after estimation; `varexo_det` at order 3 or with pruning | `forecast` after `stoch_simul` works at orders 1–3 |
| Optimal policy | OSR with optimisers other than csminwel, and `osr_params_bounds`; `mult_elimination` | |
| Perfect foresight | `varexo_det`; `var(log)` auxiliaries in `initval_file`/`histval_file` | |
| Extended path | Sparse-tree and hybrid stochastic EP, Stroud cubature, `extended_path_mc` | Order 0 and the full-tree stochastic EP work |
| Backward models | `model_inversion` with leads; `histval` with `maximum_lag` > 1 | |
| Semi-structural | `pac.estimate` with `particleswarm` (a MATLAB toolbox function; Dynare fails under Octave), PAC lags other than 1 and non-additive residuals (Dynare fails on both); `equation.evaluate`/cherrypick, calls to model-named functions (`example1.dynamic_resid`) | `pac.estimate.nls`/`iterative_ols` (incl. `annealing`), `var_model`, trend-component models, `var_expectation` and the PAC updates work |
| Toolboxes | `identification`, `dynare_sensitivity` (GSA), `method_of_moments`, `irf_calibration`/`moment_calibration`, Markov-switching SBVAR, BVAR, heterogeneous-agent models | Not started |
| Output | LaTeX (`write_latex_*`), reporting | These warn |
| Data | Full `dseries` (transformations, tags, non-MAT formats) | `dseries` covers names, dates, arithmetic, lags/leads, field/brace/paren assignment (Dynare's merge rules) and MAT files |

## 2. The MATLAB interpreter and `pydyn.dynare()`

The interpreter runs the MATLAB code people put in `.mod` and steady-state files. It is sandboxed:
- Python builtins are unreachable.
- Helper functions from `addpath` folders are interpreted, never run natively.
- Writes are confined to the model's folder and never overwrite files pydyn did not write.

It deliberately refuses these:
- **Data types and objects:** complex numbers, classes, function handles beyond simple ones, N-d cell and char arrays.
- **Environment and time:** file I/O other than the sandboxed `save`/`delete`, and clock-dependent code (`tic`/`toc`).
- **Random numbers:** generators other than Octave's Mersenne Twister. `rand`/`randn`/`randi`/`randperm` are bit-exact with Octave 8.4, not with MATLAB.
- **Places where Octave and MATLAB disagree or are undocumented:** certain deletions; `sprintf` of negative values with `%u`/`%o`/`%x`; `all(X,dim)` with NaN above the first dimension; `unique` of non-0x0 empties.

These still stop `pydyn.dynare()` (each raises `ResultsError` naming the item):
- **3-D or struct-array fields:**
  - `oo_.occbin.simul.regime_history`
  - `oo_.pfwee.controlled_paths_by_period`
  - `oo_.deterministic_simulation.sim1/sim2`
  - `oo_.dr.g_0`–`g_3` and `pruning` at order 3
  - `oo_.dr.fuu`
- **Unfilled fields:** `oo_.initval_series` and `oo_.time`.
- **Statements still refused:**
  - `load_mh_file`/`mh_recover`
  - `shock_paths`
  - `with_epilogue`
  - `filter_decomposition`

## 3. Numerical differences that are expected

- **Steady states.** Dynare stops its Newton iterations at 1e-5; pydyn polishes to about 1e-12 unless told otherwise. Values computed from a numerically solved steady state can therefore differ at the 1e-5 level. Where Dynare's own tolerance is set in the file, pydyn follows it.
- **Optimisers.** csminwel, newrat and simplex are ported and follow Dynare's path until rounding noise separates them. On flat likelihoods the stopping points can differ, by up to about 0.4 standard errors for ML without priors. The tests state each tolerance and its cause.
- **Ill-conditioned models.** For example, on `nk_baseline` the decision rules agree to about 2e-9, not 1e-12.

## 4. Dynare 7.2 defects found

pydyn reproduces these unless noted. Each port's report lists them in more detail.

**Steady state**
- `dynare_solve.m` (solve_algo 2/4) uses row indices for the columns of later blocks on structurally singular systems.

**Smoother and posterior statistics**
- Diffuse smoother `kalman_algo=3`: wrong result in a corner case.
- `pm3.m`: `FilteredVariablesKStepAheadVariances` holds the means.
- `pm3.m`: `FilteredVariables` is right only when `filter_step_ahead(1)=1`.

**Estimation**
- `check_for_calibrated_covariances` drops calibrated structural covariances when no correlation is estimated.
- `estimated_params` bounds apply during MCMC, despite the manual.
- `mcmc_jumping_covariance` files are inverted, so they are read as a Hessian.
- DSGE-VAR counts the lag observations in the sample size.

**Forecasts**
- Forecast HPD bands are not shifted by observation trends.
- Posterior measurement-error draws use the upper Cholesky factor, which is wrong when measurement errors are correlated.

**Particle filters**
- "Stratified" resampling gives most particles the same index.
- Eight further defects in the nonlinear filters; see `pydyn.nonlinear_filters.DYNARE_QUIRKS`.

**Optimal policy**
- `evaluate_planner_objective` at order 2 may order shocks and states wrongly when there are several of each. All of Dynare's tests have one shock, so this is unconfirmed.

**Epilogue**
- Results are rounded to 6 decimals.
- Leads leave the variable NaN.

**Dynare's own test suite**
- Several test files call functions absent from 7.2.
- Some files are missing from `meson.build`.
