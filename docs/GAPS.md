# pydyn vs Dynare 7.2: what is and is not supported

Audit of the snapshot `/home/claude/audit/snap` (copy of `/home/claude/pydyn` taken at the start; nothing in the original was touched). Date 2026-11-20. Test suite on the snapshot: 654 passed, 2 skipped (`test_first_order.py:72`, "no IRFs in reference").
Method: read README and all modules; read the manual (`the-model-file.rst`, `time-series.rst`, `reporting.rst`, `dynare-misc-commands.rst`); ran several hundred probe `.mod` files through `pydyn.run_preprocessor` + `pydyn.Model` and the relevant functions; ran real Dynare 7.2 under Octave for the four cases marked "(Dynare/Octave)". Probe scripts: `/home/claude/audit/probes/*.py` (names quoted as "probe hNN"), Octave runs in `/home/claude/audit/oct/`.

## Status legend

Command table (exact phrases requested):
- **V** = supported and validated against Dynare (a test in `tests/` compares with a Dynare 7.2 reference `.mat`)
- **S** = supported, not validated
- **P** = partly supported
- **IS** = parsed and ignored silently
- **PW** = parsed and warned (one `UserWarning: statements not handled: [...]` emitted at `Model()` load; nothing at call time; the test files filter this warning out)
- **ER** = raises an error
- **NA** / n/a = not applicable in Python

Option tables: HV honoured and validated; H honoured, not validated; KW honoured only as a Python keyword (the `.mod` value is silently ignored); IGN0 silently ignored with no effect on numbers; **IGNN silently ignored and can change numbers or drops an output**; ERR raises; PRE handled by the preprocessor; NA display/IO flag; P partly.

## Architecture facts that explain most gaps

- `Model()` stores only these statements' options in `m.options`: steady, check, occbin_setup, occbin_graph, rplot, stoch_simul, estimation, occbin_solver, estimated_params*, varobs, perfect_foresight_setup/solver, calib_smoother, shock_decomposition, initial_condition_decomposition, shock_groups, observation_trends. Every other statement name is appended to `unhandled_statements` and warned once.
- Only five places read `.mod` options: `moments._options` (stoch_simul), `Occbin.mod_options` (simul_* only), `Estimation.__init__` (keys in `_DEFAULTS`; everything else goes to `est.ignored_options`, **never reported**), `smoother._resolve_options` (keys in its `_DEFAULTS`; all others silently dropped) and `perfect_foresight` (only `periods`). `steady_state`, `first_order`, `k_order`, `simulate`, `irf`, `plotting.*` read no `.mod` option at all.
- Dynare 7 default `stoch_simul` order is 2 (manual, `order = INTEGER`: "Default: 2"); the preprocessor writes `order=2` into `m.options['stoch_simul']` when omitted. `first_order()` ignores it.
- The 7.2 preprocessor writes invalid JSON for `shock_paths`, `model_comparison` and `shock_decomposition(colormap=...)`; `Model()` repairs only `matched_irfs` and negative bounds, so those three raise `JSONDecodeError` at load (probes cmds2, h5).

## 1. Command-level table (every command/block of the "The model file" chapter plus time-series and reporting)

#### Variable declarations

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| var (plain) | V | Model.endo_names | every test |  |
| var(log), var(deflator=), var(log_deflator=), long_name, TeX names | S | preprocessor + Model | probe h14: var(log) steady+first_order OK; deflator forms load | not compared with Dynare numbers; pydyn never reads deflators (relies on the preprocessor stationarisation) |
| varexo | V | Model.exo_names | every test |  |
| varexo_det | ER | Model.__init__ | probe cmds2: NotImplementedError "deterministic exogenous variables are not supported" at load | clear error, no support |
| parameters / parameter initialisation (incl. MATLAB expressions) | V | Model.params; pydyn.matlab | test_first_order, test_matlab (156-file corpus), test_stage3 param loop | MATLAB subset only (see README) |
| var_remove | S | preprocessor | probe h18: loads, variable removed | not validated |
| predetermined_variables | S | preprocessor + dr_ordering | probe vs Dynare/Octave (my run): ghx/ghu identical to 4 digits on a k(+1) RBC; no repo test | no repo test |
| trend_var, log_trend_var | S | preprocessor | examples/aguiar_gopinath_2007_trend loads and first_order solves (probe ex.py); README sweep claims match, not in tests/ | no test in tests/ |
| model_local_variable | S | Model._locals | probe cmds: loads; # definitions V in test_local_variables_equal_hand_expansion |  |
| external_function | ER | Model._build | probe h16: NotImplementedError "external functions are not supported" when the function is used in the model (declaration alone is accepted) | Model(functions=) only serves steady_state_model, not the model block |
| Expressions: + - * / ^, comparisons, unary minus, parameters in expressions | V | expr.Translator | test_review::test_comparisons_are_numeric, test_chained_comparisons |  |
| Functions exp log ln log10 sqrt cbrt sign abs sin..atanh max min normcdf normpdf erf erfc | S | expr.NAMESPACE | probe cmds: all evaluated through steady+first_order; max/min derivatives V (test_max_min_derivatives) | values of the other functions not compared with Dynare |
| Constants inf, nan | S | expr | not checked | not checked |
| STEADY_STATE operator | S | Model._steady_ref | probe h15: steady+first_order OK; aux tests | not validated |
| EXPECTATION operator | S | preprocessor (aux vars) | probe h15: steady+first_order OK | not validated |
| diff(), adl() operators; leads/lags >1; lagged/led exogenous | V | preprocessor aux vars; Model.aux_names | test_auxiliary (lags/leads of endogenous and exogenous); diff probe loads | adl not probed |
| M_.params, get_param_by_name, set_param_value, get/set_shock_stderr | V | pydyn.matlab, Model.set_param(s), set_shock_stderr | test_matlab, test_stage3::test_parameter_loop_without_reloading (vs Dynare) | skewness getters/setters not checked |

#### Model declaration

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| model; (equations) | V | Model._compile | all tests |  |
| model options: linear, use_dll, block, bytecode, cutoff, mfs, static_mfs, no_static, differentiate_forward_vars, parallel_local_files, balanced_growth_test_tol | S | preprocessor | probe h11: decision rules identical to default for every option; linear V at order 2 (test_review) | see option table; differentiate_forward_vars changes n_endo, not compared with Dynare |
| equation tags: name, bind/relax (occbin), static/dynamic | V | Model, Occbin | occbin tests; probe: [name=], [static]/[dynamic] load and solve | static/dynamic tag not validated |
| equation tag mcp (lmmcp) | IS | - | probe h2: tag kept in JSON, never read; PF with lmmcp returns a path with r=-1.32 violating r>0 | no mcp/lmmcp solver for PF, EP or stoch |
| model_options (statement) | S | preprocessor | probe: loads |  |
| model_remove, model_replace | S | preprocessor | probe cmds2/h18: loads, equation set updated | not validated |
| write_latex_original_model, _dynamic_, _static_, _steady_state_model | IS | Model._exec_statement (pass) | probe: no warning, no file | no LaTeX output |
| write_latex_definitions / parameter_table / prior_table / collect_latex_files | IS | skipped (native / pass) | probe: accepted, nothing written | no LaTeX output |
| Auxiliary variables (AUX_ENDO_LEAD/LAG, AUX_EXO_*, MULT_*, LOG_*) | V | Model.aux_names, steady.fill_auxiliary | test_auxiliary | Lagrange-multiplier aux (Ramsey) mishandled: fill_auxiliary gives non-finite start (see Ramsey) |

#### Initial and terminal conditions

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| initval | V | Model.initval | steady-state tests, PF example |  |
| endval | S | Model.endval, perfect_foresight | my probe vs Dynare/Octave: max/diff/ 3.3e-5 on a 60-period permanent-shock RBC (Dynare tolf 1e-5); no repo test (README says endval is covered, PF example has none) | repo test missing |
| histval | PW | - | probe: warning "statements not handled: [histval]" | initial lags for PF / simulation not supported (Dynare: matlab/histvalf.m) |
| initval_file, histval_file, smoother2histval | PW | - | probe: warned | Dynare: matlab/histvalf_initvalf.m, smoother2histval.m |
| resid | PW | Model.static_resid exists (building block) | probe: warned | no printing command |
| load_params_and_steady_state | S | Model._exec_statement | probe: params and initval filled from file | not validated |
| save_params_and_steady_state | PW | - | probe: warned |  |

#### Shocks

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| shocks: var / stderr / variance / corr / covariance (var a,b = x) | V | Model._shocks | test_review::test_shock_blocks_follow_dynare_order, estimation/smoother tests |  |
| shocks: periods/values (deterministic) | S | Model.det_shocks, exo_path | PF example (one block, one period) V; probe with several blocks loads | overlapping/ranges not validated |
| shocks(overwrite) | S | Model._shocks | handled in code | not validated |
| shocks(surprise), shocks(surprise,overwrite) | V | Occbin.surprise_shocks | occbin tests | several surprise blocks are merged with a warning (overwrite flag missing from JSON) |
| shocks(learnt_in=) | ER | preprocessor | preprocessor refuses unless pfwee setup present | pfwee unsupported |
| shocks: skew / coskewness | PW | - | probe: "shocks: skewness" warning |  |
| shocks: measurement error variances (var obs; stderr x) | V | Model.measurement_errors | est_nk, smo_me tests |  |
| shocks: measurement-error covariances / correlations | PW | - | probe: "shocks: measurement-error covariance" warning; Estimation warns again |  |
| mshocks, mshocks(learnt_in), relative_to_initval | PW | - | probe: warned | Dynare: perfect-foresight-models/perfect_foresight_setup.m |
| heteroskedastic_shocks | PW | - | probe: warned at load; Estimation ignores it (likelihood unchanged); smoother raises NotImplementedError | Dynare: matlab/kalman/get_Qvec_heteroskedastic_filter.m |
| shock_paths block | ER | Model.__init__ | probe: JSONDecodeError at load (7.2 preprocessor writes invalid JSON: "period1" :1, "period2" :2") | JSON repair missing |
| special options (shocks: periods / stderr scale ...) | n/a | - | no separate "special options" command in 7.2 manual; covered by shocks rows |  |
| get_shock_stderr_by_name, set_shock_stderr_value | S | pydyn.matlab; Model.set_shock_stderr | test_matlab |  |
| get/set_shock_skew* | n/a | - | not checked | not checked |
| dsample | PW | - | probe: warned | irrelevant |

#### Steady state and diagnostics

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| steady (+ options) | V | pydyn.steady_state | test_first_order::test_steady_state, test_matlab nk_baseline, corpus; analytic steady_state_model V | most options ignored: see option table (solve_algo, tolf, maxit, homotopy) |
| homotopy_setup + steady(homotopy_mode=) | P | Model._exec_statement | code + probe: parameters moved to final values, no stepping | no homotopy stepping |
| steady_state_model | V | steady._from_steady_state_model | test_estimation, nk tests, test_matlab | multi-output [a,b]=f() handled (test_helper_with_several_outputs) |
| <model>_steadystate.m files | V | steady.steady_state_file_fn, matlab.SteadyStateFile | test_matlab::test_nk_baseline_with_its_matlab_steady_state_file |  |
| get_mean | PW | - | probe: skipped native with warning |  |
| check | P | first_order (BlanchardKahnError); FirstOrderSolution.eigenvalues | BK V (large-model test compares max stable root); eigenvalue list not compared to oo_.dr.eigval | eigenvalues are of the 2n pencil, not the reduced system; options ignored |
| model_diagnostics | PW | - | probe: warned | Dynare: matlab/model_diagnostics.m |
| model_info (block_static, block_dynamic, incidence) | PW | - | probe: warned |  |
| print_bytecode_dynamic_model / static_model | n/a | - | probe: skipped native, warned | no bytecode in Python |

#### Deterministic simulation

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| perfect_foresight_setup / perfect_foresight_solver | S | pydyn.perfect_foresight | test_auxiliary::test_perfect_foresight_example (Dynare perfect_foresight_rbc, initval + shock); my endval probe 3.3e-5 | validated on ONE repo example; most options ignored (option table); no mcp/lmmcp, no homotopy, no histval, no dates |
| simul (old name) | PW | - | probe: warned |  |
| perfect_foresight_with_expectation_errors_setup/_solver | PW | - | probe: warned; shipped example loads with warning | Dynare: perfect-foresight-models/perfect_foresight_with_expectation_errors_*.m |
| endval(learnt_in=), mshocks(learnt_in=) | ER | preprocessor | - |  |
| perfect_foresight_controlled_paths (exogenize/endogenize) | PW | - | probe: warned, solver ignores it (would solve the uncontrolled problem if the warning is missed) | Dynare: controlled_paths_*.m |
| extended_path (deterministic/stochastic EP) | PW | - | probe: warned | Dynare: matlab/ep/ |
| database | n/a | - | preprocessor emits nothing |  |

#### Stochastic solution

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| stoch_simul (order 1, orders 2-3, moments, IRFs) | V | first_order, k_order, simulate, irf, moments.stoch_simul_moments | test_first_order, test_korder, test_stage3, test_moments (decision rules 1e-9, moments 1e-9, order-2 IRFs with common draws) | options: see table; order-3 IRFs not validated; no oo_.irfs / graph driver / relative_irf / spectral_density / partial_information; theoretical moments only to order 2 |
| Orders >3 | ER | k_order | KOrderError / NotImplementedError | Dynare also stops at 3 for the k-order solver |
| get_irf | n/a | pydyn.irf / FirstOrderSolution.irf |  | no oo_.irfs dictionary |
| extended_path (stochastic) | PW | - | probe |  |

#### OccBin

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| occbin_constraints | V | Model.occbin_constraints, Occbin | rbc_occbin (2 constraints), borrcon (1) | lags/leads inside constraints raise NotImplementedError; >2 constraints unsupported (as Dynare) |
| shocks(surprise) | V | Occbin.surprise_shocks / run | tests |  |
| occbin_setup (simul_*) | V | Occbin.mod_options | test_*_options_from_mod |  |
| occbin_setup (smoother_*, likelihood_*, filter_*, particle, posterior_importance_sampling) | IS | - | probe h6: 22 options silently ignored; Estimation/smoother on an OccBin model silently run the linear reference regime (probe h7) | Dynare: matlab/+occbin/{DSGE_smoother,IVF_core,IVF_posterior,kalman_update_engine,pkf_conditional_density,posterior_importance_sampling}.m |
| occbin_solver | V | Occbin.simulate/run | tests | simul_debug no effect |
| occbin_graph | P | plotting.plot_occbin | test_plotting (data and layout) | noconstant option not checked; no pixel comparison with Dynare |
| occbin_write_regimes | PW | - | probe: statement occbin_write_regimes_xls warned | Dynare: +occbin/write_regimes_to_xls.m |
| OccBin IRFs (occbin.irf) | V | Occbin.irf_occbin, Occbin.irf | test_irf_occbin_*, test_existing_irf_method_matches_dynare |  |
| OccBin shock decomposition, squeeze_shock_decomposition, forecast | n/a | - | not implemented | Dynare: +occbin/shock_decomposition.m, squeeze_shock_decomposition.m, forecast.m |

#### Estimation

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| varobs | V | Model.varobs | estimation/smoother tests |  |
| observation_trends | P | Estimation._setup_trends (V in est_ur) | test_estimation est_ur | smoother.smooth raises NotImplementedError for it |
| estimated_params (single block; stderr, corr, param; all 6 prior families; ML) | V | estimation.Estimation | test_estimation (likelihood, priors vs Dynare), probe h4 | several blocks / overwrite raise |
| estimated_params_init | P | Estimation._apply_init_blocks | probe h4: values accepted | (use_calibration) raises NotImplementedError |
| estimated_params_bounds, estimated_params_remove | ER | Estimation._setup_parameters | probe: NotImplementedError (remove also warned at load) |  |
| estimation (likelihood, posterior mode, Laplace, RWMH) | P | Estimation.log_likelihood/log_posterior/mode/hessian/laplace/sample | likelihood V; mode V (schorfheide); target density V; sampler algorithm vs reimplementation | see option table; first order only; no diagnostics, MDD (MHM), posterior statistics, posterior IRFs/smoother, forecasts |
| model_comparison | ER | Model.__init__ | probe: JSONDecodeError at load (invalid JSON for the file list) | no function either; Dynare: estimation/model_comparison.m |
| data (block) | PW | - | probe: statement estimation_data warned |  |
| filter_initial_state | PW | - | probe: warned at load; Estimation likelihood unchanged; smoother raises | Dynare: get_init_state.m |
| shock_decomposition | V | smoother.shock_decomposition | test_shock_decomposition_matches_dynare (7 cases incl. loglinear, diffuse, ME, init_state) | calibrated parameters only |
| shock_groups / use_shock_groups | P | smoother.group_shocks | test_group_shocks (internal consistency only) | not validated vs Dynare; use_shock_groups option silently ignored by shock_decomposition(); Dynare: shock_decomposition/expand_group.m |
| realtime_shock_decomposition, plot_shock_decomposition, squeeze_shock_decomposition, init2shocks, annualized decompositions | PW | - | probe: warned | Dynare: matlab/shock_decomposition/*.m |
| initial_condition_decomposition | V | smoother.initial_condition_decomposition | test_initial_condition_decomposition_matches_dynare | no plotting |
| calib_smoother | V | smoother.smooth | test_smoother (14 cases, filtered, k-step, covariances, diffuse, missing data) | calibrated parameters only |
| unit_root_vars | n/a | - | not in the 7.2 manual or preprocessor (removed) |  |
| bvar_density, bvar_forecast, bvar_irf, data | PW | - | probe: warned | Dynare: matlab/+bvar |
| dsge_var (estimation option), dsge_prior_weight | IS | - | probe h3/h12: ignored silently; likelihood equals plain DSGE | Dynare: dsge_var_likelihood.m |
| evaluate_planner_objective | PW | - | probe: warned |  |
| mh options, generate_trace_plots, internals | IS | - | see estimation option table | MATLAB utilities; no diagnostics |
| method_of_moments, matched_moments, matched_irfs, matched_irfs_weights | PW | - | probe: warned; rbc_irf_matching example loads | Dynare: matlab/+mom |
| dynare_sensitivity / sensitivity | PW | - | sensitivity warned; "dynare_sensitivity" is not a 7.2 command | Dynare: matlab/+gsa |
| identification | PW | - | probe: warned | Dynare: matlab/+identification |
| irf_calibration, moment_calibration | PW | - | probe: warned |  |
| prior_function, posterior_function | PW | - | probe: statement prior_posterior_function warned |  |

#### Forecasting

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| forecast | PW | - | probe: warned | Dynare: matlab/+forecasts/*.m |
| conditional_forecast, conditional_forecast_paths, plot_conditional_forecast | PW | - | probe: warned | Dynare: matlab/+conditional_forecasts/*.m |
| bvar_forecast | PW | - | probe: warned |  |

#### Optimal policy

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| planner_objective, ramsey_model, ramsey_constraints, ramsey_policy | PW | (Model loads the preprocessor FOC model) | probe t_ramsey: warned; steady_state() default raises "initial guess ... not finite" (fill_auxiliary on MULT_*); with explicit guess=zeros first_order returns eigenvalues (unvalidated); nk_ramsey_osr example: SteadyStateError | Ramsey steady state, multipliers initialisation, planner objective evaluation; Dynare: matlab/optimal_policy/* |
| discretionary_policy | PW | - | probe: warned | Dynare: matlab/discretionary_policy/ |
| osr, osr_params, optim_weights, osr_params_bounds | PW | - | probe: warned | Dynare: matlab/+osr/ |

#### Markov-switching SBVAR

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| markov_switching, svar, sbvar, svar_identification, ms_estimation, ms_simulation, ms_compute_mdd, ms_compute_probabilities, ms_irf, ms_forecast, ms_variance_decomposition | PW | - | probes: all warned (statement names ms_sbvar_*) | Dynare: matlab/ms-sbvar |

#### Epilogue, semi-structural, heterogeneity

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| epilogue | IS | - | probe: not in modfile.json (no warning, nothing executed) | epilogue variables never computed |
| var_model | PW | - | probe: warned |  |
| trend_component_model, var_expectation_model, var_expectation, pac_model, pac_target_info, pac_expectation, pac_target_nonstationary | PW | - | pac_model example: loads only if pac.initialize lines are removed (then warned); with them: ModelError at load (var_expectation.initialize, pac.initialize) | Dynare: matlab/+pac, +var_expectation, pac-tools |
| heterogeneity_dimension, var(heterogeneity=), model(heterogeneity=), shocks(heterogeneity=), heterogeneity_* commands, rouwenhorst | PW | - | six shipped examples load with warnings; any pydyn function then fails with NameError (misleading) | Dynare: matlab/+heterogeneity |

#### Displaying and saving results

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| rplot | IS | options["rplot"] recorded only | probe |  |
| dynatype, dynasave | PW | - | probe: warned |  |
| Figures: IRF panels, OccBin graph | P | plotting.plot_irf, plot_occbin, save | test_plotting: data drawn and Dynare layout rule (pltorg) checked; styles "pydyn"/"dynare" | no decomposition, smoother, prior/posterior, forecast, mode_check, trace plots; no pixel comparison with Dynare |

#### Macro-processing language

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| @#define @#if @#for @#include @#echo @#error @#includepath, -D variables | V | preprocessor (run_preprocessor(extra_args=)) | est_nk.mod uses @#if/-DVARIANT in tests; macro_for/macro_if probes; bkk_1992 loads+solves | preprocessor is Dynare's own |

#### verbatim and misc commands

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| verbatim | S | pydyn.matlab ModSession | test_matlab | MATLAB subset only |
| Native MATLAB lines | V | pydyn.matlab | test_matlab, 156-file corpus (parameters and Sigma_e vs Dynare) | 41 of 212 corpus files refused, 15 out of scope (README, tools/sweep_matlab.py) |
| set_dynare_seed | IS | pydyn.matlab (no-op) | probe | random streams cannot match MATLAB anyway |
| compilation_setup | IS | - | probe: not in JSON |  |
| dynare_version | IS | - | probe |  |
| print_bytecode_* | n/a | - | see above |  |
| send_endogenous_variables_to_workspace, clean_current_folder, search, dplot, prior, set_dynare_threads, internals, generate_trace_plots | NA | - | (dynare-misc-commands.rst: MATLAB workspace utilities) |  |

#### Time series and reporting chapters

| Dynare command | Status | pydyn function | Evidence | Missing / note |
|---|---|---|---|---|
| dseries / dates / time-series chapter | NA | numpy arrays; estimation.load_data / smoother.load_data (.m, .mat, .csv) | probe: date options in perfect_foresight_setup raise ValueError | no dseries; no date arithmetic; no .xls; first_obs/nobs as dates unsupported |
| reporting chapter (report, page, graph, table ...) | NA | - | collard_2001_reporting example loads, 48 MATLAB statements skipped with warning | use matplotlib/pandas |

## 2. Option-level tables

Counts follow the manual's `.. option::` entries (duplicates merged). Probe ids refer to `/home/claude/audit/probes/`.

### Options of `model`

| Option (manual) | Status | Note |
|---|---|---|
| `linear` | PRE | preprocessor emits first derivatives only; pydyn solves it, order-2 on a linear model validated (test_review::test_linear_model_at_second_order) |
| `use_dll` | PRE | no effect; probe: identical decision rules |
| `block` | PRE | no effect; probe: identical decision rules (JSON is not block-ordered, pydyn does its own solve) |
| `bytecode` | PRE | no effect; probe identical |
| `cutoff = DOUBLE` | PRE | applied by the preprocessor to the derivatives; pydyn uses what it gets |
| `mfs = INTEGER` | PRE | no effect; probe identical |
| `static_mfs` | PRE | no effect; probe identical |
| `no_static` | PRE | no effect; probe identical |
| `differentiate_forward_vars` | PRE | adds auxiliary variables (n_endo changes); solves; not validated against Dynare numbers |
| `parallel_local_files = ( FILENAME [, FILENAME]... )` | IGN0 | preprocessor rejected my probe syntax; irrelevant to pydyn |
| `balanced_growth_test_tol = DOUBLE` | PRE | preprocessor only |

### Options of `steady`

| Option (manual) | Status | Note |
|---|---|---|
| `maxit = INTEGER` | IGNN | steady_state() has fixed solver loops; probe: steady(maxit=1) still solves |
| `tolf = DOUBLE` | IGNN | pydyn checks max|residual|<1e-10 (keyword tol); Dynare default eps^(1/3)=6e-6. Probe: steady_state_model residual 1e-7 is accepted by Dynare, raises SteadyStateError in pydyn |
| `tolx = DOUBLE` | IGN0 |  |
| `non_zero` | IGN0 |  |
| `solve_algo = INTEGER` | IGNN | always scipy hybr + Newton polish from initval; Dynare algorithms 0-14 (incl. block/trust-region) not reproduced; a model Dynare solves with solve_algo=4 or block may fail here |
| `homotopy_mode = INTEGER` | P | Model._exec_statement moves homotopy_setup parameters straight to their final values, then solves from initval: no stepping |
| `homotopy_steps = INTEGER` | IGNN | no stepping at all |
| `homotopy_force_continue = INTEGER` | IGN0 |  |
| `nocheck` | IGNN | ignored; the 1e-10 check is always on (see tolf) |
| `noprint` | NA |  |
| `markowitz = DOUBLE` | IGN0 |  |
| `fsolve_options = (NAME, VALUE, ...)` | IGN0 |  |

### Options of `check`

| Option (manual) | Status | Note |
|---|---|---|
| `solve_algo = INTEGER` | IGN0 | steady-state solver choice |
| `qz_zero_threshold = DOUBLE` | IGN0 | no such threshold in pydyn |

### Options of `stoch_simul`

| Option (manual) | Status | Note |
|---|---|---|
| `ar = INTEGER` | HV | moments.ar (test_moments th/th1_hp ar=2..5) |
| `drop = INTEGER` | HV | simulated moments (s1_*, s2p*); irf() has its own drop keyword (100) |
| `hp_filter = DOUBLE` | HV | theoretical (order 1) and simulated; th1_hp, s1_hp |
| `one_sided_hp_filter = DOUBLE` | HV | simulated moments only (s1_os); theoretical raises ValueError as in Dynare |
| `bandpass_filter` | HV | th1_bp, s1_bp; read from .mod keys bandpass.indicator/passband |
| `filtered_theoretical_moments_grid = INTEGER` | HV | th1_hp128 |
| `irf = INTEGER` | KW | IRF length is the periods= keyword of pydyn.irf / FirstOrderSolution.irf; .mod value ignored (length only) |
| `irf_shocks = ( VARIABLE_NAME [[,] VARIABLE_NAME ...] )` | IGN0 | no function loops over shocks; caller passes impulse vectors. Option silently ignored |
| `relative_irf` | IGNN | NOT IMPLEMENTED anywhere (grep); IRF size is whatever impulse the caller passes; no warning |
| `irf_plot_threshold = DOUBLE` | IGN0 | not implemented in plotting.plot_irf |
| `nocorr` | HV | moments |
| `nodecomposition` | HV | moments |
| `nofunctions` | NA |  |
| `nomoments` | NA | stoch_simul_moments always computes; Moments.format() prints |
| `nomodelsummary` | NA |  |
| `nograph` | NA | plotting is explicit (plotting.plot_irf) |
| `graph` | NA |  |
| `nodisplay` | NA |  |
| `graph_format = FORMAT` | NA | plotting.save(fmt=...) |
| `noprint` | NA |  |
| `print` | NA |  |
| `order = INTEGER` | HV | moments (1, 2 theoretical; 3 only simulated -> NotImplementedError otherwise); k_order(order=) keyword. NOT read by first_order/simulate/irf. NB Dynare 7 default is 2 and the preprocessor writes order=2 into m.options when omitted |
| `k_order_solver` | IGN0 | orders >=2 always use the C++ k-order library, order 1 always QZ |
| `periods = INTEGER` | HV | moments: simulated vs theoretical |
| `qz_criterium = DOUBLE` | IGNN | used only for the Lyapunov/unit-root test inside moments (validated); NOT for the QZ solve (fixed 1+1e-6 in first_order and k_order). Probe vs Dynare: x=1.02x(-1)+e, qz_criterium=1.05: Dynare solves (ghx=1.02), pydyn raises BlanchardKahnError |
| `qz_zero_threshold = DOUBLE` | IGN0 | no equivalent |
| `replic = INTEGER` | KW | irf(replic=50) keyword, default equals Dynare; .mod value ignored by stoch_simul_moments (no warning) |
| `simul_replic = INTEGER` | IGNN | probe: identical result with simul_replic=3; Dynare averages moments over replications |
| `solve_algo = INTEGER` | IGN0 | steady-state solver choice |
| `aim_solver` | IGN0 | QZ always; same solution whenever BK holds |
| `conditional_variance_decomposition = INTEGER` | HV | th1/th2 cvd cases |
| `pruning` | HV | moments (th2p, s2p). NOT read by pydyn.simulate / pydyn.irf (default pruning=False). Probe: stoch_simul(order=2,pruning) + simulate(dr,shocks) differs from pruned by up to 9e-4 |
| `partial_information` | IGNN | ignored; full-information solution is returned (code review: no reference anywhere in pydyn); not probed numerically |
| `dr = OPTION` | IGN0 | default/cycle_reduction/logarithmic_reduction all run QZ; same solution whenever it exists |
| `dr_cycle_reduction_tol = DOUBLE` | IGN0 |  |
| `dr_cycle_reduction_maxiter = INTEGER` | IGN0 |  |
| `dr_logarithmic_reduction_tol = DOUBLE` | IGN0 |  |
| `dr_logarithmic_reduction_maxiter = INTEGER` | IGN0 |  |
| `loglinear` | IGNN | VERIFIED vs Dynare/Octave: RBC with positive steady state, std(y,c,k): Dynare 0.04288,0.03374,0.04434 (loglinear) vs pydyn 0.12929,0.07782,1.25707 (levels), no warning |
| `tex` | NA |  |
| `dr_display_tol = DOUBLE` | NA |  |
| `contemporaneous_correlation` | HV | moments |
| `spectral_density` | IGN0 | not implemented (oo_.SpectralDensity absent); silently ignored |

### Options of `perfect_foresight_setup`

| Option (manual) | Status | Note |
|---|---|---|
| `periods = INTEGER` | H | read from .mod by perfect_foresight(); keyword periods= too; validated on one example (perfect_foresight_rbc) + my endval probe |
| `first_simulation_period = DATE` | ERR | periods then "not given" ValueError; dated shocks raise ValueError (int("dates(...)")) |
| `last_simulation_period = DATE` | ERR | as above |
| `datafile = FILENAME` | IGNN | silently ignored (initval_file equivalent not implemented); probe ran with the steady-state path |
| `endval_steady` | IGN0 | pydyn always recomputes the terminal steady state when endval is given |
| `endval_steady_nocheck` | IGN0 |  |

### Options of `perfect_foresight_solver`

| Option (manual) | Status | Note |
|---|---|---|
| `maxit = INTEGER` | KW | maxit=50 keyword (= Dynare default); .mod value ignored |
| `tolf = DOUBLE` | KW | tol=1e-9 keyword (Dynare default 1e-5); .mod value ignored |
| `tolx = DOUBLE` | IGN0 | only the residual is tested |
| `noprint` | NA |  |
| `print` | NA |  |
| `stack_solve_algo = INTEGER` | IGN0 | always sparse Newton on the stacked system (SuiteSparse spsolve) |
| `preconditioner = OPTION` | IGN0 |  |
| `iter_tol = DOUBLE` | IGN0 |  |
| `iter_maxit = INTEGER` | IGN0 |  |
| `gmres_restart = INTEGER` | IGN0 |  |
| `block_diagonal_lu_maxlu = INTEGER` | IGN0 |  |
| `block_diagonal_lu_nperiods = INTEGER` | IGN0 |  |
| `block_diagonal_lu_nlu = INTEGER` | IGN0 |  |
| `block_diagonal_lu_relu = DOUBLE` | IGN0 |  |
| `robust_lin_solve` | IGN0 |  |
| `allow_nonfinite_values` | IGN0 |  |
| `check_jacobian_singularity` | IGN0 |  |
| `solve_algo` | IGN0 |  |
| `no_homotopy` | IGN0 | pydyn has no homotopy fallback; non-convergence raises PerfectForesightError |
| `homotopy_initial_step_size = DOUBLE` | IGN0 |  |
| `homotopy_min_step_size = DOUBLE` | IGN0 |  |
| `homotopy_step_size_increase_success_count = INTEGER` | IGN0 |  |
| `homotopy_linearization_fallback` | IGN0 |  |
| `homotopy_marginal_linearization_fallback [= DOUBLE]` | IGN0 |  |
| `homotopy_max_completion_share = DOUBLE` | IGN0 |  |
| `homotopy_exclude_varexo = (VARIABLE_NAME...)` | IGN0 |  |
| `markowitz = DOUBLE` | IGN0 |  |
| `minimal_solving_periods = INTEGER` | IGN0 |  |
| `lmmcp` | IGNN | VERIFIED: model with [mcp='r > 0'] and perfect_foresight_solver(lmmcp): pydyn path has r = -1.32 (constraint violated), no warning. The mcp tag is parsed by the preprocessor and dropped by pydyn |
| `endogenous_terminal_period` | IGNN | fixed horizon always; Dynare would shorten the horizon after convergence |
| `linear_approximation` | IGNN | ignored; full nonlinear Newton always (same answer only for linear models) |
| `steady_solve_algo = INTEGER` | IGN0 |  |
| `steady_tolf = DOUBLE` | IGN0 |  |
| `steady_tolx = DOUBLE` | IGN0 |  |
| `steady_maxit = INTEGER` | IGN0 |  |
| `steady_markowitz = DOUBLE` | IGN0 |  |

### Options of `occbin_setup`

| Option (manual) | Status | Note |
|---|---|---|
| `simul_periods = INTEGER` | HV | Occbin.mod_options() -> simulate(periods=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_maxit = INTEGER` | HV | Occbin.mod_options() -> simulate(maxit=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_check_ahead_periods = INTEGER` | HV | Occbin.mod_options() -> simulate(check_ahead_periods=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_reset_check_ahead_periods` | HV | Occbin.mod_options() -> simulate(reset_check_ahead_periods_in_new_period=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_reset_regime_in_new_period` | HV | Occbin.mod_options() -> simulate(reset_regime_in_new_period=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_max_check_ahead_periods = INTEGER` | HV | Occbin.mod_options() -> simulate(max_check_ahead_periods=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_curb_retrench` | HV | Occbin.mod_options() -> simulate(curb_retrench=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_periodic_solution` | HV | Occbin.mod_options() -> simulate(periodic_solution=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_periodic_solution_threshold = INTEGER` | HV | Occbin.mod_options() -> simulate(periodic_solution_threshold=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_periodic_solution_strict` | HV | Occbin.mod_options() -> simulate(periodic_solution_strict=); tests test_*_options_from_mod, test_borrcon_*, test_rbc_* |
| `simul_debug` | IGN0 | accepted, no effect (debug output only) |
| `smoother_periods = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_maxit = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_check_ahead_periods = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_max_check_ahead_periods = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_curb_retrench` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_periodic_solution` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_inversion_filter` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_piecewise_kalman_filter` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_plot` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_first_period_occbin_update = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `smoother_max_number_of_iterations = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `likelihood_periods = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `likelihood_maxit = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `likelihood_check_ahead_periods = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `likelihood_curb_retrench` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `likelihood_periodic_solution` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `likelihood_inversion_filter` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `likelihood_piecewise_kalman_filter` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `likelihood_max_kalman_iterations` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_use_relaxation` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_draw_states_from_empirical_density` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_state_importance_sampling_pkf_init` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_state_importance_sampling_logpost_crit_threshold = DOUBLE` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_state_importance_sampling_slice_override_iteration = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_state_importance_sampling_slice_burnin = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_initial_state_ergodic_simul` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_diagnostics` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_diagnostics_graph_periods = INTEGER_VECTOR` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_diagnostics_nograph` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_number_of_particles  = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_number_of_shocks_per_particle = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_state_draws   = NUMERICAL_VECTOR` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `particle_filtering` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_particle_use_pkf_updated_state_threshold = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `filter_init_periods_using_particles` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `posterior_importance_sampling` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `posterior_importance_sampling_filter = QUOTED_STRING` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `posterior_importance_sampling_orig_filter = QUOTED_STRING` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `posterior_importance_sampling_sub_draws = INTEGER` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `posterior_importance_sampling_orig_dname = FILENAME` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |
| `posterior_importance_sampling_orig_fname = FILENAME` | IGNN | OccBin estimation/filter/smoother option: silently ignored; Estimation and smoother.smooth run the LINEAR reference-regime model (probe h7) |

### Options of `occbin_solver`

| Option (manual) | Status | Note |
|---|---|---|
| `simul_periods = INTEGER` | HV | same (periods) |
| `simul_maxit = INTEGER` | HV | same (maxit) |
| `simul_check_ahead_periods = INTEGER` | HV | same (check_ahead_periods) |
| `simul_reset_check_ahead_periods` | HV | same (reset_check_ahead_periods_in_new_period) |
| `simul_max_check_ahead_periods` | HV | same (max_check_ahead_periods) |
| `simul_curb_retrench` | HV | same (curb_retrench) |
| `simul_debug` | IGN0 | accepted, no effect (debug output only) |
| (`simul_reset_regime_in_new_period`, hidden/extra) | HV | same (reset_regime_in_new_period) |
| (`simul_periodic_solution`, hidden/extra) | HV | same (periodic_solution) |
| (`simul_periodic_solution_threshold`, hidden/extra) | HV | same (periodic_solution_threshold) |
| (`simul_periodic_solution_strict`, hidden/extra) | HV | same (periodic_solution_strict) |

### Options of `estimation`

| Option (manual) | Status | Note |
|---|---|---|
| `datafile = FILENAME` | HV | .m/.mat/.csv; relative to .mod dir (test_data_sources_agree) |
| `dirname = FILENAME` | NA |  |
| `xls_sheet = QUOTED_STRING` | IGN0 | .xls/.xlsx cannot be read |
| `xls_range = RANGE` | IGN0 | .xls/.xlsx cannot be read |
| `nobs = INTEGER` | HV | scalar (test variant 1); vector forms raise NotImplementedError (recursive estimation) |
| `first_obs = INTEGER` | HV | scalar (test variant 1); vector forms raise NotImplementedError (recursive estimation) |
| `prefilter = INTEGER` | HV | variant 3 |
| `presample = INTEGER` | HV | variant 1 |
| `loglinear` | HV | schorfheide model |
| `logdata` | HV | schorfheide model |
| `plot_priors = INTEGER` | NA |  |
| `nograph` | NA |  |
| `posterior_nograph` | NA |  |
| `posterior_graph` | NA |  |
| `nodisplay` | NA |  |
| `graph_format = FORMAT` | NA |  |
| `tex` | NA |  |
| `mode_compute = INTEGER | FUNCTION_NAME` | IGNN | listed in Estimation defaults but never read: mode() always runs SLSQP then L-BFGS-B ("auto"); a different optimum is possible; mode_compute=0 does not skip optimisation (caller just does not call mode()) |
| `mode_file = FILENAME` | IGN0 | silently recorded in ignored_options (cova_compute: Hessian computed by default) |
| `additional_optimizer_steps = [INTEGER]` | IGN0 | silently recorded in ignored_options (cova_compute: Hessian computed by default) |
| `optim = (NAME, VALUE, ...)` | IGN0 | silently recorded in ignored_options (cova_compute: Hessian computed by default) |
| `silent_optimizer` | IGN0 | silently recorded in ignored_options (cova_compute: Hessian computed by default) |
| `cova_compute = INTEGER` | IGN0 | silently recorded in ignored_options (cova_compute: Hessian computed by default) |
| `use_penalized_objective_for_hessian` | IGN0 | silently recorded in ignored_options (cova_compute: Hessian computed by default) |
| `analytic_derivation` | IGN0 | silently recorded in ignored_options (cova_compute: Hessian computed by default) |
| `mode_check` | IGN0 | no mode_check implementation |
| `mode_check_neighbourhood_size = DOUBLE` | IGN0 | no mode_check implementation |
| `mode_check_symmetric_plots = INTEGER` | IGN0 | no mode_check implementation |
| `mode_check_number_of_points = INTEGER` | IGN0 | no mode_check implementation |
| `no_init_estimation_check_first_obs` | IGN0 | own Lyapunov solver always; skewed filters not implemented |
| `lik_init = INTEGER` | HV | 1,2,3 validated (variants 0,4; ur model); 4 and 5 raise NotImplementedError |
| `conditional_likelihood` | IGNN | ignored: unconditional likelihood is returned |
| `kalman_algo = INTEGER` | HV | 0-4 (variants 0,8; ur) |
| `fast_kalman_filter` | IGN0 | own Lyapunov solver always; skewed filters not implemented |
| `kalman_tol = DOUBLE` | H | honoured; non-default values not validated |
| `diffuse_kalman_tol = DOUBLE` | H | honoured; non-default values not validated |
| `filter_covariance` | KW | available through pydyn.smoother.smooth(model, ...) at the current parameters (validated) or Estimation.smoother(theta); .mod flag silently ignored by Estimation; no posterior smoother |
| `filter_step_ahead = [INTEGER1:INTEGER2]` | KW | available through pydyn.smoother.smooth(model, ...) at the current parameters (validated) or Estimation.smoother(theta); .mod flag silently ignored by Estimation; no posterior smoother |
| `filter_decomposition` | IGN0 | ignored |
| `diffuse_filter` | HV | est_ur |
| `heteroskedastic_filter` | IGNN | ignored; heteroskedastic_shocks block warned at load and ignored (probe: likelihood unchanged) |
| `use_univariate_filters_if_singularity_is_detected = INTEGER` | H | honoured; 0 not validated |
| `use_univariate_smoother_if_singularity_is_detected` | IGN0 | ignored |
| `rescale_prediction_error_covariance` | H | honoured; not validated |
| `lyapunov = OPTION` | IGN0 | own Lyapunov solver always; skewed filters not implemented |
| `lyapunov_fixed_point_tol = DOUBLE` | IGN0 | own Lyapunov solver always; skewed filters not implemented |
| `lyapunov_doubling_tol = DOUBLE` | IGN0 | own Lyapunov solver always; skewed filters not implemented |
| `skewed_kalman_prune_tol = DOUBLE` | IGN0 | own Lyapunov solver always; skewed filters not implemented |
| `skewed_kalman_rank_deficiency_transform` | IGN0 | own Lyapunov solver always; skewed filters not implemented |
| `skewed_kalman_mvnlogcdf = QUOTED_STRING` | IGN0 | own Lyapunov solver always; skewed filters not implemented |
| `moments_varendo` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `ar = INTEGER` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `contemporaneous_correlation` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `no_posterior_kernel_density` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `conditional_variance_decomposition = INTEGER` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `filtered_vars` | KW | available through pydyn.smoother.smooth(model, ...) at the current parameters (validated) or Estimation.smoother(theta); .mod flag silently ignored by Estimation; no posterior smoother |
| `frequentist_smoother = BOOLEAN` | KW | available through pydyn.smoother.smooth(model, ...) at the current parameters (validated) or Estimation.smoother(theta); .mod flag silently ignored by Estimation; no posterior smoother |
| `smoother` | KW | available through pydyn.smoother.smooth(model, ...) at the current parameters (validated) or Estimation.smoother(theta); .mod flag silently ignored by Estimation; no posterior smoother |
| `smoother_redux` | KW | available through pydyn.smoother.smooth(model, ...) at the current parameters (validated) or Estimation.smoother(theta); .mod flag silently ignored by Estimation; no posterior smoother |
| `smoothed_state_uncertainty` | KW | available through pydyn.smoother.smooth(model, ...) at the current parameters (validated) or Estimation.smoother(theta); .mod flag silently ignored by Estimation; no posterior smoother |
| `forecast = INTEGER` | IGNN | no forecasts, no posterior IRFs: silently produces nothing |
| `conf_sig = DOUBLE` | IGNN | no forecasts, no posterior IRFs: silently produces nothing |
| `bayesian_irf` | IGNN | no forecasts, no posterior IRFs: silently produces nothing |
| `relative_irf` | IGNN | no forecasts, no posterior IRFs: silently produces nothing |
| `irf = INTEGER` | IGNN | no forecasts, no posterior IRFs: silently produces nothing |
| `irf_shocks = ( VARIABLE_NAME [[,] VARIABLE_NAME ...] )` | IGNN | no forecasts, no posterior IRFs: silently produces nothing |
| `irf_plot_threshold = DOUBLE` | IGNN | no forecasts, no posterior IRFs: silently produces nothing |
| `consider_all_endogenous` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `consider_all_endogenous_and_auxiliary` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `consider_only_observed` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `selected_variables_only` | IGN0 | no posterior moments; theoretical moments at a parameter vector are available via pydyn.moments after set_params |
| `solve_algo = INTEGER` | IGN0 | QZ only |
| `order = INTEGER` | ERR | order=1 only; order>=2 raises NotImplementedError |
| `dr = OPTION` | IGN0 | QZ only |
| `dr_cycle_reduction_tol = DOUBLE` | IGN0 | QZ only |
| `dr_cycle_reduction_maxiter = INTEGER` | IGN0 | QZ only |
| `dr_logarithmic_reduction_tol = DOUBLE` | IGN0 | QZ only |
| `dr_logarithmic_reduction_maxiter = INTEGER` | IGN0 | QZ only |
| `qz_zero_threshold = DOUBLE` | IGN0 | QZ only |
| `posterior_sampling_method = NAME` | IGNN | sample() is always random-walk Metropolis; slice/tarb/independent/DSMH are silently replaced |
| `posterior_sampler_options = (NAME, VALUE, ...)` | IGNN | sample() is always random-walk Metropolis; slice/tarb/independent/DSMH are silently replaced |
| `sub_draws = INTEGER` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `posterior_max_subsample_draws = INTEGER` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `prior_trunc = DOUBLE` | H | bounds checked against Dynare formula (test_prior_errors_and_bounds), not against a Dynare run |
| `huge_number = DOUBLE` | IGN0 | silently recorded in ignored_options (cova_compute: Hessian computed by default) |
| `endogenous_prior` | IGNN | ignored: prior is not augmented |
| `estimate_initial_states_endogenous_prior` | IGNN | ignored: prior is not augmented |
| `mh_conf_sig = DOUBLE` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mh_replic = INTEGER` | H | read by Estimation.sample(); target density validated vs Dynare chains (test_log_posterior_at_dynare_mcmc_draws); accept/reject rule validated vs an independent re-implementation only |
| `mh_nblocks = INTEGER` | H | read by Estimation.sample(); target density validated vs Dynare chains (test_log_posterior_at_dynare_mcmc_draws); accept/reject rule validated vs an independent re-implementation only |
| `mh_drop = DOUBLE` | H | read by Estimation.sample(); target density validated vs Dynare chains (test_log_posterior_at_dynare_mcmc_draws); accept/reject rule validated vs an independent re-implementation only |
| `mh_jscale = DOUBLE` | H | read by Estimation.sample(); target density validated vs Dynare chains (test_log_posterior_at_dynare_mcmc_draws); accept/reject rule validated vs an independent re-implementation only |
| `mh_init_scale = DOUBLE (deprecated)` | IGN0 | deprecated alias, mapped to nothing |
| `mh_init_scale_factor = DOUBLE` | H | read by Estimation.sample(); target density validated vs Dynare chains (test_log_posterior_at_dynare_mcmc_draws); accept/reject rule validated vs an independent re-implementation only |
| `mh_tune_jscale [= DOUBLE]` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mh_tune_guess = DOUBLE` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mh_recover` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mh_posterior_mode_estimation` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `load_mh_file` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `load_results_after_load_mh` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mh_initialize_from_previous_mcmc` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mh_initialize_from_previous_mcmc_directory = FILENAME` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mh_initialize_from_previous_mcmc_record = FILENAME` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mh_initialize_from_previous_mcmc_prior = FILENAME` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `mcmc_jumping_covariance = OPTION` | IGN0 | not implemented (no jscale calibration, no file-based chain management, no restart); silently ignored |
| `nodiagnostic` | IGN0 | no convergence diagnostics at all |
| `taper_steps = [INTEGER1 INTEGER2 ...]` | IGN0 | no convergence diagnostics at all |
| `brooks_gelman_plotrows = INTEGER` | IGN0 | no convergence diagnostics at all |
| `geweke_interval = [DOUBLE DOUBLE]` | IGN0 | no convergence diagnostics at all |
| `raftery_lewis_diagnostics` | IGN0 | no convergence diagnostics at all |
| `raftery_lewis_qrs = [DOUBLE DOUBLE DOUBLE]` | IGN0 | no convergence diagnostics at all |
| `dsge_var` | IGNN | VERIFIED: estimation(dsge_var=0.5) gives exactly the plain DSGE likelihood, no warning |
| `dsge_varlag = INTEGER` | IGNN | VERIFIED: estimation(dsge_var=0.5) gives exactly the plain DSGE likelihood, no warning |
| `pruning` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `number_of_particles = INTEGER` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `filter_algorithm = OPTION` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `resampling = OPTION` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `resampling_threshold = DOUBLE` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `resampling_method = OPTION` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `proposal_approximation = OPTION` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `distribution_approximation = OPTION` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `cpf_weights = OPTION` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `nonlinear_filter_initialization = INTEGER` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| `particle_filter_options = (NAME, VALUE, ...)` | IGN0 | nonlinear filters need order>=2, which raises; options silently ignored |
| (`qz_criterium`, hidden/extra) | H | honoured in Estimation (select_qz_criterium_value logic); non-default not validated |

### Options of `calib_smoother`

| Option (manual) | Status | Note |
|---|---|---|
| `datafile = FILENAME` | HV |  |
| `filtered_vars` | HV | FilteredVariables |
| `filter_step_ahead = [INTEGER1:INTEGER2]` | HV | kstep and variances |
| `prefilter = INTEGER` | HV | 0/1; error if inconsistent with non-zero steady state, as Dynare |
| `parameter_set = OPTION` | ERR | only calibration; posterior_mode/mean/median/... raise NotImplementedError (probe) |
| `loglinear` | HV | smo_fs case (loglinear) |
| `first_obs = INTEGER` | HV |  |
| `filter_decomposition` | IGN0 | silently dropped (not in _DEFAULTS); no function |
| `filter_covariance` | HV | Variance |
| `smoother_redux` | IGN0 | silently dropped; pydyn always uses the full state vector (same smoothed values) |
| `kalman_algo = INTEGER` | HV | 1-4, incl. diffuse and automatic univariate switch |
| `diffuse_filter = INTEGER` | HV | smo_ur |
| `diffuse_kalman_tol = DOUBLE` | HV |  |
| `use_univariate_smoother_if_singularity_is_detected` | IGNN | silently dropped; pydyn always switches to the univariate smoother (Dynare default); setting 0 (error in Dynare) is not honoured |
| `xls_sheet = QUOTED_STRING` | IGN0 | .xls/.xlsx data cannot be read (load_data accepts .m/.mat/.csv only) |
| `xls_range = RANGE` | IGN0 |  |
| `heteroskedastic_filter` | IGNN | silently dropped by calib_smoother option reader; heteroskedastic_shocks BLOCK raises NotImplementedError in smooth() |
| `nobs = INTEGER` | HV | scalar; vector (recursive) raises NotImplementedError |

### Options of `shock_decomposition`

| Option (manual) | Status | Note |
|---|---|---|
| `parameter_set = OPTION` | ERR | calibration only; others raise NotImplementedError |
| `datafile = FILENAME` | HV |  |
| `first_obs = INTEGER` | HV |  |
| `nobs = INTEGER` | HV |  |
| `prefilter = INTEGER` | HV |  |
| `loglinear` | HV |  |
| `diffuse_kalman_tol = DOUBLE` | HV |  |
| `diffuse_filter` | HV |  |
| `kalman_algo = INTEGER` | HV |  |
| `kalman_tol = DOUBLE` | H |  |
| `use_univariate_filters_if_singularity_is_detected` | IGNN | always switches (Dynare default); 0 not honoured |
| `xls_sheet = QUOTED_STRING` | IGN0 |  |
| `xls_range = RANGE` | IGN0 |  |
| `use_shock_groups [= NAME]` | IGNN | silently ignored by shock_decomposition(); user must call smoother.group_shocks(dec, model=m, name=...) (not validated vs Dynare) |
| `colormap = VARIABLE_NAME` | ERR | with a colormap name the 7.2 preprocessor writes invalid JSON and Model() raises JSONDecodeError (probe) |
| `nograph` | NA | no plotting for decompositions at all |
| `init_state = BOOLEAN` | HV | smo_fs case 2 (init_state=1) |
| `with_epilogue` | IGN0 | epilogue block is not in the JSON; no epilogue support |
| `forecast_type = unconditional | conditional` | IGNN | silently ignored; no forecast decomposition |
### Syntax elements of blocks without `.. option::` entries

| Block / element | Status | Note |
|---|---|---|
| `shocks(overwrite)` | H | `Model._shocks`; not validated |
| `shocks(surprise)`, `(surprise, overwrite)` | HV (occbin) | several surprise blocks merged with a warning |
| `shocks(learnt_in=)` | ER | preprocessor refuses without pfwee setup |
| `shocks`: `var x; stderr/variance`, `corr`, `var x, y = cov` | HV | |
| `shocks`: `periods`/`values` | H | PF example validated (single period) |
| `shocks`: `skew`, `coskewness` | WARN | |
| `mshocks(overwrite, relative_to_initval)` | WARN | |
| `heteroskedastic_shocks(overwrite)` | WARN at load; IGNN in Estimation; ERR in smoother | |
| `varobs` | HV | |
| `observation_trends` | HV in Estimation; ERR in smoother | |
| `estimated_params`: `stderr`, `corr`, parameter lines; 3-5 prior columns, bounds, jscale column | HV | one block only |
| `estimated_params`: `dsge_prior_weight` | ERR/IGNN | preprocessor refuses with dsge_var option; with `dsge_var` ignored |
| `estimated_params_init` values | H | `(use_calibration)` ERR |
| `estimated_params_bounds`, `estimated_params_remove`, second `estimated_params` block, `(overwrite)` | ERR | NotImplementedError |
| prior shapes beta, gamma, normal, inv_gamma(1,2), uniform, weibull | HV | `est_priors.mat` |

## Silent option hazards (ordered by how likely a user is to hit them)

"Verified" = I observed it (probe id, or Dynare/Octave run). None of these produces any warning or error from pydyn.

1. **`stoch_simul` defaults and `.mod` options not applied by the solve/simulate/IRF functions.** Dynare 7 default order is 2 and the preprocessor records it; `first_order()` always returns order 1; `simulate()`/`irf()` default `pruning=False` whatever the `.mod` says (only `moments.stoch_simul_moments` reads `order`/`pruning`). Verified (h9): `stoch_simul(order=2, pruning)` + `pydyn.simulate(dr2, shocks)` differs from the pruned path by up to 9e-4.
2. **`estimation(mode_compute=..., mode_file=, optim=, additional_optimizer_steps=, cova_compute=, mode_check, analytic_derivation)`**: `mode_compute` sits in `_DEFAULTS` but is never read; `Estimation.mode()` always runs SLSQP then L-BFGS-B (verified h12). Different optimiser, different local optimum possible. Others only land in `est.ignored_options`.
3. **`estimation(posterior_sampling_method=slice|tarb|independent|..., mh_tune_jscale, mh_conf_sig, load_mh_file, mh_recover, mh_initialize_from_previous_mcmc, mcmc_jumping_covariance, sub_draws, nodiagnostic, ...)`**: `Estimation.sample()` is always random-walk Metropolis with the inverse Hessian; none of the options is honoured or reported (h3).
4. **Post-estimation requests silently produce nothing**: `estimation(smoother, filtered_vars, filter_step_ahead, forecast=, bayesian_irf, moments_varendo, irf=, irf_shocks=, relative_irf, consider_*, selected_variables_only, conf_sig)`. The `.mod` flags are dropped; smoother outputs exist only via `smoother.smooth()` at the current parameters (h3).
5. **`loglinear` in `stoch_simul`** is ignored by `first_order`, `k_order`, `moments`, `simulate`, `irf` (honoured only by `Estimation` and `smoother`). Verified against Dynare/Octave: RBC, std(y,c,k) Dynare 0.04288/0.03374/0.04434, pydyn 0.12929/0.07782/1.25707 (levels). Dynare also errors when a steady state is not strictly positive; pydyn returns levels.
6. **`Estimation`/`smoother.smooth` on a model with `occbin_constraints`** (with or without `occbin_setup(likelihood_inversion_filter, smoother_*, ...)`): runs the linear reference-regime model as if there were no constraints and no warning (verified h7). All 22 `smoother_*`, `likelihood_*`, `filter_*`, `particle_filtering`, `posterior_importance_sampling*` options of `occbin_setup` are ignored.
7. **Perfect foresight**: `perfect_foresight_solver(lmmcp)` with `[mcp='...']` equation tags returns the unconstrained path (verified h2: r = -1.32 with `mcp='r > 0'`); `endogenous_terminal_period`, `linear_approximation`, `tolf`, `tolx`, `maxit`, `stack_solve_algo`, `no_homotopy`, `homotopy_*`, `markowitz`, `minimal_solving_periods` ignored (keyword `tol=1e-9`, `maxit=50` instead, Dynare default tolf 1e-5); `perfect_foresight_setup(datafile=...)` ignored (h19); `perfect_foresight_controlled_paths` and `histval` only warned at load.
8. **`steady(...)`** options all ignored: `tolf`, `maxit`, `solve_algo`, `markowitz`, `homotopy_steps`, `nocheck`, `fsolve_options`; pydyn demands max|residual| < 1e-10 where Dynare accepts 6e-6 (verified h10: a `steady_state_model` off by 1e-7 raises `SteadyStateError`, also with `steady(nocheck)`). `homotopy_setup` jumps to the final values without stepping.
9. **`qz_criterium` in `stoch_simul`/`check`/`estimation`-solve of the model**: honoured for Lyapunov/unit-root detection in `moments` and in `Estimation`, but not in `first_order`/`k_order` (fixed 1+1e-6). Verified against Dynare/Octave: `x = 1.02 x(-1) + e` with `qz_criterium=1.05` is solved by Dynare (ghx=1.02), pydyn raises BlanchardKahnError (h9). `qz_zero_threshold` is unused everywhere.
10. **`dsge_var`, `dsge_varlag`, `heteroskedastic_filter`, `heteroskedastic_shocks` block, `conditional_likelihood`, `endogenous_prior`, `filter_initial_state` block** in estimation: ignored, the plain DSGE likelihood is returned (verified h12: `dsge_var=0.5` gives exactly the DSGE likelihood; the blocks are warned once at load).
11. **`shock_decomposition(use_shock_groups=, forecast_type=, with_epilogue)`**, `calib_smoother(heteroskedastic_filter, filter_decomposition, smoother_redux, use_univariate_smoother_if_singularity_is_detected=0)`: dropped by the smoother's option reader without notice (h5). Grouped decompositions need an explicit `smoother.group_shocks(...)` (not validated against Dynare).
12. **`stoch_simul(simul_replic=, replic=)`**: ignored by `stoch_simul_moments` (verified h13: identical output with `simul_replic=3`); Dynare averages over replications.
13. **`stoch_simul(irf_shocks=, relative_irf, irf_plot_threshold, irf=, spectral_density)`**: no function reads them; `relative_irf` and `spectral_density` do not exist in the package.
14. **`partial_information`, `aim_solver`, `dr=cycle_reduction|logarithmic_reduction`, `k_order_solver`, `solve_algo` (stoch_simul)**: always QZ / C++ k-order. Same answer when BK holds; `partial_information` returns the full-information solution (not probed numerically).
15. **`calib_smoother`/`shock_decomposition`/`estimation` data**: `xls_sheet`, `xls_range` ignored and `.xls/.xlsx` files cannot be read (raise EstimationError); `dirname` ignored.
16. **`write_latex_*`, `collect_latex_files`, `rplot`, `set_dynare_seed`, `compilation_setup`, `epilogue`, `dynare_version`**: accepted with no warning and no effect (`epilogue` never evaluated: it is not in the JSON).

## Prioritised gaps for an applied macroeconomist (medium-scale NK, OccBin at order 1, order 1-2 solution, smoother, shock decompositions, figures)

Size: small < 1 day, medium = days, large = weeks. Dynare paths are under `/home/claude/dl/src72/`.

| # | Gap | Size | Where Dynare implements it |
|---|---|---|---|
| 1 | Guard rails: raise or warn for every recognised-but-unhonoured option (central option registry; make `Estimation`, `smoother`, `steady`, `perfect_foresight` report unknown/ignored `.mod` options at call time); refuse `Estimation`/`smooth` on a model that has `occbin_constraints`; make `first_order`/`simulate`/`irf`/`moments` consistent about `.mod` `order`, `pruning`, `loglinear`, `qz_criterium` | small | n/a (pydyn design); option lists in `preprocessor/src/ComputingTasks.cc`, `matlab/default_option_values.m` |
| 2 | OccBin estimation and smoothing: piecewise Kalman filter and inversion filter likelihood, OccBin smoother, `occbin_setup(likelihood_*, smoother_*, filter_*)`, OccBin shock decomposition and `squeeze_shock_decomposition`, `occbin_write_regimes`, posterior importance sampling | large | `matlab/+occbin/{DSGE_smoother,IVF_core,IVF_posterior,kalman_update_engine,kalman_update_algo_1,kalman_update_algo_3,pkf_conditional_density,shock_decomposition,squeeze_shock_decomposition,posterior_importance_sampling,write_regimes_to_xls}.m`, hook in `matlab/estimation/dsge_likelihood.m` |
| 3 | Shock-decomposition workflow: `plot_shock_decomposition` (stacked bars, `use_shock_groups`, `type=qoq|yoy|aoa`, `detail_plot`, `screen_shocks`, `steadystate`, `diff`, `flip`), `realtime_shock_decomposition`, `squeeze_shock_decomposition`, `init2shocks`, annualised decompositions; validate `group_shocks` against Dynare and honour `use_shock_groups` in `shock_decomposition()` | medium | `matlab/shock_decomposition/{plot_shock_decomposition,graph_decomp,graph_decomp_detail,expand_group,annualized_shock_decomposition,realtime_shock_decomposition,squeeze_shock_decomposition,epilogue_shock_decomposition}.m` |
| 4 | Posterior parameter sets feeding the smoother and decompositions (`parameter_set=posterior_mode|posterior_mean|posterior_median|prior_mode|prior_mean`), `observation_trends` and non-stationary (`lik_init!=1`) models in `Estimation.smoother`, posterior smoothed variables/shocks with HPD bands | small-medium | `matlab/shock_decomposition/shock_decomposition.m`, `matlab/estimation/{dynare_estimation_1,prior_posterior_statistics}.m`, `matlab/set_parameters_locally.m` |
| 5 | Posterior analysis after MH: convergence diagnostics (Brooks-Gelman, Geweke, Raftery-Lewis), posterior statistics tables and densities, marginal data density (modified harmonic mean, Geweke), `mh_tune_jscale`/`calibrate_mh_scale_parameter`, chain persistence/recovery, posterior IRFs (`bayesian_irf`), posterior moments, `mode_check`, `plot_priors`, `prior_function`/`posterior_function` | medium | `matlab/estimation/{marginal_density,GetPosteriorParametersStatistics,PosteriorIRF*,posterior_analysis,posterior_moments,mode_check,plot_priors,calibrate_mh_scale_parameter}.m`, `matlab/convergence_diagnostics/` |
| 6 | Optimisers behind `mode_compute` (csminwel=4, newrat=5, fminsearch/simplex 7/8, cmaes 9, simpsa 10, ...), `mode_file`, `optim` options, `use_penalized_objective_for_hessian`, `additional_optimizer_steps`; honour `mode_compute=0` | medium | `matlab/optimization/*.m`, `matlab/estimation/dynare_estimation_1.m` |
| 7 | IRF tooling for `stoch_simul`: IRFs for all shocks with Dynare defaults (one-std-dev, Cholesky), `irf_shocks`, `relative_irf`, `irf_plot_threshold`, `oo_.irfs`-style output, Dynare-layout IRF figures, order-3 IRF validation, `replic`/`simul_replic` | small-medium | `matlab/stochastic_solver/{stoch_simul,irf,getIrfShocksIndx}.m` (IRF plotting is inside `stoch_simul.m`), `matlab/dyn_figure.m` |
| 8 | `loglinear` for `stoch_simul`, steady-state display, IRFs and simulation (the smoother and estimation already have `_loglinear_rules`) | small | `matlab/stochastic_solver/{stochastic_solvers,irf,simult_,k_order_pert}.m`, `matlab/disp_steady_state.m` |
| 9 | Perfect foresight: mcp/`lmmcp` (ZLB without OccBin), `histval`, `initval_file`/`histval_file`, `perfect_foresight_controlled_paths`, `perfect_foresight_with_expectation_errors_*`, `mshocks`, dated shocks, `endogenous_terminal_period`, `linear_approximation`, `.mod` tolerances/maxit, homotopy; more validation (endval) | medium | `matlab/perfect-foresight-models/{perfect_foresight_mcp_problem,controlled_paths_*,perfect_foresight_with_expectation_errors_solver,perfect_foresight_solver_core}.m`, `matlab/lmmcp/`, `matlab/histvalf.m` |
| 10 | `forecast` and `conditional_forecast` (+ plots), including forecasts from `estimation(forecast=)` and from smoothed states | medium | `matlab/+forecasts/*.m`, `matlab/+conditional_forecasts/*.m`, `matlab/+occbin/forecast.m` |
| 11 | Honour solver options: `qz_criterium`/`qz_zero_threshold` in `first_order`/`k_order`, `dr=cycle_reduction|logarithmic_reduction`, steady-state `solve_algo`/block decomposition/tolerances/homotopy stepping, `steady(nocheck)` and Dynare-size tolerance (6e-6) instead of 1e-10 | small-medium | `matlab/stochastic_solver/{dyn_first_order_solver,select_qz_criterium_value}.m`, `matlab/matrix_solver/`, `matlab/steady.m`, `matlab/evaluate_steady_state.m`, `matlab/optimization/dynare_solve.m` |
| 12 | Estimation model features: `heteroskedastic_shocks`/`heteroskedastic_filter`, `filter_initial_state`, `estimated_params_bounds/remove`, several `estimated_params` blocks and `(overwrite)`, `use_calibration`, measurement-error covariances, `conditional_likelihood`, `endogenous_prior`, recursive estimation (`nobs=[a:b]`), `dsge_var`, second-order estimation with nonlinear filters, calibrated covariances | medium-large | `matlab/estimation/{dsge_likelihood,dsge_var_likelihood,get_init_state,endogenous_prior,set_prior,non_linear_dsge_likelihood}.m`, `matlab/kalman/get_Qvec_heteroskedastic_filter.m`, `matlab/nonlinear-filters/` |
| 13 | Optimal policy: `ramsey_model`/`ramsey_policy`/`planner_objective`/`ramsey_constraints` (multiplier initialisation in `steady`, `fill_auxiliary` mishandles `MULT_*`), `evaluate_planner_objective`, `discretionary_policy`, `osr` | medium-large | `matlab/optimal_policy/{dyn_ramsey_static,mult_elimination,evaluate_planner_objective}.m`, `matlab/discretionary_policy/`, `matlab/+osr/` |
| 14 | Model-file coverage: `varexo_det`, external functions in the model block, `epilogue`, `mshocks`, `skew`/`coskewness`, `histval`; JSON repair for `shock_paths`, `model_comparison`, `colormap`; `model_info`, `resid`, `model_diagnostics`, `write_latex_*` | small-medium | `preprocessor/src/{ParsingDriver,ModelEquationBlock,Shocks}.cc` (JSON writers), `matlab/{model_diagnostics,display_static_residuals}.m` |
| 15 | Stochastic extended path (`extended_path`) and higher-order theoretical moments / third-order estimation filters; method of moments (`method_of_moments`, GMM/SMM/IRF matching); `identification`/`sensitivity`; `bvar_*`; SBVAR; PAC/VAR-expectation; heterogeneity (lowest priority for this profile) | large each | `matlab/ep/`, `matlab/+mom/`, `matlab/+identification/`, `matlab/+gsa/`, `matlab/+bvar/`, `matlab/ms-sbvar/`, `matlab/+pac/`, `matlab/+heterogeneity/` |

## Claims in the README that the tests do not back up

Quotes are from `/home/claude/audit/snap/README.md`.

1. "Decision rules agree to about 1e-13; 28 IRFs agree to about 1e-15", "Piecewise and linear paths agree to 1e-15", "Paths agree to 1e-16", "about 1e-14", "about 1e-15": measured, not asserted. The tests assert rtol 1e-9/1e-10 (`test_first_order.py:64-82`, `test_occbin.py:36-43`, `test_occbin_options.py:56`, `test_korder.py` rtol 1e-9, `test_stage3.py` rtol 1e-10/1e-9). The tighter figures are not regression-protected.
2. "Perfect foresight (stacked Newton; `initval`, `endval`, deterministic `shocks`) ... done, validated on one example": the one example (`perfect_foresight_rbc`, `test_auxiliary.py:103`) has `initval` and a one-period `shocks` block and no `endval`; no test touches `endval`, `histval`, several shock blocks, or any solver option. (My own Dynare/Octave comparison of a 60-period permanent-shock RBC gives max|diff| 3.3e-5, consistent with Dynare's 1e-5 stopping rule, but it is not in the repo.)
3. "Simulation and IRFs at orders 2 and 3, with and without pruning ... done, validated": IRFs are validated at orders 1 and 2 only (`test_irf_order2`, `test_first_order_rules_and_irfs`); there is no order-3 IRF reference (`r_o3.m`, `r_stage3.m` hold order-3 simulation only). Order-2 IRFs are validated against a port of the same algorithm with common random draws, i.e. not against independent Dynare draws.
4. "random-walk Metropolis-Hastings ... done, validated": what is validated is (a) the log posterior at 40 stored Dynare draws and (b) the accept/reject path against an independent re-implementation of the rule (`test_metropolis_accept_reject_path`, `test_sampler_on_model_with_fixed_random_numbers`). There is no comparison of chain output (means, acceptance, mixing) with Dynare, no convergence diagnostics, and `mh_drop`, `mh_nblocks`, `mh_init_scale_factor` defaults are not compared. Posterior mode is validated for one model (`test_posterior_mode_schorfheide`) and uses a different optimiser from Dynare.
5. "MATLAB code in `.mod` files ... done, validated on Dynare's test suite": the test (`test_corpus_calibration_against_dynare`, 156 files) compares parameters and `Sigma_e` (and `ys` for flagged files) only; it says nothing about the 41 refused and 15 out-of-scope files, whose counts come from `tools/sweep_matlab.py`, which is not run by the tests.
6. "tools/sweep_examples.py loads each of the 23 example .mod files ... Twelve match": the script needs `ref_<name>.mat` files that are not in the repo and it is not run by `tests/`; the figure "twelve match" cannot be reproduced from the repository. (I confirmed independently that 22 of 23 load; the PAC example raises `ModelError`; the six heterogeneity examples load with warnings and then fail with a misleading `NameError` in any pydyn call; the two Ramsey examples fail in `steady`.)
7. "Known limits: Not handled: `loglinear` ..." contradicts the status table and the code: `loglinear` is implemented and validated in `Estimation` and `smoother` (schorfheide model), and in `stoch_simul` it is not "not handled" but silently ignored (verified against Dynare).
8. "Known limits: ... steady-state assignments with several outputs (`[a, b] = f(...)`)" is stale: `steady._from_steady_state_model` implements it and `test_matlab.py::test_helper_with_several_outputs` tests it.
9. "Known limits: `examples/estimation/rbc_irf_matching.mod`: the 7.2 preprocessor writes invalid JSON for this file, so it cannot be loaded" contradicts the Dynare-examples paragraph ("loads and solves") and `model.py` (it repairs `matched_irfs` JSON); in my sweep it loads. The same preprocessor defect exists, unrepaired, for `shock_paths`, `model_comparison` and `colormap=` (not mentioned in the README).
10. "OccBin estimation (piecewise Kalman filter, inversion filter) | not started": true, but the README does not say that `Estimation` and `smoother.smooth` accept an OccBin model and silently use the linear reference regime.
11. "Orders 2 and 3 through Dynare's C++ k-order library | done, validated": the order-2/3 tests are skipped, not failed, when `korder/libpydyn_korder.so` is absent (`test_korder.py:27`, `test_stage3.py needs_lib`, `test_moments.py needs_lib`), so a run without the library reports success without validating them. The library is a prebuilt binary that must be rebuilt per platform from the Dynare source tree.
12. "Kalman smoother, shock decomposition ... done, validated": true for `shock_decomposition`, `calib_smoother`, `initial_condition_decomposition` at calibrated parameters; `group_shocks` (used for `use_shock_groups`) is tested only for internal consistency (`test_group_shocks`), not against Dynare.
13. README status table says nothing about warnings: every "parsed and warned" statement in the command table above is reported once, at `Model()` load, as a Python `UserWarning` that the package's own test modules filter out (`pytest.mark.filterwarnings("ignore:statements not handled")`).

## Not checked

- MATLAB (as opposed to Octave) behaviour; Windows/macOS; the 6.x JSON layout.
- `Estimation` and `sample()` speed/memory on a medium-scale model (smoother on the 40-variable `sw_big` runs in 0.04 s, `first_order` including steady state in 3 ms; likelihood-evaluation throughput not measured).
- `OccBin` run time on large models; `adl()`, `get_shock_skew_by_name`, `@#include`/`@#echo`/`@#error`/`@#includepath` individually; `trend_component_model` and `var_expectation_model` statements individually (only via the PAC example and the Dynare test-suite files, which fail at `pac.initialize` / `var_expectation.initialize`).
- Numerical agreement with Dynare of the individual math functions (other than max/min derivatives) and of `differentiate_forward_vars`, `var_remove`, `model_replace`.
- Dynare's own behaviour for the options marked IGN0 (assumed numerically neutral from the manual text).
