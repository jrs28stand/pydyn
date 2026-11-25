# Refusals: classification and fix list

Generated from the full sweep of main 41c15f1 (`/home/claude/sweep_41c1`; 215 files refused). Every refused file is in exactly one class. The class is the cause of the first refusal, confirmed by reading the .mod file and what it would hit next; a file whose first blocker would be followed by an unported toolbox is classed by the toolbox (see the note under T).

| class | meaning | files |
|---|---|---|
| T | a Dynare toolbox / feature area pydyn has not ported: now `excluded` through `tools/accept_scope_pydyn.json` (reason: toolbox not ported (yet)) | 63 |
| F | fixable gap: a missing interpreter function, option or field, small feature (work queue below) | 122 |
| D | not comparable for reasons outside pydyn: depends on another test's outputs or on the Octave environment | 23 |
| O | other: explained below | 7 |
| total | | 215 |

After the scope update the sweep reports 152 refused (= F + D + O), 117 excluded (54 earlier + 63 T), and match 371 / partial 79 / wrong 26 are unchanged. Sizes: S under a day, M a few days, L a week or more (rough). Counts are files whose *first* blocker the item removes; a file may reach a further blocker afterwards, which this list cannot see.

## F: fix list, sorted by files unblocked per effort

Effort weights for the sort: S 0.5, S-M 1, M 2, M-L 3, L 5 days.

| # | group | files | size | exact missing piece |
|---|---|---|---|---|
| 1 | osr_opt_algo | 6 | S | `osr` with `opt_algo` 1, 3, 8, 9: map them to the scipy equivalents already used for `mode_compute` 1, 3, 7, 8, 9 (only csminwel, opt_algo=4, exists today). | **Status: DONE in work_f (runs; some files verdict wrong from optimiser end point, known_json entries pending).**
| 2 | char_matrix | 5 | S | char-matrix value type in the interpreter (`var_list_`, occbin `char` rows: padded rows, row indexing, `strvcat`/`char`/`deblank`). | **Status: DONE in work_f (ar_qz_test, Gali_discretion match; occbin files advance to unpack_simulations).**
| 3 | plot_noop | 4 | S | `figure`, `subplot`, `plot`, `legend`, `title`, `hold` etc. in .mod MATLAB code are refused ('pydyn draws nothing'). Policy decision: accept them as no-ops (or draw with matplotlib). Needs the owner's OK because it is a refusal by design. | **Status: DONE in work_f (accepted, draws nothing; print/saveas/savefig refused).**
| 4 | misc_small | 11 | mixed | eleven unrelated single-file gaps, listed below. |
| 5 | struct_fields | 7 | S-M | fields of `M_`, `options_`, `oo_` read by the mods: `M_.exo_partitions`, `M_.endo_partitions`, `M_.observed_exo_names`, `options_.varobs`/`varobs_id`, `oo_.initval_series`, `options_.heteroskedastic_shocks`, `M_.aux_vars` as a struct array, `oo_.FilteredVariablesShockDecomposition`. | **Status: PARTLY done in work_f (partitions, observed_exo_names, varobs/varobs_id; initval_series open).**
| 6 | consider_only_observed | 3 | S | `moments_varendo` with `consider_only_observed` after `estimation(...)`: the .mod lists no variables and uses no `consider_all_*`, so the refusal looks mis-triggered by `consider_only_observed` (to confirm); fix the option handling. | **Status: CODED in work_f, 3 files not re-run.**
| 7 | missing_functions | 11 | M | single Dynare internals or builtins called from .mod code, each small: `dynare_resolve`, `diff`, `mult_elimination`, `evaluate_planner_objective`, `shock_decomposition_backward`, `sample_hp_filter`, `generate_trace_plots`, `fataltest` (test helper), `perfect_foresight_problem` (lmmcp/sw_newton), `extended_path_mc`. |
| 8 | dseries_readers | 5 | S-M | `dseries('x.m')`, `dseries('x.csv')`, xlsx/`.xls` data files in `estimation(datafile=...)`, and `dseries('file')` with a path (only .mat is read today). | **Status: NOT done: see NOTES.md.**
| 9 | set_time | 2 | S | `set_time` statement (sets `M_.start_date`/`options_.initial_period`); needed before `dseries` data are aligned. | **Status: NOT done: see NOTES.md.**
| 10 | datafile_m_script | 2 | S | estimation data file written as a MATLAB script with statements other than `name = expression` (`clear`, a bare name, `reshape(...)`): extend the data-file mini-interpreter. |
| 11 | smoother_only_noparams | 4 | S-M | `estimation(...)` without an `estimated_params` block (smoother or filter at the calibrated parameters): allow an empty estimated set; Dynare then uses `M_.params` as the mode. |
| 12 | occbin_xls | 2 | S | `occbin_write_regimes_xls` (writes regimes to an .xls file): write csv/xlsx with the same cells, or accept as a documented no-op. |
| 13 | identification_misc (pdflatex) | 2 | S | `system('pdflatex ...')` at the end of `identification` files (as2007, fs2000_corr): incidental LaTeX build; allow `system` for a whitelist of commands that do not affect results, or skip. |
| 14 | est_params_multi | 3 | S-M | several `estimated_params` blocks in one file (second block adds to or replaces the first): merge as Dynare's `estim_params_` does. |
| 15 | mode_compute | 6 | M | `mode_compute` 2 (simulated annealing), 10 (simpsa), 101 (solvopt); `optim=(...)` option string `InitialInverseHessian='eye(9)*.0001'`; `newrat` with `Hessian=0|2` (outer-product Hessian from univariate filters); `mode_compute=@fn` custom optimiser wrapper (`fs2000_w`). |
| 16 | mh_posterior_mode | 6 | M | estimation option `mh_posterior_mode_estimation` (start an optimiser from the MH posterior mode), used by the six particle-filter files. |
| 17 | hssmc | 2 | S-M | `posterior_sampling_method='hssmc'`: the sampler returns an `SMCResult` that post-processing does `float()` on (`dsge_base_hssmc`, `particle/dsge_base_HSSMC`); wire the SMC result into the posterior statistics. |
| 18 | pruned_simult | 4 | M | `simult_` at order 2 and 3 with pruning, and the `pruned_SS` object (`pruned_SS.*`): simulations of the pruned state space. |
| 19 | particle_other | 5 | M-L | particle files: `nonlinear_filter_initialization=2` (ergodic variance from a 5000-period simulation; 2 files, M), `particle.pruning` (1, M), estimation at order 3 (2, L; only order 2 is implemented). |
| 20 | smoother_redux | 3 | M | `smoother_redux` (reduced-state smoother); also used by occbin `NKM_calib_smoother`. |
| 21 | shock_paths | 2 | M | `shock_paths` statement for controlled / anticipated shock paths in perfect foresight (`ramst_shock_paths_*`). |
| 22 | homotopy | 2 | M | `homotopy_setup` + the `homotopy` statement of `steady`/`perfect_foresight_solver`, incl. `homotopy_exclude_varexo` (`oo_.endo_simul` field `sim1`). |
| 23 | var_expectations_eval | 2 | M | `var_expectations` model package functions (`<model>.var_expectations.<name>.evaluate`). |
| 24 | dsgevar_irf | 2 | M | `bayesian_irf` with `dsge_var` (inverse-Wishart draws for the DSGE-VAR IRFs). |
| 25 | selected_vars_only | 2 | M | `selected_variables_only` combined with the posterior smoother / `filtered_vars` / `filter_covariance` / `smoothed_state_uncertainty`. |
| 26 | stack_algo7 | 2 | M | `stack_solve_algo=7` on models with exogenous lags/leads does not converge as in Dynare (`rbc_det_stack_solve_algo_7_exo_lag/lead`): diagnose. |
| 27 | corr_ME_theory | 2 | M | theoretical moments with correlated measurement errors (`fs2000_het_corr`) and the diffuse univariate smoother (`kalman_algo=4`) with correlated measurement errors (`fs2000_uncorr_ME_missing`). |
| 28 | identification_misc (diffuse) | 2 | M | `identification` with the diffuse filter (unit roots, `ident_unit_root`); `identification(parameter_set=posterior_mode)` (S, `Ramsey_Example_estimation`). |
| 29 | order_gt3 | 4 | L | stochastic simulation with `order` > 3 (burnside_k_order, neo_growth_k_order, neo_growth_ramsey_k_order, ep/burnside): the preprocessor output only has derivatives to order 3; needs preprocessor `output=` higher or pydyn's own k-order tensors. |
| 30 | sep | 4 | L | stochastic extended path: sparse tree (`ep.stochastic.order`, perfect/sparse tree), `hybrid_order`, plus `tic/toc` in `ep/integration/burnside`. |
| 31 | model_inversion_leads | 2 | M-L | `model_inversion` (`shocks` + `histval`/`inversion`) for models with leads: a perfect-foresight problem per period. |
| 32 | prior_command | 1 | M | the `prior` command: `prior plot|table|moments|simulate|optimize` (one file, gsa/ls2003prior). |
| 33 | occbin_filter | 2 | L | estimation with `occbin_constraints` (piecewise-linear Kalman / inversion filter for occbin models). |

### Files per group

**1. osr_opt_algo** (6, S)

- `optimal_policy/OSR/OSR_planner_objective`
- `optimal_policy/OSR/osr_example`
- `optimal_policy/OSR/osr_example_param_bounds`
- `optimal_policy/OSR/osr_obj_corr_algo_3`
- `optimal_policy/OSR/osr_obj_corr_algo_8`
- `optimal_policy/OSR/osr_obj_corr_algo_9`

**2. char_matrix** (5, S)

- `conditional_variance_decomposition/example1`
- `decision_rules/ar_qz_test`
- `discretionary_policy/Gali_discretion`
- `occbin/model_borrcon/borrcon`
- `occbin/model_borrcon/borrcon_0_std_shocks`

**3. plot_noop** (4, S)

- `estimation/system_prior_restriction/Gali_2015`
- `examples/estimation/gali_2015`
- `forecast/Hansen_exo_det_forecast`
- `steady_state/Gali_2015_chapter_6_4`

**4. misc_small** (11, mixed)

- `decision_rules/first_order/walsh`: S: call of a function in a namespace (`matlab.namespace.y_k(...)`) is not tokenised.
- `deterministic_simulations/Solow_no_varexo`: S: `perfect_foresight_solver` on a model without `varexo`: a None is multiplied (bug).
- `discretionary_policy/dennis_1_estim`: S-M: `estimation` boolean index size mismatch (4 vs 5): bug with discretionary policy and observables.
- `estimation/fs2000_estimated_params_remove`: S: `estimated_params_remove` statement.
- `estimation/fs2000_initialize_from_calib`: S: `estimated_params_init(use_calibration)` when a `corr` listed is not estimated.
- `estimation/slice/fs2000_estimate_initial_states`: M: option `estimate_initial_states_endogenous_prior`.
- `examples/optimal_policy/nk_ramsey_steady_file`: S: the user steady-state file is run by the interpreter and `R` is unknown there (Ramsey steady-state file).
- `histval_initval_file/ramst_data_generate`: M: `fopen`/`fprintf`/`fclose` writing data files from the .mod (file-system calls are refused today).
- `histval_initval_file/ramst_initval_file_with_dseries`: S-M: `initval_file(series=ds)` reading a dseries of the workspace.
- `kalman_filter_smoother/SOE`: S: `calib_smoother(datafile=...)`: observed variables not found in the data (data file naming/macro; diagnose).
- `prior_posterior_function/fs2000_prior_posterior_function`: M: `prior_posterior_function` statement.

**5. struct_fields** (7, S-M)

- `estimation/heteroskedastic_shocks/fs2000_het_sample_restriction`
- `exogenous-observed-variables/preprocessor`
- `histval_initval_file/sim_exo_lead_lag_initvalf`
- `kalman_filter_smoother/check_variable_dimensions/fs2000_ML`
- `partitioning/partitions`
- `smoother2histval/fs2000_simul`
- `write/example1`

**6. consider_only_observed** (3, S)

- `estimation/fs2000_with_weibull_prior`
- `kalman_initial_state/fs2000_kalman_initial`
- `kalman_initial_state/fs2000_kalman_initial_2_lag`

**7. missing_functions** (11, M)

- `deterministic_simulations/rbc_det_exo_lag_2b`
- `deterministic_simulations/rbc_det_exo_lag_2c`
- `ep/rbc_mc`
- `estimation/slice/fs2000_slice_use_prior`
- `kalman_filter_smoother/test_compute_Pinf_Pstar`
- `lmmcp/sw_newton`
- `moments/example1_hp_test`
- `optimal_policy/Ramsey/Gali_commitment`
- `optimal_policy/mult_elimination_test`
- `optimal_policy/neo_growth_ramsey`
- `shock_decomposition/shock_decomp_backward`

**8. dseries_readers** (5, S-M)

- `data/mod1a`
- `dates/dseries_interact`
- `dates/fs2000`
- `estimation/heteroskedastic_shocks/fs2000_het_dates`
- `fs2000/fs2000_calib_dseries`

**9. set_time** (2, S)

- `fs2000/fs2000_dseries_a`
- `fs2000/fs2000_dseries_b`

**10. datafile_m_script** (2, S)

- `estimation/no_init_estimation_check_first_obs/fs2000_init_check`
- `parallel/ls2003`

**11. smoother_only_noparams** (4, S-M)

- `kalman_filter_smoother/fs2000_smoother_only`
- `kalman_filter_smoother/fs2000_smoother_only_ns`
- `kalman_initial_state/fs2000_ns_smoother_only_initial_state`
- `kalman_initial_state/fs2000_smoother_only_initial_state`

**12. occbin_xls** (2, S)

- `occbin/model_irrcap_twoconstraints/dynrbc`
- `occbin/model_irrcap_twoconstraints/dynrbc_0_std_shocks`

**13. identification_misc:pdflatex** (2, S)

- `identification/as2007/as2007`
- `identification/correlated_errors/fs2000_corr`

**14. est_params_multi** (3, S-M)

- `analytic_derivatives/fs2000_analytic_derivation`
- `analytic_derivatives/fs2000_analytic_derivation_diag_H`
- `moments/fs2000_post_moments`

**15. mode_compute** (6, M)

- `optimizers/fs2000_10`
- `optimizers/fs2000_101`
- `optimizers/fs2000_2`
- `optimizers/fs2000_4_with_optim`
- `optimizers/fs2000_5`
- `optimizers/fs2000_w`

**16. mh_posterior_mode** (6, M)

- `particle/dsge_base_apf`
- `particle/dsge_base_cpf`
- `particle/dsge_base_gf`
- `particle/dsge_base_gmf`
- `particle/dsge_base_nlkf`
- `particle/dsge_base_sis`

**17. hssmc** (2, S-M)

- `estimation/hssmc/dsge_base_hssmc`
- `particle/dsge_base_HSSMC`

**18. pruned_simult** (4, M)

- `stochastic_simulations/fs2000_k_order_simul`
- `stochastic_simulations/pruning/AS_pruned_state_space_red_shock`
- `stochastic_simulations/pruning/AnSchorfheide_pruned_state_space`
- `stochastic_simulations/pruning/fs2000_pruning`

**19. particle_other** (5, M-L)

- `particle/dsge_base_Liu_West`
- `particle/first_spec`
- `particle/first_spec_MCMC`
- `particle/local_state_space_iteration_3_test`
- `particle/local_state_space_iteration_k_test`

**20. smoother_redux** (3, M)

- `kalman_filter_smoother/fs2000_smoother_redux`
- `kalman_filter_smoother/trend_cycle_smoother_redux`
- `occbin/filter/NKM_calib_smoother`

**21. shock_paths** (2, M)

- `deterministic_simulations/ramst_shock_paths_controlled`
- `deterministic_simulations/ramst_shock_paths_pfwee`

**22. homotopy** (2, M)

- `deterministic_simulations/homotopy_exclude_varexo`
- `steady_state/homotopy/homotopy_from_initval_to_endval`

**23. var_expectations_eval** (2, M)

- `var-expectations/10/example1`
- `var-expectations/12/example1`

**24. dsgevar_irf** (2, M)

- `dsge-var/dsgevar_forward_calibrated_lambda`
- `dsge-var/dsgevar_forward_estimated_lambda`

**25. selected_vars_only** (2, M)

- `estimation/fs2000`
- `observation_trends_and_prefiltering/ML/Trend_no_prefilter_selected_var`

**26. stack_algo7** (2, M)

- `deterministic_simulations/rbc_det_stack_solve_algo_7_exo_lag`
- `deterministic_simulations/rbc_det_stack_solve_algo_7_exo_lead`

**27. corr_ME_theory** (2, M)

- `estimation/heteroskedastic_shocks/fs2000_het_corr`
- `kalman/likelihood_from_dynare/fs2000_uncorr_ME_missing`

**28. identification_misc:diffuse** (2, M)

- `identification/ident_unit_root/ident_unit_root`
- `optimal_policy/Ramsey/Ramsey_Example_estimation`

**29. order_gt3** (4, L)

- `decision_rules/k_order/burnside_k_order`
- `ep/burnside`
- `optimal_policy/neo_growth_k_order`
- `optimal_policy/neo_growth_ramsey_k_order`

**30. sep** (4, L)

- `ep/hybrid/rs`
- `ep/integration/burnside`
- `ep/zeroshocks/rs`
- `ep/zeroshocks/rstrue`

**31. model_inversion_leads** (2, M-L)

- `model-inversion/nk-1/invert`
- `model-inversion/nk-2/invert`

**32. prior_command** (1, M)

- `gsa/ls2003prior`

**33. occbin_filter** (2, L)

- `occbin/filter/NKM_0_std_shocks`
- `occbin/filter/NKM_non_restrict_state_space`

## T: toolboxes not ported (excluded through the scope file)

Scope entries in `tools/accept_scope_pydyn.json` (feature, ids and an `expect` regex of the refusal message). A file counts as T when the toolbox is the first blocker, or when the file is a test of that toolbox whose first blocker is a small gap that would not unblock it (noted).

| toolbox | files |
|---|---|
| method of moments (`method_of_moments`, `matched_moments`, `matched_irfs`) | 24 |
| global sensitivity analysis (`dynare_sensitivity`) | 3 |
| `irf_calibration` / `moment_calibration` | 1 |
| Markov-switching SBVAR / DSGE (`svar_identification`, `markov_switching`) | 11 |
| BVAR (`bvar_density`, `bvar_forecast`; `bvar_and_dsge` first meets a data-file `clear`) | 2 |
| recursive estimation / rolling forecasts | 5 |
| partial information solution | 4 |
| skewed shocks, skewed Kalman filter | 4 |
| reporting package (first meets a dseries frequency mix in its data preparation) | 1 |
| balanced growth path package | 2 |
| `plot_contributions` (first meets a dseries parse/index gap) | 2 |
| semi-structural tools (existing entry, widened: first meets the dseries csv reader / `firstdate`) | 4 |

Method-of-moments files `RBC_MoM_prefilter`, `RBC_MoM_SMM_ME`, `RBC_MoM_SMM_ME_filter` first read `options_.varobs`, a field pydyn does not expose; fixing that alone does not run them (the `method_of_moments` command follows), so they are T.

## D: outside pydyn (23 files)

- **mode_file** (11): `mode_file` points at the output of another test (`algo1`, `algo3`, `algoH1`, `fs2000_*_ME`, `fs2000_het`...) that the sweep does not run first; a harness fix (run the producing test first, as the Dynare run order does) would let them compare. Not a pydyn gap.
  - `estimation/heteroskedastic_shocks/fs2000_het`
  - `kalman/likelihood_from_dynare/fs2000_corr_ME`
  - `kalman/likelihood_from_dynare/fs2000_corr_ME_missing`
  - `kalman/likelihood_from_dynare/fs2000_uncorr_ME`
  - `kalman/likelihood_from_dynare/fs2000ns_corr_ME`
  - `kalman/likelihood_from_dynare/fs2000ns_corr_ME_missing`
  - `kalman/likelihood_from_dynare/fs2000ns_uncorr_ME`
  - `kalman/likelihood_from_dynare/fs2000ns_uncorr_ME_missing`
  - `kalman_filter_smoother/algo2`
  - `kalman_filter_smoother/algo4`
  - `kalman_filter_smoother/algoH2`
- **mh_recover** (5): `MH_recover/*` and `fs2000_MCMC_jumping_covariance` copy/load Metropolis files written by an earlier run.
  - `estimation/MH_recover/fs2000_recover`
  - `estimation/MH_recover/fs2000_recover_2`
  - `estimation/MH_recover/fs2000_recover_3`
  - `estimation/MH_recover/fs2000_recover_tarb`
  - `estimation/fs2000_MCMC_jumping_covariance`
- **env_branch** (4): `optimizers/fs2000_1`, `fs2000_3`, `fs2000_7`, `kalman/lyapunov/fs2000_lyap` choose a branch with `user_has_octave_package(...)` or `exist("fminunc","file")`: the result depends on which packages the Octave of the reference had; pydyn refuses to guess.
  - `kalman/lyapunov/fs2000_lyap`
  - `optimizers/fs2000_1`
  - `optimizers/fs2000_3`
  - `optimizers/fs2000_7`
- **other_test_output** (3): `var-expectations/4`, `4-with-time-shift` load `../3/weights.mat`; `preprocessor_checks/ramst_model_edit` loads `../deterministic_simulations/ramst` results.
  - `preprocessor_checks/ramst_model_edit`
  - `var-expectations/4-with-time-shift/example1`
  - `var-expectations/4/example1`

## O: other (1 file after the follow-up)

Follow-up (work copy /home/claude/work_det, not yet regenerated into the counts above): 6 of the 7 O files were investigated.

- `kalman_filter_smoother/compare_results_simulation/{fs2000, fs2000_loglinear, fs2000_ML, fs2000_ML_loglinear}`: two pydyn bugs found and fixed. (1) The smoother read the `.m` data file with a plain-number reader and ignored the `gy_obs = log(gy_obs)` lines at the end of `fsdat_simul_logged.m`, so the smoothed observed variables were levels instead of logs. (2) `resol` under `loglinear` returned `dr.ys` equal to the live `oo_.steady_state` instead of the steady state at the parameters of the `M_` it was given. The files now run; `fs2000` and `fs2000_loglinear` are partial (MCMC end point, as for other MH files) and the two ML files are documented in `tools/accept_known.json` as optimiser end point differences (the ML likelihood is nearly degenerate, psi -> 1: a 1e-7 change of the starting alp moves newrat's end point from alp=0.683, rho=-0.40 to alp=0.707, rho=0.735; Dynare's is 0.726, 0.730).
- `forecast/example1_varexo_det`, `forecast/forecast_test_AR1_varexo_det_order_2`: the refusal at order > 1 was not needed. `pydyn.forecast` already replays Dynare's draw order; the results layer now passes its Octave stream to it. Both files run and every `oo_.forecast` field matches the Octave reference; the verdict is partial only because of `dr.exo_det_length`, `dr.ghud`, `dr.ghxud` (F item: fields of the varexo_det decision rule).
- `stochastic_simulations/example2long` (still refused): the mod compares `ghxx` of two equivalent models to 1e-12 absolute. pydyn gives a difference of 1.6e-12 (ghx 3.6e-14, ghxx values up to 43, so about 4e-14 relative) against Octave's 1.4e-14, and Octave's own `use_dll` path gives 3.3e-13. Each pydyn dr matches the Octave dr of the same model to 3e-13 (example1long) and 1.9e-12 (example2long). This is rounding noise in a threshold tuned to Octave's linear algebra, not a wrong result. `example2long_use_dll` passes in pydyn.
