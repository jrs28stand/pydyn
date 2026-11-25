# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Which options of each Dynare statement pydyn acts on: nothing is silently ignored.

Every option the Dynare 7.2 preprocessor can emit for a statement is in exactly one of three sets of ``SPECS[statement]``:

* ``honoured``: pydyn reads it and the results depend on it (verified by tools/options_probe.py and the sweep, or by review);
* ``harmless``: it cannot change a number or a field pydyn computes (printing, figures, the name of an output folder, the
  choice of a numerical algorithm that gives the same solution), with the reason in ``HARMLESS_REASONS``;
* ``refused``: Dynare would act on it and pydyn does not; ``pydyn.dynare()`` stops with a ``ResultsError`` naming the
  statement and the option (``refused_options``).

An option that is in none of them (not known to the grammar of Dynare 7.2 for that statement) is refused as well, so a new
option of a later Dynare is refused until someone classifies it.  tools/options_audit.py rebuilds the ground truth from
DynareBison.yy and reports options missing from, or extra in, these tables (tests/test_option_policy.py runs it when the
preprocessor sources are available).  Statements that are not listed here either have no options pydyn reads (declarations
whose options the model layer handles) or are refused as a whole.

Plain-API users (pydyn.Model and the functions of the package) get an ``UnsupportedOptionWarning`` at load time for the
refused options of the statements in the .mod file (``warn_model``) instead of an error, since they may use only the
parts of the file pydyn supports.
"""
from __future__ import annotations

HARMLESS_REASONS = {
    'TeX': 'LaTeX output only',
    'block_dynamic': 'printing only',
    'block_static': 'printing only',
    'console_mode': 'printing only',
    'debug': 'printing only',
    'dirname': "name of the folder of Dynare's output files (pydyn writes its own, see docs/ACCEPTANCE.md)",
    'dr_display_tol': 'printing only',
    'graph_format': 'figure file format only',
    'graph_save_formats': 'figure file format only',
    'impulse_responses.plot_threshold': 'figures only',
    'incidence': 'printing only',
    'irf_plot_threshold': 'figures only',
    'no_graph.plot_shock_decomposition': 'no figures',
    'no_graph.posterior': 'no figures',
    'no_graph.shock_decomposition': 'no figures',
    'nodisplay': 'no figures are shown',
    'nofunctions': 'printing only',
    'nograph': 'no figures are made',
    'nomodelsummary': 'printing only',
    'noprint': 'printing only',
    'parallel_info.use_pct.estimation.': 'parallel execution does not change the results',
    'plot_priors': 'figures only',
    'silent_optimizer': 'printing only',
    'simul.debug': 'printing only',
    'steady.non_zero': 'printing only (non-zero residuals)',
    'tex': 'LaTeX output only',
    'verbosity': 'printing only',
    'dr_cycle_reduction': 'numerical algorithm or its tuning; the same solution when it converges',
    'dr_cycle_reduction_maxiter': 'numerical algorithm or its tuning; the same solution when it converges',
    'dr_cycle_reduction_tol': 'numerical algorithm or its tuning; the same solution when it converges',
    'dr_logarithmic_reduction': 'numerical algorithm or its tuning; the same solution when it converges',
    'dr_logarithmic_reduction_maxiter': 'numerical algorithm or its tuning; the same solution when it converges',
    'dr_logarithmic_reduction_tol': 'numerical algorithm or its tuning; the same solution when it converges',
    'lyapunov_db': 'numerical algorithm or its tuning; the same solution when it converges',
    'lyapunov_fixed_point_tol': 'numerical algorithm or its tuning; the same solution when it converges',
    'lyapunov_fp': 'numerical algorithm or its tuning; the same solution when it converges',
    'lyapunov_srs': 'numerical algorithm or its tuning; the same solution when it converges',
    'simul.block_diagonal_lu_maxlu': 'numerical algorithm or its tuning; the same solution when it converges',
    'simul.block_diagonal_lu_nlu': 'numerical algorithm or its tuning; the same solution when it converges',
    'simul.block_diagonal_lu_nperiods': 'numerical algorithm or its tuning; the same solution when it converges',
    'simul.block_diagonal_lu_relu': 'numerical algorithm or its tuning; the same solution when it converges',
    'simul.gmres_restart': 'numerical algorithm or its tuning; the same solution when it converges',
    'simul.iter_maxit': 'numerical algorithm or its tuning; the same solution when it converges',
    'simul.iter_tol': 'numerical algorithm or its tuning; the same solution when it converges',
    'simul.preconditioner': 'numerical algorithm or its tuning; the same solution when it converges',
}
#: figure options of plot_shock_decomposition / initial_condition_decomposition (the figures are not what pydyn computes)
HARMLESS_PREFIXES = {'irf_opt.': 'accepted by the preprocessor but read by no code of Dynare 7.2 (no consumer of options_.irf_opt in matlab/; Octave gives identical IRFs at order 2 with and without them)', 'plot_shock_decomp.': 'figure option only', 'initial_condition_decomp.': 'figure option only'}

SPECS = {
    'steady': dict(
        honoured=frozenset({
            'homotopy_force_continue', 'homotopy_mode', 'homotopy_steps', 'solve_algo', 'solve_tolf',
            'solve_tolx', 'steady.maxit', 'steadystate.nocheck'
        }),
        harmless=frozenset({
            'markowitz', 'noprint', 'steady.non_zero'
        }),
        refused=frozenset({
            'fsolve_options'
        }),
    ),
    'check': dict(
        honoured=frozenset({
            'homotopy_force_continue', 'homotopy_mode', 'homotopy_steps', 'qz_zero_threshold', 'solve_algo', 'solve_tolf', 'solve_tolx', 'steady.maxit', 'steadystate.nocheck'
        }),
        harmless=frozenset({
            'markowitz', 'noprint', 'steady.non_zero'
        }),
        refused=frozenset({
            'fsolve_options'
        }),
    ),
    'stoch_simul': dict(
        honoured=frozenset({
            'SpectralDensity.trigger', 'aim_solver', 'ar', 'bandpass.indicator', 'bandpass.passband', 'conditional_variance_decomposition', 'contemporaneous_correlation', 'drop', 'filtered_theoretical_moments_grid', 'hp_filter', 'irf', 'irf_shocks', 'k_order_solver', 'loglinear', 'nocorr', 'nodecomposition', 'nomoments', 'one_sided_hp_filter', 'order', 'partial_information', 'periods', 'pruning', 'qz_criterium', 'qz_zero_threshold', 'relative_irf', 'replic', 'simul_replic', 'solve_algo'
        }),
        harmless=frozenset({
            'TeX', 'dr_cycle_reduction', 'dr_cycle_reduction_maxiter', 'dr_cycle_reduction_tol', 'dr_display_tol', 'dr_logarithmic_reduction', 'dr_logarithmic_reduction_maxiter', 'dr_logarithmic_reduction_tol', 'graph_format', 'impulse_responses.plot_threshold', 'irf_opt.EM.drop', 'irf_opt.EM.iter', 'irf_opt.EM.tolf', 'irf_opt.analytical_GIRF', 'irf_opt.diagonal_only', 'irf_opt.ergodic_mean_irf', 'irf_opt.percent', 'irf_opt.stderr_multiples', 'nodisplay', 'nofunctions', 'nograph', 'nomodelsummary', 'noprint'
        }),
        refused=frozenset(),
    ),
    'calib_smoother': dict(
        honoured=frozenset({
            'datafile', 'diffuse_filter', 'diffuse_kalman_tol', 'filter_covariance', 'filter_decomposition',
            'filter_step_ahead', 'filtered_vars', 'first_obs', 'heteroskedastic_filter', 'kalman_algo',
            'kalman_tol', 'lik_init', 'loglinear', 'nobs', 'parameter_set', 'prefilter',
            'smoothed_state_uncertainty', 'smoother_redux',
            'use_univariate_filters_if_singularity_is_detected',
            'use_univariate_smoother_if_singularity_is_detected', 'xls_range', 'xls_sheet'
        }),
        harmless=frozenset(),
        refused=frozenset({
            'skewed_kalman.mvnlogcdf', 'skewed_kalman.prune_tol', 'skewed_kalman.rank_deficiency_transform',
            'updated_covariance'
        }),
    ),
    'estimation': dict(
        honoured=frozenset({
            'MCMC_jumping_covariance', 'additional_optimizer_steps', 'aim_solver', 'analytic_derivation', 'ar', 'bayesian_irf', 'conditional_variance_decomposition', 'contemporaneous_correlation', 'convergence.brooksgelman.plotrows', 'convergence.geweke.geweke_interval', 'convergence.geweke.taper_steps', 'convergence.rafterylewis.indicator', 'convergence.rafterylewis.qrs', 'cova_compute', 'datafile', 'diffuse_filter', 'diffuse_kalman_tol', 'dsge_var', 'dsge_varlag', 'endo_vars_for_moment_computations_in_estimation', 'endogenous_prior', 'estimate_initial_states_endogenous_prior', 'estimation.moments_posterior_density.indicator', 'filter_covariance', 'filter_decomposition', 'filter_step_ahead', 'filtered_vars', 'first_obs', 'forecast', 'forecasts.conf_sig', 'frequentist_smoother', 'hessian.use_penalized_objective', 'heteroskedastic_filter', 'huge_number', 'irf', 'irf_shocks', 'kalman_algo', 'kalman_tol', 'lik_init', 'load_mh_file', 'logdata', 'loglinear', 'lyapunov_doubling_tol', 'mh_conf_sig', 'mh_drop', 'mh_init_scale', 'mh_init_scale_factor', 'mh_jscale', 'mh_nblck', 'mh_posterior_mode_estimation', 'mh_recover', 'mh_replic', 'mh_tune_jscale.guess', 'mh_tune_jscale.status', 'mh_tune_jscale.target', 'mode_check.neighbourhood_size', 'mode_check.number_of_points', 'mode_check.status', 'mode_check.symmetric_plots', 'mode_compute', 'mode_file', 'moments_varendo', 'no_init_estimation_check_first_obs', 'nobs', 'noconstant', 'nodecomposition', 'nodiagnostic', 'optim_opt', 'order', 'partial_information', 'posterior_max_subsample_draws', 'posterior_sampler_options.posterior_sampling_method', 'posterior_sampler_options.sampling_opt', 'prefilter', 'presample', 'prior_trunc', 'pruning', 'qz_zero_threshold', 'relative_irf', 'rescale_prediction_error_covariance', 'smoothed_state_uncertainty', 'smoother', 'smoother_redux', 'solve_algo', 'sub_draws', 'use_univariate_filters_if_singularity_is_detected', 'use_univariate_smoother_if_singularity_is_detected', 'xls_range', 'xls_sheet'
        }),
        harmless=frozenset({
            'TeX', 'dirname', 'dr_cycle_reduction', 'dr_cycle_reduction_maxiter', 'dr_cycle_reduction_tol', 'dr_logarithmic_reduction', 'dr_logarithmic_reduction_maxiter', 'dr_logarithmic_reduction_tol', 'graph_format', 'impulse_responses.plot_threshold', 'irf_opt.EM.drop', 'irf_opt.EM.iter', 'irf_opt.EM.tolf', 'irf_opt.analytical_GIRF', 'irf_opt.diagonal_only', 'irf_opt.ergodic_mean_irf', 'irf_opt.percent', 'irf_opt.stderr_multiples', 'lyapunov_db', 'lyapunov_fixed_point_tol', 'lyapunov_fp', 'lyapunov_srs', 'no_graph.posterior', 'nodisplay', 'nograph', 'parallel_info.use_pct.estimation.', 'plot_priors', 'silent_optimizer'
        }),
        refused=frozenset({
            'conditional_likelihood.status', 'fast_kalman_filter', 'load_results_after_load_mh', 'mh_initialize_from_previous_mcmc.directory', 'mh_initialize_from_previous_mcmc.prior', 'mh_initialize_from_previous_mcmc.record', 'mh_initialize_from_previous_mcmc.status', 'particle.cpf_weights_method.amisanotristani', 'particle.cpf_weights_method.murrayjonesparslow', 'particle.distribution_approximation.cubature', 'particle.distribution_approximation.montecarlo', 'particle.distribution_approximation.unscented', 'particle.filter_algorithm', 'particle.initialization', 'particle.number_of_particles', 'particle.particle_filter_options', 'particle.proposal_approximation.cubature', 'particle.proposal_approximation.montecarlo', 'particle.proposal_approximation.unscented', 'particle.resampling.method.kitagawa', 'particle.resampling.method.multinomial', 'particle.resampling.method.residual', 'particle.resampling.method.smooth', 'particle.resampling.method.stratified', 'particle.resampling.status.generic', 'particle.resampling.status.none', 'particle.resampling.status.systematic', 'particle.resampling.threshold', 'posterior_sampler_options.posterior_sampling_method.proposal_distribution', 'selected_variables_only', 'skewed_kalman.mvnlogcdf', 'skewed_kalman.prune_tol', 'skewed_kalman.rank_deficiency_transform', 'updated_covariance'
        }),
    ),
    'forecast': dict(
        honoured=frozenset({
            'forecasts.conf_sig', 'periods'
        }),
        harmless=frozenset({
            'graph_format', 'nodisplay', 'nograph'
        }),
        refused=frozenset(),
    ),
    'conditional_forecast': dict(
        honoured=frozenset({
            'conditional_forecast.conf_sig', 'controlled_varexo', 'parameter_set', 'periods', 'replic'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'shock_decomposition': dict(
        honoured=frozenset({
            'datafile', 'diffuse_filter', 'diffuse_kalman_tol', 'filter_covariance', 'filter_decomposition',
            'first_obs', 'heteroskedastic_filter', 'kalman_algo', 'kalman_tol', 'lik_init', 'loglinear',
            'nobs', 'parameter_set', 'plot_shock_decomp.use_shock_groups', 'prefilter',
            'shock_decomp.forecast_type', 'shock_decomp.init_state', 'shock_decomp.with_epilogue',
            'smoothed_state_uncertainty', 'smoother_redux',
            'use_univariate_filters_if_singularity_is_detected',
            'use_univariate_smoother_if_singularity_is_detected', 'xls_range', 'xls_sheet'
        }),
        harmless=frozenset({
            'no_graph.shock_decomposition', 'plot_shock_decomp.colormap'
        }),
        refused=frozenset({
            'skewed_kalman.mvnlogcdf', 'skewed_kalman.prune_tol', 'skewed_kalman.rank_deficiency_transform',
            'updated_covariance'
        }),
    ),
    'plot_shock_decomposition': dict(
        honoured=frozenset({
            'plot_shock_decomp.init2shocks', 'plot_shock_decomp.realtime',
            'plot_shock_decomp.use_shock_groups'
        }),
        harmless=frozenset({
            'no_graph.plot_shock_decomposition', 'plot_shock_decomp.colormap',
            'plot_shock_decomp.detail_plot', 'plot_shock_decomp.diff', 'plot_shock_decomp.fig_name',
            'plot_shock_decomp.flip', 'plot_shock_decomp.interactive', 'plot_shock_decomp.max_nrows',
            'plot_shock_decomp.nodisplay', 'plot_shock_decomp.plot_end_date',
            'plot_shock_decomp.plot_init_date', 'plot_shock_decomp.screen_shocks',
            'plot_shock_decomp.steadystate', 'plot_shock_decomp.type', 'plot_shock_decomp.vintage',
            'plot_shock_decomp.write_xls'
        }),
        refused=frozenset(),
    ),
    'initial_condition_decomposition': dict(
        honoured=frozenset({
            'initial_condition_decomp.with_epilogue'
        }),
        harmless=frozenset({
            'initial_condition_decomp.colormap', 'initial_condition_decomp.detail_plot',
            'initial_condition_decomp.diff', 'initial_condition_decomp.fig_name',
            'initial_condition_decomp.flip', 'initial_condition_decomp.max_nrows',
            'initial_condition_decomp.nodisplay', 'initial_condition_decomp.plot_end_date',
            'initial_condition_decomp.plot_init_date', 'initial_condition_decomp.steadystate',
            'initial_condition_decomp.type'
        }),
        refused=frozenset({
            'initial_condition_decomp.write_xls'
        }),
    ),
    'realtime_shock_decomposition': dict(
        honoured=frozenset({
            'datafile', 'first_obs', 'nobs', 'parameter_set', 'plot_shock_decomp.use_shock_groups',
            'shock_decomp.fast_realtime', 'shock_decomp.forecast', 'shock_decomp.presample',
            'shock_decomp.save_realtime', 'shock_decomp.with_epilogue'
        }),
        harmless=frozenset({
            'no_graph.shock_decomposition', 'plot_shock_decomp.colormap'
        }),
        refused=frozenset(),
    ),
    'occbin_setup': dict(
        honoured=frozenset({
            'filter.init_periods_using_particles', 'filter.particle.status', 'filter.use_relaxation',
            'likelihood.brute_force_extra_regime_guess', 'likelihood.brute_force_regime_guess',
            'likelihood.check_ahead_periods', 'likelihood.curb_retrench',
            'likelihood.first_period_binding_regime_allowed', 'likelihood.first_period_occbin_update',
            'likelihood.inversion_filter', 'likelihood.max_check_ahead_periods',
            'likelihood.max_number_of_iterations', 'likelihood.maxit', 'likelihood.periodic_solution',
            'likelihood.periods', 'simul.check_ahead_periods', 'simul.curb_retrench',
            'simul.max_check_ahead_periods', 'simul.maxit', 'simul.periodic_solution',
            'simul.periodic_solution_strict', 'simul.periodic_solution_threshold', 'simul.periods',
            'simul.reset_check_ahead_periods_in_new_period', 'simul.reset_regime_in_new_period',
            'smoother.check_ahead_periods', 'smoother.curb_retrench', 'smoother.debug',
            'smoother.first_period_occbin_update', 'smoother.inversion_filter',
            'smoother.max_check_ahead_periods', 'smoother.max_number_of_iterations', 'smoother.maxit',
            'smoother.periodic_solution', 'smoother.periods', 'smoother.plot'
        }),
        harmless=frozenset({
            'simul.debug'
        }),
        refused=frozenset({
            'filter.particle.diagnostics.graph_periods', 'filter.particle.diagnostics.nograph',
            'filter.particle.diagnostics.status', 'filter.particle.draw_states_from_empirical_density',
            'filter.particle.initial_state_ergodic_simul', 'filter.particle.number_of_particles',
            'filter.particle.number_of_shocks_per_particle', 'filter.particle.state_draws',
            'filter.particle.state_importance_sampling.logpost_crit_threshold',
            'filter.particle.state_importance_sampling.pkf_init',
            'filter.particle.state_importance_sampling.slice_burnin',
            'filter.particle.state_importance_sampling.slice_override_iteration',
            'filter.particle.use_pkf_updated_state_threshold', 'posterior_importance_sampling.filter',
            'posterior_importance_sampling.orig_dname', 'posterior_importance_sampling.orig_filter',
            'posterior_importance_sampling.orig_fname', 'posterior_importance_sampling.status',
            'posterior_importance_sampling.sub_draws'
        }),
    ),
    'occbin_solver': dict(
        honoured=frozenset({
            'simul.check_ahead_periods', 'simul.curb_retrench', 'simul.max_check_ahead_periods',
            'simul.maxit', 'simul.periodic_solution', 'simul.periodic_solution_strict',
            'simul.periodic_solution_threshold', 'simul.periods',
            'simul.reset_check_ahead_periods_in_new_period', 'simul.reset_regime_in_new_period'
        }),
        harmless=frozenset({
            'simul.debug'
        }),
        refused=frozenset(),
    ),
    'occbin_graph': dict(
        honoured=frozenset({
            'graph.steady_state'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'ramsey_model': dict(
        honoured=frozenset({
            'instruments'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'ramsey_policy': dict(
        honoured=frozenset({
            'aim_solver', 'ar', 'conditional_variance_decomposition', 'contemporaneous_correlation', 'drop', 'filtered_theoretical_moments_grid', 'hp_filter', 'instruments', 'irf', 'irf_shocks', 'k_order_solver', 'nocorr', 'nomoments', 'order', 'partial_information', 'periods', 'pruning', 'qz_criterium', 'qz_zero_threshold', 'relative_irf', 'replic', 'simul_replic', 'solve_algo'
        }),
        harmless=frozenset({
            'TeX', 'dr_cycle_reduction', 'dr_cycle_reduction_maxiter', 'dr_cycle_reduction_tol', 'dr_display_tol', 'dr_logarithmic_reduction', 'dr_logarithmic_reduction_maxiter', 'dr_logarithmic_reduction_tol', 'graph_format', 'impulse_responses.plot_threshold', 'irf_opt.EM.drop', 'irf_opt.EM.iter', 'irf_opt.EM.tolf', 'irf_opt.analytical_GIRF', 'irf_opt.diagonal_only', 'irf_opt.ergodic_mean_irf', 'irf_opt.percent', 'irf_opt.stderr_multiples', 'nodisplay', 'nofunctions', 'nograph', 'nomodelsummary', 'noprint'
        }),
        refused=frozenset(),
    ),
    'evaluate_planner_objective': dict(
        honoured=frozenset(),
        harmless=frozenset(),
        refused=frozenset({
            'ramsey.drop', 'ramsey.periods'
        }),
    ),
    'osr': dict(
        honoured=frozenset({
            'aim_solver', 'analytic_derivation', 'analytic_derivation_mode', 'ar', 'conditional_variance_decomposition', 'contemporaneous_correlation', 'drop', 'filtered_theoretical_moments_grid', 'hp_filter', 'huge_number', 'irf', 'irf_shocks', 'k_order_solver', 'nocorr', 'nomoments', 'optim_opt', 'order', 'osr.maxit', 'osr.opt_algo', 'osr.tolf', 'partial_information', 'periods', 'pruning', 'qz_criterium', 'qz_zero_threshold', 'relative_irf', 'replic', 'simul_replic', 'solve_algo'
        }),
        harmless=frozenset({
            'TeX', 'dr_cycle_reduction', 'dr_cycle_reduction_maxiter', 'dr_cycle_reduction_tol', 'dr_display_tol', 'dr_logarithmic_reduction', 'dr_logarithmic_reduction_maxiter', 'dr_logarithmic_reduction_tol', 'graph_format', 'impulse_responses.plot_threshold', 'irf_opt.EM.drop', 'irf_opt.EM.iter', 'irf_opt.EM.tolf', 'irf_opt.analytical_GIRF', 'irf_opt.diagonal_only', 'irf_opt.ergodic_mean_irf', 'irf_opt.percent', 'irf_opt.stderr_multiples', 'nodisplay', 'nofunctions', 'nograph', 'nomodelsummary', 'noprint', 'silent_optimizer'
        }),
        refused=frozenset(),
    ),
    'discretionary_policy': dict(
        honoured=frozenset({
            'aim_solver', 'ar', 'conditional_variance_decomposition', 'contemporaneous_correlation', 'discretionary_tol', 'drop', 'filtered_theoretical_moments_grid', 'hp_filter', 'instruments', 'irf', 'irf_shocks', 'k_order_solver', 'nocorr', 'nomoments', 'order', 'partial_information', 'periods', 'pruning', 'qz_criterium', 'qz_zero_threshold', 'relative_irf', 'replic', 'simul_replic', 'solve_algo'
        }),
        harmless=frozenset({
            'TeX', 'dr_cycle_reduction', 'dr_cycle_reduction_maxiter', 'dr_cycle_reduction_tol', 'dr_display_tol', 'dr_logarithmic_reduction', 'dr_logarithmic_reduction_maxiter', 'dr_logarithmic_reduction_tol', 'graph_format', 'impulse_responses.plot_threshold', 'irf_opt.EM.drop', 'irf_opt.EM.iter', 'irf_opt.EM.tolf', 'irf_opt.analytical_GIRF', 'irf_opt.diagonal_only', 'irf_opt.ergodic_mean_irf', 'irf_opt.percent', 'irf_opt.stderr_multiples', 'nodisplay', 'nofunctions', 'nograph', 'nomodelsummary', 'noprint'
        }),
        refused=frozenset(),
    ),
    'identification': dict(
        honoured=frozenset({
            'advanced', 'analytic_derivation', 'analytic_derivation_mode', 'ar', 'checks_via_subsets',
            'diffuse_filter', 'diffuse_kalman_tol', 'grid_nbr', 'gsa_sample_file', 'heteroskedastic_filter',
            'kalman_algo', 'kalman_tol', 'lik_init', 'load_ident_files', 'max_dim_cova_group',
            'max_dim_subsets_groups', 'no_identification_minimal', 'no_identification_moments',
            'no_identification_reducedform', 'no_identification_spectrum', 'no_identification_strength',
            'normalize_jacobians', 'order', 'parameter_set', 'periods', 'prior_mc', 'prior_range',
            'prior_trunc', 'replic', 'schur_vec_tol', 'tol_deriv', 'tol_rank', 'tol_sv',
            'use_univariate_filters_if_singularity_is_detected', 'useautocorr'
        }),
        harmless=frozenset({
            'TeX', 'graph_format', 'nodisplay', 'nograph'
        }),
        refused=frozenset({
            'skewed_kalman.mvnlogcdf', 'skewed_kalman.prune_tol', 'skewed_kalman.rank_deficiency_transform'
        }),
    ),
    'smoother_2_histval': dict(
        honoured=frozenset({
            'invars', 'outfile', 'outvars', 'period'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'extended_path': dict(
        honoured=frozenset({
            'ep.periods', 'ep.stochastic.IntegrationAlgorithm', 'ep.stochastic.algo',
            'ep.stochastic.hybrid_order', 'ep.stochastic.order', 'ep.stochastic.quadrature.nodes',
            'ep.use_first_order_solution_as_initial_guess', 'lmmcp.status', 'periods'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'perfect_foresight_setup': dict(
        honoured=frozenset({
            'datafile', 'periods', 'simul.endval_steady', 'simul.endval_steady_nocheck',
            'simul.first_simulation_period', 'simul.last_simulation_period'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'perfect_foresight_solver': dict(
        honoured=frozenset({
            'dynatol.f', 'dynatol.x', 'endogenous_terminal_period', 'linear_approximation', 'lmmcp.status',
            'markowitz', 'minimal_solving_periods', 'no_homotopy', 'simul.allow_nonfinite_values',
            'simul.homotopy_exclude_varexo', 'simul.homotopy_initial_step_size',
            'simul.homotopy_linearization_fallback', 'simul.homotopy_marginal_linearization_fallback',
            'simul.homotopy_max_completion_share', 'simul.homotopy_min_step_size',
            'simul.homotopy_step_size_increase_success_count', 'simul.maxit', 'simul.robust_lin_solve',
            'simul.steady_markowitz', 'simul.steady_maxit', 'simul.steady_solve_algo', 'simul.steady_tolf',
            'simul.steady_tolx', 'solve_algo', 'stack_solve_algo'
        }),
        harmless=frozenset({
            'noprint', 'simul.block_diagonal_lu_maxlu', 'simul.block_diagonal_lu_nlu',
            'simul.block_diagonal_lu_nperiods', 'simul.block_diagonal_lu_relu', 'simul.gmres_restart',
            'simul.iter_maxit', 'simul.iter_tol', 'simul.preconditioner'
        }),
        refused=frozenset({
            'simul.check_jacobian_singularity'
        }),
    ),
    'perfect_foresight_with_expectation_errors_setup': dict(
        honoured=frozenset({
            'datafile', 'periods', 'simul.endval_steady_nocheck', 'simul.first_simulation_period',
            'simul.last_simulation_period'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'perfect_foresight_with_expectation_errors_solver': dict(
        honoured=frozenset({
            'dynatol.f', 'dynatol.x', 'endogenous_terminal_period', 'linear_approximation', 'lmmcp.status',
            'markowitz', 'minimal_solving_periods', 'no_homotopy', 'pfwee.constant_simulation_length',
            'simul.allow_nonfinite_values', 'simul.homotopy_exclude_varexo',
            'simul.homotopy_initial_step_size', 'simul.homotopy_linearization_fallback',
            'simul.homotopy_marginal_linearization_fallback', 'simul.homotopy_max_completion_share',
            'simul.homotopy_min_step_size', 'simul.homotopy_step_size_increase_success_count', 'simul.maxit',
            'simul.robust_lin_solve', 'simul.steady_markowitz', 'simul.steady_maxit',
            'simul.steady_solve_algo', 'simul.steady_tolf', 'simul.steady_tolx', 'solve_algo',
            'stack_solve_algo'
        }),
        harmless=frozenset({
            'noprint', 'simul.block_diagonal_lu_maxlu', 'simul.block_diagonal_lu_nlu',
            'simul.block_diagonal_lu_nperiods', 'simul.block_diagonal_lu_relu', 'simul.gmres_restart',
            'simul.iter_maxit', 'simul.iter_tol', 'simul.preconditioner'
        }),
        refused=frozenset({
            'simul.check_jacobian_singularity'
        }),
    ),
    'initval_file': dict(
        honoured=frozenset({
            'datafile', 'filename', 'first_obs', 'first_simulation_period', 'last_obs',
            'last_simulation_period', 'nobs', 'series'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'histval_file': dict(
        honoured=frozenset({
            'datafile', 'filename', 'first_obs', 'first_simulation_period', 'last_obs', 'series'
        }),
        harmless=frozenset(),
        refused=frozenset(),
    ),
    'model_info': dict(
        honoured=frozenset(),
        harmless=frozenset({
            'block_dynamic', 'block_static', 'incidence'
        }),
        refused=frozenset(),
    ),
    'resid': dict(
        honoured=frozenset(),
        harmless=frozenset({
            'steady.non_zero'
        }),
        refused=frozenset(),
    ),
    'occbin_write_regimes_xls': dict(
        honoured=frozenset(),
        harmless=frozenset({
            'write_regimes.filename', 'write_regimes.periods', 'write_regimes.type'
        }),
        refused=frozenset(),
    ),
}


#: why a refused option is refused (first matching prefix; the default says it is not implemented)
REFUSAL_REASONS = (
    ("particle.", "particle filters are not implemented"),
    ("filter.particle.", "particle filters are not implemented"),
    ("posterior_importance_sampling.", "importance sampling of the posterior is not implemented"),
    ("skewed_kalman.", "the skewed Kalman filter is not implemented"),
    ("irf_opt.", "generalised impulse responses (EM / analytical GIRF) are not implemented"),
    ("mh_initialize_from_previous_mcmc.", "initialising the chains from a previous MCMC is not implemented"),
    ("shock_decomp.", "this realtime shock decomposition option is not implemented"),
)


#: options refused unless their value is one that cannot change the result: key -> predicate on the preprocessor's value
def _fsolve_options_harmless(value):
    """fsolve_options = ('Display','iter','AutoScaling','on'): only options of fsolve that do not move its solution."""
    import re
    toks = [t.strip() for t in str(value).split(",")]
    if len(toks) % 2:
        return False
    names = []
    for t in toks[0::2]:
        m = re.fullmatch(r"''(\w+)''", t)
        if not m:
            return False
        names.append(m.group(1))
    return bool(names) and all(n in {"Display", "AutoScaling", "Diagnostics", "FunValCheck", "OutputFcn", "PlotFcns"} for n in names)


def _num(value):
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _at_most(limit):
    return lambda value: _num(value) is not None and _num(value) <= limit


def _at_least(limit):
    return lambda value: _num(value) is not None and _num(value) >= limit


#: Dynare's defaults (matlab/default_option_values.m).  A looser tolerance (or fewer iterations) changes the stored
#: solution: decision_rules/first_order/walsh.mod shows cycle reduction at dr_cycle_reduction_tol=1e-1 differing from QZ.
#: At the default or tighter, Dynare's own cycle and logarithmic reduction agree with QZ to 4e-15 (checked under Octave on a
#: small New Keynesian model), so the algorithm choice is harmless there.
VALUE_OK = {
    "fsolve_options": _fsolve_options_harmless,
    "dr_cycle_reduction_tol": _at_most(1e-7), "dr_cycle_reduction_maxiter": _at_least(100),
    "dr_logarithmic_reduction_tol": _at_most(1e-12), "dr_logarithmic_reduction_maxiter": _at_least(100),
}
#: lyapunov=fixed_point at its default tolerance (1e-10) differs from the doubling algorithm by 1.6e-8 relative in the smoothed
#: variables (Octave, tests above 1e-8), so the fixed-point method and its tolerance are refused; doubling and square_root_solver
#: agree with pydyn to 6e-13 (est_conv model, lik_init default) and stay harmless.
_ALWAYS_REFUSED = ("lyapunov_fp", "lyapunov_fixed_point_tol")
for _spec in SPECS.values():
    for _k in _ALWAYS_REFUSED:
        if _k in _spec["harmless"]:
            _spec["harmless"] = _spec["harmless"] - {_k}
            _spec["refused"] = _spec["refused"] | {_k}
for _spec in SPECS.values():                 # these were classified harmless as algorithm tuning: only so at the default or tighter
    for _k in VALUE_OK:
        if _k in _spec["harmless"]:
            _spec["harmless"] = _spec["harmless"] - {_k}
            _spec["refused"] = _spec["refused"] | {_k}


def _reason(key):
    for prefix, why in REFUSAL_REASONS:
        if key.startswith(prefix):
            return why
    return "pydyn does not implement it"


def classify(statement, key):
    """``'honoured'``, ``'harmless'`` or ``'refused'`` for option ``key`` of ``statement`` (None: statement not in SPECS)."""
    spec = SPECS.get(statement)
    if spec is None:
        return None
    if key in spec["honoured"]:
        return "honoured"
    if key in spec["harmless"] or any(key.startswith(p) for p in HARMLESS_PREFIXES):
        return "harmless"
    return "refused"


def refused_options(statement, options):
    """[(option, reason)] for the options of ``statement`` that pydyn does not act on (empty for a statement not in SPECS)."""
    out = []
    for key, value in (options or {}).items():
        if classify(statement, key) == "refused" and not (key in VALUE_OK and VALUE_OK[key](value)):
            known = key in SPECS[statement]["refused"]
            why = "not an option of this statement in Dynare 7.2"
            if key in VALUE_OK:
                why = ("pydyn solves with QZ; a value looser than Dynare's default would make Dynare's stored solution "
                       "differ from it" if key != "fsolve_options" else "fsolve options other than Display, AutoScaling, ... "
                       "can move the solution")
            elif known:
                why = _reason(key)
            out.append((key, why))
    return out


def check_statement(statement, options):
    """Message that refuses the statement because of its options (None: all of them are acted on)."""
    bad = refused_options(statement, options)
    if not bad:
        return None
    return ("pydyn does not act on " + ", ".join(f"{k} ({why})" for k, why in bad)
            + f"; the {statement} statement is refused rather than run without it")


def warn_model(model):
    """Plain API: warn (once per model) about the refused options of the statements of the .mod file."""
    from .options import warn_once
    bad = {}
    for st in getattr(model, "statements", ()) or ():
        for key, why in refused_options(st.get("statementName"), st.get("options")):
            bad.setdefault(st["statementName"], set()).add(key)
    for name, keys in sorted(bad.items()):
        warn_once(model, f"pydyn does not act on these {name} options of the .mod file: {', '.join(sorted(keys))} "
                         "(pydyn.dynare() refuses the statement)", stacklevel=4)
    return bad
