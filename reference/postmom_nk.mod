// Small New Keynesian model of reference/post_nk.mod with the estimation commands used to validate pydyn.posterior_moments
// (posterior distributions of theoretical moments, smoothed and filtered variables) against Dynare 7.2.
// Cases are selected with -DPM=n; see reference/r_postmom.m.
@#define VARIANT = 0
@#ifndef PM
@#define PM = 1
@#endif
@#if PM == 6
@#define VARIANT = 3
@#endif
var y pi r a u yobs piobs robs;
varexo ea eu er;
parameters beta kappa phipi phiy rhoa rhou rhor sig pibar rbar;

beta = 0.99; kappa = 0.1; phipi = 1.5; phiy = 0.25; rhoa = 0.8; rhou = 0.5; rhor = 0.7;
sig = 1.5; pibar = 0.5; rbar = 0;

model;
y = y(+1) - (1/sig)*(r - pi(+1)) + a;
pi = beta*pi(+1) + kappa*y + u;
r = rhor*r(-1) + (1-rhor)*(phipi*pi + phiy*y) + er;
a = rhoa*a(-1) + ea;
u = rhou*u(-1) + eu;
yobs = y;
@#if VARIANT == 3
piobs = pi;
robs = r;
@#else
piobs = pibar + pi;
robs = rbar + pibar + r;
@#endif
end;

steady_state_model;
// rbar is a parameter that depends on the estimated parameter beta
rbar = 100*(1/beta - 1);
y = 0; pi = 0; r = 0; a = 0; u = 0;
yobs = 0;
@#if VARIANT == 3
piobs = 0; robs = 0;
@#else
piobs = pibar;
robs = rbar + pibar;
@#endif
end;

shocks;
var ea; stderr 0.5;
var eu; stderr 0.3;
var er; stderr 0.2;
corr ea, eu = 0.1;
var yobs; stderr 0.1;
@#if VARIANT == 9
var piobs; stderr 0.05;
@#endif
end;

estimated_params;
stderr ea, inv_gamma_pdf, 0.5, 2;
stderr eu, inv_gamma2_pdf, 0.1, 1;
stderr er, 0.2, 0.01, 2, gamma_pdf, 0.2, 0.1;
stderr yobs, 0.12, uniform_pdf, , , 0, 1;
@#if VARIANT == 9
stderr piobs, inv_gamma_pdf, 0.05, inf;
@#endif
corr ea, eu, 0.05, -0.9, 0.9, normal_pdf, 0, 0.3;
@#if VARIANT == 9
corr yobs, piobs, 0.2, uniform_pdf, , , -0.9, 0.9;
@#endif
beta, beta_pdf, 0.99, 0.002;
kappa, gamma_pdf, 0.1, 0.05;
phipi, 1.6, 1.01, 4, normal_pdf, 1.5, 0.25;
rhoa, beta_pdf, 0.7, 0.1;
rhor, uniform_pdf, 0.5, 0.25;
sig, 1.4, weibull_pdf, 1.5, 0.5;
pibar, beta_pdf, 0.6, 0.2, -1, 3;
end;

varobs yobs piobs robs;


@#if PM == 6
// observation trends and prefilter (zero-mean observables: variant 3 of the model), smoother and k-step ahead forecasts
observation_trends;
yobs (0.05);
piobs (-0.02);
end;
@#endif
@#if PM == 1
// moments_varendo: explicit variable list (a variable with measurement error, observed variables, unobserved ones),
// correlated shocks, non-default ar, conditional variance decompositions, mh_conf_sig, contemporaneous correlations
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=2400, mh_nblocks=3, mh_jscale=0.9, mh_drop=0.4,
           mh_conf_sig=0.8, sub_draws=90, moments_varendo, ar=4, conditional_variance_decomposition=[1 4 10],
           contemporaneous_correlation, nodisplay) y pi r robs yobs;
@#endif
@#if PM == 2
// consider_only_observed, no unconditional decomposition, more lags, default mh_conf_sig
// (nodecomposition together with conditional_variance_decomposition and a measurement error stops Dynare 7.2:
// compute_moments_varendo.m uses observable_name_requested_vars, defined only when the unconditional decomposition is computed)
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=2400, mh_nblocks=3, mh_jscale=0.9, mh_drop=0.4,
           sub_draws=60, moments_varendo, ar=7, nodecomposition, consider_only_observed, nodisplay);
@#endif
@#if PM == 5
// consider_only_observed with conditional variance decompositions (with and without measurement error)
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=2400, mh_nblocks=3, mh_jscale=0.9, mh_drop=0.4,
           sub_draws=60, moments_varendo, ar=1, consider_only_observed, conditional_variance_decomposition=[2 3], nodisplay);
@#endif
@#if PM == 3
// smoother, filtered variables and k-step ahead filtered variables
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=2400, mh_nblocks=3, mh_jscale=0.9, mh_drop=0.4,
           mh_conf_sig=0.8, sub_draws=40, smoother, filtered_vars, filter_step_ahead=[1 3], nodisplay) y pi r yobs piobs;
@#endif
@#if PM == 4
// consider_all_endogenous, all moments (no ME on non-observed), fewer draws
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=2400, mh_nblocks=3, mh_jscale=0.9, mh_drop=0.4,
           sub_draws=30, moments_varendo, ar=2, consider_all_endogenous, conditional_variance_decomposition=[1 8], nodisplay);
@#endif
@#if PM == 6
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=2400, mh_nblocks=3, mh_jscale=0.9, mh_drop=0.4,
           sub_draws=30, prefilter=1, smoother, filtered_vars, filter_step_ahead=[2 1], nodisplay) y pi r yobs piobs robs;
@#endif
@#if PM == 7
// prior draws: the "prior moments(distribution)" command is called from r_postmom.m after this estimation (no sampling)
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=0, nodisplay);
@#endif
