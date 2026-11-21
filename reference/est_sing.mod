// pydyn test (finding R25, likelihood): one shock, two observed variables, so the forecast-error
// variance of the multivariate Kalman filter is singular. Dynare 7.2's dsge_likelihood then switches
// to the univariate filter (use_univariate_filters_if_singularity_is_detected, default 1), or returns
// info 50 with the option set to 0 (VARIANT 2). (Dynare's estimation command itself stops in
// initial_estimation_checks; reference/r_est_sing.m calls dsge_likelihood directly.)
@#ifndef VARIANT
@#define VARIANT = 1
@#endif
var x y1 y2;
varexo e;
parameters rho;
rho = 0.8;
model(linear);
x = rho*x(-1) + e;
y1 = x;
y2 = 2*x;
end;
shocks; var e; stderr 0.5; end;
estimated_params;
stderr e, 0.5, 0.01, 3;
rho, 0.8, 0, 0.99;
end;
varobs y1 y2;
@#if VARIANT == 2
estimation(datafile=est_sing_data, mode_compute=0, mh_replic=0, use_univariate_filters_if_singularity_is_detected=0);
@#else
estimation(datafile=est_sing_data, mode_compute=0, mh_replic=0);
@#endif
