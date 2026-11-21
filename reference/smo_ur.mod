// Linear model with two stochastic trends (unit roots), a forward-looking variable and static
// variables, for the diffuse Kalman smoother. Written for pydyn's smoother tests (2026).
// Data: smo_ur_data.m, simulated (reference/smo_make_data.py); it contains missing observations.

@#ifndef CASE
@#define CASE = 1
@#endif

var y x y2 mu tau z w s;
varexo e_mu e_tau e_z e_w;
parameters rho_z rho_w lam;

rho_z = 0.8;
rho_w = 0.5;
lam = 0.3;

model(linear);
mu = mu(-1) + e_mu;
tau = tau(-1) + lam*z(-1) + e_tau;
z = rho_z*z(-1) + e_z;
w = 0.4*w(+1) + 0.3*w(-1) + 0.5*z + e_w;
s = rho_w*s(-1) + 0.2*w + 0.1*e_mu;
y = mu + z + w;
x = tau + 0.5*z + s;
y2 = mu + 0.4*w + s(-1);
end;

shocks;
var e_mu; stderr 0.3;
var e_tau; stderr 0.2;
var e_z; stderr 0.5;
var e_w; stderr 0.4;
@#if CASE == 4
var x; stderr 0.15;
@#endif
end;

@#if CASE == 3
varobs y y2;
@#elseif CASE == 6 || CASE == 7
varobs y w;
@#else
varobs y x;
@#endif

stoch_simul(order=1, irf=0, nomoments, noprint);

@#if CASE == 1
// multivariate diffuse filter (kalman_algo=3)
calib_smoother(datafile=smo_ur_data, diffuse_filter, filtered_vars, filter_step_ahead=[1,2], filter_covariance, smoothed_state_uncertainty);
shock_decomposition(parameter_set=calibration, nograph);
@#endif

@#if CASE == 2
// univariate diffuse filter
calib_smoother(datafile=smo_ur_data, diffuse_filter, kalman_algo=4, filtered_vars, filter_covariance, smoothed_state_uncertainty, first_obs=4);
@#endif

@#if CASE == 3
// cointegrated observables: Dynare switches from the multivariate to the univariate diffuse filter
calib_smoother(datafile=smo_ur_data, diffuse_filter, filtered_vars, smoothed_state_uncertainty);
shock_decomposition(parameter_set=calibration, nograph);
@#endif

@#if CASE == 4
// measurement error on x
calib_smoother(datafile=smo_ur_data, diffuse_filter, filtered_vars, smoothed_state_uncertainty, first_obs=3, nobs=70);
@#endif

@#if CASE == 5
// as case 1 with the univariate diffuse filter: the two differ in Dynare 7.2 because period 1 has
// no observation (see pydyn.smoother.smooth, replicate_dynare_bugs)
calib_smoother(datafile=smo_ur_data, diffuse_filter, kalman_algo=4, filtered_vars, filter_step_ahead=[1,2], filter_covariance, smoothed_state_uncertainty);
@#endif

@#if CASE == 6
// Observables y (unit root) and w (stationary), sample starting in period 20: y is missing in the
// first period and w in the second, so F_inf = 0 in the first period while the filter is diffuse.
// Dynare 7.2's multivariate diffuse smoother mishandles this period (see pydyn.smoother.smooth).
calib_smoother(datafile=smo_ur_data, diffuse_filter, kalman_algo=3, filtered_vars, smoothed_state_uncertainty, first_obs=20, nobs=40);
@#endif

@#if CASE == 7
// as case 6 with the univariate diffuse filter
calib_smoother(datafile=smo_ur_data, diffuse_filter, kalman_algo=4, filtered_vars, smoothed_state_uncertainty, first_obs=20, nobs=40);
@#endif
