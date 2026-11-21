// Small linear New Keynesian model with three observables, measurement error on two of them
// and missing observations in the data. Written for pydyn's smoother tests (2026).
// Data: smo_me_data.m, simulated (reference/smo_make_data.py).

@#ifndef CASE
@#define CASE = 1
@#endif

var x pi i a v g yobs piobs iobs;
varexo e_a e_v e_g;
parameters beta sigma kappa phi_pi phi_x rho_i rho_a rho_v rho_g;

beta = 0.99;
sigma = 1.5;
kappa = 0.12;
phi_pi = 1.6;
phi_x = 0.2;
rho_i = 0.6;
rho_a = 0.85;
rho_v = 0.4;
rho_g = 0.7;

model(linear);
x = x(+1) - (1/sigma)*(i - pi(+1)) + g - 0.3*a;
pi = beta*pi(+1) + kappa*x + 0.1*a(-1);
i = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi + phi_x*x) + v;
a = rho_a*a(-1) + e_a;
v = rho_v*v(-1) + e_v;
g = rho_g*g(-1) + e_g;
yobs = x - x(-1) + a;
piobs = pi;
iobs = i;
end;

shocks;
var e_a; stderr 0.6;
var e_v; stderr 0.25;
var e_g; stderr 0.4;
var e_a, e_g = 0.05;
var yobs; stderr 0.3;
var piobs; stderr 0.1;
end;

varobs yobs piobs iobs;

steady;
stoch_simul(order=1, irf=0, nomoments, noprint);

@#if CASE == 1
calib_smoother(datafile=smo_me_data, filtered_vars, filter_step_ahead=[1,3], filter_covariance, smoothed_state_uncertainty);
shock_decomposition(parameter_set=calibration, nograph);
@#endif

@#if CASE == 2
calib_smoother(datafile=smo_me_data, kalman_algo=2, filtered_vars, filter_covariance, smoothed_state_uncertainty, first_obs=5);
@#endif

@#if CASE == 3
set_param_value('kappa', 0.3);
set_param_value('rho_a', 0.5);
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=smo_me_data, prefilter=1, filtered_vars, nobs=90);
shock_decomposition(parameter_set=calibration, nograph);
@#endif
