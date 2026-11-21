// Small New Keynesian model (the model of est_nk.mod, written for pydyn) used to validate the
// smoother and the shock-decomposition workflow of pydyn against Dynare 7.2 at estimated
// parameter sets. Cases are selected with -DCASE=n (default 1); data: est_nk_data.m (simulated).
// References: reference/r_shock_decomposition.m -> sd_nk_case*.mat.
@#ifndef CASE
@#define CASE = 1
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
piobs = pibar + pi;
robs = rbar + pibar + r;
end;

steady_state_model;
// rbar is a parameter that depends on the estimated parameter beta
rbar = 100*(1/beta - 1);
y = 0; pi = 0; r = 0; a = 0; u = 0;
yobs = 0;
piobs = pibar;
robs = rbar + pibar;
end;

shocks;
var ea; stderr 0.5;
var eu; stderr 0.3;
var er; stderr 0.2;
corr ea, eu = 0.1;
var yobs; stderr 0.1;
end;

varobs yobs piobs robs;

shock_groups(name=grp);
supply = eu;
'demand and policy' = ea, er;
end;

shock_groups(name=partial);
policy = er;
end;

init2shocks(name=i2s);
a ea;
u eu;
end;

@#if CASE == 1
// Bayesian estimation, posterior mode and a short chain
estimated_params;
stderr ea, inv_gamma_pdf, 0.5, 2;
stderr eu, inv_gamma2_pdf, 0.1, 1;
stderr er, 0.2, 0.01, 2, gamma_pdf, 0.2, 0.1;
stderr yobs, 0.12, uniform_pdf, , , 0, 1;
corr ea, eu, 0.05, -0.9, 0.9, normal_pdf, 0, 0.3;
beta, beta_pdf, 0.99, 0.002;
kappa, gamma_pdf, 0.1, 0.05;
phipi, 1.6, 1.01, 4, normal_pdf, 1.5, 0.25;
rhoa, beta_pdf, 0.7, 0.1;
rhor, uniform_pdf, 0.5, 0.25;
sig, 1.4, weibull_pdf, 1.5, 0.5;
pibar, beta_pdf, 0.6, 0.2, -1, 3;
end;
estimation(datafile=est_nk_data, mode_compute=4, mh_replic=1200, mh_nblocks=1, mh_jscale=0.35, nodisplay, nograph);
shock_decomposition(parameter_set=posterior_mode, nograph);
plot_shock_decomposition(use_shock_groups=grp, nograph);
@#endif

@#if CASE == 2
// maximum likelihood
estimated_params;
stderr ea, 0.5, 0.01, 3;
stderr er, 0.2, 0.01, 2;
stderr yobs, 0.12, 0.001, 1;
kappa, 0.1, 0.01, 1;
rhoa, 0.7, 0.01, 0.99;
pibar, 0.6, -1, 3;
end;
estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, nodisplay, nograph);
shock_decomposition(parameter_set=mle_mode, nograph);
@#endif

@#if CASE == 3
// calibration: realtime decomposition, groups, init2shocks, initial-condition decomposition
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=est_nk_data, first_obs=5, nobs=60);
shock_decomposition(parameter_set=calibration, nograph);
realtime_shock_decomposition(parameter_set=calibration, forecast=4, presample=20, save_realtime=[30,45,60], nograph);
plot_shock_decomposition(use_shock_groups=grp, steadystate, nograph);
@#endif

@#if CASE == 4
// calibration with a prescribed initial state of the filter
filter_initial_state;
a(0) = 0.4;
r(0) = 2*kappa + 0.1;
end;
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=est_nk_data, filtered_vars, nobs=50);
shock_decomposition(parameter_set=calibration, nograph);
@#endif

@#if CASE == 5
// as case 4 with the univariate filter
filter_initial_state;
a(0) = 0.4;
r(0) = 2*kappa + 0.1;
end;
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=est_nk_data, filtered_vars, nobs=50, kalman_algo=2);
@#endif
