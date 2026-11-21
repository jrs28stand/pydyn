// Adapted in 2026 from reference/est_nk.mod (pydyn): conditional_forecast (Dynare +conditional_forecasts/run.m), validated by tests/test_forecast.py.
// Small New Keynesian model; -DVARIANT=n selects the case (default 0).
// Variants are selected with -DVARIANT=n on the command line (default 0).
@#ifndef VARIANT
@#define VARIANT = 0
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
@#if VARIANT == 3 || VARIANT == 11
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
@#if VARIANT == 3 || VARIANT == 11
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

@#if VARIANT == 0 || VARIANT == 2 || VARIANT == 3
estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, nodisplay) y pi r a u yobs piobs robs;
@#else
stoch_simul(order=1, nomoments, irf=0);
@#endif

conditional_forecast_paths;
@#if VARIANT == 3
var robs;
periods 1:3;
values 2.5;
var yobs;
periods 2:4;
values 0.4;
@#else
var robs;
periods 1:2 3;
values 2.5 2.2;
var yobs;
periods 1:3;
values 0.4;
@#endif
end;

randn('state', 77);
@#if VARIANT == 0 || VARIANT == 3
conditional_forecast(parameter_set=posterior_mode, controlled_varexo=(ea,er), periods=8, replic=300, conf_sig=0.7);
@#endif
@#if VARIANT == 1
conditional_forecast(parameter_set=calibration, controlled_varexo=(ea,er), periods=8, replic=300, conf_sig=0.7);
@#endif
@#if VARIANT == 2
conditional_forecast(parameter_set=prior_mean, controlled_varexo=(eu,er), periods=6, replic=250);
@#endif
cf = oo_.conditional_forecast;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
else
  xp = [];
end
