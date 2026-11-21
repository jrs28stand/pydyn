// Adapted in 2026 from reference/est_nk.mod: no measurement error, uncorrelated shocks, DSGE-VAR estimation.
// Small New Keynesian model used to validate pydyn.estimation against Dynare 7.2.
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
@#if VARIANT == 3
piobs = pi;
robs = r;
@#else
piobs = pibar + pi;
robs = rbar + pibar + r;
@#endif
end;

@#if VARIANT != 2
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
@#else
// Dynare 7.2 refuses dsge_var=Inf with a steady_state_model block (the weight is an infinite parameter)
initval;
piobs = pibar; robs = rbar + pibar;
end;
@#endif

shocks;
var ea; stderr 0.5;
var eu; stderr 0.3;
var er; stderr 0.2;
@#if VARIANT == 9
var piobs; stderr 0.05;
@#endif
end;

estimated_params;
stderr ea, inv_gamma_pdf, 0.5, 2;
stderr eu, inv_gamma2_pdf, 0.1, 1;
stderr er, 0.2, 0.01, 2, gamma_pdf, 0.2, 0.1;
@#if VARIANT == 9
stderr piobs, inv_gamma_pdf, 0.05, inf;
@#endif
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
@#if VARIANT == 1
dsge_prior_weight, 0.9, gamma_pdf, 1, 0.4;
@#endif
end;

varobs yobs piobs robs;

@#if VARIANT == 0
estimation(datafile=est_nk_data, first_obs=3, dsge_var=0.8, dsge_varlag=2, mode_compute=0, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 1
estimation(datafile=est_nk_data, first_obs=5, nobs=100, dsge_var, mode_compute=4, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 2
estimation(datafile=est_nk_data, first_obs=2, dsge_var=Inf, dsge_varlag=1, mode_compute=0, mh_replic=0, nodisplay);
@#endif
