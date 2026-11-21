// Adapted in 2026 from reference/est_nk.mod: variant 20 adds endogenous_prior and mode_compute=4.
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

@#if VARIANT == 20
estimation(datafile=est_nk_data, endogenous_prior, mode_compute=4, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 0
estimation(datafile=est_nk_data, mode_compute=0, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 1
estimation(datafile=est_nk_data, first_obs=11, nobs=80, presample=4, mode_compute=0, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 2
estimation(datafile=est_nk_miss, mode_compute=0, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 3
estimation(datafile=est_nk_data, prefilter=1, mode_compute=0, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 4
estimation(datafile=est_nk_data, lik_init=2, mode_compute=0, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 5
estimation(datafile=est_nk_data, mode_compute=0, mh_replic=0, smoother, nodisplay) y pi r a u yobs piobs robs;
@#endif
@#if VARIANT == 6
estimation(datafile=est_nk_data, mode_compute=4, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 7
estimation(datafile=est_nk_miss, mode_compute=0, mh_replic=0, smoother, nodisplay) y pi r a u yobs piobs robs;
@#endif
@#if VARIANT == 8
estimation(datafile=est_nk_miss, kalman_algo=2, mode_compute=0, mh_replic=0, nodisplay);
@#endif
@#if VARIANT == 9
// correlated measurement errors, with missing observations
estimation(datafile=est_nk_miss, mode_compute=0, mh_replic=0, nodisplay);
@#endif
