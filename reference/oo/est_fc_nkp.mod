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
piobs = pi;
robs = r;
end;

steady_state_model;
y = 0; pi = 0; r = 0; a = 0; u = 0;
yobs = 0; piobs = 0; robs = 0;
end;

shocks;
var ea; stderr 0.5;
var eu; stderr 0.3;
var er; stderr 0.2;
corr ea, eu = 0.1;
var yobs; stderr 0.1;
end;

estimated_params;
stderr ea, inv_gamma_pdf, 0.5, 2;
stderr eu, inv_gamma2_pdf, 0.1, 1;
stderr er, 0.2, 0.01, 2, gamma_pdf, 0.2, 0.1;
stderr yobs, 0.12, uniform_pdf, , , 0, 1;
corr ea, eu, 0.05, -0.9, 0.9, normal_pdf, 0, 0.3;
kappa, gamma_pdf, 0.1, 0.05;
phipi, 1.6, 1.01, 4, normal_pdf, 1.5, 0.25;
rhoa, beta_pdf, 0.7, 0.1;
rhor, uniform_pdf, 0.5, 0.25;
sig, 1.4, weibull_pdf, 1.5, 0.5;
end;

varobs yobs piobs robs;
options_.estimation.moments_posterior_density.indicator = 0;
steady;
check;
estimation(datafile=est_nk_data, nobs=50, first_obs=4, prefilter=1, mode_compute=0, mode_file=est_fc_nkp_mode, mh_replic=50, mh_nblocks=2, mh_jscale=0.9, mh_drop=0.3, mh_conf_sig=0.8, sub_draws=10, forecast=4, smoother, filtered_vars, filter_step_ahead=[1 2], silent_optimizer, nograph, nodiagnostic) yobs piobs robs y;
