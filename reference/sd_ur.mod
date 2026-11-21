// Unit-root model (the model of est_ur.mod, written for pydyn) used to validate observation_trends
// and lik_init=2 in pydyn's smoother against Dynare 7.2. Cases: -DCASE=n (default 1).
// Data: est_ur_data.m / est_ur_miss.m (simulated). References: reference/r_shock_decomposition.m.
@#ifndef CASE
@#define CASE = 1
@#endif
var g y x yobs xobs;
varexo eg ey ex;
parameters rho lam phi mu;
rho = 0.6; lam = 0.3; phi = 0.5; mu = 0.05;

model(linear);
g = rho*g(-1) + eg;
y = y(-1) + g + ey;
x = phi*x(-1) + lam*g + ex;
yobs = y;
xobs = x;
end;

shocks;
var eg; stderr 0.4;
var ey; stderr 0.2;
var ex; stderr 0.3;
var xobs; stderr 0.05;
end;

varobs yobs xobs;

observation_trends;
yobs (mu);
xobs (2*mu - 0.1);
end;

@#if CASE == 1
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=est_ur_data, diffuse_filter, first_obs=5, filtered_vars, filter_step_ahead=[1,3]);
shock_decomposition(parameter_set=calibration, nograph);
@#endif

@#if CASE == 2
// prefilter: the trend is demeaned as well; missing observations
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=est_ur_miss, diffuse_filter, prefilter=1, filtered_vars, filter_step_ahead=[2]);
shock_decomposition(parameter_set=calibration, nograph);
@#endif

@#if CASE == 3
// lik_init=2 with an estimated trend slope; the smoother runs at the initial values (mode_compute=0)
estimated_params;
stderr eg, inv_gamma_pdf, 0.4, inf;
stderr ey, inv_gamma_pdf, 0.2, inf;
stderr ex, 0.25, inv_gamma2_pdf, 0.3, 0.5;
rho, beta_pdf, 0.5, 0.2;
lam, normal_pdf, 0.2, 0.2;
phi, 0.4, beta_pdf, 0.5, 0.2;
mu, 0.08, normal_pdf, 0.05, 0.05;
end;
estimation(datafile=est_ur_data, lik_init=2, mode_compute=0, mh_replic=0, smoother, filtered_vars, nodisplay, nograph) g y x yobs xobs;
shock_decomposition(parameter_set=prior_mean, nograph);
@#endif

@#if CASE == 4
// as case 3 with the diffuse filter and the prior mode
estimated_params;
stderr eg, inv_gamma_pdf, 0.4, inf;
stderr ey, inv_gamma_pdf, 0.2, inf;
stderr ex, 0.25, inv_gamma2_pdf, 0.3, 0.5;
rho, beta_pdf, 0.5, 0.2;
lam, normal_pdf, 0.2, 0.2;
phi, 0.4, beta_pdf, 0.5, 0.2;
mu, 0.08, normal_pdf, 0.05, 0.05;
end;
estimation(datafile=est_ur_data, diffuse_filter, mode_compute=0, mh_replic=0, smoother, nodisplay, nograph) g y x yobs xobs;
shock_decomposition(parameter_set=prior_mode, nograph);
@#endif
