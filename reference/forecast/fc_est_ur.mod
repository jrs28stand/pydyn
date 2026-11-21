// Adapted in 2026 from reference/est_ur.mod (pydyn): forecasts after estimation, variants as there; variant 10 adds prefilter to variant 5.
// Model with a unit root, used to validate the diffuse filter (lik_init=3) and lik_init=2
// of pydyn.estimation against Dynare 7.2. Variants: -DVARIANT=n (default 0).
@#ifndef VARIANT
@#define VARIANT = 0
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

@#if VARIANT == 4
// maximum likelihood: initial values and bounds only (stderr ey stays calibrated: the
// likelihood of this sample is maximised at ey -> 0, on the bound)
estimated_params;
stderr eg, 0.3, 0.01, 3;
stderr ex, 0.25, 0.01, 3;
rho, 0.5, 0, 0.99;
lam, 0.2, -2, 2;
phi, 0.4, 0, 0.99;
end;
@#else
estimated_params;
stderr eg, inv_gamma_pdf, 0.4, inf;
stderr ey, inv_gamma_pdf, 0.2, inf;
stderr ex, 0.25, inv_gamma2_pdf, 0.3, 0.5;
rho, beta_pdf, 0.5, 0.2;
lam, normal_pdf, 0.2, 0.2;
phi, 0.4, beta_pdf, 0.5, 0.2;
@#if VARIANT == 5 || VARIANT == 10
mu, normal_pdf, 0.05, 0.05;
@#endif
end;
@#endif

varobs yobs xobs;

@#if VARIANT == 5 || VARIANT == 10
// linear trends in the measurement equations, with an estimated slope
observation_trends;
yobs (mu);
xobs (2*mu - 0.1);
end;
@#endif

@#if VARIANT == 0
estimation(forecast=6, smoother, conf_sig=0.8, datafile=est_ur_data, diffuse_filter, mode_compute=0, mh_replic=0, nodisplay) g y x yobs xobs;
fc = oo_.forecast;
ys_ = oo_.dr.ys;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
elseif isfield(oo_, 'mle_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
else
  xp = [];
end

@#endif
@#if VARIANT == 1
estimation(forecast=6, smoother, conf_sig=0.8, datafile=est_ur_data, lik_init=2, mode_compute=0, mh_replic=0, nodisplay) g y x yobs xobs;
fc = oo_.forecast;
ys_ = oo_.dr.ys;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
elseif isfield(oo_, 'mle_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
else
  xp = [];
end

@#endif
@#if VARIANT == 2
estimation(forecast=6, smoother, conf_sig=0.8, datafile=est_ur_miss, diffuse_filter, presample=3, mode_compute=0, mh_replic=0, nodisplay) g y x yobs xobs;
fc = oo_.forecast;
ys_ = oo_.dr.ys;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
elseif isfield(oo_, 'mle_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
else
  xp = [];
end

@#endif
@#if VARIANT == 3
estimation(forecast=6, smoother, conf_sig=0.8, datafile=est_ur_data, diffuse_filter, mode_compute=4, mh_replic=0, nodisplay) g y x yobs xobs;
fc = oo_.forecast;
ys_ = oo_.dr.ys;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
elseif isfield(oo_, 'mle_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
else
  xp = [];
end

@#endif
@#if VARIANT == 4
estimation(forecast=6, smoother, conf_sig=0.8, datafile=est_ur_data, diffuse_filter, mode_compute=4, mh_replic=0, nodisplay) g y x yobs xobs;
fc = oo_.forecast;
ys_ = oo_.dr.ys;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
elseif isfield(oo_, 'mle_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
else
  xp = [];
end

@#endif
@#if VARIANT == 5
estimation(forecast=6, smoother, conf_sig=0.8, datafile=est_ur_data, diffuse_filter, first_obs=5, mode_compute=0, mh_replic=0, nodisplay) g y x yobs xobs;
fc = oo_.forecast;
ys_ = oo_.dr.ys;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
elseif isfield(oo_, 'mle_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
else
  xp = [];
end

@#endif
@#if VARIANT == 6
estimation(forecast=6, smoother, conf_sig=0.8, datafile=est_ur_miss, kalman_algo=4, presample=2, mode_compute=0, mh_replic=0, nodisplay) g y x yobs xobs;
fc = oo_.forecast;
ys_ = oo_.dr.ys;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
elseif isfield(oo_, 'mle_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
else
  xp = [];
end

@#endif
@#if VARIANT == 10
estimation(forecast=6, smoother, conf_sig=0.8, datafile=est_ur_data, diffuse_filter, prefilter=1, first_obs=5, mode_compute=0, mh_replic=0, nodisplay) g y x yobs xobs;

fc = oo_.forecast;
ys_ = oo_.dr.ys;
if isfield(oo_, 'posterior_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
elseif isfield(oo_, 'mle_mode')
  xp = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
else
  xp = [];
end
@#endif
