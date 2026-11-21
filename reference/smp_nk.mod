// Small New Keynesian model of reference/post_nk.mod with the estimation commands used to validate pydyn.samplers
// against Dynare 7.2 (slice, TaRB, independent and Student random-walk Metropolis-Hastings, SMC). Cases are selected
// with -DCASE=n; see reference/r_samplers.m.
@#define VARIANT = 0
@#ifndef CASE
@#define CASE = 0
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
@#if CASE == 8
// tight estimated_params bounds on rhoa (posterior mode 0.857): Dynare's samplers reject proposals above 0.86
rhoa, 0.85, 0.5, 0.86, beta_pdf, 0.7, 0.1;
@#else
rhoa, beta_pdf, 0.7, 0.1;
@#endif
rhor, uniform_pdf, 0.5, 0.25;
sig, 1.4, weibull_pdf, 1.5, 0.5;
pibar, beta_pdf, 0.6, 0.2, -1, 3;
end;

varobs yobs piobs robs;

@#if CASE == 0
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=0, mh_jscale=0.45, nodisplay, nograph);
@#endif
// slice, default options: two chains from prior draws, univariate slice within the estimation bounds (prior_trunc=1e-10)
@#if CASE == 1
estimation(datafile=est_nk_data, mh_replic=40, mh_nblocks=2, posterior_sampling_method='slice', sub_draws=0, nodisplay, nograph);
@#endif
// rotated slice initialised at the mode (mode file), widths 3 sqrt(eigenvalues)
@#if CASE == 3
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=30, mh_nblocks=2, mh_jscale=0.5,
           posterior_sampling_method='slice', posterior_sampler_options=('rotated',1,'slice_initialize_with_mode',1),
           sub_draws=0, nodisplay, nograph);
@#endif
// TaRB with default options (csminwel on each block)
@#if CASE == 5
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=12, mh_nblocks=1, mh_jscale=0.6,
           posterior_sampling_method='tailored_random_block_metropolis_hastings', sub_draws=0, nodisplay, nograph);
@#endif
// independent Metropolis-Hastings, normal proposal
@#if CASE == 6
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=400, mh_nblocks=2, mh_jscale=1.2,
           posterior_sampling_method='independent_metropolis_hastings', sub_draws=0, nodisplay, nograph);
@#endif
// RWMH with the estimated_params bounds of CASE 8 (rhoa <= 0.86)
@#if CASE == 8
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=600, mh_nblocks=1, mh_jscale=0.5,
           sub_draws=0, nodisplay, nograph);
@#endif
// TaRB as CASE 5, four draws, with every block logged (r_samplers.m instruments posterior_sampler_iteration.m)
@#if CASE == 13
estimation(datafile=est_nk_data, mode_compute=0, mode_file=post_nk_mode0, mh_replic=4, mh_nblocks=1, mh_jscale=0.6,
           posterior_sampling_method='tailored_random_block_metropolis_hastings', sub_draws=0, nodisplay, nograph);
@#endif
