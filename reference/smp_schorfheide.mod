// Schorfheide (2000) cash-in-advance model (Dynare's examples/estimation/schorfheide_2000.mod, Copyright (C) 2004-2023
// Dynare Team, GPL-3.0-or-later; as reference/post_schorfheide.mod) with the estimation commands used to validate
// pydyn.samplers against Dynare 7.2. Cases are selected with -DCASE=n; see reference/r_samplers.m.
@#ifndef CASE
@#define CASE = 0
@#endif

var m       ${m}$           (long_name='money growth')
    P       ${P}$           (long_name='Price level')
    c       ${c}$           (long_name='consumption')
    e       ${e}$           (long_name='capital stock')
    W       ${W}$           (long_name='Wage rate')
    R       ${R}$           (long_name='interest rate')
    k       ${k}$           (long_name='capital stock')
    d       ${d}$           (long_name='dividends')
    n       ${n}$           (long_name='labor')
    l       ${l}$           (long_name='loans')
    gy_obs  ${\Delta \ln GDP}$  (long_name='detrended capital stock')
    gp_obs  ${\Delta \ln P}$    (long_name='detrended capital stock')
    y       ${y}$           (long_name='detrended output')
    dA      ${\Delta A}$    (long_name='TFP growth')
    ;
varexo e_a  ${\epsilon_A}$      (long_name='TFP shock')
    e_m     ${\epsilon_M}$      (long_name='Money growth shock')
    ;

parameters alp  ${\alpha}$       (long_name='capital share')
    bet         ${\beta}$        (long_name='discount factor')
    gam         ${\gamma}$       (long_name='long-run TFP growth')
    logmst         ${\log(m^*)}$ (long_name='long-run money growth')
    rho         ${\rho}$         (long_name='autocorrelation money growth')
    phi         ${\phi}$         (long_name='labor weight in consumption')
    del         ${\delta}$       (long_name='depreciation rate')
    ;

% roughly picked values to allow simulating the model before estimation
alp = 0.33;
bet = 0.99;
gam = 0.003;
logmst = log(1.011);
rho = 0.7;
phi = 0.787;
del = 0.02;

model;
[name='NC before eq. (1), TFP growth equation']
dA = exp(gam+e_a);
[name='NC eq. (2), money growth rate']
log(m) = (1-rho)*logmst + rho*log(m(-1))+e_m;
[name='NC eq. (A1), Euler equation']
-P/(c(+1)*P(+1)*m)+bet*P(+1)*(alp*exp(-alp*(gam+log(e(+1))))*k^(alp-1)*n(+1)^(1-alp)+(1-del)*exp(-(gam+log(e(+1)))))/(c(+2)*P(+2)*m(+1))=0;
[name='NC below eq. (A1), firm borrowing constraint']
W = l/n;
[name='NC eq. (A2), intratemporal labour market condition']
-(phi/(1-phi))*(c*P/(1-n))+l/n = 0;
[name='NC below eq. (A2), credit market clearing']
R = P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(-alp)/W;
[name='NC eq. (A3), credit market optimality']
1/(c*P)-bet*P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)/(m*l*c(+1)*P(+1)) = 0;
[name='NC eq. (18), aggregate resource constraint']
c+k = exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)+(1-del)*exp(-(gam+e_a))*k(-1);
[name='NC eq. (19), money market condition']
P*c = m;
[name='NC eq. (20), credit market equilibrium condition']
m-1+d = l;
[name='Definition TFP shock']
e = exp(e_a);
[name='Implied by NC eq. (18), production function']
y = k(-1)^alp*n^(1-alp)*exp(-alp*(gam+e_a));
[name='Observation equation GDP growth']
gy_obs = dA*y/y(-1);
[name='Observation equation price level']
gp_obs = (P/P(-1))*m(-1)/dA;
end;

shocks;
var e_a; stderr 0.014;
var e_m; stderr 0.005;
end;

steady_state_model;
  dA = exp(gam);
  gst = 1/dA;
  m = exp(logmst);
  khst = ( (1-gst*bet*(1-del)) / (alp*gst^alp*bet) )^(1/(alp-1));
  xist = ( ((khst*gst)^alp - (1-gst*(1-del))*khst)/m )^(-1);
  nust = phi*m^2/( (1-alp)*(1-phi)*bet*gst^alp*khst^alp );
  n  = xist/(nust+xist);
  P  = xist + nust;
  k  = khst*n;

  l  = phi*m*n/( (1-phi)*(1-n) );
  c  = m/P;
  d  = l - m + 1;
  y  = k^alp*n^(1-alp)*gst^alp;
  R  = m/bet;
  W  = l/n;
  ist  = y-c;
  q  = 1 - d;

  e = 1;

  gp_obs = m/dA;
  gy_obs = dA;
end;


steady;
check;

% Table 1 of Schorfheide (2000)
estimated_params;
alp, beta_pdf, 0.356, 0.02;
bet, beta_pdf, 0.993, 0.002;
gam, normal_pdf, 0.0085, 0.003;
logmst, normal_pdf, 0.0002, 0.007;
rho, beta_pdf, 0.129, 0.223;
phi, beta_pdf, 0.65, 0.05;
del, beta_pdf, 0.01, 0.005;
stderr e_a, inv_gamma_pdf, 0.035449, inf;
stderr e_m, inv_gamma_pdf, 0.008862, inf;
end;

varobs gp_obs gy_obs;

@#if CASE == 0
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, mode_compute=0, mode_file=post_schorfheide_mode0,
           mh_replic=0, mh_jscale=0.8, nodisplay, nograph);
@#endif
// slice, one chain from the prior mode, wider initial step
@#if CASE == 2
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, mh_replic=25, mh_nblocks=1,
           posterior_sampling_method='slice', posterior_sampler_options=('initial_step_size',0.5), sub_draws=0,
           nodisplay, nograph);
@#endif
// multimodal slice: two "modes" read from smp_schorfheide_modes.mat (xparams, one column per mode)
@#if CASE == 4
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, mode_compute=0, mode_file=post_schorfheide_mode0,
           mh_replic=12, mh_nblocks=1, posterior_sampling_method='slice',
           posterior_sampler_options=('mode_files','smp_schorfheide_modes','slice_initialize_with_mode',1), sub_draws=0,
           nodisplay, nograph);
@#endif
// RWMH with a Student proposal (4 degrees of freedom)
@#if CASE == 7
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, mode_compute=0, mode_file=post_schorfheide_mode0,
           mh_replic=400, mh_nblocks=2, mh_jscale=0.8,
           posterior_sampler_options=('proposal_distribution','rand_multivariate_student','student_degrees_of_freedom',4),
           sub_draws=0, nodisplay, nograph);
@#endif
// TaRB with a Student proposal (6 degrees of freedom) and new_block_probability 0.5
@#if CASE == 9
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, mode_compute=0, mode_file=post_schorfheide_mode0,
           mh_replic=6, mh_nblocks=2, mh_jscale=0.8, posterior_sampling_method='tailored_random_block_metropolis_hastings',
           posterior_sampler_options=('proposal_distribution','rand_multivariate_student','student_degrees_of_freedom',6,
                                      'new_block_probability',0.5),
           sub_draws=0, nodisplay, nograph);
@#endif
// independent Metropolis-Hastings with a Student proposal (5 degrees of freedom)
@#if CASE == 10
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, mode_compute=0, mode_file=post_schorfheide_mode0,
           mh_replic=300, mh_nblocks=1, mh_jscale=1.1, posterior_sampling_method='independent_metropolis_hastings',
           posterior_sampler_options=('proposal_distribution','rand_multivariate_student','student_degrees_of_freedom',5),
           sub_draws=0, nodisplay, nograph);
@#endif
// Herbst-Schorfheide SMC, small
@#if CASE == 11
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, posterior_sampling_method='hssmc',
           posterior_sampler_options=('particles',300,'steps',8,'lambda',2,'target',0.3,'scale',0.4), sub_draws=0,
           nodisplay, nograph);
@#endif
// dynamic striated Metropolis-Hastings, small (with fewer particles or stages Dynare stops: the weighted covariance degenerates and chol fails)
@#if CASE == 12
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, posterior_sampling_method='dsmh',
           posterior_sampler_options=('N',30,'G',10,'M',5,'K',3,'H',15,'tau',2), sub_draws=0, nodisplay, nograph);
@#endif
// TaRB as CASE 9, two draws per chain, with every block logged (r_samplers.m instruments posterior_sampler_iteration.m)
@#if CASE == 14
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, mode_compute=0, mode_file=post_schorfheide_mode0,
           mh_replic=2, mh_nblocks=2, mh_jscale=0.8, posterior_sampling_method='tailored_random_block_metropolis_hastings',
           posterior_sampler_options=('proposal_distribution','rand_multivariate_student','student_degrees_of_freedom',6,
                                      'new_block_probability',0.5),
           sub_draws=0, nodisplay, nograph);
@#endif
// TaRB with default options, two chains of 150 draws (statistical comparison with pydyn, reference/smp_compare.py);
// -DSEED=k sets the seed
@#ifdef SEED
set_dynare_seed(@{SEED});
@#endif
@#if CASE == 15
estimation(order=1, datafile=est_schorfheide_data, loglinear, logdata, mode_compute=0, mode_file=post_schorfheide_mode0,
           mh_replic=150, mh_nblocks=2, mh_jscale=0.8, posterior_sampling_method='tailored_random_block_metropolis_hastings',
           sub_draws=0, nodisplay, nograph);
@#endif
