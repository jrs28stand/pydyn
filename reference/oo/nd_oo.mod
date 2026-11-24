// N-d fields of oo_ (tests/test_results.py; Dynare 7.2 under Octave 8.4): conditional variance decomposition,
// smoother variances and state uncertainty, k-step-ahead filtered variables and variances, shock and
// initial-condition decompositions.  The model, priors and data of cs1.mod.
var m P c e W R k d n l gy_obs gp_obs y dA;
varexo e_a e_m;

parameters alp bet gam mst rho psi del;

alp = 0.33;
bet = 0.99;
gam = 0.003;
mst = 1.011;
rho = 0.7;
psi = 0.787;
del = 0.02;

model;
dA = exp(gam+e_a);
log(m) = (1-rho)*log(mst) + rho*log(m(-1))+e_m;
-P/(c(+1)*P(+1)*m)+bet*P(+1)*(alp*exp(-alp*(gam+log(e(+1))))*k^(alp-1)*n(+1)^(1-alp)+(1-del)*exp(-(gam+log(e(+1)))))/(c(+2)*P(+2)*m(+1))=0;
W = l/n;
-(psi/(1-psi))*(c*P/(1-n))+l/n = 0;
R = P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(-alp)/W;
1/(c*P)-bet*P*(1-alp)*exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)/(m*l*c(+1)*P(+1)) = 0;
c+k = exp(-alp*(gam+e_a))*k(-1)^alp*n^(1-alp)+(1-del)*exp(-(gam+e_a))*k(-1);
P*c = m;
m-1+d = l;
e = exp(e_a);
y = k(-1)^alp*n^(1-alp)*exp(-alp*(gam+e_a));
gy_obs = dA*y/y(-1);
gp_obs = (P/P(-1))*m(-1)/dA;
end;

steady_state_model;
  dA = exp(gam);
  gst = 1/dA;
  m = mst;
  khst = ( (1-gst*bet*(1-del)) / (alp*gst^alp*bet) )^(1/(alp-1));
  xist = ( ((khst*gst)^alp - (1-gst*(1-del))*khst)/mst )^(-1);
  nust = psi*mst^2/( (1-alp)*(1-psi)*bet*gst^alp*khst^alp );
  n  = xist/(nust+xist);
  P  = xist + nust;
  k  = khst*n;

  l  = psi*mst*n/( (1-psi)*(1-n) );
  c  = mst/P;
  d  = l - mst + 1;
  y  = k^alp*n^(1-alp)*gst^alp;
  R  = mst/bet;
  W  = l/n;
  ist  = y-c;
  q  = 1 - d;

  e = 1;

  gp_obs = m/dA;
  gy_obs = dA;
end;


shocks;
var e_a; stderr 0.014;
var e_m; stderr 0.005;
end;

steady;

check;

estimated_params;
alp, beta_pdf, 0.356, 0.02;
bet, beta_pdf, 0.993, 0.002;
gam, normal_pdf, 0.0085, 0.003;
mst, normal_pdf, 1.0002, 0.007,1.0002-3*0.007,1.0002+3*0.007;
rho, beta_pdf, 0.129, 0.05;
psi, beta_pdf, 0.65, 0.05;
del, beta_pdf, 0.01, 0.005;
stderr e_a, inv_gamma_pdf, 0.035449, inf;
stderr e_m, inv_gamma_pdf, 0.008862, inf;
end;

varobs gp_obs gy_obs;

options_.solve_tolf = 1e-12;


stoch_simul(order=1, irf=0, periods=0, conditional_variance_decomposition=[1 4 10], nograph) gy_obs gp_obs y;
cvd = oo_.conditional_variance_decomposition;
cvd_sz = size(cvd);
calib_smoother(datafile=fsdat_simul,nobs=192,loglinear,filtered_vars,filter_step_ahead=[1 2 4],filter_covariance,smoothed_state_uncertainty) gp_obs gy_obs;
kv = oo_.FilteredVariablesKStepAheadVariances;
k2 = squeeze(oo_.FilteredVariablesKStepAhead(2, :, 10));
su = squeeze(oo_.Smoother.State_uncertainty(3, 3, :));
shock_decomposition(parameter_set=calibration, nograph) y gy_obs;
sd = oo_.shock_decomposition;
sd_sum = squeeze(sum(sd(:, 1:end-1, :), 2));
initial_condition_decomposition y;
icd_sz = size(oo_.initval_decomposition);
realtime_shock_decomposition(parameter_set=calibration, forecast=2, save_realtime=[5 10 190], presample=3, fast_realtime=180) y;
rt_sz = size(oo_.realtime_shock_decomposition.pool);
shock_decomposition(parameter_set=calibration) gy_obs y;
