/*
 * This file replicates the estimation of the cash in advance model (termed M1
 * in the paper) described in Frank Schorfheide (2000): "Loss function-based
 * evaluation of DSGE models", Journal of Applied Econometrics, 15(6), 645-670.
 *
 * The data are taken from the replication package at
 * http://dx.doi.org/10.15456/jae.2022314.0708799949
 *
 * The prior distribution follows the one originally specified in Schorfheide's
 * paper. Note that the elicited beta prior for rho in the paper
 * implies an asymptote and corresponding prior mode at 0. It is generally
 * recommended to avoid this extreme type of prior.
 *
 * Because the data are already logged and we use the loglinear option to conduct
 * a full log-linearization, we need to use the logdata option.
 *
 * The equations are taken from J. Nason and T. Cogley (1994): "Testing the
 * implications of long-run neutrality for monetary business cycle models",
 * Journal of Applied Econometrics, 9, S37-S70, NC in the following.
 * Note that there is an initial minus sign missing in equation (A1), p. S63.
 *
 * This implementation was originally written by Michel Juillard. Please note that the
 * following copyright notice only applies to this Dynare implementation of the
 * model.
 */

/*
 * Copyright © 2004-2023 Dynare Team
 *
 * This file is part of Dynare.
 *
 * Dynare is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Dynare is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Dynare.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * Modified in 2026 for pydyn (reference/smo_fs.mod): priors and estimation removed, tags removed,
 * calib_smoother / shock_decomposition cases added, selected by the macro variable CASE.
 * The data files smo_fs_data.m / smo_fs_logdata.m are simulated (reference/smo_make_data.py).
 */

@#ifndef CASE
@#define CASE = 1
@#endif

var m P c e W R k d n l gy_obs gp_obs y dA;
varexo e_a e_m;
parameters alp bet gam logmst rho phi del;

alp = 0.33;
bet = 0.99;
gam = 0.003;
logmst = log(1.011);
rho = 0.7;
phi = 0.787;
del = 0.02;

model;
dA = exp(gam+e_a);
log(m) = (1-rho)*logmst + rho*log(m(-1))+e_m;
-P/(c(+1)*P(+1)*m)+bet*P(+1)*(alp*exp(-alp*(gam+log(e(+1))))*k^(alp-1)*n(+1)^(1-alp)+(1-del)*exp(-(gam+log(e(+1)))))/(c(+2)*P(+2)*m(+1))=0;
W = l/n;
-(phi/(1-phi))*(c*P/(1-n))+l/n = 0;
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
  e = 1;
  gp_obs = m/dA;
  gy_obs = dA;
end;

varobs gp_obs gy_obs;

@#if CASE == 1
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=smo_fs_data, loglinear, filtered_vars, filter_step_ahead=[1,2,4], filter_covariance, smoothed_state_uncertainty);
shock_decomposition(parameter_set=calibration, nograph);
@#endif

@#if CASE == 2
set_param_value('rho', 0.5);
set_param_value('alp', 0.36);
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=smo_fs_data, loglinear, first_obs=11, nobs=80, kalman_algo=2, filtered_vars, smoothed_state_uncertainty);
shock_decomposition(parameter_set=calibration, init_state=1, nograph);
@#endif

@#if CASE == 3
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
options_.logdata = 1;   // calib_smoother has no logdata option
calib_smoother(datafile=smo_fs_logdata, loglinear, nobs=60);
@#endif

@#if CASE == 4
steady;
stoch_simul(order=1, irf=0, nomoments, noprint);
calib_smoother(datafile=smo_fs_data, filtered_vars, nobs=100);
shock_decomposition(parameter_set=calibration, nograph);
@#endif
