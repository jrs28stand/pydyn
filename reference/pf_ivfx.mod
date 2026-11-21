// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/histval_initval_file/sim_exo_lead_lag_initvalf.mod (the dseries of the
// original written to pf_ivfx_data.csv; tolerances added).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// initval_file with first_obs, last_obs and nobs; auxiliary variables for leads and lags up to 4.
var c cmav k z_backward z_forward;
varexo x;
parameters alph gam delt bet aa;
alph=0.5;
gam=0.5;
delt=0.02;
bet=0.05;
aa=0.5;
model;
  c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
  c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
  z_backward=0.1*1+0.9*z_backward(-1) + (x(-4) - 1);
  z_forward=0.2*1+0.8*z_forward(+1) + (x(+4) - 1);
  cmav = 0.2*(c(-2) + c(-1) + c + c(+1) + c(+2));
end;
initval_file(datafile = 'pf_ivfx_data.csv', first_obs = 3, last_obs = 210, nobs = 208);
perfect_foresight_setup(periods=200);
perfect_foresight_solver(maxit=100, tolf=1e-8, tolx=1e-8);
