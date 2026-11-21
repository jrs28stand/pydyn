// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/histval_initval_file/sim_exo_lead_lag_histvalf.mod (the dseries of the
// original written to pf_hvfx_data.csv; tolerances added).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// histval_file with first_obs and last_obs: initial conditions for lags up to 4 (auxiliary variables).
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
initval;
  x = 1;
end;
steady_state_model;
  k = ((bet + delt)/(aa*alph*x))^(1/(alph - 1));
  c = aa*x*k^alph - delt*k;
  z_backward = x;
  z_forward = x;
  cmav = c;
end;
steady;
shocks;
  var x;
  periods 2;
  values 0.9;
end;
histval_file(datafile = 'pf_hvfx_data.csv', first_obs = 3, last_obs = 6);
perfect_foresight_setup(periods=200);
perfect_foresight_solver(tolf=1e-8, tolx=1e-8);
