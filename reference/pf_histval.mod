// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/deterministic_simulations/histval_det.mod (tolerances and one shock added).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// histval: initial conditions for lags beyond one (through auxiliary variables) and for a lagged
// exogenous variable. Variables the block does not mention (c, and z in period 0) start at 0.
var c k z;
varexo x;
parameters alph gam delt bet aa;
alph=0.5; gam=0.5; delt=0.02; bet=0.05; aa=0.5;
model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
z = 0.9*z(-2);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(-1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;
initval;
x = 1;
k = 12.75;
c = 1.53;
z = 0;
end;
steady(tolf=1e-14);
histval;
k(0) = 60;
x(0) = 0.9;
z(-1) = 0.1;
end;
shocks;
var x; periods 2:3; values 1.1;
end;
perfect_foresight_setup(periods=200);
perfect_foresight_solver(tolf=1e-8, tolx=1e-8);
