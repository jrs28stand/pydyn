// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/deterministic_simulations/ramst_vec.mod and ramst_mshocks_vec.mod (both vector-valued blocks in one file; tolf 1e-8).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
var c k;
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
end;
initval;
x = 1;
k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
c = aa*k^alph-delt*k;
end;
steady;
a=[1.2; 1.1];
shocks;
var x;
periods 1:2;
values (a);
end;
xvalues = [1.2; 0.8; 0.8];
mshocks;
var x;
periods 4:6;
values (xvalues);
end;
perfect_foresight_setup(periods=200);
perfect_foresight_solver(tolf=1e-8);
