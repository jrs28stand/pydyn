// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/histval_initval_file/ramst_initval_file.mod and ramst_initval_file_data.m.
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// The data of ramst_initval_file_data.m (x = 1 except 1.2 in period 1, k = 13, c = 1.5 in every period)
// written as an initval block without steady and a one-period shock: the same problem as
// initval_file(datafile = ramst_initval_file_data_col_vec_mat) followed by perfect_foresight_setup,
// which this Dynare cannot run (its dseries class is missing).
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
k = 13;
c = 1.5;
end;
shocks;
var x; periods 1; values 1.2;
end;
perfect_foresight_setup(periods=200);
perfect_foresight_solver(tolf=1e-8, tolx=1e-8);
