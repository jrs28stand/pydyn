// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/deterministic_simulations/ramst_pf_controlled_paths.mod and ramst_pf_controlled_paths_common.inc (inlined) (dated periods 2003Y:2005Y written as 3:5 (first_simulation_period removed); tolf 1e-8; checks removed).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// Performs a simulation with a perfect_foresight_controlled_paths block
// with the default stack_solve_algo=0

var c k y;
varexo x e_y;

parameters alph gam delt bet aa;
alph=0.5;
gam=0.5;
delt=0.02;
bet=0.05;
aa=0.5;


model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
y = 0.98*y(-1)+e_y;
end;

initval;
x = 1;
k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
c = aa*k^alph-delt*k;
y = 0;
end;

steady;

perfect_foresight_controlled_paths;
  exogenize c;
  periods 2 4:5;
  values 1.6 1.7;
  endogenize x;

  exogenize k;
  periods 7:9;
  values 13;
  endogenize x;

  exogenize y;
  periods 3:5;
  values 0.5;
  endogenize e_y;
end;

shocks;
  var x;
  periods 1;
  values 1.2;
end;


perfect_foresight_setup(periods=30);
perfect_foresight_solver(stack_solve_algo = 0, tolf=1e-8);


if oo_.endo_simul(1, 3) ~= 1.6 || any(oo_.endo_simul(1, 5:6) ~= 1.7) || any(oo_.endo_simul(2, 8:10) ~= 13) ...
   || any(oo_.endo_simul(3, 4:6) ~= 0.5) || oo_.exo_simul(2, 1) ~= 1.2
   error('Perfect foresight with controlled paths failed')
end

