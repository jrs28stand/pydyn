// Adapted in 2026 from reference/pf_rbcdet.mod: shocks dated with first/last_simulation_period.
// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/deterministic_simulations/rbc_det.mod (its steady_state_model block replaced by initval values, so that 'CapitalSS = oo_.steady_state(ik)' reads a numerical steady state; solver tolf 1e-8; checks removed).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
var Capital, Output, Labour, Consumption, Efficiency, efficiency, ExpectedTerm;

varexo EfficiencyInnovation;

parameters beta, theta, tau, alpha, psi, delta, rho, effstar, sigma2;

beta    =   0.9900;
theta   =   0.3570;
tau     =   2.0000;
alpha   =   0.4500;
psi     =  -0.1000;
delta   =   0.0200;
rho     =   0.8000;
effstar =   1.0000;
sigma2  =   0;

model;

  // Eq. n°1:
  efficiency = rho*efficiency(-1) + EfficiencyInnovation;

  // Eq. n°2:
  Efficiency = effstar*exp(efficiency);

  // Eq. n°3:
  Output = Efficiency*(alpha*(Capital(-1)^psi)+(1-alpha)*(Labour^psi))^(1/psi);

  // Eq. n°4:
  Capital = Output-Consumption + (1-delta)*Capital(-1);

  // Eq. n°5:
  ((1-theta)/theta)*(Consumption/(1-Labour)) - (1-alpha)*(Output/Labour)^(1-psi);

  // Eq. n°6:
  (((Consumption^theta)*((1-Labour)^(1-theta)))^(1-tau))/Consumption  = ExpectedTerm(1);

  // Eq. n°7:
  ExpectedTerm = beta*((((Consumption^theta)*((1-Labour)^(1-theta)))^(1-tau))/Consumption)*(alpha*((Output/Capital(-1))^(1-psi))+(1-delta));

end;

initval;
Capital = 35; Output = 3.4; Labour = 0.33; Consumption = 2.7; Efficiency = 1; efficiency = 0; ExpectedTerm = 0.12;
end;

steady;

steady;
shocks;
var EfficiencyInnovation;
periods 2000Q1 2000Q3:2001Q2 2003Q4;
values 0.02 -0.01 0.015;
end;
perfect_foresight_setup(first_simulation_period=2000Q1, last_simulation_period=2049Q4);
perfect_foresight_solver(tolf=1e-8);
