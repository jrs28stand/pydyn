// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/deterministic_simulations/homotopy.mod (dates removed).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// 80 percent of the shock, then extrapolation by linearization.
var Consumption, Capital, LoggedProductivity;
varexo LoggedProductivityInnovation;
parameters beta, alpha, delta, rho;
beta = .985; alpha = 1/3; delta = alpha/10; rho = .9;
model;
1/Consumption = beta/Consumption(1)*(alpha*exp(LoggedProductivity(1))*Capital^(alpha-1)+1-delta);
Capital = exp(LoggedProductivity)*Capital(-1)^alpha+(1-delta)*Capital(-1)-Consumption;
LoggedProductivity = rho*LoggedProductivity(-1)+LoggedProductivityInnovation;
end;
steady_state_model;
LoggedProductivity = LoggedProductivityInnovation/(1-rho);
Capital = (exp(LoggedProductivity)*alpha/(1/beta-1+delta))^(1/(1-alpha));
Consumption = exp(LoggedProductivity)*Capital^alpha-delta*Capital;
end;
initval;
LoggedProductivityInnovation = 0;
end;
steady;
endval;
LoggedProductivityInnovation = 1;
end;
steady;
perfect_foresight_setup(periods=200);
perfect_foresight_solver(tolf=1e-6, homotopy_max_completion_share=0.8, homotopy_linearization_fallback);
