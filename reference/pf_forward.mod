// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/deterministic_simulations/purely_forward/ar1.mod (a second variable added).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// Purely forward model: solved backwards in time, period by period.
var y w;
varexo eps;
model;
y = sqrt(y(1))*exp(eps);
w = 0.5*w(1) + 0.2*y(1)^2 + eps;
end;
initval;
y = 1; w = 0.4; eps = 0;
end;
steady(tolf=1e-14);
shocks;
var eps; periods 1 2 4:5; values 1 -1 0.3;
end;
perfect_foresight_setup(periods=12);
perfect_foresight_solver(tolf=1e-12, tolx=1e-12);
