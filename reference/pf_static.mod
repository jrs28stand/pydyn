// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Static model (no lag, no lead): one nonlinear system per period.
var y w;
varexo e;
model;
y = exp(e)*w^0.5;
w + y^1.5 = 3;
end;
initval;
y = 1; w = 2; e = 0;
end;
steady(tolf=1e-14);
shocks;
var e; periods 2:4 7; values 0.3 -0.5;
end;
perfect_foresight_setup(periods=10);
perfect_foresight_solver(tolf=1e-12, tolx=1e-12);
