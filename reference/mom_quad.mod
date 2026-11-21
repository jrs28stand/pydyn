// AR(1) state x and a static quadratic variable y = x + 0.5 x^2: the pruned second-order moments
// are known in closed form (used by tests/test_moments.py; no Dynare reference needed).
var x y;
varexo e;
parameters rho;
rho = 0.8;
model;
x = rho*x(-1) + e;
y = x + 0.5*x^2;
end;
steady_state_model;
x = 0;
y = 0;
end;
shocks;
var e; stderr 0.1;
end;
stoch_simul(order=2, pruning, irf=0, nograph);
