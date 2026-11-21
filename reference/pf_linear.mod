// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// model(linear): Dynare solves the stacked system in one linear step.
var y pi i u;
varexo e eu;
parameters beta kappa rho phipi phiy rhou;
beta=0.99; kappa=0.1; rho=0.7; phipi=1.5; phiy=0.25; rhou=0.8;
model(linear);
y = y(+1) - (i - pi(+1)) + u;
pi = beta*pi(+1) + kappa*y + 0.05*pi(-1);
i = rho*i(-1) + (1-rho)*(phipi*pi + phiy*y) + e;
u = rhou*u(-1) + eu;
end;
shocks;
var e; periods 1:3; values 0.01;
var eu; periods 2; values -0.02;
end;
perfect_foresight_setup(periods=80);
perfect_foresight_solver;
