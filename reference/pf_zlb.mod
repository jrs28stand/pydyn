// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Zero lower bound through a complementarity condition ([mcp = ...] tag) and the lmmcp option:
// the bound on the interest rate binds for several periods.
var y pi i rn;
varexo e;
parameters beta sigma kappa phipi phiy rho;
beta=0.99; sigma=1; kappa=0.05; phipi=1.5; phiy=0.5; rho=0.8;
model;
y = y(+1) - (1/sigma)*(i - pi(+1) - rn);
pi = beta*pi(+1) + kappa*y + 0.5*kappa*y^2;
rn = rho*rn(-1) + e;
[mcp = 'i > -0.01']
i = phipi*pi + phiy*y;
end;
initval;
y=0; pi=0; i=0; rn=0; e=0;
end;
steady;
shocks;
var e; periods 1:3; values -0.03;
end;
options_.lmmcp.TolFun = 1e-10;
perfect_foresight_setup(periods=60);
perfect_foresight_solver(lmmcp, no_homotopy);
