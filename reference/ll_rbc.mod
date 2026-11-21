// pydyn validation of the loglinear option: an RBC model whose steady state is strictly positive,
// with a lag of two on capital (an endogenous auxiliary variable, logged by Dynare) and lagged and
// led shocks (exogenous auxiliary variables of types 2/3, which Dynare keeps in levels).
var c k y z inv;
varexo e u;
parameters beta alpha delta rho sigma theta;
beta = 0.99; alpha = 0.33; delta = 0.025; rho = 0.9; sigma = 2; theta = 0.3;
model;
c^(-sigma) = beta*c(+1)^(-sigma)*(alpha*z(+1)*k^(alpha-1) + 1 - delta);
y = z*k(-1)^alpha;
inv = k - (1-delta)*k(-1) + 0.01*(k(-1) - k(-2));
c + inv = y;
z = 1 - rho + rho*z(-1) + e + theta*e(-1) + 0.1*u(-2) + 0.05*u(+1);
end;
steady_state_model;
z = 1;
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));
y = k^alpha;
inv = delta*k;
c = y - inv;
end;
shocks;
var e; stderr 0.01;
var u; stderr 0.02;
end;
steady;
check;
@#ifndef ORDER
@#define ORDER = 1
@#endif
stoch_simul(order=@{ORDER}, loglinear, irf=12, ar=3, nograph, conditional_variance_decomposition=[1 4]);
