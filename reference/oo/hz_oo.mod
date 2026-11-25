// A model that is linear in its variables, stoch_simul at order 2: Dynare reverts to order 1 because M_.hessian_eq_zero (stoch_simul.m);
// oo_.gamma_y then has ar+2 cells and no order-2 mean correction
var y pi r;
varexo e u;
parameters b k rho phi;
b = 0.99; k = 0.1; rho = 0.7; phi = 1.5;
model;
pi = b*pi(+1) + k*y + u;
y = y(+1) - (r - pi(+1));
r = rho*r(-1) + (1-rho)*phi*pi + e;
end;
shocks;
var e; stderr 0.01;
var u; stderr 0.005;
end;
steady;
stoch_simul(order=2, ar=2, irf=8, periods=0, nograph);
