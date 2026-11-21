// pydyn test (finding R21): a parameter declared but never set is NaN in M_.params. The model does
// not use it: Dynare 7.2 solves the model at orders 1 and 2 (its MATLAB solvers check the
// derivatives), and refuses at order 3 (k_order_perturbation checks every parameter).
var y c k a;
varexo e;
parameters alpha beta delta rho unused;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9;
model;
1/c = beta/c(+1)*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
y = exp(a)*k(-1)^alpha;
k = y - c + (1-delta)*k(-1);
a = rho*a(-1) + e;
end;
shocks; var e; stderr 0.01; end;
steady_state_model;
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1)); y = k^alpha; c = y - delta*k; a = 0;
end;
stoch_simul(order=2, irf=0, nomoments, noprint);
dr2 = oo_.dr;
stoch_simul(order=3, irf=0, nomoments, noprint);
