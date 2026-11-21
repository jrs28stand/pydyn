// order=3 copy (2026) of the file of the same name without "3", for tests/test_moments3.py.
// Model-local variables (#) and leads/lags beyond one period, which the preprocessor turns into
// auxiliary variables (copy of reference/aux_a.mod); measurement errors on c and z added.
var c k a r z;
varexo e u;
parameters beta alpha delta rho sigma theta;
beta=0.99; alpha=0.33; delta=0.025; rho=0.9; sigma=2; theta=0.3;

model;
# MUC  = c^(-sigma);
# MUC1 = c(+1)^(-sigma);
# RK   = alpha*exp(a)*k(-1)^(alpha-1);
MUC = beta*MUC1*(r(+1) + 1 - delta);
r = RK;
c + k = exp(a)*k(-1)^alpha + (1-delta)*k(-1);
a = rho*a(-1) + 0.05*a(-3) + e + theta*e(-1) + 0.2*u(-2);
z = 0.5*(c(+2) + c(+1)) - c + 0.1*k(-2) + 0.05*u(+1);
end;

steady_state_model;
a = 0;
r = 1/beta - 1 + delta;
k = (r/alpha)^(1/(alpha-1));
c = k^alpha - delta*k;
z = 0.1*k;
end;

varobs c z k;
shocks;
var e; stderr 0.01;
var u; stderr 0.02;
var c; stderr 0.05;
var z; stderr 0.003;
end;

steady;
check;
stoch_simul(order=3, nograph, irf=0);
