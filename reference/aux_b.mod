// Model-local variables (#) and leads/lags beyond one period, which the preprocessor
// turns into auxiliary variables.
var c k a r z;
varexo e u;
parameters beta alpha delta rho sigma theta;
beta=0.99; alpha=0.33; delta=0.025; rho=0.9; sigma=2; theta=0.3;

model;
c^(-sigma) = beta*c(+1)^(-sigma)*(r(+1) + 1 - delta);
r = alpha*exp(a)*k(-1)^(alpha-1);
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

shocks;
var e; stderr 0.01;
var u; stderr 0.02;
end;

steady;
check;
stoch_simul(order=2, irf=12, nograph, nocorr, nomoments);
