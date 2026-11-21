// pydyn test: options of steady, check and stoch_simul accumulate in options_ in file order (a later
// statement only overrides the options it sets); check takes steady options too.
var y c k a;
varexo e;
parameters alpha beta delta rho;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9;
model;
1/c = beta/c(+1)*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
y = exp(a)*k(-1)^alpha;
k = y - c + (1-delta)*k(-1);
a = rho*a(-1) + e;
end;
shocks; var e; stderr 0.01; end;
initval; k = 28; c = 2.3; y = 3; a = 0; end;
steady(solve_algo=9);
stoch_simul(order=1, qz_zero_threshold=1e-7, hp_filter=1600, ar=2, irf=0, nograph);
check(qz_zero_threshold=1e-8, tolf=1e-9);
steady(maxit=30);
stoch_simul(order=1, irf=0, nograph);
