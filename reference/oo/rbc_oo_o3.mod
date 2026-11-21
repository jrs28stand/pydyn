var y c k a;
varexo e u;
parameters alpha beta delta rho;
alpha = 0.36; beta = 0.99; delta = 0.025; rho = 0.9;

model;
c + k = exp(a)*k(-1)^alpha + (1-delta)*k(-1) + u;
1/c = beta*(1/c(+1))*(alpha*exp(a(+1))*k^(alpha-1)+1-delta);
a = rho*a(-1) + e;
y = exp(a)*k(-1)^alpha;
end;

steady_state_model;
a = 0;
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));
y = k^alpha;
c = y - delta*k;
end;

shocks;
var e; stderr 0.01;
var u; stderr 0.004;
end;

stoch_simul(order=3, pruning, irf=8, replic=10, drop=30, periods=0, nograph);
