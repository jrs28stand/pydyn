// Random numbers drawn in MATLAB code and by Dynare's simulations: one stream, in the order of the file
// (tests/test_results.py; Dynare 7.2 under Octave 8.4)
var y c k a;
varexo e u;
parameters alpha beta delta rho;
alpha = 0.36; beta = 0.99; delta = 0.025;
z0 = rand(1, 3);
rho = 0.85 + 0.1*rand;
n0 = randn(2, 2);
ri = randi(5, 1, 4);
rp = randperm(6);

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

steady;
stoch_simul(order=1, irf=0, periods=60, drop=10, nograph) y c k;
sim1 = oo_.endo_simul;
n1 = randn(1, 3);
u1 = rand(1, 2);
set_dynare_seed(7);
n2 = randn(1, 2);
stoch_simul(order=1, irf=0, periods=30, drop=5, nograph) y c k;
sim2 = oo_.endo_simul;
rng(3);
n3 = randn;
stoch_simul(order=1, irf=0, periods=12, drop=2, nograph) y;
n4 = randn(1, 2);
