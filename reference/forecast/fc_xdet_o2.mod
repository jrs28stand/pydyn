// Adapted in 2026 from Dynare's tests/forecast/example1_varexo_det.mod (GPL-3.0-or-later, Dynare Team): forecasts with varexo_det.
// Test for varexo_det and forecast command at order 1

var y, c, k, a, h, b;
varexo e,u;
varexo_det ahat, bhat;

parameters beta, rho, rho2, alpha, delta, theta, psi, tau;

alpha = 0.36;
rho   = 0.95;
tau   = 0.025;
beta  = 0.99;
delta = 0.025;
psi   = 0;
theta = 2.95;
rho2 = -0.1;

phi   = 0.1;

model;
c*theta*h^(1+psi)=(1-alpha)*y;
k = beta*(((exp(b)*c)/(exp(b(+1))*c(+1)))
    *(exp(b(+1))*alpha*y(+1)+(1-delta)*k));
y = exp(a)*(k(-1)^alpha)*(h^(1-alpha));
k = exp(b)*(y-c)+(1-delta)*k(-1);
a = ahat + rho*a(-1) + rho2*a(-2) + tau*b(-1) + e;
b = bhat + tau*a(-1) + rho*b(-1) + u;
end;

initval;
y = 1.08068253095672;
c = 0.80359242014163;
h = 0.29175631001732;
k = 5;
a = 0;
b = 0;
e = 0;
u = 0;
ahat = 0;
bhat = 0;
end;

shocks;
var e; stderr 0.009;
var u; stderr 0.009;
var e, u = phi*0.009*0.009;
var ahat;
periods 1;
values 0.1;
var bhat;
periods 2 3;
values 0.2 -0.05;
end;

stoch_simul(order=2, irf=0, noprint);
ys_ = oo_.dr.ys;
options_.forecast_replic=150;
randn('state', 5);
forecast(periods=5, conf_sig=0.8) y c k a;
fc1 = oo_.forecast;
varobs y c;
shocks;
var y = 0.01^2;
var c = 0.02^2;
end;
randn('state', 6);
forecast(periods=7, conf_sig=0.9) y c k;
fc2 = oo_.forecast;

