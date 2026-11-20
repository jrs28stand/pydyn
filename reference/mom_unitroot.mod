// Unit root in x (random walk): x, q = x + w and p are non-stationary or depend on x; d = x - x(-1),
// w, z and v-driven z are stationary. v has zero variance.
var x w q d z p s;
varexo e u v;
parameters rho_w;
rho_w = 0.8;
model;
x = x(-1) + e;
w = rho_w*w(-1) + u;
q = x + w;
d = x - x(-1);
z = 0.5*z(-1) + w + 0.1*v;
p = 0.5*p(+1) + w;
s = 0.3*s(-1) + 0.2*x(-1) + 0.1*u;
end;
steady_state_model;
x = 0; w = 0; q = 0; d = 0; z = 0; p = 0; s = 0;
end;
shocks;
var e; stderr 0.01;
var u; stderr 0.02;
var v; stderr 0;
end;
steady;
stoch_simul(order=1, nograph, irf=0, ar=3);
