// Two interacting constraints, both complementary pairs: a lower bound on the policy rate and a floor on
// a spending term s that supports demand. Test model for pydyn.obc (not a Dynare file).
var y pi r rn a s sn;
varexo e;
parameters beta phi rho rbar sbar;
beta=0.99; phi=1.5; rho=0.8; rbar=-0.01; sbar=-0.02;
model;
y = y(+1) - (r - pi(+1)) + a + 0.3*s;
pi = beta*pi(+1) + 0.1*y;
rn = phi*pi + 0.2*y;
[name='zlb', relax='zlb']
r = rn;
[name='zlb', bind='zlb']
r = rbar;
sn = 0.5*y;
[name='floor', relax='floor']
s = sn;
[name='floor', bind='floor']
s = sbar;
a = rho*a(-1) + e;
end;
occbin_constraints;
name 'zlb'; bind rn < rbar; relax rn >= rbar;
name 'floor'; bind sn < sbar;
end;
shocks; var e; stderr 0.01; end;
shocks(surprise); var e; periods 1 5; values -0.05 0.03; end;
steady; check;
occbin_setup;
occbin_solver(simul_periods=40);
