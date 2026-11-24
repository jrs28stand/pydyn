var y pi r rn a;
varexo e;
parameters beta phi rho rbar;
beta=0.99; phi=1.5; rho=0.8; rbar=-0.01;
model;
y = y(+1) - (r - pi(+1)) + a;
pi = beta*pi(+1) + 0.1*y;
rn = phi*pi;
[name='zlb', relax='zlb']
r = rn;
[name='zlb', bind='zlb']
r = rbar;
a = rho*a(-1) + e;
end;
occbin_constraints;
name 'zlb'; bind rn < rbar; relax rn >= rbar;
end;
shocks; var e; stderr 0.01; end;
shocks(surprise); var e; periods 1; values -0.05; end;
steady; check;
occbin_setup;
occbin_solver(simul_periods=30);
