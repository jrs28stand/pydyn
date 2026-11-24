var i r pi p;
varexo epsilon;
parameters r_ phi chi;
chi = 0.01;
r_  = log( 1 / 0.99 );
phi = 1.5;
model;
    r = r_ + epsilon;
    [name='zlb', relax='zlb']
    i = r + phi * pi + chi * p;
    [name='zlb', bind='zlb']
    i = 0;
    i = r + pi(+1);
    p = p(-1) + pi;
end;
occbin_constraints;
name 'zlb'; bind r + phi*pi + chi*p < 0; relax r + phi*pi + chi*p >= 0;
end;
shocks; var epsilon = 0.01; end;
steady_state_model; i = r_; r = r_; pi = 0; p = 0; end;
steady; check;
occbin_setup;
occbin_solver(simul_periods=20);
