// The same model in OccBin form.
var c pi R Rn a;
varexo e;
parameters beta kappa phi phiy rho sig;
beta = 0.995; kappa = 0.1; phi = 1.5; phiy = 0.25; rho = 0.8; sig = 0.004;
model;
    1 / c = beta * exp( a ) * R / ( c(+1) * pi(+1) );
    ( pi - 1 ) * pi = beta * ( pi(+1) - 1 ) * pi(+1) + kappa * ( c - 1 );
    Rn = ( 1 / beta ) * pi ^ phi * c ^ phiy;
    [name='zlb', relax='zlb']
    R = Rn;
    [name='zlb', bind='zlb']
    R = 1;
    a = rho * a(-1) + sig * e;
end;
occbin_constraints;
name 'zlb'; bind Rn < 1; relax Rn >= 1;
end;
steady_state_model;
    c = 1; pi = 1; Rn = 1 / beta; R = 1 / beta; a = 0;
end;
shocks; var e = 1; end;
steady; check;
stoch_simul( order = 2, irf = 0, periods = 0, nograph );
occbin_setup;
