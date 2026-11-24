// Small nonlinear New Keynesian model with a lower bound on the gross policy rate, in DynareOBC's form.
var c pi R Rn a;
varexo e;
parameters beta kappa phi phiy rho sig;
beta = 0.995; kappa = 0.1; phi = 1.5; phiy = 0.25; rho = 0.8; sig = 0.004;
model;
    1 / c = beta * exp( a ) * R / ( c(+1) * pi(+1) );
    ( pi - 1 ) * pi = beta * ( pi(+1) - 1 ) * pi(+1) + kappa * ( c - 1 );
    Rn = ( 1 / beta ) * pi ^ phi * c ^ phiy;
    R = max( 1, Rn );
    a = rho * a(-1) + sig * e;
end;
steady_state_model;
    c = 1; pi = 1; Rn = 1 / beta; R = 1 / beta; a = 0;
end;
shocks; var e = 1; end;
stoch_simul( order = ORDERHERE, irf = 0, periods = 300, drop = 1, nograph );
