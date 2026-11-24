// The Simple1 example in OccBin form (cf. simple1_max.mod).
var i r pi;

varexo epsilon;

parameters r_ phi;

r_  = log( 1 / 0.99 );
phi = 1.5;

model;
    r = r_ + epsilon;
    [name='zlb', relax='zlb']
    i = r + phi * pi;
    [name='zlb', bind='zlb']
    i = 0;
    i = r + pi(+1);
end;

occbin_constraints;
name 'zlb'; bind r + phi * pi < 0; relax r + phi * pi >= 0;
end;

shocks;
    var epsilon = 0.01;
end;

steady_state_model;
    i  = r_;
    r  = r_;
    pi = 0;
end;

steady; check;
occbin_setup;
