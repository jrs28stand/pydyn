// perfect foresight re-solved with other shock paths
var c k;
varexo x;
parameters alph gam delta bet aa;
alph=0.5; gam=0.5; delta=0.02; bet=0.05; aa=0.5;
model;
c + k - aa*x*k(-1)^alph - (1-delta)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delta)*c(+1)^(-gam);
end;
initval;
x = 1;
k = ((delta+bet)/(1.0*aa*alph))^(1/(alph-1));
c = aa*k^alph-delta*k;
end;
steady;
shocks;
var x;
periods 1:3;
values 1.2;
end;
perfect_foresight_setup(periods=50);
perfect_foresight_solver;
x0 = oo_.exo_simul;
path1 = oo_.endo_simul(2, :);
status1 = oo_.deterministic_simulation.status;
for s = [0.5 2]
    oo_.exo_simul = 1 + s*(x0 - 1);
    perfect_foresight_solver;
    paths(find([0.5 2] == s), :) = oo_.endo_simul(2, :);
end
last_status = oo_.deterministic_simulation.status;
