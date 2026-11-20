// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Unanticipated shocks: each is learnt in the period in which it hits (shocks(learnt_in=...)).
var c k;
varexo x;
parameters alph gam delt bet aa;
alph=0.5; gam=0.5; delt=0.02; bet=0.05; aa=0.5;
model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;
steady_state_model;
k = ((delt+bet)/(x*aa*alph))^(1/(alph-1));
c = aa*x*k^alph-delt*k;
end;
initval;
x = 1;
end;
steady;
shocks(learnt_in = 1);
var x; periods 1; values 1.2;
end;
shocks(learnt_in = 3);
var x; periods 3; values 0.9;
end;
shocks(learnt_in = 4);
var x; periods 4; values 1.1;
end;
perfect_foresight_with_expectation_errors_setup(periods = 60);
perfect_foresight_with_expectation_errors_solver(tolf=1e-8, constant_simulation_length);
