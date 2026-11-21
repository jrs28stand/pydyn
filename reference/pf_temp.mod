// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Temporary shocks on two exogenous variables over period ranges; lags and leads beyond one
// period on endogenous variables, a lag and a lead on an exogenous variable; numerical steady state.
var c k y i;
varexo z g;
parameters alph gam delt bet;
alph=0.36; gam=2; delt=0.05; bet=0.97;
model;
y = z*k(-1)^alph*(1+0.1*(z(-1)-1));
i = 0.6*(k-(1-delt)*k(-1)) + 0.4*(k(-1)-(1-delt)*k(-2));
c + i + g = y;
c^(-gam) = bet*(alph*z(+1)*k^(alph-1)+1-delt)*(0.8*c(+1)+0.2*c(+2))^(-gam);
end;
initval;
z=1; g=0.2; k=7; c=1.4; y=2; i=0.35;
end;
steady(tolf=1e-14);
shocks;
var z; periods 1:4 6 10:12; values 1.05 0.97 1.02;
var g; periods 3:8; values 0.25;
end;
perfect_foresight_setup(periods=150);
perfect_foresight_solver(tolf=1e-8, tolx=1e-8);
