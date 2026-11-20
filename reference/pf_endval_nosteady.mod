// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// endval without steady: the terminal condition is the set of values written in the block
// (not a steady state); the exogenous variables stay at their endval values.
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
endval;
z=1.05; g=0.2; k=7.5; c=1.45; y=2.1; i=0.37;
end;
perfect_foresight_setup(periods=60);
perfect_foresight_solver(tolf=1e-8, tolx=1e-8);
