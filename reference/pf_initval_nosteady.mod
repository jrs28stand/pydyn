// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// initval without steady: the path starts from, and ends at, the values of the block; the
// auxiliary variables (k(-1), z, c(+1) carried for the long lags and leads) are zero at both ends.
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
z=1; g=0.2; k=6; c=1.3; y=1.9; i=0.3;
end;
shocks;
var z; periods 2:3; values 1.03;
end;
perfect_foresight_setup(periods=80);
perfect_foresight_solver(tolf=1e-8, tolx=1e-8);
