// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Purely backward model with a second lag, started from histval: solved period by period.
var k y;
varexo s;
parameters a d;
a=0.3; d=0.1;
model;
y = k(-1)^a;
k = (1-d)*k(-1) + s*y + 0.05*(k(-1)-k(-2));
end;
initval;
s=0.2; k=2.5; y=1.3;
end;
steady(tolf=1e-14);
histval;
k(0) = 1.5;
k(-1) = 1.4;
end;
shocks;
var s; periods 3:6; values 0.25;
end;
perfect_foresight_setup(periods=60);
perfect_foresight_solver(tolf=1e-12, tolx=1e-12);
