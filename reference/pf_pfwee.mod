// Reference model for tests/test_perfect_foresight.py (pydyn), run with Dynare 7.2 by r_perfect_foresight.m.
// Adapted from Dynare 7.2, tests/deterministic_simulations/pfwee_learnt_in.inc (integer periods, tolerances added).
// Copyright (C) Dynare Team. Dynare is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.
// Perfect foresight with expectation errors: news about temporary and permanent shocks.
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
shocks;
var x; periods 1:2; values 1.2;
end;
endval;
x = 1.05;
end;
shocks(learnt_in = 2);
var x; periods 2; add 0.1;
end;
endval(learnt_in = 2);
x = 1.1;
end;
shocks(learnt_in = 3);
var x; periods 3; values 1.4;
end;
mshocks(learnt_in = 3);
var x; periods 7; values (1.5/1.2);
end;
endval(learnt_in = 3);
x += 0.1;
end;
shocks(learnt_in = 6);
var x; periods 6:8; values 10;
end;
shocks(learnt_in = 6, overwrite);
var x; periods 6:7; multiply 0.8;
end;
endval(learnt_in = 6);
x *= 0.75;
end;
perfect_foresight_with_expectation_errors_setup(periods = 40);
perfect_foresight_with_expectation_errors_solver(tolf=1e-8);
