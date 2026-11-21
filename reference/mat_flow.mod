// pydyn test model: MATLAB control flow and calibration code interleaved with Dynare statements.
// Written for pydyn (tests/test_matlab.py); reference values in mat_flow.mat come from Dynare 7.2
// under Octave (see r_matlab.m).
var c k;
varexo e u;
parameters rho s a1 a2 a3 b1 b2 b3 g1 g2 g3 g4 g5;

rho = 0.5; s = 2;
calib = 1;
if calib == 1
    rho = 0.9; s = 5;
else
    rho = 0.1;
end
if calib == 2
    s = 7;
elseif calib == 1 && rho > 0.5
    p3 = 2^3^2 + 0*calib(1);
    a1 = rho^2 + p3;
else
    a1 = -1;
end

v = [0.1 0.2; 0.3 0.4]';
vlast = v(end);
a2 = v(2, 1) + vlast;
w = linspace(0, 1, 5);
for i = 1:3
    w(i) = w(i) + i;
    if i == 2
        continue
    end
    a3 = 10*i;
end
names = {'b1', 'b2', 'b3'};
vals = [1.5 -2.5 3.5];
for i = 1:numel(names)
    set_param_value(names{i}, vals(i) * w(i));
end
x1 = b1;                               // workspace copy made by set_param_value
M_.params(6) = 2 * get_param_by_name('b2');   // b1 by index: the workspace variable b1 (and so x2) keeps the old value
x2 = b1;
g1 = x1 + x2 + b1;                     // parameter b1 itself is read from M_.params in a Dynare statement
if calib == 1, s = 99; end              // a one-line block is passed to MATLAB as it is: only the workspace variable s changes
x3 = s;
k0 = 0;
while k0 < 5
    k0 = k0 + 2;
    if k0 > 3, break, end
end
switch k0
  case {1, 2}
    g2 = 1;
  case 4
    g2tmp = sum(w(:)) + max(vals) + numel(find(vals > 0));
    g2 = g2tmp;
  otherwise
    g2 = 3;
end
tmp.a = 3; tmp.b = [4 5 6];
g3tmp = tmp.a * tmp.b(end) + mod(-7, 3);
g3 = g3tmp;
f = @(z) z.^2 + rho;
g4tmp = f(3) + fzero(@(z) z^3 - s, 1);
g4 = g4tmp + w(2);
try
    error('not reached further');
    g5 = 1;
catch
    g5 = 2;
end

model;
c = rho*c(-1) + s*e;
k = a1*k(-1) + u + g1*0;
end;

shocks;
var e; stderr 0.1;
var u; stderr 0.2;
end;
M_.Sigma_e(1, 2) = 0.005;
M_.Sigma_e(2, 1) = M_.Sigma_e(1, 2);
set_shock_stderr_value('u', 0.3);
options_.irf = 7;
if options_.irf > 5
    g5 = g5 + 10;
end

stoch_simul(order=1, irf=0, noprint);
