% Regenerates reference/ext_<case>.mat (Dynare 7.2, tests/external_function models without their
% final checks) for tests/test_external.py. Run with the case name in the variable CASE, in a fresh
% directory that holds <CASE>.mod and the function files:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "CASE='no_deriv_given'; r_external"
addpath /home/claude/dl/src72/matlab
eval(['dynare ' CASE ' nograph']);
r = struct();
d = oo_.dr;
f = {'ghx','ghu','ghxx','ghxu','ghuu','ghs2','ys','order_var'};
for k = 1:numel(f), r.(f{k}) = d.(f{k}); end
r.steady_state = oo_.steady_state; r.var = oo_.var; r.mean = oo_.mean;
save('-mat7-binary', ['ext_' CASE '.mat'], '-struct', 'r');
disp('EXT_DONE')
