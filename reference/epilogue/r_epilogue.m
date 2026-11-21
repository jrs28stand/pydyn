% Regenerates reference/epilogue/epi_example.mat (Dynare 7.2 epilogue_dynamic / epilogue_static) for
% tests/test_epilogue.py. Run in a fresh directory holding epi_example.mod:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_epilogue.m
addpath /home/claude/dl/src72/matlab
dynare epi_example nograph
randn('state', 3);
Y = oo_.endo_simul';
Y(1:3, 4) = NaN;                       % h starts at period 4
X = oo_.exo_simul;
r = struct();
r.endo_in = Y; r.exo_in = X(end-rows(Y)+1:end, :); r.x_in = randn(7,1);
ds = dseries(Y, dates('2000Q1'), M_.endo_names);
ds = [ds dseries(r.exo_in, dates('2000Q1'), M_.exo_names)];
ds = [ds dseries(r.x_in, dates('2000Q1'), 'x')];
ds = epi_example.epilogue_dynamic(M_.params, ds);
for n = M_.epilogue_names', r.(['dyn_' n{1}]) = ds{n{1}}.data; end
ss = dseries(repmat([oo_.steady_state' oo_.exo_steady_state'], 3, 1), dates('2000Q1'), [M_.endo_names; M_.exo_names]);
ss = [ss dseries([1; 2; 3], dates('2000Q1'), 'x') dseries([0.1; 0.2; 0.3], dates('2000Q1'), 'z')];
ss = epi_example.epilogue_static(M_.params, ss);
for n = M_.epilogue_names', r.(['sta_' n{1}]) = ss{n{1}}.data; end
r.steady_state = oo_.steady_state; r.params = M_.params;
r.names = strjoin(M_.epilogue_names', ',');
save('-mat7-binary', 'epi_example.mat', '-struct', 'r');
disp('EPI_DONE')
