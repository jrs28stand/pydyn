% Regenerates the nr_*.mat reference files of tests/test_newrat.py with Dynare 7.2 under Octave 8.4 (mode_compute=5,
% newrat). Run in a scratch copy of this directory (it needs post_nk.mod, post_schorfheide.mod, est_nk_data.m and
% est_schorfheide_data.m):
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_newrat.m > log 2>&1
% nr_testfun.mat: newrat on a least-squares objective with a "prior" term whose contributions are computed with the same
%   floating-point operations here and in tests/test_newrat.py (only + - * /): every evaluation and the result, for
%   three settings (default; robust with a smaller gradient tolerance; MaxIter 3).
% nr_post_nk.mat, nr_post_schorfheide.mat: newrat as dynare_minimize_objective calls it (mode_compute=5, default
%   options) on dsge_likelihood from the initial values: every function value (a sample of the points) and the result.
addpath /home/claude/dl/src72/matlab
dynare_config();
function write_file(name, body)
  fid = fopen(name, 'w'); fprintf(fid, '%s\n', body); fclose(fid);
end
write_file('nr_logged_objective.m', sprintf(['function [fval, info, exit_flag, a1, a2] = nr_logged_objective(x, varargin)\n' ...
  'global POSTLOG\n[fval, info, exit_flag, a1, a2] = dsge_likelihood(x, varargin{:});\np = 0; if numel(info) >= 4, p = info(4); end\n' ...
  'POSTLOG.X(:, end+1) = x(:); POSTLOG.F(end+1) = fval; POSTLOG.info(:, end+1) = [info(1); p];\nend']));
write_file('nr_testfun.m', sprintf(['function [fval, info, exit_flag, a1, a2] = nr_testfun(x, varargin)\n' ...
  'global POSTLOG\nT = 60; info = zeros(4, 1); exit_flag = 1; a2 = [];\n' ...
  'if x(4) <= 0.05\n  fval = Inf; info(1) = 42; info(4) = (0.05 - x(4))*(0.05 - x(4)); exit_flag = 0; a1 = [];\nelse\n' ...
  '  mu = [0.5; 1; 0; 1]; sd = [2; 2; 2; 1]; c0 = 0;\n' ...
  '  for j = 1:4, d = (x(j) - mu(j))/sd(j); c0 = c0 + 0.5*d*d; end\n' ...
  '  c = zeros(T, 1); fval = c0;\n' ...
  '  for t = 1:T\n    s = t/T; y = 1 + 2*s - 0.5*s*s/(1 + 0.8*s) + 0.1*(mod(t*7919, 13)/13 - 0.5);\n' ...
  '    m = x(1) + x(2)*s + x(3)*s*s/(1 + x(4)*x(4)*s); r = y - m; c(t) = 0.5*r*r/0.01; fval = fval + c(t);\n  end\n' ...
  '  a1 = [c0; c];\nend\n' ...
  'POSTLOG.X(:, end+1) = x(:); POSTLOG.F(end+1) = fval; POSTLOG.info(:, end+1) = [info(1); info(4)];\nend']));
rehash;
global POSTLOG
t = struct(); t.x0 = [0; 0; 0; 1]; t.bounds = [-10 10; -10 10; -10 10; 0.01 5]; t.prior_std = [2; 2; 2; 1];
% settings: robust, TolGstep (hess_info.htol), MaxIter
t.settings = [0 1e-5 1000; 1 1e-6 1000; 0 1e-5 3];
for v = 1:rows(t.settings)
  POSTLOG = struct('X', zeros(4, 0), 'F', [], 'info', zeros(2, 0));
  hi = struct('gstep', [1e-2 1], 'htol', t.settings(v,2), 'htol_rel', NaN, 'h1', 1e-6*ones(4, 1), 'robust', logical(t.settings(v,1)));
  [xh, hh, gg, fval, igg, hio, jit, fcount, exitflag] = newrat(@nr_testfun, t.x0, t.bounds, 0, 1e-5, t.settings(v,3), 0, 0, 0, ...
                                                              hi, t.prior_std, 1e-6, {'a', 'b', 'c', 'd'});
  p = sprintf('s%d_', v);
  t.([p 'xh']) = xh; t.([p 'hh']) = hh; t.([p 'gg']) = gg; t.([p 'fval']) = fval; t.([p 'igg']) = igg; t.([p 'h1']) = hio.h1;
  t.([p 'iterations']) = jit; t.([p 'fcount']) = fcount; t.([p 'exitflag']) = exitflag;
  t.([p 'X']) = POSTLOG.X; t.([p 'F']) = POSTLOG.F; t.([p 'info']) = POSTLOG.info;
end
save('-mat7-binary', 'nr_testfun.mat', '-struct', 't');
for nmc = {'post_nk', 'post_schorfheide'}
  nm = nmc{1};
  pause(1.5); rehash;
  eval(sprintf('dynare %s nograph -DPOST=0', nm));
  [dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
      dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
  global POSTLOG          % the driver cleared the global variables
  POSTLOG = struct('X', zeros(numel(xparam1), 0), 'F', [], 'info', zeros(2, 0));
  o = options_; o.silent_optimizer = true;
  t0 = tic;
  [xo, fo, ef, hm, ~, nri, info] = dynare_minimize_objective(@nr_logged_objective, xparam1, 5, o, [bounds.lb bounds.ub], ...
      bayestopt_.name, bayestopt_, [], dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_.dr, ...
      oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  r = struct('elapsed', toc(t0), 'x0', xparam1, 'xh', xo, 'fh', fo, 'exitflag', ef, 'opg', hm, 'iterations', info.iterations, ...
             'fcount', info.funcCount, 'F', POSTLOG.F, 'info', POSTLOG.info, 'lb', bounds.lb, 'ub', bounds.ub, ...
             'names', strjoin(bayestopt_.name', ';'));
  k = unique([1:min(columns(POSTLOG.X), 300), round(linspace(1, columns(POSTLOG.X), 300))]);
  r.Xidx = k; r.X = POSTLOG.X(:, k);
  save('-mat7-binary', sprintf('nr_%s.mat', nm), '-struct', 'r');
  printf('NR_DONE %s %d evals %.1fs\n', nm, columns(POSTLOG.X), r.elapsed);
end
