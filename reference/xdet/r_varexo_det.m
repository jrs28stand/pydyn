% Regenerates reference/xdet_o<order>.mat (Dynare 7.2 varexo_det decision rules and simult_varexo_det paths)
% for tests/test_varexo_det.py. Run in a fresh directory holding xdet_o<ORDER>.mod:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "ORDER=1; r_varexo_det"
addpath /home/claude/dl/src72/matlab
eval(sprintf('dynare xdet_o%d nograph', ORDER));
r = struct();
d = oo_.dr;
f = {'ghx','ghu','ys','order_var','exo_det_length'};
if ORDER == 2, f = [f {'ghxx','ghxu','ghuu','ghs2'}]; end
for k = 1:numel(f), r.(f{k}) = d.(f{k}); end
L = d.exo_det_length;
for j = 1:L
  r.(sprintf('ghud_%d', j)) = d.ghud{j};
  if ORDER == 2
    r.(sprintf('ghxud_%d', j)) = d.ghxud{j};
    r.(sprintf('ghuud_%d', j)) = d.ghuud{j};
    for k = 1:L
      if ~isempty(d.ghudud{k,j}), r.(sprintf('ghudud_%d_%d', k, j)) = d.ghudud{k,j}; end
    end
  end
end
r.exo_det_steady_state = oo_.exo_det_steady_state;
% simulation: random stochastic shocks, deterministic exogenous path of the shocks block plus noise
randn('state', 7);
horizon = 6;
ex = 0.01*randn(horizon, M_.exo_nbr);
ml = M_.maximum_lag;
xs = oo_.exo_det_simul(M_.maximum_lag+1:end, :);
ex_det = [zeros(ml, M_.exo_det_nbr); xs; zeros(horizon-size(xs,1), M_.exo_det_nbr)];
ex_det(ml+1, 1) = 0.1; ex_det(ml+2, 2) = 0.2; ex_det(ml+3, 2) = -0.05;   % the shocks block
ex_det = ex_det + 0.02*randn(size(ex_det));
y0 = repmat(oo_.dr.ys, 1, ml) + 0.01*randn(M_.endo_nbr, ml);
opt = options_; opt.order = ORDER;
r.sim = simult_varexo_det(y0, ex, ex_det, ORDER, {}, M_, oo_.dr, oo_.exo_det_steady_state, opt);
r.sim_ex = ex; r.sim_ex_det = ex_det; r.sim_y0 = y0;
r.exo_det_simul = oo_.exo_det_simul; r.maximum_lag = ml;
save('-mat7-binary', sprintf('xdet_o%d.mat', ORDER), '-struct', 'r');
disp('XDET_DONE')
