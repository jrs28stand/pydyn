addpath /home/claude/dl/src72/matlab
dynare ll_rbc nograph -DORDER=1
d = oo_.dr; r = struct();
r.ys = d.ys; r.steady_state = oo_.steady_state; r.ghx = d.ghx; r.ghu = d.ghu; r.order_var = d.order_var;
r.params = M_.params; r.Sigma_e = M_.Sigma_e; r.endo_names = strjoin(cellstr(M_.endo_names)', ',');
r.aux_types = [M_.aux_vars.type];
fn = fieldnames(oo_.irfs); r.irf_names = strjoin(fn', ','); r.irfs = cell2mat(struct2cell(oo_.irfs));
r.mean = oo_.mean; r.var = oo_.var; r.autocorr = cell2mat(oo_.autocorr); r.vd = oo_.variance_decomposition;
r.cvd = oo_.conditional_variance_decomposition;
randn('state', 4); r.ex = randn(50, M_.exo_nbr) * chol(M_.Sigma_e);
y = simult_(M_, options_, d.ys, d, r.ex, 1); r.sim = y';
save('-mat7-binary', 'clean_loglinear.mat', '-struct', 'r');
try
  dynare ll_rbc nograph -DORDER=2
  disp('ORDER2_NO_ERROR')
catch e
  disp(['ORDER2_ERROR: ' e.message])
end
