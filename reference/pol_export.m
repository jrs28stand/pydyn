function pol_export(outfile)
% Saves what pydyn's optimal-policy tests compare (Dynare 7.2 under Octave): steady state, decision
% rules, planner objective values, OSR results. Called after "dynare <case>" in a fresh folder.
global M_ oo_ options_
r = struct();
r.endo_names = strjoin(cellstr(M_.endo_names)', ',');
r.params = M_.params; r.Sigma_e = M_.Sigma_e;
r.param_names = strjoin(cellstr(M_.param_names)', ',');
if isfield(M_, 'ramsey_orig_endo_nbr'), r.ramsey_orig_endo_nbr = M_.ramsey_orig_endo_nbr; r.ramsey_orig_eq_nbr = M_.ramsey_orig_eq_nbr; end
if isfield(oo_, 'steady_state'), r.steady_state = oo_.steady_state; end
if isfield(oo_, 'dr') && isfield(oo_.dr, 'ghx')
  d = oo_.dr; r.ys = d.ys; r.ghx = d.ghx; r.ghu = d.ghu; r.order_var = d.order_var;
  for f = {'ghxx','ghxu','ghuu','ghs2'}
    if isfield(d, f{1}), r.(f{1}) = d.(f{1}); end
  end
end
r.order = options_.order;
if isfield(oo_, 'exo_simul') && ~isempty(oo_.exo_simul), r.exo_simul1 = oo_.exo_simul(1,:); end
if isfield(oo_, 'planner_objective_value')
  p = oo_.planner_objective_value;
  r.W_unc = p.unconditional;
  if isstruct(p.conditional)
    r.W_cond_L0 = p.conditional.zero_initial_multiplier; r.W_cond_LSS = p.conditional.steady_initial_multiplier;
  else
    r.W_cond = p.conditional;
  end
end
if isfield(oo_, 'osr')
  r.osr_f = oo_.osr.objective_function;
  fn = fieldnames(oo_.osr.optim_params); r.osr_names = strjoin(fn', ',');
  r.osr_x = cellfun(@(f) oo_.osr.optim_params.(f), fn);
  r.osr_info = oo_.osr.optimization_info;
end
if isfield(oo_, 'mean'), r.mean = oo_.mean; end
if isfield(oo_, 'var'), r.var = oo_.var; end
save('-mat7-binary', outfile, '-struct', 'r');
