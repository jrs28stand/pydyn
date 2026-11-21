function pf_save(name, M_, oo_, options_)
% Plain-array copy of a deterministic simulation (reference for pydyn's tests).
r = struct();
r.endo_simul = oo_.endo_simul; r.exo_simul = oo_.exo_simul;
r.steady_state = oo_.steady_state; r.exo_steady_state = oo_.exo_steady_state;
if isfield(oo_, 'initial_steady_state') && ~isempty(oo_.initial_steady_state)
    r.initial_steady_state = oo_.initial_steady_state; r.initial_exo_steady_state = oo_.initial_exo_steady_state;
end
d = oo_.deterministic_simulation;
r.status = double(d.status);
f = {'error', 'iterations', 'homotopy_completion_share', 'homotopy_iterations'};
for i = 1:numel(f)
    if isfield(d, f{i}) && ~isempty(d.(f{i})), r.(f{i}) = double(d.(f{i})); end
end
r.homotopy_linearization = double(isfield(d, 'homotopy_linearization') && d.homotopy_linearization);
r.homotopy_marginal_linearization = double(isfield(d, 'homotopy_marginal_linearization') && d.homotopy_marginal_linearization);
r.endo_names = strjoin(cellstr(M_.endo_names)', ','); r.exo_names = strjoin(cellstr(M_.exo_names)', ',');
r.params = M_.params; r.maximum_lag = M_.maximum_lag; r.maximum_lead = M_.maximum_lead;
r.tolf = options_.dynatol.f;
if isfield(oo_, 'pfwee') && isfield(oo_.pfwee, 'shocks_info')
    r.shocks_info = oo_.pfwee.shocks_info; r.terminal_info = oo_.pfwee.terminal_info;
end
save('-mat7-binary', [name '.mat'], '-struct', 'r');
end
