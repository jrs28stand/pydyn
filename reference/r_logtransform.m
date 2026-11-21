% var(log): Dynare's tests/log_transform models (ramst: deterministic, solver tolf=1e-8; example1 and
% fs2000_nonstationary: stoch_simul with irf=0, the latter with steady(tolf=1e-12)). Steady states, rules, simulation.
addpath /home/claude/dl/src72/matlab
r = struct();
dynare log_ramst nograph
r.ramst_ys = oo_.steady_state; r.ramst_endo = oo_.endo_simul; r.ramst_status = oo_.deterministic_simulation.status; r.ramst_iterations = oo_.deterministic_simulation.iterations;
r.ramst_names = strjoin(cellstr(M_.endo_names)', ',');
clear -global M_ oo_ options_
dynare log_example1 nograph
r.ex1_ys = oo_.dr.ys; r.ex1_ghx = oo_.dr.ghx; r.ex1_ghu = oo_.dr.ghu; r.ex1_order_var = oo_.dr.order_var;
r.ex1_names = strjoin(cellstr(M_.endo_names)', ',');
clear -global M_ oo_ options_
dynare log_fs2000ns nograph
r.fs_ys = oo_.dr.ys; r.fs_ghx = oo_.dr.ghx; r.fs_ghu = oo_.dr.ghu; r.fs_order_var = oo_.dr.order_var;
r.fs_names = strjoin(cellstr(M_.endo_names)', ',');
save('-mat7-binary','clean_log_transform.mat','-struct','r');
