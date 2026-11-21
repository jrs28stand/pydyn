% korder_nanparam.mod: second-order rules with an unused NaN parameter; the order-3 solution fails.
addpath /home/claude/dl/src72/matlab
dynare korder_nanparam nograph
r = struct('ghx', dr2.ghx, 'ghu', dr2.ghu, 'ghxx', dr2.ghxx, 'ghxu', dr2.ghxu, 'ghuu', dr2.ghuu, ...
           'ghs2', dr2.ghs2, 'ys', dr2.ys, 'order_var', dr2.order_var, 'params', M_.params, ...
           'order3_solved', isfield(oo_.dr, 'g_3'));
save('-mat7-binary', 'clean_korder_nanparam.mat', '-struct', 'r');
