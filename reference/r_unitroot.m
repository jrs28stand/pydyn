% Unit-root model (continuum of steady states): Dynare's default steady state (solve_algo=4) and
% first-order rules. ur_ss: block a and the unit-root equation hold at initval, the block of
% [y c k] (with z) does not: Dynare solves it for z, y, c (k keeps its initval). ur_ss_fail: a does
% not hold at initval, and Dynare 7.2 fails.
addpath /home/claude/dl/src72/matlab
dynare ur_ss nograph
d=oo_.dr; r=struct('ys',oo_.steady_state,'ghx',d.ghx,'ghu',d.ghu,'order_var',d.order_var);
try
  dynare ur_ss_fail nograph
  r.fail_failed = 0;
catch
  r.fail_failed = 1;
end
save('-mat7-binary','clean_ur_ss.mat','-struct','r');
