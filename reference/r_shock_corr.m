% Estimated stderr of correlated shocks (est_corr.mod variants 1-3): Sigma_e from Dynare's
% set_all_parameters and the likelihood (dsge_likelihood) at several points; set_shock_stderr_value.
addpath /home/claude/dl/src72/matlab
for v = 1:3
  pause(1.5); rehash;
  eval(sprintf('dynare est_corr nograph -DVARIANT=%d', v));
  pause(1.5); rehash;
  [dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
      dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
  n = numel(xparam1);
  X = [xparam1, xparam1 + [0.4; 0.1*ones(n-1,1)], xparam1 + [-0.2; -0.05*ones(n-1,1)]];
  r = struct('X', X, 'names', strjoin(bayestopt_.name', ';'), 'Sigma_e0', M_.Sigma_e);
  r.fval = zeros(1, columns(X)); r.Sigma_e = zeros(3, 3, columns(X));
  for k = 1:columns(X)
    Mk = set_all_parameters(X(:,k), estim_params_, M_);
    r.Sigma_e(:,:,k) = Mk.Sigma_e;
    r.fval(k) = dsge_likelihood(X(:,k), dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                                oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  end
  if v == 1
    set_shock_stderr_value('e', 0.8);
    r.Sigma_e_set = M_.Sigma_e;
  end
  save('-mat7-binary', sprintf('clean_est_corr_v%d.mat', v), '-struct', 'r');
  printf('OK %d\n', v);
end
