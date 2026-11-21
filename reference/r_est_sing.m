% est_sing.mod: likelihood of a stochastically singular model at three points (dsge_likelihood,
% called after dynare_estimation_init since the estimation command stops in initial_estimation_checks).
addpath /home/claude/dl/src72/matlab
for v = 1:2
  pause(1.5); rehash;
  try
    eval(sprintf('dynare est_sing nograph -DVARIANT=%d', v));
  catch err
    disp(err.message);
  end
  pause(1.5); rehash;
  [dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
      dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
  X = [xparam1, xparam1 + [0.1; -0.1], xparam1 + [-0.2; 0.15]];
  res = struct('X', X, 'names', strjoin(bayestopt_.name', ';'), 'fval', zeros(1, 3), 'info', zeros(1, 3));
  for k = 1:3
    [fval, info] = dsge_likelihood(X(:,k), dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                                   oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    res.fval(k) = fval; res.info(k) = info(1);
  end
  save('-mat7-binary', sprintf('clean_est_sing_v%d.mat', v), '-struct', 'res');
  disp(res.fval); disp(res.info);
end
