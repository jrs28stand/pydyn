% Regenerates the est_*.mat reference files of tests/test_estimation.py with Dynare 7.2 under Octave.
% Run in a scratch copy of this directory:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_estimation.m
% Set run_full = true to redo the full Schorfheide (2000) estimation with the settings of
% examples/estimation/schorfheide_2000.mod (mode_compute=4, 2 x 20000 draws; about 40 minutes).
addpath /home/claude/dl/src72/matlab
run_full = false;

function r = est_points(M_, options_, oo_, estim_params_, bayestopt_, extras)
  % likelihood and prior at several parameter vectors, called as in dynare_estimation_1.m
  [dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
      dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
  n = numel(xparam1); K = 6;
  sd = bayestopt_.p2; k = ~isfinite(sd); sd(k) = 0.1*abs(xparam1(k));
  X = zeros(n, K);
  for k = 0:K-1
    X(:,k+1) = xparam1 + (k>0)*0.2*sin(k*(1:n)').*sd;
  end
  for e = 1:numel(extras)
    x = xparam1; pairs = extras{e};
    for j = 1:2:numel(pairs), x(strcmp(bayestopt_.name, pairs{j})) = pairs{j+1}; end
    X = [X x];
  end
  r = struct('X', X, 'x0', xparam1, 'lb', bounds.lb, 'ub', bounds.ub);
  r.names = strjoin(bayestopt_.name', ';');
  r.fval = zeros(1, columns(X)); r.info = r.fval; r.lnprior = r.fval;
  for k = 1:columns(X)
    [fval, info] = dsge_likelihood(X(:,k), dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                                   oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    r.fval(k) = fval; r.info(k) = info(1);
    r.lnprior(k) = priordens(X(:,k), bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
  end
  r.loglik = -(r.fval + r.lnprior);      % fval = -(log likelihood + log prior)
  for f = {'pshape','p1','p2','p3','p4','p6','p7','jscale'}
    r.(f{1}) = bayestopt_.(f{1});
  end
  r.data = dataset_.data; r.nobs = dataset_.nobs;
  r.lik_init = options_.lik_init; r.qz_criterium = options_.qz_criterium; r.presample = options_.presample;
end

% ---- likelihood and prior at several points
cases = { ...
  'est_schorfheide', 0, {{'alp', 1.2}, {'rho', 0.999999}, {'stderr e_a', -0.01}}; ...
  'est_nk', 0, {{'phipi', 0.8}, {'beta', 1.2}, {'corr ea, eu', 0.95}, {'rhor', 0.2}, {'stderr yobs', 0}, {'sig', 0}}; ...
  'est_nk', 1, {}; ...
  'est_nk', 2, {}; ...
  'est_nk', 3, {}; ...
  'est_nk', 4, {}; ...
  'est_nk', 8, {}; ...
  'est_nk', 9, {{'corr yobs, piobs', -0.5}, {'corr yobs, piobs', 0.95}}; ...
  'est_ur', 0, {}; ...
  'est_ur', 1, {}; ...
  'est_ur', 2, {}; ...
  'est_ur', 5, {}; ...
  'est_ur', 6, {}};
for c = 1:rows(cases)
  nm = cases{c,1}; v = cases{c,2}; ex = cases{c,3};
  try
    % the generated model functions change between variants: make Octave notice the new files
    pause(1.5); rehash;
    eval(sprintf('dynare %s nograph -DVARIANT=%d', nm, v));
    pause(1.5); rehash;
    r = est_points(M_, options_, oo_, estim_params_, bayestopt_, ex);
    r.params0 = M_.params; r.param_names = strjoin(M_.param_names', ';');
    save('-mat7-binary', sprintf('est_%s_v%d.mat', nm(5:end), v), '-struct', 'r');
    printf('OK %s %d\n', nm, v);
  catch err
    printf('FAIL %s %d: %s\n', nm, v, err.message);
  end
end

% ---- smoother at the initial parameter values (complete data, missing observations, loglinear)
function s = smoother_ref(M_, options_, oo_)
  n = M_.orig_endo_nbr;
  s = struct();
  s.endo_names = strjoin(M_.endo_names(1:n)', ';'); s.exo_names = strjoin(M_.exo_names', ';');
  s.smoothed = zeros(options_.nobs, n); s.updated = s.smoothed;
  for i = 1:n
    s.smoothed(:,i) = oo_.SmoothedVariables.(M_.endo_names{i});
    s.updated(:,i) = oo_.UpdatedVariables.(M_.endo_names{i});
  end
  s.shocks = zeros(options_.nobs, M_.exo_nbr);
  for i = 1:M_.exo_nbr, s.shocks(:,i) = oo_.SmoothedShocks.(M_.exo_names{i}); end
  if isfield(oo_, 'SmoothedMeasurementErrors'), s.me_yobs = oo_.SmoothedMeasurementErrors.yobs; end
  s.steady_state = oo_.steady_state;
end
pause(1.5); rehash;
dynare est_nk nograph -DVARIANT=5
s = smoother_ref(M_, options_, oo_); save('-mat7-binary', 'est_nk_smoother.mat', '-struct', 's');
pause(1.5); rehash;
dynare est_nk nograph -DVARIANT=7
s = smoother_ref(M_, options_, oo_); save('-mat7-binary', 'est_nk_smoother_miss.mat', '-struct', 's');
pause(1.5); rehash;
dynare est_schorfheide nograph -DSMOOTHER
s = smoother_ref(M_, options_, oo_); save('-mat7-binary', 'est_schorfheide_smoother.mat', '-struct', 's');

% ---- posterior modes found by Dynare (mode_compute=4), Hessian and Laplace approximation
modes = {'est_nk', 6; 'est_ur', 3; 'est_ur', 4};     % est_ur 4 is maximum likelihood
for c = 1:rows(modes)
  pause(1.5); rehash;
  eval(sprintf('dynare %s nograph -DVARIANT=%d', modes{c,1}, modes{c,2}));
  r = struct('xparam1', oo_.posterior.optimization.mode, 'Vmode', oo_.posterior.optimization.Variance, ...
             'logpost', oo_.posterior.optimization.log_density, 'laplace', NaN);
  if isfield(oo_, 'MarginalDensity'), r.laplace = oo_.MarginalDensity.LaplaceApproximation; end
  r.names = strjoin(bayestopt_.name', ';');
  save('-mat7-binary', sprintf('est_%s_mode_v%d.mat', modes{c,1}(5:end), modes{c,2}), '-struct', 'r');
end

% ---- prior densities and hyper-parameters on a grid
% columns: shape, mean, std, p3, p4 (NaN = default)
P = [1 0.5 0.2 NaN NaN; 1 0.129 0.223 NaN NaN; 1 0.6 0.2 -1 3; ...
     2 0.1 0.05 NaN NaN; 2 2 1 0.5 NaN; ...
     3 0 1 NaN NaN; 3 1.5 0.25 NaN NaN; ...
     4 0.5 2 NaN NaN; 4 0.035449 Inf NaN NaN; 4 1.2 0.3 0.2 NaN; ...
     5 NaN NaN 0 1; 5 0.5 0.25 NaN NaN; ...
     6 0.1 1 NaN NaN; 6 0.5 0.2 NaN NaN; 6 1.2 0.3 0.2 NaN; ...
     8 1.5 0.5 NaN NaN; 8 2 1 0.5 NaN; 8 1 1 NaN NaN];
grid = [linspace(-1.5, 4.5, 61), 0.01*linspace(0.1, 12, 30), 0, 0.5, 1, 0.2];
q = struct('spec', P, 'grid', grid, 'p6', zeros(rows(P),1), 'p7', zeros(rows(P),1), 'p3', P(:,4), 'p4', P(:,5), ...
           'lpdf', zeros(rows(P), numel(grid)));
for i = 1:rows(P)
  sh = P(i,1); m = P(i,2); s2 = P(i,3)^2; p3 = P(i,4); p4 = P(i,5);
  switch sh
    case 1
      if isnan(p3), p3 = 0; end; if isnan(p4), p4 = 1; end
      [a, b] = beta_specification(m, s2, p3, p4);
    case 2
      if isnan(p3), p3 = 0; end; if isnan(p4), p4 = Inf; end
      [a, b] = gamma_specification(m, s2, p3);
    case 3
      if isnan(p3), p3 = -Inf; end; if isnan(p4), p4 = Inf; end
      a = m; b = P(i,3);
    case {4, 6}
      if isnan(p3), p3 = 0; end; if isnan(p4), p4 = Inf; end
      [a, b] = inverse_gamma_specification(m, s2, p3, 1 + (sh==6), false);
    case 5
      [~, ~, a, b] = uniform_specification(m, P(i,3), p3, p4); p3 = a; p4 = b;
    case 8
      if isnan(p3), p3 = 0; end; if isnan(p4), p4 = Inf; end
      [a, b] = weibull_specification(m, s2, p3);
  end
  q.p6(i) = a; q.p7(i) = b; q.p3(i) = p3; q.p4(i) = p4;
  for j = 1:numel(grid)
    q.lpdf(i,j) = priordens(grid(j), sh, a, b, p3, p4);
  end
end
save('-mat7-binary', 'est_priors.mat', '-struct', 'q');

% ---- full estimation of the Schorfheide (2000) example: mode, Hessian, Laplace, posterior moments
if run_full
  dynare est_schorfheide nograph -DFULL
  r = struct();
  r.xparam1 = oo_.posterior.optimization.mode; r.Vmode = oo_.posterior.optimization.Variance;
  r.logpost = oo_.posterior.optimization.log_density; r.laplace = oo_.MarginalDensity.LaplaceApproximation;
  r.mhm = oo_.MarginalDensity.ModifiedHarmonicMean;
  md = load('est_schorfheide/Output/est_schorfheide_mode.mat'); r.hh = md.hh;
  nm = bayestopt_.name; r.names = strjoin(nm', ';'); pm = []; pv = [];
  for i = 1:numel(nm)
    s = nm{i};
    if strncmp(s, 'stderr ', 7)
      pm(i) = oo_.posterior_mean.shocks_std.(s(8:end)); pv(i) = oo_.posterior_std.shocks_std.(s(8:end));
    else
      pm(i) = oo_.posterior_mean.parameters.(s); pv(i) = oo_.posterior_std.parameters.(s);
    end
  end
  r.post_mean = pm; r.post_std = pv;
  save('-mat7-binary', 'est_schorfheide_full.mat', '-struct', 'r');
end
% ---- a sample of Dynare's own Metropolis draws with their log posterior (needs the output of the full run)
if exist('est_schorfheide/metropolis/est_schorfheide_mh1_blck1.mat', 'file')
  d = struct('X', [], 'LP', []);
  for b = 1:2
    q = load(sprintf('est_schorfheide/metropolis/est_schorfheide_mh1_blck%d.mat', b));
    k = round(linspace(1, rows(q.x2), 20));
    d.X = [d.X; q.x2(k,:)]; d.LP = [d.LP; q.logpo2(k)];
  end
  save('-mat7-binary', 'est_schorfheide_draws.mat', '-struct', 'd');
end
disp('ESTIMATION_REFS_DONE')
