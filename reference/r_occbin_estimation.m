% Regenerates reference/occest_*.mat: Dynare 7.2 OccBin likelihoods (piecewise Kalman filter and
% inversion filter), filtered states, regime histories and OccBin smoother results, for
% tests/test_occbin_estimation.py.
%
%   OCCEST_CASE=nkm GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_occbin_estimation.m
%
% Run it in a scratch directory holding copies of occest_<case>.mod and of the data/mode files named
% below (tests/occbin/filter/{dataobsfile.mat,NKM_mh_mode_saved.mat} of the Dynare 7.2 sources).
% The script writes an instrumented copy of matlab/kalman/likelihood/missing_observations_kalman_filter.m
% into the scratch directory (it shadows Dynare's) that keeps the per-period likelihood, the filtered
% states and the regimes of the piecewise Kalman filter in a global variable; nothing else is changed.
1;

function patch_filter()
  src = fileread('/home/claude/dl/src72/matlab/kalman/likelihood/missing_observations_kalman_filter.m');
  key = sprintf('lik(1:s) = .5*lik(1:s);');
  k = strfind(src, key);
  ins = sprintf(['global OCCEST_PKF\nif occbin_.status\n  OCCEST_PKF = struct(''lik'', lik, ''a0'', a0, ''a1'', a1, ' ...
                 '''P0'', P0, ''regimes'', {regimes_}, ''t'', t);\nend\n']);
  src = [src(1:k(1)-1) ins src(k(1):end)];
  fid = fopen('missing_observations_kalman_filter.m', 'w'); fwrite(fid, src); fclose(fid);
  rehash;
end

function patch_shock_decomposition()
  % copy of +occbin/shock_decomposition.m without the dates (Octave's datatypes package is missing):
  % the decomposition starts in period 1 (TINIT = first period)
  src = fileread('/home/claude/dl/src72/matlab/+occbin/shock_decomposition.m');
  src = strrep(src, 'function oo_ = shock_decomposition(', 'function oo_ = occest_shock_decomposition(');
  src = strrep(src, 'load (options_.datafile, ''T'');', '');
  src = regexprep(src, 'TT1= dates\([^\n]*\n', '\n');
  src = regexprep(src, 'if exist\(''T'',''var''\)\s*\n\s*TT1=TT1\+T\(options_.first_obs\)\*4;\s*\nend', '');
  src = regexprep(src, 'shock_decomp_options = set_default_option\(shock_decomp_options,''TINIT'',TT1\(1\)\);', '');
  src = regexprep(src, 'tinit = max\(\[1,find\(TT1==shock_decomp_options.TINIT\)\]\);', 'tinit = 1;');
  fid = fopen('occest_shock_decomposition.m', 'w'); fwrite(fid, src); fclose(fid);
  rehash;
end

function c = regimes_to_cell(rh)
  % struct array of regimes -> cell (fields x periods); empty -> {}
  if isempty(rh), c = {}; else, c = struct2cell(rh(:)); c = reshape(c, size(c, 1), []); end
end

function r = points(r, X, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_, do_ivf, do_pkf)
  global OCCEST_PKF
  K = columns(X);
  r.X = X;
  r.pkf_fval = NaN(1, K); r.pkf_info = zeros(1, K); r.ivf_fval = NaN(1, K); r.ivf_info = zeros(1, K);
  r.lnprior = zeros(1, K);
  for k = 1:K
    r.lnprior(k) = priordens(X(:,k), bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
    if do_pkf
      OCCEST_PKF = [];
      t0 = time();
      [fval, info] = dsge_likelihood(X(:,k), dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                                     oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
      printf('  PKF point %d: fval %.10g info %d (%.1f s)\n', k, fval, info(1), time() - t0);
      r.pkf_fval(k) = fval; r.pkf_info(k) = info(1);
      if ~isempty(OCCEST_PKF)
        r.(sprintf('pkf_lik_%d', k)) = OCCEST_PKF.lik;
        r.(sprintf('pkf_a0_%d', k)) = OCCEST_PKF.a0;
        r.(sprintf('pkf_a1_%d', k)) = OCCEST_PKF.a1;
        r.(sprintf('pkf_regimes_%d', k)) = regimes_to_cell(OCCEST_PKF.regimes);
      end
    end
    if do_ivf
      o = options_;
      o.occbin.likelihood.inversion_filter = true;
      o.occbin.likelihood.IVF_shock_observable_mapping = find(diag(M_.Sigma_e) ~= 0);
      t0 = time();
      [fval, info, ~, ~, ~, ~, ~, ~, ~, atT, innov, rh] = occbin.IVF_posterior(X(:,k), dataset_, dataset_info, o, M_, ...
          estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
      printf('  IVF point %d: fval %.10g info %d (%.1f s)\n', k, fval, info(1), time() - t0);
      r.ivf_fval(k) = fval; r.ivf_info(k) = info(1);
      if info(1) == 0
        r.(sprintf('ivf_atT_%d', k)) = atT;      % decision-rule order, deviations from steady state
        r.(sprintf('ivf_innov_%d', k)) = innov;
        r.(sprintf('ivf_regimes_%d', k)) = regimes_to_cell(rh);
      end
    end
  end
end

function s = smoother_ref(prefix, s, M_, oo_)
  n = M_.orig_endo_nbr;
  f = {'SmoothedVariables', 'UpdatedVariables', 'FilteredVariables'};
  for j = 1:numel(f)
    if isfield(oo_, f{j}) && isstruct(oo_.(f{j})) && isfield(oo_.(f{j}), M_.endo_names{1})
      T = numel(oo_.(f{j}).(M_.endo_names{1}));
      A = NaN(T, n);
      for i = 1:n, A(:, i) = oo_.(f{j}).(M_.endo_names{i}); end
      s.([prefix '_' f{j}]) = A;
    end
  end
  if isfield(oo_, 'SmoothedShocks')
    T = numel(oo_.SmoothedShocks.(M_.exo_names{1}));
    A = NaN(T, M_.exo_nbr);
    for i = 1:M_.exo_nbr, A(:, i) = oo_.SmoothedShocks.(M_.exo_names{i}); end
    s.([prefix '_SmoothedShocks']) = A;
  end
  if isfield(oo_, 'occbin') && isfield(oo_.occbin, 'smoother')
    sm = oo_.occbin.smoother;
    if isfield(sm, 'regime_history'), s.([prefix '_regimes']) = regimes_to_cell(sm.regime_history); end
    if isfield(sm, 'realtime_regime_history'), s.([prefix '_rt_regimes']) = regimes_to_cell(sm.realtime_regime_history); end
    if isfield(sm, 'error_flag'), s.([prefix '_error_flag']) = sm.error_flag; end
  end
  if isfield(oo_, 'occbin') && isfield(oo_.occbin, 'linear_smoother') && isfield(oo_.occbin.linear_smoother, 'alphahat')
    s.([prefix '_linear_alphahat']) = oo_.occbin.linear_smoother.alphahat;
    s.([prefix '_linear_etahat']) = oo_.occbin.linear_smoother.etahat;
  end
end

% ------------------------------------------------------------------------------------------------
addpath /home/claude/dl/src72/matlab
cname = getenv('OCCEST_CASE');
if isempty(cname), cname = 'nkm'; end
patch_filter();
global OCCEST_PKF

switch cname
  case 'nkm'
    dynare occest_nkm nograph
    md = load('NKM_mh_mode_saved.mat');
    x0 = md.xparam1(:);
    rel = [0 0.05 -0.05 0.1];
  case 'nkms'
    % posterior mode with mode_compute=4 (csminwel) from the prior mean, piecewise Kalman filter
    t0 = time();
    try
      dynare occest_nkms nograph
    catch err
      % the smoother run after the mode tries to plot (occbin smoother_plot cannot be switched off
      % from occbin_setup in Dynare 7.2): the mode file is written before that
      printf('dynare stopped after the mode: %s\n', err.message);
    end
    printf('estimation (mode_compute=4): %.1f s\n', time() - t0);
    md = load(fullfile(M_.dname, 'Output', [M_.fname '_mode.mat']));
    x0 = md.xparam1(:);
    rel = [0 0.1 -0.1];
    mres = struct('xparam1', md.xparam1, 'hh', md.hh, 'fval', md.fval, 'x_start', bayestopt_.p1);
    save('-mat7-binary', 'occest_nkms_mode.mat', '-struct', 'mres');
  case 'rbc'
    % two constraints; data: c from the OccBin simulation of the .mod file's shocks(surprise)
    dynare occest_rbc nograph
    ic = find(strcmp(M_.endo_names, 'c'));
    c = oo_.occbin.simul.piecewise(1:169, ic);
    save('datasim.mat', 'c');
    options_.datafile = 'datasim'; options_.nobs = 169; options_.first_obs = 1;
    options_.mode_compute = 0; options_.mh_replic = 0;
    x0 = [0.015; 0.9];
    rel = [0 0.05 -0.1];
  case 'nkmopt'
    % the NKM model with filter options changed: filter_use_relaxation, likelihood_max_kalman_iterations=3,
    % likelihood_periodic_solution, likelihood_check_ahead_periods=100
    dynare occest_nkm nograph
    options_.occbin.filter.use_relaxation = true;
    options_.occbin.likelihood.max_number_of_iterations = 3;
    options_.occbin.likelihood.periodic_solution = true;
    options_.occbin.likelihood.check_ahead_periods = 100;
    md = load('NKM_mh_mode_saved.mat');
    x0 = md.xparam1(:);
    rel = [0 0.1];
  otherwise
    error('unknown case');
end
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
n = numel(x0);
X = zeros(n, numel(rel));
for k = 1:numel(rel)
  X(:, k) = x0 .* (1 + rel(k) * sin(k * (1:n)'));
end
r = struct();
r.names = strjoin(bayestopt_.name', ';');
r.data = dataset_.data;
r.order_var = oo_.dr.order_var; r.restrict_var_list = oo_.dr.restrict_var_list;
r.lb = bounds.lb; r.ub = bounds.ub;
r = points(r, X, dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_, true, true);
save('-mat7-binary', sprintf('occest_%s_lik.mat', cname), '-struct', 'r');
printf('saved likelihood points\n');
if any(strcmp(cname, {'nkms', 'nkmopt'})), return; end

% ---- smoothers at x0
s = struct();
s.x0 = x0;
o = options_;
o.smoother = true; o.filtered_vars = true; o.filter_step_ahead = 1; o.nk = 1;
o.occbin.smoother.plot = false; o.nograph = true;
t0 = time();
oo1 = oo_;
[oo1, M1] = evaluate_smoother(x0, {}, M_, oo1, o, bayestopt_, estim_params_);
printf('PKF smoother: %.1f s\n', time() - t0);
s = smoother_ref('pkf', s, M1, oo1);
patch_shock_decomposition();
o.occbin.shock_decomp.init_names_ = {};
oo1 = occest_shock_decomposition(oo1, M1, o, []);
if strcmp(cname, 'nkm')
  rows = [find(strcmp(M_.endo_names, 'yg')), find(strcmp(M_.endo_names, 'inom')), find(strcmp(M_.endo_names, 'pi'))];
else
  rows = 1:M_.endo_nbr;
end
s.sd_rows = rows;
s.sd_decomp = oo1.occbin.smoother.decomp(rows, :, :);
s.sd_wdecomp = oo1.occbin.smoother.wdecomp(rows, :, :);
s.sd_linear = oo1.occbin.linear_smoother.decomp(rows, :, :);
s.sd_init = oo1.occbin.smoother.init_decomp(rows, :, :);
s.sd_linear_init = oo1.occbin.linear_smoother.init_decomp(rows, :, :);
o.occbin.smoother.inversion_filter = true; o.occbin.likelihood.inversion_filter = true;
oo2 = oo_;
t0 = time();
[oo2, M2] = evaluate_smoother(x0, {}, M_, oo2, o, bayestopt_, estim_params_);
printf('IVF smoother: %.1f s\n', time() - t0);
s = smoother_ref('ivf', s, M2, oo2);
s.ys = oo_.steady_state; s.endo_names = strjoin(M_.endo_names', ';'); s.exo_names = strjoin(M_.exo_names', ';');
save('-mat7-binary', sprintf('occest_%s_smoother.mat', cname), '-struct', 's');
printf('saved smoother\n');
