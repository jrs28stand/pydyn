% Regenerates the postmom_*.mat reference files of tests/test_posterior_moments.py with Dynare 7.2 under Octave 8.4:
% posterior distributions of theoretical moments of endogenous variables (moments_varendo) and of smoothed / filtered
% variables (smoother, filtered_vars, filter_step_ahead). Run in a scratch directory holding postmom_nk.mod,
% postmom_schorfheide.mod, est_nk_data.m, est_schorfheide_data.m, post_nk_mode0.mat and post_schorfheide_mode0.mat:
%   PM_CASES="nk:1,nk:2,nk:3,nk:4,schorfheide:1,schorfheide:2" GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_postmom.m
% (each case is a fresh Metropolis-Hastings run with Dynare's default seed, so the chains are those of post_<m>_chains.mat for
% the same settings). Files written, one per case: postmom_<m>_<k>.mat with the parameter draws Dynare used for the statistics
% (pdraws of metropolis/<m>_posterior_draws1.mat for moments, stock of <m>_param1.mat for the smoother branch), the
% statistics of oo_.PosteriorTheoreticalMoments (flattened) or of oo_.SmoothedVariables etc., and the per-draw moments
% Dynare stored in its metropolis folder (first draws). Density estimates are stored every DT-th grid point.
addpath /home/claude/dl/src72/matlab
dynare_config();
cs = getenv('PM_CASES'); if isempty(cs), cs = 'nk:1,nk:2,nk:3,nk:4,schorfheide:1,schorfheide:2'; end
cases = strsplit(cs, ',');

function write_stub(name, body)
  fid = fopen([name '.m'], 'w'); fprintf(fid, '%s\n', body); fclose(fid);
end
write_stub('plot', sprintf('function h = plot(varargin)\nh = 1;\nend'));
for f = {'dyn_figure', 'subplot'}
  write_stub(f{1}, sprintf('function h = %s(varargin)\nh = 1;\nend', f{1}));
end
for f = {'dyn_saveas', 'hold', 'xlim', 'title', 'drawnow', 'box'}
  write_stub(f{1}, sprintf('function %s(varargin)\nend', f{1}));
end
rehash;

function d = thin(x, dt)
  % density arrays: grid points (dimension 1) every dt-th point
  d = x(1:dt:end, :);
end

function s = pack_pairs(T, v, dens, dt)
  % covariance-like family: one value per pair i <= j (row-major), fields Mean.(v1).(v2)
  n = numel(v); np = n*(n+1)/2;
  s.mean = NaN(np,1); s.median = s.mean; s.var = s.mean; s.hpdinf = s.mean; s.hpdsup = s.mean; s.deciles = NaN(np,9);
  if dens, s.density = NaN(np, ceil(512/dt), 2); end
  k = 0;
  for i = 1:n
    for j = i:n
      k = k+1; a = v{i}; b = v{j};
      s.mean(k) = T.Mean.(a).(b); s.median(k) = T.Median.(a).(b); s.var(k) = T.Variance.(a).(b);
      s.hpdinf(k) = T.HPDinf.(a).(b); s.hpdsup(k) = T.HPDsup.(a).(b); s.deciles(k,:) = T.deciles.(a).(b)(:)';
      if dens, s.density(k,:,:) = reshape(thin(T.density.(a).(b), dt), 1, [], 2); end
    end
  end
end

function s = pack_corr(T, v, dens, dt)
  % autocorrelations: (nvar, nvar, nar)
  n = numel(v); nar = numel(T.Mean.(v{1}).(v{1}));
  s.mean = NaN(n,n,nar); s.median = s.mean; s.var = s.mean; s.hpdinf = s.mean; s.hpdsup = s.mean; s.deciles = NaN(n,n,nar,9);
  if dens, s.density = NaN(n,n,nar,ceil(512/dt),2); end
  for i = 1:n
    for j = 1:n
      a = v{i}; b = v{j};
      s.mean(i,j,:) = T.Mean.(a).(b); s.median(i,j,:) = T.Median.(a).(b); s.var(i,j,:) = T.Variance.(a).(b);
      s.hpdinf(i,j,:) = T.HPDinf.(a).(b); s.hpdsup(i,j,:) = T.HPDsup.(a).(b);
      for l = 1:nar
        s.deciles(i,j,l,:) = T.deciles.(a).(b){l}(:)';
        if dens, s.density(i,j,l,:,:) = reshape(thin(T.density.(a).(b){l}, dt), 1, 1, 1, [], 2); end
      end
    end
  end
end

function s = pack_vd(T, v, x, dens, dt)
  % variance decomposition: (nvar, nshock)
  n = numel(v); m = numel(x);
  s.mean = NaN(n,m); s.median = s.mean; s.var = s.mean; s.hpdinf = s.mean; s.hpdsup = s.mean; s.deciles = NaN(n,m,9);
  if dens, s.density = NaN(n,m,ceil(512/dt),2); end
  for i = 1:n
    for j = 1:m
      a = v{i}; b = x{j};
      s.mean(i,j) = T.Mean.(a).(b); s.median(i,j) = T.Median.(a).(b); s.var(i,j) = T.Variance.(a).(b);
      s.hpdinf(i,j) = T.HPDinf.(a).(b); s.hpdsup(i,j) = T.HPDsup.(a).(b); s.deciles(i,j,:) = T.deciles.(a).(b)(:)';
      if dens, s.density(i,j,:,:) = reshape(thin(T.density.(a).(b), dt), 1, 1, [], 2); end
    end
  end
end

function s = pack_cvd(T, v, x, dens, dt)
  % conditional variance decomposition: (nvar, nshock, nsteps)
  n = numel(v); m = numel(x); ns = numel(T.Steps);
  s.steps = T.Steps(:)';
  s.mean = NaN(n,m,ns); s.median = s.mean; s.var = s.mean; s.hpdinf = s.mean; s.hpdsup = s.mean; s.deciles = NaN(n,m,ns,9);
  if dens, s.density = NaN(n,m,ns,ceil(512/dt),2); end
  for i = 1:n
    for j = 1:m
      a = v{i}; b = x{j};
      s.mean(i,j,:) = T.Mean.(a).(b); s.median(i,j,:) = T.Median.(a).(b); s.var(i,j,:) = T.Variance.(a).(b);
      s.hpdinf(i,j,:) = T.HPDinf.(a).(b); s.hpdsup(i,j,:) = T.HPDsup.(a).(b);
      for l = 1:ns
        s.deciles(i,j,l,:) = T.deciles.(a).(b)(:,l)';
        if dens, s.density(i,j,l,:,:) = reshape(thin(T.density.(a).(b)(:,:,l), dt), 1, 1, 1, [], 2); end
      end
    end
  end
end

function r = copy_fields(r, prefix, s)
  f = fieldnames(s);
  for k = 1:numel(f), r.([prefix '_' f{k}]) = s.(f{k}); end
end

function r = pack_series(r, prefix, T, names, dens, dt, pidx)
  % pm3 output: struct with Mean.(name) (n2 x 1) ... deciles.(name) (9 x n2), density (512 x 2 x n2); stored at the periods pidx
  % (every 10th period and the last) and, for the density, every dt-th grid point, to keep the files small
  n = numel(names); n2 = numel(T.Mean.(names{1}));
  if n2 == 1, pidx = 1; end
  np = numel(pidx);
  m = NaN(np,n); md = m; v = m; hi = m; hs = m; dc = NaN(9,np,n);
  if dens, de = NaN(ceil(512/dt), 2, np, n); end
  for i = 1:n
    a = names{i};
    x = T.Mean.(a); m(:,i) = x(pidx); x = T.Median.(a); md(:,i) = x(pidx); x = T.Var.(a); v(:,i) = x(pidx);
    x = T.HPDinf.(a); hi(:,i) = x(pidx); x = T.HPDsup.(a); hs(:,i) = x(pidx);
    x = T.deciles.(a); dc(:,:,i) = x(:,pidx);
    if dens, x = T.density.(a); de(:,:,:,i) = x(1:dt:end,:,pidx); end
  end
  r.([prefix '_mean']) = m; r.([prefix '_median']) = md; r.([prefix '_var']) = v; r.([prefix '_hpdinf']) = hi;
  r.([prefix '_hpdsup']) = hs; r.([prefix '_deciles']) = dc;
  if dens, r.([prefix '_density']) = de; end
end

DT = 16; DTS = 32;
dtenv = str2double(getenv('PM_DT')); if ~isnan(dtenv), DT = dtenv; DTS = dtenv; end
for ic = 1:numel(cases)
  parts = strsplit(cases{ic}, ':'); nm = parts{1}; k = str2double(parts{2});
  pause(1.5); rehash;
  if exist(sprintf('postmom_%s', nm), 'dir'), confirm_recursive_rmdir(false, 'local'); rmdir(sprintf('postmom_%s', nm), 's'); end   % no stale files of an earlier case
  tic; eval(sprintf('dynare postmom_%s nograph -DPM=%d', nm, k)); elapsed = toc;
  r = struct('elapsed', elapsed, 'dt', DT, 'dts', DTS);
  if k == 7
    % prior moments(distribution): prior draws, the moments Dynare computed at each, and the draws that did not solve
    options_.prior_mc = 150; options_.noprint = 1;
    prior('moments(distribution)');
    q = load(sprintf('postmom_%s/prior/postmom_%s_endogenous_variables_prior_draws.mat', nm, nm));
    t = load(sprintf('postmom_%s/prior/draws/prior_draws1.mat', nm)); pd = t.pdraws;
    r.param_names = strjoin(bayestopt_.name', ';'); r.endo_names = strjoin(M_.endo_names', ';');
    r.prior_params = cell2mat(cellfun(@(x) x(:)', pd(:,1), 'UniformOutput', false));
    r.prior_info = cell2mat(pd(:,2));
    r.first_order = q.FirstOrderMoments; r.second_order = q.SecondOrderMoments;
    r.orig_endo_names = strjoin(M_.endo_names(1:M_.orig_endo_nbr)', ';');
    save('-mat7-binary', sprintf('postmom_%s_%d.mat', nm, k), '-struct', 'r');
    printf('CASE_DONE %s %d (%.0f s)\n', nm, k, elapsed);
    continue
  end
  md = sprintf('postmom_%s/metropolis', nm); fn = sprintf('postmom_%s', nm);
  r.mh_conf_sig = options_.mh_conf_sig; r.sub_draws = options_.sub_draws;
  r.param_names = strjoin(bayestopt_.name', ';'); r.exo_names = strjoin(M_.exo_names', ';');
  r.varlist = strjoin(options_.varlist', ';'); r.varobs = strjoin(options_.varobs', ';');
  r.endo_names = strjoin(M_.endo_names', ';'); r.nexo = M_.exo_nbr;
  if options_.moments_varendo
    q = load(sprintf('%s/%s_posterior_draws1.mat', md, fn));
    nd = rows(q.pdraws); r.nd = nd; r.pdraws = cell2mat(q.pdraws(:,1));
    T = oo_.PosteriorTheoreticalMoments.dsge;
    v = options_.varlist; r.vartan = strjoin(v', ';'); nv = numel(v);
    r.ar = options_.ar; r.steps = options_.conditional_variance_decomposition; r.nodecomposition = options_.nodecomposition;
    r.contemporaneous_correlation = options_.contemporaneous_correlation;
    r = copy_fields(r, 'cov', pack_pairs(T.covariance, v, true, 8));
    if options_.contemporaneous_correlation
      r = copy_fields(r, 'ccorr', pack_pairs(T.contemporeaneous_correlation, v, true, 8));
    end
    if options_.ar > 0, r = copy_fields(r, 'corr', pack_corr(T.correlation, v, true, DT)); end
    ex = M_.exo_names;
    if isfield(T, 'VarianceDecomposition')
      r = copy_fields(r, 'vd', pack_vd(T.VarianceDecomposition, v, ex, true, DT));
    end
    obsv = intersect(v, options_.varobs, 'stable'); r.me_vars = strjoin(obsv', ';');
    if isfield(T, 'VarianceDecompositionME')
      r = copy_fields(r, 'vdme', pack_vd(T.VarianceDecompositionME, obsv, [ex; 'ME'], true, DT));
    end
    if isfield(T, 'ConditionalVarianceDecomposition')
      r = copy_fields(r, 'cvd', pack_cvd(T.ConditionalVarianceDecomposition, v, ex, true, DT));
    end
    if isfield(T, 'ConditionalVarianceDecompositionME')
      r = copy_fields(r, 'cvdme', pack_cvd(T.ConditionalVarianceDecompositionME, obsv, [ex; 'ME'], true, DT));
    end
    % the per-draw moments Dynare stored (first 6 draws, full precision)
    nk = min(6, nd);
    f = sprintf('%s/%s_Posterior2ndOrderMoments1.mat', md, fn);
    if exist(f, 'file'), t = load(f); r.draw_cov = t.Covariance_matrix(1:nk,:); end
    f = sprintf('%s/%s_PosteriorCorrelations1.mat', md, fn);
    if exist(f, 'file'), t = load(f); r.draw_corr = t.Correlation_array(1:nk,:,:,:); end
    f = sprintf('%s/%s_PosteriorVarianceDecomposition1.mat', md, fn);
    if exist(f, 'file'), t = load(f); r.draw_vd = t.Decomposition_array(1:nk,:); end
    f = sprintf('%s/%s_PosteriorVarianceDecompME1.mat', md, fn);
    if exist(f, 'file'), t = load(f); r.draw_vdme = t.Decomposition_array_ME(1:nk,:); end
    f = sprintf('%s/%s_PosteriorConditionalVarianceDecomposition1.mat', md, fn);
    if exist(f, 'file'), t = load(f); r.draw_cvd = t.Conditional_decomposition_array(:,:,:,1:nk); end
    f = sprintf('%s/%s_PosteriorConditionalVarianceDecompME1.mat', md, fn);
    if exist(f, 'file'), t = load(f); r.draw_cvdme = t.Conditional_decomposition_array_ME(:,:,:,1:nk); end
    r.H_diag = diag(M_.H);
  end
  if options_.smoother || ~isempty(options_.filter_step_ahead)
    ddir = sprintf('%s/metropolis', nm); ddir = md;
    t = load(sprintf('%s/%s_param1.mat', md, fn)); r.sm_params = t.stock; r.sm_logpo = t.stock_logpo; r.sm_ys = t.stock_ys;
    nd = rows(t.stock); r.sm_nd = nd;
    gend = size(oo_.SmoothedVariables.Mean.(options_.varlist{1}),1); if isfield(oo_,'FilteredVariables') && ~options_.smoother, gend = size(oo_.FilteredVariables.Mean.(options_.varlist{1}),1); end
    pidx = unique([1:10:gend, gend]); r.pidx = pidx; r.gend = gend;
    v = options_.varlist; r.sm_varlist = strjoin(v', ';');
    if options_.smoother
      r = pack_series(r, 'smv', oo_.SmoothedVariables, v, true, DTS, pidx);
      r = pack_series(r, 'upd', oo_.UpdatedVariables, v, true, DTS, pidx);
      r = pack_series(r, 'inn', oo_.SmoothedShocks, M_.exo_names, true, DTS, pidx);
      if isfield(oo_, 'SmoothedMeasurementErrors')
        me = fieldnames(oo_.SmoothedMeasurementErrors.Mean); r.me_names = strjoin(me', ';');
        r = pack_series(r, 'sme', oo_.SmoothedMeasurementErrors, me, true, DTS, pidx);
      end
      r = pack_series(r, 'cst', oo_.Constant, v, true, DTS, pidx);
      if isfield(oo_.Smoother, 'TrendCoeff')
        r = pack_series(r, 'tco', oo_.Smoother.TrendCoeff, v, true, DTS, pidx);
        r = pack_series(r, 'ini', oo_.Smoother.Init_State, v, true, DTS, pidx);
        r = pack_series(r, 'trd', oo_.Smoother.Trend, v, true, DTS, pidx);
      end
      % first draws of the stocks
      t = load(sprintf('%s/%s_smooth1.mat', md, fn)); r.stock_smooth = t.stock(:,:,1:2);
      t = load(sprintf('%s/%s_update1.mat', md, fn)); r.stock_update = t.stock(:,:,1:2);
      t = load(sprintf('%s/%s_inno1.mat', md, fn)); r.stock_inno = t.stock(:,:,1:2);
      f = sprintf('%s/%s_error1.mat', md, fn);
      if exist(f, 'file'), t = load(f); r.stock_error = t.stock(:,:,1:2); end
      t = load(sprintf('%s/%s_init_state1.mat', md, fn)); r.stock_init = t.stock;
      t = load(sprintf('%s/%s_trend_coeff1.mat', md, fn)); r.stock_tcoef = t.stock;
      t = load(sprintf('%s/%s_smoothed_constant1.mat', md, fn)); r.stock_const = t.stock(:,:,1:2);
      t = load(sprintf('%s/%s_smoothed_trend1.mat', md, fn)); r.stock_trend = t.stock(:,:,1:2);
    end
    if options_.filtered_vars
      r = pack_series(r, 'flt', oo_.FilteredVariables, v, true, DTS, pidx);
      r.filter_step_ahead = options_.filter_step_ahead;
      for ks = options_.filter_step_ahead(:)'
        nf = sprintf('Filtered_Variables_%d_step_ahead', ks);
        r = pack_series(r, sprintf('fk%d', ks), oo_.(nf), v, true, DTS, pidx);
      end
      r.kstep_mean = oo_.FilteredVariablesKStepAhead; r.kstep_var = oo_.FilteredVariablesKStepAheadVariances;
      t = load(sprintf('%s/%s_filter_step_ahead1.mat', md, fn)); r.stock_fsa = t.stock(:,:,:,1:2);
    end
    r.smoother_loglinear = options_.loglinear;
  end
  save('-mat7-binary', sprintf('postmom_%s_%d.mat', nm, k), '-struct', 'r');
  printf('CASE_DONE %s %d (%.0f s)\n', nm, k, elapsed);
end
disp('POSTMOM_REFS_DONE')
