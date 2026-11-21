% Regenerates the post_*.mat reference files of tests/test_posterior.py and tests/test_estimation.py with
% Dynare 7.2 under Octave 8.4. Run in a scratch copy of this directory (it needs post_nk.mod,
% post_schorfheide.mod, est_nk_data.m and est_schorfheide_data.m):
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_posterior.m > log 2>&1
% Environment variable POST_STAGES selects what is run (default "12"):
%   1  objective log of csminwel (mode_compute=4), Dynare's own mode, mode_check data, prior density curves
%   2  short Metropolis-Hastings runs: chains, the random numbers that produced them, posterior statistics,
%      marginal density, convergence diagnostics, posterior IRFs, mh_tune_jscale
%   3  long runs (2 x 20000 draws per model, about 15 minutes each): batch statistics for the comparison of
%      pydyn's sampler with Dynare's (post_sampler_compare.py)
%   4  the long run repeated with other seeds (POST_SEEDS, default "1000,2000,3000"): post_<m>_seed<k>.mat, the
%      Monte Carlo spread between independent Dynare runs
%   5  no estimation: model_comparison.m and set_mcmc_jumping_covariance.m on small inputs (post_misc.mat)
% Stages 2 and 3 need the mode files written by stage 1 in the same directory. POST_MODELS (default
% "post_nk,post_schorfheide") selects the models.
% Reference files written (copy them to reference/): post_csminwel_fun.mat; per model post_<m>_mode.mat,
% post_<m>_mode0.mat (Dynare's mode file, read back through mode_file), post_<m>_chains.mat, post_<m>_replay.mat,
% post_<m>_stats.mat, post_<m>_long.mat, post_<m>_seed<k>.mat and post_misc.mat. Scratch files, not references: post_<m>_csminwel_full.mat and
% post_<m>_long_chains.mat. After stage 3 run post_sampler_compare.py to redo pydyn's side of the comparison
% (post_<m>_long_pydyn.npz).
%
% There is no graphics toolkit: no-op versions of the figure functions are written to the working
% directory; the stand-in for plot() records what Dynare would have drawn (the Brooks-Gelman series are
% not stored anywhere else by Dynare).
addpath /home/claude/dl/src72/matlab
dynare_config();          % puts Dynare's sub-folders on the path
stages = getenv('POST_STAGES'); if isempty(stages), stages = '12'; end
models = {'post_nk', 'post_schorfheide'};
only = getenv('POST_MODELS'); if ~isempty(only), models = strsplit(only, ','); end

function write_stub(name, body)
  fid = fopen([name '.m'], 'w'); fprintf(fid, '%s\n', body); fclose(fid);
end
write_stub('plot', sprintf('function h = plot(varargin)\nglobal PLOTLOG\nif isempty(PLOTLOG), PLOTLOG = {}; end\nPLOTLOG{end+1} = varargin;\nh = 1;\nend'));
for f = {'dyn_figure', 'subplot'}
  write_stub(f{1}, sprintf('function h = %s(varargin)\nh = 1;\nend', f{1}));
end
for f = {'dyn_saveas', 'hold', 'xlim', 'title', 'drawnow', 'box'}
  write_stub(f{1}, sprintf('function %s(varargin)\nend', f{1}));
end
% objective function that records every evaluation (for the comparison of optimiser paths)
write_stub('post_logged_objective', sprintf(['function [fval, info, exit_flag, a1, a2] = post_logged_objective(x, varargin)\n' ...
  'global POSTLOG\n[fval, info, exit_flag, a1, a2] = dsge_likelihood(x, varargin{:});\n' ...
  'p = 0; if numel(info) >= 4, p = info(4); end\n' ...
  'POSTLOG.X(:, end+1) = x(:); POSTLOG.F(end+1) = fval; POSTLOG.info(:, end+1) = [info(1); p];\nend']));
% an objective evaluated with the same floating-point operations in Octave and in Python (tests/test_estimation.py):
% chained Rosenbrock function on x <= 1.5, with Dynare's penalty convention outside
write_stub('post_testfun', sprintf(['function [fval, info, exit_flag, a1, a2] = post_testfun(x, varargin)\n' ...
  'global POSTLOG\na1 = []; a2 = []; exit_flag = 1; info = zeros(4, 1);\nk = find(x > 1.5);\n' ...
  'if ~isempty(k)\n  fval = Inf; exit_flag = 0; info(1) = 42; s = 0;\n  for j = k(:)'', s = s + (x(j)-1.5)*(x(j)-1.5); end\n  info(4) = s;\n' ...
  'else\n  fval = 0;\n  for i = 1:numel(x)-1\n    a = x(i+1) - x(i)*x(i); b = 1 - x(i);\n    fval = fval + 100*(a*a) + b*b;\n  end\nend\n' ...
  'POSTLOG.X(:, end+1) = x(:); POSTLOG.F(end+1) = fval; POSTLOG.info(:, end+1) = [info(1); info(4)];\nend']));
rehash;

function v = name_field(s, name, M_, options_)
  % value of oo_.posterior_xxx for the parameter called name in bayestopt_.name
  if strncmp(name, 'stderr ', 7)
    n = name(8:end);
    if any(strcmp(n, M_.exo_names)), v = s.shocks_std.(n); else, v = s.measurement_errors_std.(n); end
  elseif strncmp(name, 'corr ', 5)
    p = strsplit(name(6:end), ', '); n = [p{1} '_' p{2}];
    if any(strcmp(p{1}, M_.exo_names)), v = s.shocks_corr.(n); else, v = s.measurement_errors_corr.(n); end
  else
    v = s.parameters.(name);
  end
end

function args = objective_args(dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_)
  args = {dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_.dr, oo_.steady_state, ...
          oo_.exo_steady_state, oo_.exo_det_steady_state};
end

function r = csminwel_log(x0, args, numgrad, epsilon, nit, crit)
  global POSTLOG
  POSTLOG = struct('X', zeros(numel(x0), 0), 'F', [], 'info', zeros(2, 0));
  n = numel(x0);
  [fh, xh, gh, H, itct, fcount, retcode] = csminwel1(@post_logged_objective, x0, 1e-4*eye(n), [], crit, nit, numgrad, ...
                                                     epsilon, 1, false, args{:});
  r = struct('fh', fh, 'xh', xh, 'gh', gh, 'H', H, 'itct', itct, 'fcount', fcount, 'retcode', retcode, ...
             'X', POSTLOG.X, 'F', POSTLOG.F, 'info', POSTLOG.info);
end

function b = batch_stats(x2, nbatch)
  % batch means of the second half of a chain: mean, mean of squares, 5% and 95% quantiles, share of moves
  T = rows(x2); first = floor(0.5*T) + 1; L = floor((T - first + 1)/nbatch); n = columns(x2);
  b = zeros(nbatch, n, 5);
  for k = 1:nbatch
    rng_ = first + (k-1)*L : first + k*L - 1;
    x = x2(rng_, :); s = sort(x);
    b(k,:,1) = mean(x); b(k,:,2) = mean(x.^2);
    b(k,:,3) = s(round(0.05*L), :); b(k,:,4) = s(round(0.95*L), :);
    b(k,:,5) = mean(any(x2(rng_, :) ~= x2(rng_ - 1, :), 2));
  end
end

if any(stages == '1')
  % csminwel on the test function: every evaluation and the result, for several settings
  global POSTLOG
  t = struct(); t.x0 = [-1.2; 1; -1.2; 1; 0.5];
  t.settings = [2 1e-6 1000 1e-7; 3 1e-6 1000 1e-9; 5 1e-5 1000 1e-10; 13 1e-6 40 1e-7; 15 1e-6 40 1e-7; 2 1e-7 12 1e-7];
  for v = 1:rows(t.settings)
    POSTLOG = struct('X', zeros(5, 0), 'F', [], 'info', zeros(2, 0));
    rand('state', 0); cl = rand(5, 40); rand('state', 0);
    [fh, xh, gh, H, itct, fcount, retcode] = csminwel1(@post_testfun, t.x0, 1e-4*eye(5), [], t.settings(v,4), t.settings(v,3), ...
                                                       t.settings(v,1), t.settings(v,2), 0, false);
    p = sprintf('s%d_', v);
    t.([p 'fh']) = fh; t.([p 'xh']) = xh; t.([p 'gh']) = gh; t.([p 'H']) = H; t.([p 'itct']) = itct; t.([p 'fcount']) = fcount;
    t.([p 'retcode']) = retcode; t.([p 'F']) = POSTLOG.F; t.([p 'npen']) = sum(POSTLOG.info(1,:) ~= 0);
    k = unique(round(linspace(1, columns(POSTLOG.X), 200))); t.([p 'Xidx']) = k; t.([p 'X']) = POSTLOG.X(:, k);
    u = rand; t.([p 'ncliff']) = find(cl(1,:) == u) - 1; t.([p 'cliff_rand']) = cl(:, 1:max(t.([p 'ncliff']), 1));
  end
  save('-mat7-binary', 'post_csminwel_fun.mat', '-struct', 't');
end
if any(stages == '5')
  % model_comparison.m on stored marginal densities (three made-up models, with and without priors over models,
  % Laplace and modified harmonic mean) and set_mcmc_jumping_covariance.m with every kind of proposal covariance
  q = struct(); mnames = {'mca', 'mcb', 'mcc'}; q.laplace = [-310.0, -312.5, -309.0]; q.mhm = [-310.4, -311.9, -309.2];
  for i = 1:3
    [~, ~] = mkdir(mnames{i}, 'Output');
    oo_ = struct(); oo_.MarginalDensity.LaplaceApproximation = q.laplace(i); oo_.MarginalDensity.ModifiedHarmonicMean = q.mhm(i);
    save('-mat7-binary', sprintf('%s/Output/%s_results.mat', mnames{i}, mnames{i}), 'oo_');
  end
  o = struct('noprint', true, 'TeX', false); fl = {'Prior', 'Log_Marginal_Density', 'Bayes_Ratio', 'Posterior_Model_Probability'};
  cases = {{[], 'Laplace'}, {[0.2; 0.5; 0.3], 'Laplace'}, {[1; 3; 1], 'ModifiedHarmonicMean'}};
  for c = 1:numel(cases)
    o.mc_marginal_density = cases{c}{2};
    r = model_comparison(mnames, cases{c}{1}, struct(), o, 'none');
    f = fl; if isempty(cases{c}{1}), f = [fl(1:3), {'Posterior_Odds_Ratio'}, fl(4)]; end
    v = NaN(5, 3);
    for i = 1:3, for k = 1:numel(f), v(k, i) = r.Model_Comparison.(mnames{i}).(f{k}); end, end
    q.(sprintf('mc%d', c)) = v; q.(sprintf('mc%d_fields', c)) = strjoin(f, ';');
  end
  q.mc_priors2 = [0.2; 0.5; 0.3]; q.mc_priors3 = [1; 3; 1];
  % jumping covariances: Hessian (returned as is), prior variances, identity, a matrix read from a file
  bt = struct('p2', [0.1; 0.5; 2]); V0 = [2 0.3 0.1; 0.3 1 -0.2; 0.1 -0.2 0.5]; jumping_covariance = [1 0.2 0; 0.2 0.5 0.1; 0 0.1 0.3];
  save('-mat7-binary', 'post_jc.mat', 'jumping_covariance');
  q.jc_hessian = set_mcmc_jumping_covariance(V0, 3, 'hessian', bt, 'r_posterior');
  q.jc_prior = set_mcmc_jumping_covariance(V0, 3, 'prior_variance', bt, 'r_posterior');
  q.jc_identity = set_mcmc_jumping_covariance(V0, 3, 'identity_matrix', bt, 'r_posterior');
  q.jc_file = set_mcmc_jumping_covariance(V0, 3, 'post_jc', bt, 'r_posterior'); q.jc_file_matrix = jumping_covariance;
  q.jc_V0 = V0; q.jc_p2 = bt.p2;
  save('-mat7-binary', 'post_misc.mat', '-struct', 'q');
  printf('STAGE5_DONE\n');
end
for im = 1:numel(models)
  nm = models{im};
  % ================================================================= stage 1
  if any(stages == '1')
    pause(1.5); rehash;
    eval(sprintf('dynare %s nograph -DPOST=0', nm));
    pause(1.5); rehash;
    [dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    args = objective_args(dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_);
    r = struct('x0', xparam1, 'names', strjoin(bayestopt_.name', ';'), 'lb', bounds.lb, 'ub', bounds.ub);
    n = numel(xparam1);
    % csminwel as called by dynare_minimize_objective (mode_compute=4), default options
    % csminwel perturbs its search direction with rand(n,1) when it meets a "cliff": record those numbers
    % (Dynare seeds rand with 0 at the start of a run, so its own optimisation below uses the same ones)
    rand('state', 0); r.c2_cliff_rand = rand(n, 6); rand('state', 0);
    c = csminwel_log(xparam1, args, options_.gradient_method, options_.gradient_epsilon, options_.csminwel.maxiter, ...
                     options_.csminwel.tolerance.f);
    full = c; save('-mat7-binary', sprintf('%s_csminwel_full.mat', nm), '-struct', 'full');
    r.c2_fh = c.fh; r.c2_xh = c.xh; r.c2_gh = c.gh; r.c2_H = c.H; r.c2_itct = c.itct; r.c2_fcount = c.fcount;
    r.c2_retcode = c.retcode; r.c2_F = c.F; r.c2_info = c.info;
    k = unique([1:min(columns(c.X), 60), round(linspace(1, columns(c.X), 150))]);       % a sample of the points
    r.c2_Xidx = k; r.c2_X = c.X(:, k);
    % other gradient formulas and option values (a few iterations each)
    variants = [3 1e-6 4 1e-7; 5 1e-6 3 1e-7; 13 1e-5 3 1e-7; 15 1e-6 2 1e-7; 2 1e-4 6 1e-3];
    for v = 1:rows(variants)
      rand('state', 0);
      c = csminwel_log(xparam1, args, variants(v,1), variants(v,2), variants(v,3), variants(v,4));
      p = sprintf('v%d_', v);
      r.([p 'fh']) = c.fh; r.([p 'xh']) = c.xh; r.([p 'H']) = c.H; r.([p 'itct']) = c.itct; r.([p 'fcount']) = c.fcount;
      r.([p 'retcode']) = c.retcode; r.([p 'F']) = c.F; r.([p 'X']) = c.X(:, round(linspace(1, columns(c.X), 25)));
      r.([p 'Xidx']) = round(linspace(1, columns(c.X), 25));
    end
    r.variants = variants;
    % penalties of the objective function outside its domain (base penalty + info(4) in csminwel)
    XP = repmat(xparam1, 1, 2*n); pinfo = zeros(2, 2*n); pf = zeros(1, 2*n);
    for j = 1:n
      w = bounds.ub(j) - bounds.lb(j); if ~isfinite(w), w = 1; end
      if isfinite(bounds.lb(j)), XP(j, 2*j-1) = bounds.lb(j) - 0.01*w; end
      if isfinite(bounds.ub(j)), XP(j, 2*j) = bounds.ub(j) + 0.02*w; end
    end
    for j = 1:2*n
      [fval, info] = dsge_likelihood(XP(:,j), args{:}); pf(j) = fval; pinfo(1,j) = info(1);
      if numel(info) >= 4, pinfo(2,j) = info(4); end
    end
    r.pen_X = XP; r.pen_info = pinfo; r.pen_f = pf;
    % prior density curves (plot_priors)
    r.prior_x = zeros(n, 200); r.prior_f = zeros(n, 200);
    for j = 1:n, [r.prior_x(j,:), r.prior_f(j,:)] = draw_prior_density(j, bayestopt_); end
    r.pshape = bayestopt_.pshape; r.p1 = bayestopt_.p1; r.p2 = bayestopt_.p2; r.p5 = bayestopt_.p5;
    % Dynare's own run of the same optimiser, its Hessian, mode_check
    pause(1.5); rehash;
    eval(sprintf('dynare %s nograph -DPOST=1', nm));
    md = load(sprintf('%s/Output/%s_mode.mat', nm, nm));
    r.xparam1 = md.xparam1; r.hh = md.hh; r.fval = md.fval;
    r.Vmode = oo_.posterior.optimization.Variance; r.logpost = oo_.posterior.optimization.log_density;
    r.laplace = oo_.MarginalDensity.LaplaceApproximation;
    xparam1 = md.xparam1; hh = md.hh; fval = md.fval; parameter_names = md.parameter_names;
    save('-mat7-binary', sprintf('%s_mode0.mat', nm), 'xparam1', 'hh', 'fval', 'parameter_names');
    ck = load(sprintf('%s/graphs/%s_check_plot_data.mat', nm, nm));
    fn = fieldnames(ck.mcheck.cross); r.check_names = strjoin(fn', ';');
    r.check = NaN(n, 25, 3);
    for j = 1:n, z = ck.mcheck.cross.(fn{j}); r.check(j, 1:rows(z), :) = z; end
    % mode_check with other settings, called as in dynare_estimation_1
    [dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    args = objective_args(dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, oo_);
    o2 = options_; o2.mode_check.neighbourhood_size = 0.2; o2.mode_check.symmetric_plots = false;
    o2.mode_check.number_of_points = 8; o2.nograph = true;
    mc = mode_check(@dsge_likelihood, md.xparam1, md.hh, o2, M_, estim_params_, bayestopt_, bounds, false, args{:});
    r.check2 = NaN(n, 12, 3);
    for j = 1:n, z = mc.cross.(fn{j}); r.check2(j, 1:rows(z), :) = z; end
    o2.mode_check.neighbourhood_size = Inf; o2.mode_check.number_of_points = 6;
    mc = mode_check(@dsge_likelihood, md.xparam1, md.hh, o2, M_, estim_params_, bayestopt_, bounds, false, args{:});
    r.check3 = NaN(n, 10, 3);
    for j = 1:n, z = mc.cross.(fn{j}); r.check3(j, 1:rows(z), :) = z; end
    save('-mat7-binary', sprintf('%s_mode.mat', nm), '-struct', 'r');
    printf('STAGE1_DONE %s\n', nm);
  end
  % ================================================================= stage 2
  if any(stages == '2')
    pause(1.5); rehash;
    eval(sprintf('dynare %s nograph -DPOST=2', nm));      % clears the global variables, PLOTLOG included
    global PLOTLOG
    plotlog = PLOTLOG;
    names = bayestopt_.name; n = numel(names);
    rec = load(sprintf('%s/metropolis/%s_mh_history_0.mat', nm, nm)); rec = rec.record;
    nb = rec.Nblck; T = rec.MhDraws(1,1);
    s = struct('names', strjoin(names', ';'), 'nblck', nb, 'replic', T, 'mh_drop', options_.mh_drop, ...
               'mh_conf_sig', options_.mh_conf_sig, 'jscale', bayestopt_.jscale, 'mh_jscale', options_.mh_jscale, ...
               'mh_init_scale_factor', options_.mh_init_scale_factor);
    md = load(sprintf('%s_mode0.mat', nm)); s.xparam1 = md.xparam1; s.hh = md.hh;
    s.proposal_chol = rec.ProposalCovariance; s.x_init = rec.InitialParameters; s.lp_init = rec.InitialLogPost;
    s.acceptance = rec.AcceptanceRatio; s.ifac = rec.InefficiencyFactorsPerBlock;
    mb = set_mcmc_prior_bounds(md.xparam1, bayestopt_, options_, 'r_posterior');
    s.mh_lb = mb.lb; s.mh_ub = mb.ub;
    % chains, stored as the distinct points and, for every draw, the index of its point
    X = zeros(T, n, nb); LP = zeros(T, nb);
    for b = 1:nb
      q = load(sprintf('%s/metropolis/%s_mh1_blck%d.mat', nm, nm, b)); X(:,:,b) = q.x2; LP(:,b) = q.logpo2;
    end
    pts = zeros(0, n); plp = zeros(0, 1); idx = zeros(T, nb);
    for b = 1:nb
      mv = [true; any(diff(X(:,:,b)) ~= 0, 2)];
      idx(:,b) = rows(pts) + cumsum(mv);
      pts = [pts; X(mv,:,b)]; plp = [plp; LP(mv,b)];
    end
    s.points = pts; s.points_lp = plp; s.index = idx;
    % the random numbers behind the first NSEG draws of every chain: Dynare seeds rand and randn with
    % InitialSeeds, then draws randn(1,n) for every proposal and one rand for every proposal that has a
    % finite posterior. The starting points come from randn(1,n) draws made after seeding with 0.
    NSEG = 400;
    s.seg_normal = zeros(NSEG, n, nb); s.seg_uniform = zeros(NSEG, nb);
    for b = 1:nb
      rand('state', rec.InitialSeeds(b).Unifor); randn('state', rec.InitialSeeds(b).Normal);
      s.seg_normal(:,:,b) = randn(n, NSEG)'; s.seg_uniform(:,b) = rand(NSEG, 1);
    end
    rand('state', 0); randn('state', 0); s.init_normal = randn(n, 30)';
    % posterior statistics
    for f = {'posterior_mean', 'posterior_median', 'posterior_std', 'posterior_variance', 'posterior_hpdinf', 'posterior_hpdsup'}
      v = zeros(n, 1); for j = 1:n, v(j) = name_field(oo_.(f{1}), names{j}, M_, options_); end
      s.(f{1}) = v;
    end
    s.posterior_deciles = zeros(n, 9); dens = zeros(n, 512, 2);
    for j = 1:n
      s.posterior_deciles(j,:) = name_field(oo_.posterior_deciles, names{j}, M_, options_);
      dens(j,:,:) = name_field(oo_.posterior_density, names{j}, M_, options_);
    end
    s.density_x = dens(:, 1:4:end, 1); s.density_f = dens(:, 1:4:end, 2);       % every fourth grid point
    s.density_first = dens(:, 1, 1); s.density_last = dens(:, end, 1);
    mm = load(sprintf('%s/Output/%s_mh_mode.mat', nm, nm)); s.mh_mode = mm.xparam1; s.mh_mode_lp = mm.fval; s.mh_mode_hh = mm.hh;
    mm = load(sprintf('%s/Output/%s_mean.mat', nm, nm)); s.mh_mean = mm.xparam1; s.mh_cov = mm.SIGMA;
    % marginal density
    s.mhm = oo_.MarginalDensity.ModifiedHarmonicMean; s.laplace = oo_.MarginalDensity.LaplaceApproximation;
    [s.mhm_table, tmp] = marginal_density(M_, options_, estim_params_, oo_, bayestopt_);
    s.mhm_again = tmp.MarginalDensity.ModifiedHarmonicMean;
    o2 = options_; o2.marginal_data_density.harmonic_mean.tolerance = 1e-5;     % forces the loop over wider supports
    [s.mhm_table_tight, tmp] = marginal_density(M_, o2, estim_params_, oo_, bayestopt_);
    s.mhm_tight = tmp.MarginalDensity.ModifiedHarmonicMean;
    % convergence diagnostics
    taper = options_.convergence.geweke.taper_steps(:)'; s.taper_steps = taper; s.geweke_interval = options_.convergence.geweke.geweke_interval;
    gf = {'posteriormean', 'posteriorstd', 'nse_iid', 'rne_iid'};
    for t = taper(:)', gf{end+1} = sprintf('nse_taper_%u', t); gf{end+1} = sprintf('rne_taper_%u', t); end
    s.geweke_fields = strjoin(gf, ';');
    s.geweke = zeros(nb, n, numel(gf)); s.geweke_pooled_mean = zeros(nb, n, numel(taper)+1);
    s.geweke_pooled_nse = s.geweke_pooled_mean; s.geweke_prob = s.geweke_pooled_mean;
    gn = fieldnames(oo_.convergence.geweke(1)); s.dynare_names = strjoin(gn', ';');
    for b = 1:nb
      for j = 1:n
        g = oo_.convergence.geweke(b).(gn{j});
        for k = 1:numel(gf), s.geweke(b,j,k) = g.(gf{k}); end
        s.geweke_pooled_mean(b,j,:) = g.pooled_mean; s.geweke_pooled_nse(b,j,:) = g.pooled_nse; s.geweke_prob(b,j,:) = g.prob_chi2_test;
      end
    end
    if options_.convergence.rafterylewis.indicator
      rf = {'M_burn', 'N_prec', 'k_thin', 'k_ind', 'I_stat', 'N_total'};
      s.raftery = zeros(nb, n, numel(rf)); s.raftery_fields = strjoin(rf, ';'); s.raftery_qrs = options_.convergence.rafterylewis.qrs;
      for b = 1:nb
        for k = 1:numel(rf), s.raftery(b,:,k) = oo_.convergence.raftery_lewis(b).(rf{k}); end
      end
      s.raftery_N_min = oo_.convergence.raftery_lewis(1).N_min;
    end
    % Brooks-Gelman series as plotted: for every parameter interval/m2/m3, each pooled then within; then
    % the same six series for the posterior kernel (multivariate diagnostic)
    s.bg_x = plotlog{1}{1};
    s.bg = zeros(numel(s.bg_x), numel(plotlog));
    for k = 1:numel(plotlog), s.bg(:,k) = plotlog{k}{2}; end
    % posterior IRFs and the parameter draws they were computed from
    q = load(sprintf('%s/metropolis/%s_param_irf1.mat', nm, nm)); s.irf_params = q.stock;
    fi = fieldnames(oo_.PosteriorIRF.dsge.Mean); s.irf_names = strjoin(fi', ';');
    H = options_.irf; s.irf_mean = zeros(H, numel(fi)); s.irf_median = s.irf_mean; s.irf_var = s.irf_mean;
    s.irf_hpdinf = s.irf_mean; s.irf_hpdsup = s.irf_mean; s.irf_deciles = zeros(H, 9, numel(fi));
    for k = 1:numel(fi)
      s.irf_mean(:,k) = oo_.PosteriorIRF.dsge.Mean.(fi{k}); s.irf_median(:,k) = oo_.PosteriorIRF.dsge.Median.(fi{k});
      s.irf_var(:,k) = oo_.PosteriorIRF.dsge.Var.(fi{k}); s.irf_hpdinf(:,k) = oo_.PosteriorIRF.dsge.HPDinf.(fi{k});
      s.irf_hpdsup(:,k) = oo_.PosteriorIRF.dsge.HPDsup.(fi{k}); s.irf_deciles(:,:,k) = oo_.PosteriorIRF.dsge.deciles.(fi{k});
    end
    q = load(sprintf('%s/metropolis/%s_IRF_DSGEs1.mat', nm, nm));
    s.irf_draws_first = q.STOCK_IRF_DSGE(:, :, :, 1:3);       % periods x variables x shocks x draws
    s.irf_varlist = strjoin(options_.varlist', ';'); s.exo_names = strjoin(M_.exo_names', ';');
    % mh_tune_jscale: calibrate_mh_scale_parameter called as by tune_mcmc_mh_jscale_wrapper, with known random numbers
    [dataset_, dataset_info, ~, ~, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
        dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
    args = objective_args(dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, mb, oo_);
    opt = options_.mh_tune_jscale; opt.console_mode = true; opt.rwmh = options_.posterior_sampler_options.rwmh;
    opt.maxiter = 900; opt.stepsize = 60; opt.target = 0.3; opt.c1 = 0.1; opt.c2 = 0.1; opt.c3 = 2;
    V = rec.ProposalCovariance' * rec.ProposalCovariance;
    rand('state', 77); randn('state', 77); s.tune_normal = randn(n, opt.maxiter)'; s.tune_uniform = rand(opt.maxiter, 1);
    rand('state', 77); randn('state', 77);
    s.tune_scale = calibrate_mh_scale_parameter(@dsge_likelihood, V, md.xparam1, [mb.lb, mb.ub], opt, args{:});
    u_next = rand; s.tune_used_uniform = find(s.tune_uniform == u_next) - 1;      % how many uniforms were consumed
    if isempty(s.tune_used_uniform), s.tune_used_uniform = opt.maxiter; end
    s.tune_cov = V; s.tune_opt = [opt.maxiter opt.stepsize opt.target opt.rho opt.c1 opt.c2 opt.c3];
    opt.guess = 0.9; opt.target = 0.5; opt.maxiter = 300; opt.stepsize = 50; opt.rho = 0.3;
    rand('state', 77); randn('state', 77);
    s.tune_scale2 = calibrate_mh_scale_parameter(@dsge_likelihood, V, md.xparam1, [mb.lb, mb.ub], opt, args{:});
    s.tune_opt2 = [opt.maxiter opt.stepsize opt.target opt.rho opt.c1 opt.c2 opt.c3 opt.guess];
    % three files: the chains, the random numbers behind them, and everything Dynare computed from the chains
    c = struct(); p = struct();
    for f = {'names', 'nblck', 'replic', 'mh_drop', 'mh_conf_sig', 'jscale', 'mh_jscale', 'mh_init_scale_factor', 'xparam1', 'hh', ...
             'proposal_chol', 'x_init', 'lp_init', 'acceptance', 'mh_lb', 'mh_ub', 'points', 'points_lp', 'index'}
      c.(f{1}) = s.(f{1}); s = rmfield(s, f{1});
    end
    for f = {'seg_normal', 'seg_uniform', 'init_normal', 'tune_normal', 'tune_uniform', 'tune_scale', 'tune_used_uniform', 'tune_cov', ...
             'tune_opt', 'tune_scale2', 'tune_opt2'}
      p.(f{1}) = s.(f{1}); s = rmfield(s, f{1});
    end
    save('-mat7-binary', sprintf('%s_chains.mat', nm), '-struct', 'c');
    save('-mat7-binary', sprintf('%s_replay.mat', nm), '-struct', 'p');
    save('-mat7-binary', sprintf('%s_stats.mat', nm), '-struct', 's');
    printf('STAGE2_DONE %s\n', nm);
  end
  % ================================================================= stage 3
  if any(stages == '3')
    pause(1.5); rehash;
    tic; eval(sprintf('dynare %s nograph -DPOST=3', nm)); elapsed = toc;
    rec = load(sprintf('%s/metropolis/%s_mh_history_0.mat', nm, nm)); rec = rec.record;
    nb = rec.Nblck; T = rec.MhDraws(1,1); names = bayestopt_.name; n = numel(names);
    md = load(sprintf('%s_mode0.mat', nm));
    L = struct('names', strjoin(names', ';'), 'nblck', nb, 'replic', T, 'acceptance', rec.AcceptanceRatio, ...
               'x_init', rec.InitialParameters, 'xparam1', md.xparam1, 'hh', md.hh, 'proposal_chol', rec.ProposalCovariance, ...
               'jscale', bayestopt_.jscale, 'elapsed', elapsed);
    nbatch = 20; L.batch = zeros(nbatch, n, 5, nb); chains = zeros(T, n, nb); logpo = zeros(T, nb);
    for b = 1:nb
      q = load(sprintf('%s/metropolis/%s_mh1_blck%d.mat', nm, nm, b));
      L.batch(:,:,:,b) = batch_stats(q.x2, nbatch); chains(:,:,b) = q.x2; logpo(:,b) = q.logpo2;
    end
    for f = {'posterior_mean', 'posterior_std', 'posterior_hpdinf', 'posterior_hpdsup'}
      v = zeros(n, 1); for j = 1:n, v(j) = name_field(oo_.(f{1}), names{j}, M_, options_); end
      L.(f{1}) = v;
    end
    L.mhm = oo_.MarginalDensity.ModifiedHarmonicMean; L.ifac = rec.InefficiencyFactorsPerBlock;
    save('-mat7-binary', sprintf('%s_long.mat', nm), '-struct', 'L');
    save('-mat7-binary', sprintf('%s_long_chains.mat', nm), 'chains', 'logpo');      % scratch only, not a reference file
    printf('STAGE3_DONE %s\n', nm);
  end
  % ================================================================= stage 4
  % the long run of stage 3 repeated with other seeds (set_dynare_seed through -DSEED=k; POST_SEEDS, default
  % "1000,2000,3000"): batch statistics only, to measure the Monte Carlo spread between independent Dynare runs.
  % Dynare seeds chain j of a run with seed k by k + 1 + 2 + ... + j (posterior_sampler_initialization.m adds j to
  % the seed cumulatively; checked against the seeds stored in the mh_history files: chains 1 and 2 of the default
  % run use seeds 1 and 3), and both rand and randn of a chain start from that same state. Runs with neighbouring
  % seeds can therefore share the random numbers of a chain (seed 2: chain 1 is chain 2 of the default run); chains
  % driven by the same random numbers from different starting points tend to coalesce, so such runs are not
  % independent. Hence seeds far apart (the default run has seed 0).
  if any(stages == '4')
    seeds = getenv('POST_SEEDS'); if isempty(seeds), seeds = '1000,2000,3000'; end
    seeds = str2double(strsplit(seeds, ','));
    for sd = seeds(:)'
      pause(1.5); rehash;
      tic; eval(sprintf('dynare %s nograph -DPOST=3 -DSEED=%d', nm, sd)); elapsed = toc;
      rec = load(sprintf('%s/metropolis/%s_mh_history_0.mat', nm, nm)); rec = rec.record;
      nb = rec.Nblck; T = rec.MhDraws(1,1); names = bayestopt_.name; n = numel(names);
      S = struct('names', strjoin(names', ';'), 'seed', sd, 'nblck', nb, 'replic', T, 'acceptance', rec.AcceptanceRatio, ...
                 'x_init', rec.InitialParameters, 'elapsed', elapsed, 'ifac', rec.InefficiencyFactorsPerBlock);
      nbatch = 20; S.batch = zeros(nbatch, n, 5, nb); S.chain_mean = zeros(nb, n);
      for b = 1:nb
        q = load(sprintf('%s/metropolis/%s_mh1_blck%d.mat', nm, nm, b));
        S.batch(:,:,:,b) = batch_stats(q.x2, nbatch); S.chain_mean(b,:) = mean(q.x2(floor(0.5*T)+1:end, :));
      end
      v = zeros(n, 1); for j = 1:n, v(j) = name_field(oo_.posterior_mean, names{j}, M_, options_); end
      S.posterior_mean = v;
      save('-mat7-binary', sprintf('%s_seed%d.mat', nm, sd), '-struct', 'S');
      printf('STAGE4_DONE %s seed %d\n', nm, sd);
    end
  end
end
disp('POSTERIOR_REFS_DONE')
