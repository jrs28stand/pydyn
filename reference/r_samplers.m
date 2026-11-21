% Regenerates the smp_*.mat reference files of tests/test_samplers.py with Dynare 7.2 under Octave 8.4.
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_samplers.m > log 2>&1
% Run from a scratch directory that holds copies of smp_nk.mod, smp_schorfheide.mod, smp_schorfheide_modes.mat,
% post_nk_mode0.mat, post_schorfheide_mode0.mat, est_nk_data.m and est_schorfheide_data.m; every case runs in a fresh
% sub-directory. Environment variable SMP_CASES selects the cases (default: all), e.g. SMP_CASES="1,3".
%
% Cases (see the .mod files): 1 slice (smp_nk, two chains from prior draws), 2 slice (smp_schorfheide, one chain from
% the prior mode), 3 rotated slice initialised at the mode (smp_nk), 4 multimodal slice (smp_schorfheide, mode_files),
% 5 TaRB (smp_nk), 6 independent MH (smp_nk), 7 RWMH with a Student proposal (smp_schorfheide), 8 RWMH with tight
% estimated_params bounds (smp_nk), 9 TaRB with a Student proposal (smp_schorfheide), 10 independent MH with a Student
% proposal (smp_schorfheide), 11 hssmc (smp_schorfheide), 12 dsmh (smp_schorfheide), 13 and 14: TaRB as 5 and 9 (fewer
% draws) with a copy of posterior_sampler_iteration.m in the case directory that logs every block: the current draw,
% the states of rand and randn before the block, the optimum, the proposal, the kernel there, the acceptance
% log-probability and the uniform (smp_case13.mat, smp_case14.mat; fields b<k>_*).
%
% Written per case (smp_case<k>.mat): the chains (x2, logpo2 per chain: chains x draws x parameters), the starting
% points and log posteriors, acceptance ratios and function evaluations per iteration from the mh_history file, the
% sampler's bounds (sampler_options.bounds) and the bounds of the posterior kernel (set_mcmc_prior_bounds), W1, V1, WR,
% the proposal covariance factor and scales, the prior mode p5, and the run time. SMC: the particles of the first,
% a middle and the last stage, and the marginal density estimate.
% smp_tarb_seed1000.mat and smp_tarb_seed2000.mat (statistics of TaRB chains for tests/test_samplers.py) come from
% `dynare smp_schorfheide -DCASE=15 -DSEED=k` in fresh directories (k = 1000, 2000): per chain the mean and std(.,1) of
% draws 76..150 (fields mean1, std1, mean2, std2) and the acceptance ratios.
% The random numbers are not stored: pydyn regenerates them from Dynare's seeds (pydyn.octave_rng). Case 0 writes
% smp_rng.mat, the check of that emulation: for seeds 1 and 7, two million draws of rand and randn (every 997th value,
% every normal beyond the ziggurat's base strip |z| > 3.6541528853610088, the first 50 values, the sums), the state
% vectors after seeding and after 1001 uniforms, and randperm interleaved with rand and randn.
addpath /home/claude/dl/src72/matlab
dynare_config();
function write_stub(name, body)
  fid = fopen([name '.m'], 'w'); fprintf(fid, '%s\n', body); fclose(fid);
end
cases = 1:14;
sel = getenv('SMP_CASES'); if ~isempty(sel), cases = str2double(strsplit(sel, ',')); end
model_of = {'smp_nk', 'smp_schorfheide', 'smp_nk', 'smp_schorfheide', 'smp_nk', 'smp_nk', 'smp_schorfheide', 'smp_nk', ...
            'smp_schorfheide', 'smp_schorfheide', 'smp_schorfheide', 'smp_schorfheide', 'smp_nk', 'smp_schorfheide'};
function instrument_tarb()
  % posterior_sampler_iteration.m of Dynare with a log of every TaRB block in the global TARBLOG
  s = fileread('/home/claude/dl/src72/matlab/estimation/posterior_sampler_iteration.m');
  rep = {sprintf('    for block_iter=1:nblocks\n        blocked_draws_counter=blocked_draws_counter+1;'), ...
         sprintf(['    global TARBLOG\n    for block_iter=1:nblocks\n        blocked_draws_counter=blocked_draws_counter+1;\n' ...
                  '        L = struct(); L.idx = indices(blocks==block_iter,1); L.current = current_draw; ' ...
                  'L.last_posterior = last_posterior; L.su = rand(''state''); L.sn = randn(''state'');']); ...
         '        [xopt_current_block] = dynare_minimize_objective(', ...
         '        [xopt_current_block,L.fopt,~,~,~,~,oinfo] = dynare_minimize_objective('; ...
         '        %% covariance for proposal density', ...
         sprintf('        L.xopt = xopt_current_block; L.fc = oinfo.funcCount; L.it = oinfo.iterations;\n        %%%% covariance for proposal density'); ...
         '        % check whether draw is valid and compute posterior', ...
         sprintf(['        L.chol = proposal_covariance_Cholesky_decomposition_upper; L.prop = proposed_par; L.accprob = NaN; L.u = NaN;\n' ...
                  '        %% check whether draw is valid and compute posterior']); ...
         '            if  (log(rand) < accprob)', sprintf('            L.u = rand; L.accprob = accprob;\n            if  (log(L.u) < accprob)'); ...
         sprintf('        end\n    end\n    accepted=accepted_draws_counter/blocked_draws_counter;'), ...
         sprintf(['        end\n        L.logpost = logpost; L.after = current_draw;\n' ...
                  '        if isempty(TARBLOG), TARBLOG = {L}; else, TARBLOG{end+1} = L; end\n    end\n' ...
                  '    accepted=accepted_draws_counter/blocked_draws_counter;'])};
  for i = 1:rows(rep)
    if isempty(strfind(s, rep{i,1})), error('instrument_tarb: anchor %d not found', i); end
    s = strrep(s, rep{i,1}, rep{i,2});
  end
  fid = fopen('posterior_sampler_iteration.m', 'w'); fprintf(fid, '%s', s); fclose(fid);
end
top = pwd;
if any(cases == 0)
  q = struct(); N = 2000000;
  for sd = [1 7]
    rand('state', sd); u = rand(N, 1); randn('state', sd); z = randn(N, 1);
    p = sprintf('s%d_', sd);
    q.([p 'u_sub']) = u(1:997:end); q.([p 'z_sub']) = z(1:997:end); q.([p 'u_first']) = u(1:50); q.([p 'z_first']) = z(1:50);
    k = find(abs(z) > 3.6541528853610088); q.([p 'tail_idx']) = k; q.([p 'tail_z']) = z(k);
    q.([p 'u_sum']) = sum(u); q.([p 'z_sum']) = sum(z);
    rand('state', sd); q.([p 'state0']) = rand('state'); rand(1001, 1); q.([p 'state1001']) = rand('state');
  end
  rand('state', 7); randn('state', 7);
  q.mix_a = randn(3, 1); q.mix_p1 = randperm(6); q.mix_b = rand(2, 1); q.mix_p2 = randperm(6); q.mix_c = randn(1);
  q.mix_p3 = randperm(5, 3); q.mix_d = rand(1);
  % small functions: chol_SE on a positive definite and two indefinite matrices, the proposal densities, a Student draw
  addpath /home/claude/dl/src72/matlab/estimation
  q.cse_A1 = [4 2 1; 2 5 3; 1 3 6]; q.cse_A2 = [1 2 0.5; 2 1 0.3; 0.5 0.3 -0.2]; q.cse_A3 = [2 -1 0 0.4; -1 1e-9 0.2 0; 0 0.2 -3 1; 0.4 0 1 0.5];
  for i = 1:3
    [R, indef, E] = chol_SE(q.(sprintf('cse_A%d', i)), 0);
    q.(sprintf('cse_R%d', i)) = R; q.(sprintf('cse_indef%d', i)) = indef; q.(sprintf('cse_E%d', i)) = E;
  end
  C = chol([2 0.3 0.1; 0.3 1 -0.2; 0.1 -0.2 0.5]); q.pdf_C = C; q.pdf_X = [0.3 -1.2 0.8]; q.pdf_M = [0.1 0.2 -0.1];
  q.pdf_normal = multivariate_normal_pdf(q.pdf_X, q.pdf_M, C, 3);
  q.pdf_student = multivariate_student_pdf(q.pdf_X, q.pdf_M, C, 5);
  rand('state', 11); randn('state', 11); q.student_draw = rand_multivariate_student(q.pdf_M, C, 5); q.student_next = randn(1);
  save('-mat7-binary', 'smp_rng.mat', '-struct', 'q');
  cases = cases(cases ~= 0);
end
for k = cases
  nm = model_of{k};
  d = sprintf('case%d', k);
  if isfolder(d), confirm_recursive_rmdir(false); rmdir(d, 's'); end
  mkdir(d);
  for f = {[nm '.mod'], 'smp_schorfheide_modes.mat', 'post_nk_mode0.mat', 'post_schorfheide_mode0.mat', 'est_nk_data.m', ...
           'est_schorfheide_data.m'}
    copyfile(f{1}, d);
  end
  cd(d);
  for f = {'dyn_figure', 'subplot', 'plot'}
    write_stub(f{1}, sprintf('function h = %s(varargin)\nh = 1;\nend', f{1}));
  end
  for f = {'dyn_saveas', 'hold', 'xlim', 'title', 'drawnow', 'box'}
    write_stub(f{1}, sprintf('function %s(varargin)\nend', f{1}));
  end
  global TARBLOG
  TARBLOG = {};
  if k >= 13, instrument_tarb(); end
  rehash; pause(1.5);
  t0 = tic;
  eval(sprintf('dynare %s nograph -DCASE=%d', nm, k));
  s = struct('elapsed', toc(t0), 'case_number', k, 'names', strjoin(bayestopt_.name', ';'), 'p5', bayestopt_.p5, ...
             'prior_trunc', options_.prior_trunc);
  co = options_.posterior_sampler_options.current_options;
  if isfield(co, 'bounds'), s.sampler_lb = co.bounds.lb; s.sampler_ub = co.bounds.ub; end
  for f = {'W1', 'V1', 'WR', 'new_block_probability', 'student_degrees_of_freedom'}
    if isfield(co, f{1}) && ~isempty(co.(f{1})), s.(f{1}) = co.(f{1}); end
  end
  mb = prior_bounds(bayestopt_, options_.prior_trunc); s.mh_lb = mb.lb; s.mh_ub = mb.ub;
  if k <= 10 || k >= 13
    rec = load(sprintf('%s/metropolis/%s_mh_history_0.mat', nm, nm)); rec = rec.record;
    nb = rec.Nblck; T = rec.MhDraws(1,1); n = numel(bayestopt_.name);
    s.x_init = rec.InitialParameters; s.lp_init = rec.InitialLogPost; s.acceptance = rec.AcceptanceRatio;
    s.feval = rec.FunctionEvalPerIteration; s.proposal_chol = rec.ProposalCovariance; s.jscale = rec.ProposalScaleVec;
    s.x2 = zeros(nb, T, n); s.logpo2 = zeros(nb, T);
    for b = 1:nb
      q = load(sprintf('%s/metropolis/%s_mh1_blck%d.mat', nm, nm, b));
      s.x2(b,:,:) = q.x2; s.logpo2(b,:) = q.logpo2;
    end
    s.seeds = [rec.InitialSeeds(1).Unifor(1:3)'];
    if k >= 13
      global TARBLOG
      s.nblocks_logged = numel(TARBLOG);
      for b = 1:numel(TARBLOG)
        f = fieldnames(TARBLOG{b});
        for j = 1:numel(f), s.(sprintf('b%d_%s', b, f{j})) = TARBLOG{b}.(f{j}); end
      end
    end
  elseif k == 11
    H = options_.posterior_sampler_options.current_options.steps;
    q = load(sprintf('%s/hssmc/particles-1-%d.mat', nm, H)); s.p_first = q.particles'; s.lp_first = q.tlogpostkernel;
    s.ll_first = q.loglikelihood;
    mid = 4;
    q = load(sprintf('%s/hssmc/particles-%d-%d.mat', nm, mid, H)); s.p_mid = q.particles'; s.lp_mid = q.tlogpostkernel;
    s.ll_mid = q.loglikelihood; if isfield(q, 'weights'), s.w_mid = q.weights; end
    s.mid = mid;
    q = load(sprintf('%s/hssmc/particles-%d-%d.mat', nm, H, H)); s.p_last = q.particles'; s.lp_last = q.tlogpostkernel;
    s.ll_last = q.loglikelihood;
    s.mdd = oo_.MarginalDensity.hssmc;
  else
    H = options_.posterior_sampler_options.current_options.H;
    q = load(sprintf('%s/dsmh/particles-1-%d.mat', nm, H)); s.p_first = q.particles'; s.lp_first = q.tlogpostkernel;
    s.ll_first = q.loglikelihood;
    q = load(sprintf('%s/dsmh/parameters_particles_final.mat', nm)); s.p_last = q.particles'; s.lp_last = q.tlogpostkernel;
    s.ll_last = q.loglikelihood;
    s.mdd = oo_.MarginalDensity.dsmh;
  end
  cd(top);
  save('-mat7-binary', sprintf('smp_case%d.mat', k), '-struct', 's');
  printf('SMP_CASE_DONE %d %.1fs\n', k, s.elapsed);
  clear -global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info
end
disp('SMP_REFS_DONE')
