% Regenerates reference/nl_rbc_*.mat: Dynare 7.2 likelihoods of the second-order approximated RBC model
% of tests/particle (nl_rbc.mod, data nl_rbc_data.m) with every nonlinear filter of
% matlab/nonlinear-filters (non_linear_dsge_likelihood), for tests/test_nonlinear_estimation.py.
%
%   NL_PART=lik GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_nonlinear_filters.m
%   NL_PART=seeds ...   (SIS likelihood at the initial values for seeds 1..30: distribution check)
%   NL_PART=timing ...  (timings with many particles)
%   NL_PART=simplex ... (Dynare's simplex, mode_compute=8, through dynare_minimize_objective, from the
%                        initial values: gf (full run and 300 calls) and sis with 200 particles (300 calls))
%   NL_PART=smooth ...  (inputs and outputs of multivariate_smooth_resampling in the first periods of
%                        sis with resampling_method=smooth, 200 particles, on the reduced form saved by
%                        NL_PART=lik; needs nl_rbc_lik.mat in the directory)
%
% Run it in a scratch directory holding copies of nl_rbc.mod and nl_rbc_data.m. The script writes an
% instrumented copy of matlab/estimation/non_linear_dsge_likelihood.m into the scratch directory (it
% shadows Dynare's) that keeps the per-period likelihood and the reduced form in a global variable;
% for NL_PART=seeds also a copy of set_dynare_seed_local_options.m where the 'default' seed is
% replaced by the global NL_SEED. Nothing else is changed.
1;

function patch_likelihood()
  src = fileread('/home/claude/dl/src72/matlab/estimation/non_linear_dsge_likelihood.m');
  key = 'LIK = feval(options_.particle.algorithm, ReducedForm, Y, start, options_.particle, options_.threads, options_, M_);';
  k = strfind(src, key);
  if numel(k) ~= 1, error('patch failed'); end
  new = ['[LIK, lik_] = feval(options_.particle.algorithm, ReducedForm, Y, start, options_.particle, options_.threads, options_, M_);' ...
         sprintf('\nglobal NL_OUT\nNL_OUT = struct(''lik'', lik_, ''ReducedForm'', ReducedForm);\n')];
  src = [src(1:k-1) new src(k+numel(key):end)];
  fid = fopen('non_linear_dsge_likelihood.m', 'w'); fwrite(fid, src); fclose(fid);
  rehash;
end

function patch_seed()
  src = fileread('/home/claude/dl/src72/matlab/set_dynare_seed_local_options.m');
  key = 'DynareRandomStreams.seed = 0;';
  k = strfind(src, key);
  % second occurrence: the Octave ("old MATLAB interface") branch
  new = sprintf('global NL_SEED\n            DynareRandomStreams.seed = NL_SEED;');
  src = [src(1:k(2)-1) new src(k(2)+numel(key):end)];
  fid = fopen('set_dynare_seed_local_options.m', 'w'); fwrite(fid, src); fclose(fid);
  rehash;
end

function patch_smooth()
  src = fileread('/home/claude/dl/src72/matlab/nonlinear-filters/multivariate_smooth_resampling.m');
  key = 'new_particles = new_particles*(vectors'') ;';
  k = strfind(src, key);
  if numel(k) ~= 1, error('patch failed'); end
  ins = sprintf(['\nglobal NL_SMOOTH\nA_ = particles''*(bsxfun(@times,1/number_of_particles,particles));\n' ...
                 'NL_SMOOTH{end+1} = struct(''P'', particles, ''w'', weights, ''out'', new_particles, ''sym'', issymmetric(A_));\n']);
  src = [src(1:k+numel(key)-1) ins src(k+numel(key):end)];
  fid = fopen('multivariate_smooth_resampling.m', 'w'); fwrite(fid, src); fclose(fid);
  rehash;
end

function p = set_flag(p, group, name)
  % one of the mutually exclusive flags of p.(group) (as the preprocessor writes them)
  f = fieldnames(p.(group));
  for j = 1:numel(f)
    if islogical(p.(group).(f{j})) || isnumeric(p.(group).(f{j}))
      p.(group).(f{j}) = false;
    end
  end
  p.(group).(name) = true;
end

function p = apply(p, spec)
  for j = 1:2:numel(spec)
    key = spec{j}; val = spec{j+1};
    switch key
      case 'proposal', p = set_flag(p, 'proposal_approximation', val);
      case 'distribution', p = set_flag(p, 'distribution_approximation', val);
      case 'resampling', p.resampling.status = struct('systematic', false, 'none', false, 'generic', false); p.resampling.status.(val) = true;
      case 'method', p.resampling.method = struct('kitagawa', false, 'smooth', false, 'stratified', false); p.resampling.method.(val) = true;
      case 'cpf_weights', p.cpf_weights_method = struct('amisanotristani', false, 'murrayjonesparslow', false); p.cpf_weights_method.(val) = true;
      otherwise
        parts = strsplit(key, '.');
        p = setfield(p, parts{:}, val);
    end
  end
end

addpath /home/claude/dl/src72/matlab
part = getenv('NL_PART');
if isempty(part), part = 'lik'; end
patch_likelihood();
dynare nl_rbc nograph
global NL_OUT NL_SEED   % after dynare: the driver clears the global variables
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
x0 = xparam1(:);
n = numel(x0);
rel = [0 0.002 -0.002];
X = zeros(n, numel(rel));
for k = 1:numel(rel)
  X(:, k) = x0 .* (1 + rel(k) * sin(k * (1:n)'));
end

% name, filter_algorithm, number_of_particles, other options (applied to options_.particle)
cases = { ...
  'gf_uu', 'gf', 1000, {}; ...
  'gf_cu', 'gf', 1000, {'proposal', 'cubature'}; ...
  'gf_uc', 'gf', 1000, {'distribution', 'cubature'}; ...
  'gf_cc', 'gf', 1000, {'proposal', 'cubature', 'distribution', 'cubature'}; ...
  'gf_uu_abk', 'gf', 1000, {'unscented.alpha', 0.8, 'unscented.beta', 1, 'unscented.kappa', 2}; ...
  'nlkf_u', 'nlkf', 1000, {}; ...
  'nlkf_c', 'nlkf', 1000, {'proposal', 'cubature'}; ...
  'nlkf_u_abk', 'nlkf', 1000, {'unscented.alpha', 0.8, 'unscented.beta', 1, 'unscented.kappa', 2}; ...
  'gf_init3', 'gf', 1000, {'initialization', 3, 'initial_state_prior_std', 0.5}; ...
  'nlkf_m', 'nlkf', 500, {'proposal', 'montecarlo'}; ...
  'gf_mu', 'gf', 500, {'proposal', 'montecarlo'}; ...
  'gf_um', 'gf', 500, {'distribution', 'montecarlo'}; ...
  'gf_mm', 'gf', 500, {'proposal', 'montecarlo', 'distribution', 'montecarlo'}; ...
  'gf_um_generic', 'gf', 500, {'distribution', 'montecarlo', 'resampling', 'generic'}; ...
  'gf_um_none', 'gf', 500, {'distribution', 'montecarlo', 'resampling', 'none'}; ...
  'sis', 'sis', 1000, {}; ...
  'sis_strat', 'sis', 1000, {'method', 'stratified'}; ...
  'sis_smooth', 'sis', 1000, {'method', 'smooth'}; ...
  'sis_generic', 'sis', 1000, {'resampling', 'generic'}; ...
  'sis_generic_t', 'sis', 1000, {'resampling', 'generic', 'resampling.threshold', 19}; ...
  'sis_generic_low', 'sis', 1000, {'resampling', 'generic', 'resampling.threshold', 0.001}; ...
  'sis_none', 'sis', 1000, {'resampling', 'none'}; ...
  'sis_pruning', 'sis', 1000, {'pruning', true}; ...
  'sis_init3', 'sis', 1000, {'initialization', 3}; ...
  'apf', 'apf', 1000, {}; ...
  'apf_none', 'apf', 1000, {'resampling', 'none'}; ...
  'apf_strat', 'apf', 1000, {'method', 'stratified'}; ...
  'apf_generic', 'apf', 1000, {'resampling', 'generic', 'resampling.threshold', 19}; ...
  'apf_generic_low', 'apf', 1000, {'resampling', 'generic', 'resampling.threshold', 0.001}; ...
  'apf_pruning', 'apf', 1000, {'pruning', true}; ...
  'cpf_u', 'cpf', 100, {}; ...
  'cpf_c', 'cpf', 100, {'proposal', 'cubature'}; ...
  'cpf_m', 'cpf', 100, {'proposal', 'montecarlo'}; ...
  'cpf_u_mjp', 'cpf', 100, {'cpf_weights', 'murrayjonesparslow'}; ...
  'cpf_u_none', 'cpf', 100, {'resampling', 'none'}; ...
  };

if strcmp(part, 'lik')
  r = struct();
  r.names = strjoin(bayestopt_.name', ';');
  r.X = X; r.data = dataset_.data;
  r.order_var = oo_.dr.order_var; r.restrict_var_list = oo_.dr.restrict_var_list;
  r.mf0 = bayestopt_.mf0; r.mf1 = bayestopt_.mf1;
  r.case_names = strjoin(cases(:, 1)', ';');
  for c = 1:rows(cases)
    o = options_;
    o.particle.filter_algorithm = cases{c, 2};
    o.particle.number_of_particles = cases{c, 3};
    o.particle = apply(o.particle, cases{c, 4});
    o.particle = check_particle_filter_options(o.particle);
    fv = NaN(1, columns(X)); inf_ = zeros(1, columns(X)); tm = zeros(1, columns(X));
    L = NaN(rows(dataset_.data), columns(X));
    for k = 1:columns(X)
      rand('state', 0); randn('state', 0);   % proposal_approximation=montecarlo in gf does not reset the seed
      NL_OUT = [];
      t0 = time();
      try
        [fval, info] = non_linear_dsge_likelihood(X(:,k), dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, ...
                                                  oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
      catch err
        printf('  %s point %d: ERROR %s\n', cases{c, 1}, k, err.message);
        fval = NaN; info = -1;
        r.([cases{c, 1} '_error']) = err.message;
      end
      tm(k) = time() - t0;
      fv(k) = fval; inf_(k) = info(1);
      if ~isempty(NL_OUT), L(:, k) = NL_OUT.lik; end
      if c == 1 && k == 1
        rf = NL_OUT.ReducedForm;
        rf = rmfield(rf, intersect(fieldnames(rf), {'dr', 'udr'}));
        r.rf = rf;
      end
      printf('  %s point %d: fval %.12g info %d (%.2f s)\n', cases{c, 1}, k, fval, info(1), tm(k));
    end
    r.([cases{c, 1} '_fval']) = fv; r.([cases{c, 1} '_info']) = inf_;
    r.([cases{c, 1} '_lik']) = L; r.([cases{c, 1} '_time']) = tm;
  end
  r.lnprior = zeros(1, columns(X));
  for k = 1:columns(X)
    r.lnprior(k) = priordens(X(:,k), bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
  end
  save('-mat7-binary', 'nl_rbc_lik.mat', '-struct', 'r');
  printf('saved\n');
elseif strcmp(part, 'seeds')
  patch_seed();
  r = struct();
  seeds = 1:30;
  for c = {'sis', 'apf'}
    o = options_;
    o.particle.filter_algorithm = c{1};
    o.particle.number_of_particles = 1000;
    o.particle = check_particle_filter_options(o.particle);
    fv = NaN(size(seeds));
    for s = 1:numel(seeds)
      NL_SEED = seeds(s);
      fv(s) = non_linear_dsge_likelihood(x0, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, ...
                                         oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    end
    printf('%s: mean %.6f std %.6f\n', c{1}, mean(fv), std(fv));
    r.([c{1} '_fval']) = fv;
  end
  r.seeds = seeds; r.x0 = x0;
  r.lnprior = priordens(x0, bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
  save('-mat7-binary', 'nl_rbc_seeds.mat', '-struct', 'r');
elseif strcmp(part, 'simplex')
  r = struct('x0', x0);
  runs = {'gf_full', 'gf', 1000, ''; 'gf_300', 'gf', 1000, '''MaxFunEvals'',300'; 'sis_300', 'sis', 200, '''MaxFunEvals'',300'};
  for c = 1:rows(runs)
    o = options_;
    o.particle.filter_algorithm = runs{c, 2};
    o.particle.number_of_particles = runs{c, 3};
    o.particle = check_particle_filter_options(o.particle);
    o.optim_opt = runs{c, 4};
    o.silent_optimizer = true;
    t0 = time();
    [xopt, fval, exitflag, ~, ~, ~, info_] = dynare_minimize_objective(@non_linear_dsge_likelihood, x0, 8, o, ...
        [bounds.lb bounds.ub], bayestopt_.name, bayestopt_, [], dataset_, dataset_info, o, M_, estim_params_, ...
        bayestopt_, bounds, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    printf('%s: fval %.10g, %d calls, %d iterations, %.1f s\n', runs{c, 1}, fval, info_.funcCount, info_.iterations, time() - t0);
    r.([runs{c, 1} '_x']) = xopt; r.([runs{c, 1} '_fval']) = fval; r.([runs{c, 1} '_exitflag']) = exitflag;
    r.([runs{c, 1} '_funcCount']) = info_.funcCount; r.([runs{c, 1} '_iterations']) = info_.iterations;
  end
  save('-mat7-binary', 'nl_rbc_simplex.mat', '-struct', 'r');
elseif strcmp(part, 'smooth')
  patch_smooth();
  global NL_SMOOTH
  NL_SMOOTH = {};
  r = load('nl_rbc_lik.mat');
  rf = r.rf; rf.use_k_order_solver = false;
  o = options_; o.particle.number_of_particles = 200;
  o.particle = apply(o.particle, {'method', 'smooth'});
  [LIK, lik] = sequential_importance_particle_filter(rf, r.data', 1, o.particle, o.threads, o, M_);
  s = struct('LIK', LIK, 'lik', lik);
  for k = 1:6
    s.(sprintf('P_%d', k)) = NL_SMOOTH{k}.P; s.(sprintf('w_%d', k)) = NL_SMOOTH{k}.w;
    s.(sprintf('out_%d', k)) = NL_SMOOTH{k}.out; s.(sprintf('sym_%d', k)) = NL_SMOOTH{k}.sym;
  end
  save('-mat7-binary', 'nl_rbc_smooth.mat', '-struct', 's');
elseif strcmp(part, 'timing')
  for c = {{'sis', 20000}, {'apf', 20000}, {'gf', 1}, {'nlkf', 1}, {'cpf', 1000}}
    o = options_;
    o.particle.filter_algorithm = c{1}{1};
    o.particle.number_of_particles = c{1}{2};
    o.particle = check_particle_filter_options(o.particle);
    t0 = time();
    fval = non_linear_dsge_likelihood(x0, dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, ...
                                      oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    printf('TIMING %s N=%d: %.3f s (fval %.10g)\n', c{1}{1}, c{1}{2}, time() - t0, fval);
  end
end
