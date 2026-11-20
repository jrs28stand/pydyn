% Regenerates reference/occ_*.mat: Dynare 7.2 OccBin simulations that exercise the simul.* options.
%
%   octave --no-gui --quiet r_occbin_options.m            (all cases, one child Octave per case)
%   OCC_CASE=borrcon octave --no-gui --quiet r_occbin_options.m
%
% Run it in a scratch directory that holds copies of the occ_*.mod files; it writes occ_<case>.mat
% there. Each case runs in its own process (the occbin solver keeps persistent state), and each
% variant calls occbin.solver directly with options_.occbin.simul.* changed (not every option can be
% set from the .mod file). Octave's "datatypes" package and a graphics toolkit are not needed.
1;

function M_ = reset_state(M_)
  % occbin.solver caches the decision rules and the regime dictionaries (persistent variables) and
  % only refreshes them when M_.params changes: alter a parameter that the model does not use
  persistent cnt
  if isempty(cnt), cnt = 0; end
  cnt = cnt + 1;
  ip = find(strcmp(strtrim(cellstr(M_.param_names)), 'DUMMY'));
  M_.params(ip) = 1 + cnt * 1e-3;
end

function out = runsim(M_, options_, oo_, ov)
  % one occbin.solver call with the fields of struct ov written into options_.occbin.simul
  options_.noprint = 1;
  options_.occbin.simul.waitbar = false;
  M_ = reset_state(M_);
  f = fieldnames(ov);
  for i = 1:numel(f)
    options_.occbin.simul.(f{i}) = ov.(f{i});
  end
  try
    [~, out] = occbin.solver(M_, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  catch e
    out = struct('error_flag', -1, 'msg', e.message);
    disp(['  runsim exception: ' e.message]);
  end
end

function r = store(r, name, out)
  % flat fields name_flag, name_piecewise, name_linear, name_regime (cell: fields x periods), name_shocks
  r.([name '_flag']) = out.error_flag;
  if out.error_flag == 0
    r.([name '_piecewise']) = out.piecewise;
    r.([name '_linear']) = out.linear;
    r.([name '_shocks']) = out.shocks_sequence;
    r.([name '_exo_pos']) = out.exo_pos;
    r.([name '_regime']) = struct2cell(out.regime_history(:));
  end
  printf('  %-28s error_flag %d\n', name, out.error_flag);
end

function r = sweep(r, name, M_, options_, oo_, ov, maxits)
  % error flags for a range of maxit; path at the smallest and largest maxit that succeed
  flags = zeros(1, numel(maxits));
  for i = 1:numel(maxits)
    ov.maxit = maxits(i);
    out = runsim(M_, options_, oo_, ov);
    flags(i) = out.error_flag;
    if out.error_flag == 0 && (isempty(strfind(strjoin(fieldnames(r)', ' '), [name '_first_piecewise'])))
      r.([name '_first_piecewise']) = out.piecewise;
      r.([name '_first_maxit']) = maxits(i);
    end
    last = out;
  end
  r.([name '_flags']) = flags;
  r.([name '_maxits']) = maxits;
  if last.error_flag == 0
    r.([name '_last_piecewise']) = last.piecewise;
    r.([name '_last_regime']) = struct2cell(last.regime_history(:));
  end
  printf('  %-28s flags %s\n', name, mat2str(flags));
end

function r = store_irf(r, name, irfs, M_, signs)
  % irfs.piecewise.<endo>_<exo>_<sign>: store as name_<exo>_<sign>_piecewise (periods x endo)
  for ix = 1:M_.exo_nbr
    for is = 1:numel(signs)
      for kind = {'piecewise', 'linear'}
        mat = [];
        for j = 1:M_.endo_nbr
          fld = [strtrim(M_.endo_names{j}) '_' strtrim(M_.exo_names{ix}) '_' signs{is}];
          mat = [mat irfs.(kind{1}).(fld)];
        end
        r.([name '_' strtrim(M_.exo_names{ix}) '_' signs{is} '_' kind{1}]) = mat;
      end
    end
  end
  printf('  %-28s stored irfs\n', name);
end

function r = base_info(M_, oo_)
  r = struct('ys', oo_.steady_state, 'params', M_.params, 'endo_names', strjoin(cellstr(M_.endo_names)', ','));
end

caseid = getenv('OCC_CASE');
dyn = getenv('DYNARE_MATLAB');
if isempty(dyn), dyn = '/home/claude/dl/src72/matlab'; end
addpath(dyn);
all_cases = {'borrcon', 'borrcon_opts', 'rbc_cyc', 'rbc_err', 'rbc1'};

if isempty(caseid)
  for i = 1:numel(all_cases)
    system(sprintf('OCC_CASE=%s GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_occbin_options.m > log_%s 2>&1', all_cases{i}, all_cases{i}));
    printf('case %s done\n', all_cases{i});
  end
  return
end

% ======================================================================================== borrcon
if strcmp(caseid, 'borrcon')
  dynare occ_borrcon nograph
  r = base_info(M_, oo_);
  S = zeros(30, 2);               % columns: u, v
  S(1,1) = 0.12; S(7,1) = 0.2; S(15,2) = -0.03; S(19,1) = 0.15; S(19,2) = 0.05; S(26,1) = -0.08;
  r.S = S;
  r.init = [0.02 -0.01 0.03 0.05 -0.04 0.02 -0.01 0.03]';       % endo_init: deviations, declaration order
  base = struct('SHOCKS', S, 'exo_pos', [1 2], 'periods', 80);

  out = runsim(M_, options_, oo_, base);           r = store(r, 'default', out);
  r = sweep(r, 'sw_default', M_, options_, oo_, base, 1:18);
  ov = base; ov.curb_retrench = true;               out = runsim(M_, options_, oo_, ov); r = store(r, 'curb', out);
  r = sweep(r, 'sw_curb', M_, options_, oo_, ov, 1:18);

  % endo_init
  ov = base; ov.endo_init = r.init;                 out = runsim(M_, options_, oo_, ov); r = store(r, 'endo_init', out);
  ov.SHOCKS = zeros(1, 2); ov.exo_pos = [1 2];      out = runsim(M_, options_, oo_, ov); r = store(r, 'endo_init_noshock', out);

  % exo_pos
  ov = base; ov.SHOCKS = S(:,2); ov.exo_pos = 2;    out = runsim(M_, options_, oo_, ov); r = store(r, 'exo_v_only', out);
  ov = base; ov.SHOCKS = S(:,[2 1]); ov.exo_pos = [2 1]; out = runsim(M_, options_, oo_, ov); r = store(r, 'exo_swapped', out);

  % max_check_ahead_periods
  ov = base; ov.max_check_ahead_periods = 8; ov.check_ahead_periods = 8;  out = runsim(M_, options_, oo_, ov); r = store(r, 'maxcheck8', out);
  r = sweep(r, 'sw_maxcheck8', M_, options_, oo_, ov, 1:12);
  ov = base; ov.max_check_ahead_periods = 20; ov.check_ahead_periods = 10;  out = runsim(M_, options_, oo_, ov); r = store(r, 'maxcheck20_check10', out);
  r = sweep(r, 'sw_maxcheck20_check10', M_, options_, oo_, ov, 1:25);
  ov = base; ov.max_check_ahead_periods = 8; ov.check_ahead_periods = 5;    out = runsim(M_, options_, oo_, ov); r = store(r, 'maxcheck8_check5', out);
  ov = base; ov.check_ahead_periods = 12;           out = runsim(M_, options_, oo_, ov); r = store(r, 'check12', out);
  r = sweep(r, 'sw_check12', M_, options_, oo_, ov, 1:25);

  % reset_regime_in_new_period / reset_check_ahead_periods_in_new_period
  ov = base; ov.reset_regime_in_new_period = true;  out = runsim(M_, options_, oo_, ov); r = store(r, 'reset_regime', out);
  r = sweep(r, 'sw_reset_regime', M_, options_, oo_, ov, 1:25);
  ov = base; ov.check_ahead_periods = 12; ov.reset_regime_in_new_period = true; ov.reset_check_ahead_periods_in_new_period = true;
  out = runsim(M_, options_, oo_, ov);              r = store(r, 'reset_both_check12', out);
  r = sweep(r, 'sw_reset_both_check12', M_, options_, oo_, ov, 1:25);
  ov = base; ov.check_ahead_periods = 12; ov.reset_check_ahead_periods_in_new_period = true;   % no effect without reset_regime
  out = runsim(M_, options_, oo_, ov);              r = store(r, 'reset_check_only_check12', out);

  % init_regime: the converged regime history as guess; a truncated guess; maxit too small without a guess
  def = runsim(M_, options_, oo_, base);
  rh = def.regime_history;
  ov = base; ov.maxit = 3;                          out = runsim(M_, options_, oo_, ov); r = store(r, 'maxit3', out);
  ov.init_regime = rh;                              out = runsim(M_, options_, oo_, ov); r = store(r, 'maxit3_guess_full', out);
  rh2 = rh;                                         % guess of half the length for every period
  for k = 1:numel(rh2)
    if numel(rh2(k).regime) > 1, rh2(k).regimestart(2) = ceil(rh2(k).regimestart(2) / 2); end
  end
  ov = base; ov.maxit = 8; ov.init_regime = rh2;    out = runsim(M_, options_, oo_, ov); r = store(r, 'maxit8_guess_half', out);
  ov = base; ov.maxit = 8;                          out = runsim(M_, options_, oo_, ov); r = store(r, 'maxit8', out);
  r = sweep(r, 'sw_guess_half', M_, options_, oo_, setfield(base, 'init_regime', rh2), 1:18);
  rh3 = rh(1:2);                                    % guess only for the first two shock periods
  r = sweep(r, 'sw_guess_two', M_, options_, oo_, setfield(base, 'init_regime', rh3), 1:18);
  % guess whose last regime is binding -> error 312
  g = rh; g(1).regime = [0 1]; g(1).regimestart = [1 5];
  ov = base; ov.init_regime = g;                    out = runsim(M_, options_, oo_, ov); r = store(r, 'guess_binding_end', out);
  % guess starting beyond the check-ahead rows (grows the indicator)
  g = rh; g(1).regime = [1 0]; g(1).regimestart = [1 300];
  ov = base; ov.init_regime = g;                    out = runsim(M_, options_, oo_, ov); r = store(r, 'guess_long', out);

  % init_binding_indicator: guess for the first shock period only
  bi = false(40, 1); bi(1:12) = true;
  ov = base; ov.maxit = 3; ov.init_binding_indicator = bi;   out = runsim(M_, options_, oo_, ov); r = store(r, 'maxit3_bi12', out);
  r = sweep(r, 'sw_bi12', M_, options_, oo_, setfield(base, 'init_binding_indicator', bi), 1:18);
  r.bi12 = false(40, 1); r.bi12(1:12) = true; r.bi12 = double(r.bi12);

  % (check_ahead_periods > max_check_ahead_periods: occbin.solver intends to clamp, but its message refers to
  % an undefined variable 'simul' and so fails in Dynare 7.2; not used here)
  % algo_truncation: maxit = 1 accepts the first guess
  ov = base; ov.maxit = 1;                          out = runsim(M_, options_, oo_, ov); r = store(r, 'maxit1', out);

  % IRFs (Dynare's occbin.irf): all shocks, both signs, default size sqrt(diag(Sigma_e)); exo_pos is reset
  % to all shocks because setup.m restricts it to the shocked variables of the surprise-shock block
  options_.occbin.simul.exo_pos = 1:M_.exo_nbr;
  options_.occbin.simul.SHOCKS = zeros(1, M_.exo_nbr);
  options_.occbin.simul.waitbar = false;
  options_.noprint = 1;
  options_.occbin.simul.waitbar = false;
  M_ = reset_state(M_);
  irfs = occbin.irf(M_, oo_, options_);
  r = store_irf(r, 'irf_default', irfs, M_, {'pos','neg'});
  % with the exo_pos left as set by setup.m (1 = u only) Dynare fails for a two-shock model
  options_.occbin.simul.exo_pos = 1;
  M_ = reset_state(M_);
  try
    occbin.irf(M_, oo_, options_);
    r.irf_stale_exo_pos_error = 0;
  catch e
    disp(['  irf with stale exo_pos: ' e.message]);
    r.irf_stale_exo_pos_error = 1;
  end
  options_.occbin.simul.exo_pos = 1:M_.exo_nbr;
  options_.occbin.irf.shocksize = [0.3 0.1];
  options_.occbin.irf.check_ahead_periods = 12;
  options_.occbin.irf.maxit = 25;
  options_.irf = 30;
  M_ = reset_state(M_);
  irfs = occbin.irf(M_, oo_, options_);
  r = store_irf(r, 'irf_big', irfs, M_, {'pos','neg'});
  save('-mat7-binary', 'occ_borrcon.mat', '-struct', 'r');
end

% ================================================================================== borrcon, options in the .mod
if strcmp(caseid, 'borrcon_opts')
  dynare occ_borrcon_opts nograph
  r = base_info(M_, oo_);
  s = oo_.occbin.simul;
  out = s; r = store(r, 'mod', out);
  r.mod_opts = [options_.occbin.simul.maxit options_.occbin.simul.curb_retrench options_.occbin.simul.check_ahead_periods ...
                options_.occbin.simul.max_check_ahead_periods options_.occbin.simul.reset_regime_in_new_period ...
                options_.occbin.simul.reset_check_ahead_periods_in_new_period options_.occbin.simul.periods];
  save('-mat7-binary', 'occ_borrcon_opts.mat', '-struct', 'r');
end

% ================================================================================== rbc, cycling guesses
if strcmp(caseid, 'rbc_cyc')
  dynare occ_rbc_cyc nograph
  r = base_info(M_, oo_);
  r = store(r, 'mod', oo_.occbin.simul);
  r.mod_opts = [options_.occbin.simul.maxit options_.occbin.simul.periodic_solution options_.occbin.simul.periodic_solution_threshold ...
                options_.occbin.simul.periodic_solution_strict options_.occbin.simul.periods];
  S = oo_.occbin.simul.shocks_sequence; r.S = S;
  base = struct('SHOCKS', S, 'exo_pos', 1, 'periods', 80, 'periodic_solution', false, 'periodic_solution_threshold', 1, ...
                'periodic_solution_strict', true, 'maxit', 30, 'curb_retrench', false);

  % three parameter sets (psi, phi, rho) and shock paths for which the guesses cycle (found by random search)
  pset = {
    'c1', [10.30721957177062 0.9879452800064452 0.7469193213904868], [0.17885813149410787 -0.18518412993264557 0.09168812685568793]'
    'c3', [4.6525347234318675 0.950930792986048 0.5850521699901848], [0.03831108818687395 0.08817287153074853 -0.0932996254397529]'
    'c4', [13.436741484971341 0.9608110490106523 0.5495106530269722], [0.1246837534942502 0.037652096093670165 -0.3040040918655403]'
    'e1', [4.038590202884413 0.9538230658935829 0.5028849753522318], [0.278665132869575 -0.4631084100053847 -0.09791875946873171]'};
  variants = {
    'default',        struct();
    'per',            struct('periodic_solution', true);
    'per_loose',      struct('periodic_solution', true, 'periodic_solution_strict', false);
    'per_loose_thr0', struct('periodic_solution', true, 'periodic_solution_strict', false, 'periodic_solution_threshold', 0);
    'per_loose_thr3', struct('periodic_solution', true, 'periodic_solution_strict', false, 'periodic_solution_threshold', 3);
    'loose_only',     struct('periodic_solution_strict', false);
    'per_thr0',       struct('periodic_solution', true, 'periodic_solution_threshold', 0);
    'curb',           struct('curb_retrench', true);
    'curb_per',       struct('curb_retrench', true, 'periodic_solution', true);
    'curb_per_loose', struct('curb_retrench', true, 'periodic_solution', true, 'periodic_solution_strict', false);
    'maxit3',         struct('maxit', 3);
    'maxit2_per',     struct('maxit', 2, 'periodic_solution', true);
    'maxit1',         struct('maxit', 1)};
  names = strtrim(cellstr(M_.param_names));
  M0 = M_;
  for ip = 1:size(pset, 1)
    M_ = M0;
    M_.params(strcmp(names, 'psi')) = pset{ip, 2}(1);
    M_.params(strcmp(names, 'phi')) = pset{ip, 2}(2);
    M_.params(strcmp(names, 'rho')) = pset{ip, 2}(3);
    Sp = pset{ip, 3};
    r.([pset{ip, 1} '_S']) = Sp;
    for iv = 1:size(variants, 1)
      ov = base; ov.SHOCKS = Sp;
      f = fieldnames(variants{iv, 2});
      for j = 1:numel(f), ov.(f{j}) = variants{iv, 2}.(f{j}); end
      out = runsim(M_, options_, oo_, ov);
      r = store(r, [pset{ip, 1} '_' variants{iv, 1}], out);
    end
    ov = base; ov.SHOCKS = Sp; ov.periodic_solution = true; ov.periodic_solution_strict = false;
    r = sweep(r, [pset{ip, 1} '_sw_per_loose'], M_, options_, oo_, ov, 1:8);
    ov.periodic_solution = false;
    r = sweep(r, [pset{ip, 1} '_sw_loose_only'], M_, options_, oo_, ov, 1:8);
  end
  % IRF at a size that makes the constraints bind (negative shock hits the investment floor), with the
  % parameters of Dynare's rbc_occbin example
  M_ = M0;
  M_.params(strcmp(names, 'psi')) = 5;
  M_.params(strcmp(names, 'phi')) = 0.975;
  M_.params(strcmp(names, 'rho')) = 0.9;
  options_.occbin.simul.exo_pos = 1;
  options_.occbin.simul.waitbar = false;
  options_.noprint = 1;
  options_.occbin.irf.shocksize = 0.04;
  options_.occbin.irf.check_ahead_periods = 200;   % the INEG constraint binds for ~100 periods
  options_.irf = 40;
  M_ = reset_state(M_);
  irfs = occbin.irf(M_, oo_, options_);
  r = store_irf(r, 'irf', irfs, M_, {'pos','neg'});
  options_.occbin.irf.shocksize = [];             % default: stderr 0.015
  M_ = reset_state(M_);
  irfs = occbin.irf(M_, oo_, options_);
  r = store_irf(r, 'irf_dflt', irfs, M_, {'pos','neg'});
  save('-mat7-binary', 'occ_rbc_cyc.mat', '-struct', 'r');
end

% ================================================================================== rbc, error_bind / error_relax
if strcmp(caseid, 'rbc_err')
  dynare occ_rbc_err nograph
  r = base_info(M_, oo_);
  r = store(r, 'mod', oo_.occbin.simul);          % periodic_solution on, custom errors
  S = oo_.occbin.simul.shocks_sequence; r.S = S;
  base = struct('SHOCKS', S, 'exo_pos', 1, 'periods', 80, 'periodic_solution', false, 'periodic_solution_threshold', 1, ...
                'periodic_solution_strict', true, 'maxit', 30, 'curb_retrench', false);
  variants = {'default', struct(); 'per', struct('periodic_solution', true); ...
              'per_loose', struct('periodic_solution', true, 'periodic_solution_strict', false); ...
              'per_thr0', struct('periodic_solution', true, 'periodic_solution_threshold', 0)};
  for iv = 1:size(variants, 1)
    ov = base; f = fieldnames(variants{iv, 2});
    for j = 1:numel(f), ov.(f{j}) = variants{iv, 2}.(f{j}); end
    out = runsim(M_, options_, oo_, ov);
    r = store(r, variants{iv, 1}, out);
  end
  save('-mat7-binary', 'occ_rbc_err.mat', '-struct', 'r');
end

% ================================================================================== one-constraint cycling guesses
if strcmp(caseid, 'rbc1')
  dynare occ_rbc1 nograph
  r = base_info(M_, oo_);
  r = store(r, 'mod', oo_.occbin.simul);
  base = struct('SHOCKS', oo_.occbin.simul.shocks_sequence, 'exo_pos', 1, 'periods', 80, 'periodic_solution', false, 'periodic_solution_threshold', 1, ...
                'periodic_solution_strict', true, 'maxit', 30, 'curb_retrench', false);
  pset = {
    'a', [8.20453071654218 0.43376653874397436 4.162224515697595], [-0.006401074156627856 0.21600250881228591 -0.1255342530145265]',
    'b', [29.06953055729189 0.3407807947098334 0.7619883020628548], [-0.06332487508731809 -0.06344217665694422]',
    'c', [1.5460870251082188 0.4426697608503545 1.5078256168794189], [0.060713476541191136]'};
  variants = {
    'default',        struct();
    'per',            struct('periodic_solution', true);
    'per_loose',      struct('periodic_solution', true, 'periodic_solution_strict', false);
    'per_loose_thr0', struct('periodic_solution', true, 'periodic_solution_strict', false, 'periodic_solution_threshold', 0);
    'loose_only',     struct('periodic_solution_strict', false);
    'curb',           struct('curb_retrench', true);
    'curb_per',       struct('curb_retrench', true, 'periodic_solution', true);
    'maxit3',         struct('maxit', 3)};
  names = strtrim(cellstr(M_.param_names));
  M0 = M_;
  for ip = 1:size(pset, 1)
    M_ = M0;
    M_.params(strcmp(names, 'psi')) = pset{ip, 2}(1);
    M_.params(strcmp(names, 'rho')) = pset{ip, 2}(2);
    M_.params(strcmp(names, 'sigma')) = pset{ip, 2}(3);
    Sp = pset{ip, 3};
    for iv = 1:size(variants, 1)
      ov = base; ov.SHOCKS = Sp;
      f = fieldnames(variants{iv, 2});
      for j = 1:numel(f), ov.(f{j}) = variants{iv, 2}.(f{j}); end
      out = runsim(M_, options_, oo_, ov);
      r = store(r, [pset{ip, 1} '_' variants{iv, 1}], out);
    end
  end
  save('-mat7-binary', 'occ_rbc1.mat', '-struct', 'r');
end
