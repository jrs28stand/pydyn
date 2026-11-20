% Regenerates the reference/sd_*.mat files used by tests/test_shock_decomposition.py and
% tests/test_plotting.py. Run inside a scratch copy of reference/ (Dynare writes many files), then
% copy the sd_*.mat files back:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_shock_decomposition.m
% Dynare 7.2, Octave 8.4. Each case is one run of sd_nk.mod / sd_ur.mod with the macro variable CASE.
%
% Octave has no graphics toolkit here, so Dynare's figure routines are run against recording
% stubs (written to ./sd_stubs and put on the path only while a figure routine runs): every
% fill/bar/plot/text call is stored, which gives exactly the numbers Dynare draws.
addpath /home/claude/dl/src72/matlab
global SD_REC

% ---------------------------------------------------------------- recording stubs
function sd_write_stubs()
  d = fullfile(pwd, 'sd_stubs');
  if ~exist(d, 'dir'), mkdir(d); end
  rec = {'fill', 'bar', 'plot', 'text', 'title'};
  for k = 1:numel(rec)
    f = fopen(fullfile(d, [rec{k} '.m']), 'w');
    fprintf(f, 'function h = %s(varargin)\n  global SD_REC\n  h = 1;\n', rec{k});
    fprintf(f, '  SD_REC.calls{end+1} = [{''%s'', SD_REC.fig, SD_REC.panel}, varargin];\nend\n', rec{k});
    fclose(f);
  end
  f = fopen(fullfile(d, 'dyn_figure.m'), 'w');
  fprintf(f, 'function h = dyn_figure(varargin)\n  global SD_REC\n  SD_REC.fig = SD_REC.fig + 1; SD_REC.panel = 0; h = 1;\n');
  fprintf(f, '  SD_REC.names{end+1} = varargin{3};\nend\n');
  fclose(f);
  f = fopen(fullfile(d, 'subplot.m'), 'w');
  fprintf(f, 'function h = subplot(varargin)\n  global SD_REC\n  SD_REC.panel = varargin{3}; SD_REC.grid = [varargin{1} varargin{2}]; h = 1;\nend\n');
  fclose(f);
  f = fopen(fullfile(d, 'get.m'), 'w');
  fprintf(f, 'function v = get(h, p)\n  if strcmpi(p, ''screensize''), v = [1 1 1600 900]; elseif strcmpi(p, ''ytick''), v = -10:1:10; else, v = []; end\nend\n');
  fclose(f);
  f = fopen(fullfile(d, 'axis.m'), 'w');
  fprintf(f, 'function v = axis(varargin)\n  v = [0 1 0 1];\nend\n');
  fclose(f);
  f = fopen(fullfile(d, 'ylim.m'), 'w');
  fprintf(f, 'function v = ylim(varargin)\n  v = [0 1];\nend\n');
  fclose(f);
  noop = {'set', 'axes', 'hold', 'colormap', 'dyn_saveas', 'gca', 'gcf', 'figure', 'legend', 'annotation', 'uicontextmenu', 'uimenu', 'groot'};
  for k = 1:numel(noop)
    f = fopen(fullfile(d, [noop{k} '.m']), 'w');
    fprintf(f, 'function h = %s(varargin)\n  h = 1;\nend\n', noop{k});
    fclose(f);
  end
end

function sd_stubs(on)
  global SD_REC
  d = fullfile(pwd, 'sd_stubs');
  warning('off', 'Octave:shadowed-function');
  if on
    SD_REC = struct('fig', 0, 'panel', 0, 'calls', {{}}, 'names', {{}}, 'grid', [0 0]);
    addpath(d);
  else
    rmpath(d);
  end
  rehash;
end

function g = sd_drawn()
  % turn the recorded calls into arrays: per figure/panel the bar heights (component x period),
  % the line, and the legend / title texts
  global SD_REC
  g = struct('names', strjoin(SD_REC.names, '|'), 'nfig', SD_REC.fig, 'grid', SD_REC.grid);
  nb = zeros(SD_REC.fig, 1);
  for k = 1:numel(SD_REC.calls)
    c = SD_REC.calls{k}; fig = c{2}; pan = c{3};
    key = sprintf('f%d_p%d', fig, pan);
    switch c{1}
      case 'fill'
        xx = c{4}; yy = c{5}; comp = c{6};
        if (abs(xx(1)) < 1e-12 && abs(xx(3) - 0.2) < 1e-12) || isequal(yy(:)', [0 1 1 0])
          continue                                   % legend patch of graph_decomp
        end
        if ~isfield(g, [key '_bars']), g.([key '_bars']) = []; end
        g.([key '_bars'])(end+1, :) = [comp, xx(1), xx(3), yy(1), yy(2)];
      case 'bar'
        if ~isfield(g, [key '_bar']), g.([key '_bar']) = []; g.([key '_barx']) = c{4}; end
        g.([key '_bar']) = [g.([key '_bar']); c{5}'];
      case 'plot'
        if numel(c) >= 7 && ischar(c{7}) && strcmp(c{7}, 'k-')          % plot(ax, x, y, 'k-', ...)
          g.([key '_line_x']) = c{5}; g.([key '_line']) = c{6};
        elseif ischar(c{6}) && strcmp(c{6}, 'k-')                        % plot(x, y, 'k-', ...)
          g.([key '_line_x']) = c{4}; g.([key '_line']) = c{5};
        end
      case 'text'
        if ~isfield(g, [key '_labels']), g.([key '_labels']) = ''; end
        g.([key '_labels']) = [g.([key '_labels']) '|' strtrim(c{6})];
      case 'title'
        g.([key '_title']) = c{4};
    end
  end
end

% ---------------------------------------------------------------- smoother results
function r = sd_collect(oo_, M_, options_, lite)
  if nargin < 4, lite = 0; end
  r = struct();
  r.endo_names = strjoin(cellstr(M_.endo_names)', ',');
  r.exo_names = strjoin(cellstr(M_.exo_names)', ',');
  r.varobs = strjoin(options_.varobs(:)', ',');
  r.ys = oo_.dr.ys; r.params = M_.params; r.Sigma_e = M_.Sigma_e; r.H = M_.H;
  n = M_.endo_nbr; T = length(oo_.SmoothedShocks.(M_.exo_names{1}));
  r.smoothed = zeros(T, n); r.updated = zeros(T, n); r.constant = zeros(T, n); r.init_state = zeros(1, n);
  r.trend = zeros(T, n);
  if isfield(oo_, 'FilteredVariables'), r.filtered = zeros(T, n); end
  for i = 1:n
    nm = M_.endo_names{i};
    r.smoothed(:, i) = oo_.SmoothedVariables.(nm);
    r.updated(:, i) = oo_.UpdatedVariables.(nm);
    r.constant(:, i) = oo_.Smoother.Constant.(nm);
    r.init_state(i) = oo_.Smoother.Init_State.(nm);
    r.trend(:, i) = oo_.Smoother.Trend.(nm);
    if isfield(oo_, 'FilteredVariables'), r.filtered(:, i) = oo_.FilteredVariables.(nm); end
  end
  r.trend_coeffs = oo_.Smoother.TrendCoeffs;
  r.constant = r.constant(1, :);
  if ~any(r.trend(:)), r = rmfield(r, 'trend'); end
  if lite >= 1, r = rmfield(r, {'updated'}); if isfield(r, 'filtered'), r = rmfield(r, 'filtered'); end, end
  r.shocks = zeros(T, M_.exo_nbr);
  for i = 1:M_.exo_nbr, r.shocks(:, i) = oo_.SmoothedShocks.(M_.exo_names{i}); end
  if isfield(oo_, 'SmoothedMeasurementErrors')
    r.meas_err = zeros(T, length(options_.varobs));
    for i = 1:length(options_.varobs), r.meas_err(:, i) = oo_.SmoothedMeasurementErrors.(options_.varobs{i}); end
  end
  r.init_state_vector = oo_.Smoother.Init_State_Vector;
  if isfield(oo_, 'FilteredVariablesKStepAhead')
    r.kstep = oo_.FilteredVariablesKStepAhead;
    r.filter_step_ahead = options_.filter_step_ahead;
  end
end

function r = sd_at(pset, M_, oo_, options_, bayestopt_, estim_params_, with_filtered, lite)
  if nargin < 8, lite = 0; end
  % smoother and shock decomposition at a named parameter set, as Dynare's commands compute them
  options_.parameter_set = pset;
  if with_filtered, options_.filtered_vars = 1; options_.nk = max(options_.nk, 1); end
  [oo1, M1, options1] = evaluate_smoother(pset, M_.endo_names, M_, oo_, options_, bayestopt_, estim_params_);
  r = sd_collect(oo1, M1, options1, lite);
  if lite >= 2, return, end
  options_.no_graph.shock_decomposition = true;
  oo2 = shock_decomposition(M_, oo_, options_, {}, bayestopt_, estim_params_);
  r.shock_decomposition = oo2.shock_decomposition;
  if lite >= 1, return, end
  options_ = set_default_initial_condition_decomposition_options(options_);
  oo3 = initial_condition_decomposition(M_, oo_, options_, 0, bayestopt_, estim_params_);
  T = size(oo3.initval_decomposition, 3);
  r.icd_idx = [1:6 T];
  r.initval_decomposition = oo3.initval_decomposition(:, :, r.icd_idx);
end

function sd_run(nm, cs)
  pause(1.5); rehash;
  evalin('base', sprintf('dynare %s nograph -DCASE=%d', nm, cs));
end

sd_write_stubs();

% ================================================================ sd_nk case 1: Bayesian estimation
try
  sd_run('sd_nk', 1);
  r = struct();
  r.names = strjoin(bayestopt_.name', ';');
  r.theta_mode = get_posterior_parameters('mode', M_, estim_params_, oo_, options_);
  r.theta_mean = get_posterior_parameters('mean', M_, estim_params_, oo_, options_);
  r.theta_median = get_posterior_parameters('median', M_, estim_params_, oo_, options_);
  r.theta_prior_mode = bayestopt_.p5(:); r.theta_prior_mean = bayestopt_.p1(:);
  r.mh_replic = options_.mh_replic; r.mh_drop = options_.mh_drop;
  r.sd_in_mod = oo_.shock_decomposition;                 % shock_decomposition(parameter_set=posterior_mode)
  sets = {'posterior_mode', 'posterior_mean', 'posterior_median', 'prior_mode', 'prior_mean'};
  for k = 1:numel(sets)
    try
      lite = [0 1 2 2 2];
      r.(sets{k}) = sd_at(sets{k}, M_, oo_, options_, bayestopt_, estim_params_, k == 1, lite(k));
    catch e
      % prior_mode: Dynare's mode of a uniform prior is NaN, so this parameter set fails here
      r.(['error_' sets{k}]) = e.message;
      disp(['NOTE ' sets{k} ': ' e.message]);
    end
  end
  r.ys_last_solved = oo_.dr.ys;             % what Dynare's plots use as "steady state" (see p_*_ss)
  save('-mat7-binary', 'sd_nk_case1.mat', '-struct', 'r');
  % ---- what plot_shock_decomposition draws at the posterior mode
  r = struct();
  options_.parameter_set = 'posterior_mode';
  oo_ = shock_decomposition(M_, oo_, options_, {}, bayestopt_, estim_params_);
  base = options_;
  base = set_default_plot_shock_decomposition_options(base);
  base.no_graph.plot_shock_decomposition = false;
  vl = {'y'; 'pi'; 'robs'};
  % (a) numbers only (get_decomp_only), several option sets
  o = base; o.plot_shock_decomp.use_shock_groups = 'grp';
  [r.p_grp, r.p_grp_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  o = base; o.plot_shock_decomp.use_shock_groups = 'partial';
  [r.p_partial, r.p_partial_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  o = base; o.plot_shock_decomp.type = 'yoy'; o.plot_shock_decomp.steadystate = true;
  [r.p_yoy, r.p_yoy_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  o = base; o.plot_shock_decomp.type = 'qoq'; o.plot_shock_decomp.diff = true; o.plot_shock_decomp.flip = true;
  o.plot_shock_decomp.use_shock_groups = 'grp';
  [r.p_diff_flip, r.p_diff_flip_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  o = base; o.plot_shock_decomp.type = 'yoy'; o.plot_shock_decomp.diff = true; o.plot_shock_decomp.flip = true;
  [r.p_yoy_diff_flip, r.p_yoy_diff_flip_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  o = base; o.plot_shock_decomp.init2shocks = 'i2s';
  [r.p_i2s, r.p_i2s_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  o = base; o.plot_shock_decomp.init2shocks = 'i2s'; o.plot_shock_decomp.use_shock_groups = 'grp';
  [r.p_i2s_grp, r.p_i2s_grp_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  try
    o = base; o.plot_shock_decomp.type = 'aoa';
    [r.p_aoa, r.p_aoa_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  catch e
    disp(['NOTE aoa: ' e.message]);
  end
  % (b) the figure routines against the recording stubs
  save('-mat7-binary', 'sd_nk_case1_plot.mat', '-struct', 'r');
  r = struct();
  % (type=yoy cannot be drawn here: with the yearly default dates Dynare stops with "All input
  % arguments must have the same frequency"; its numbers are in p_yoy above)
  drawn = {'grp', 'plain', 'detail', 'ss', 'partial', 'full'};
  vl = {'pi'; 'robs'};
  for k = 1:numel(drawn)
    o = base;
    % periods 11 to 40 only (plot_init_date / plot_end_date), which also keeps the file small
    o.plot_shock_decomp.plot_init_date = dates('11Y'); o.plot_shock_decomp.plot_end_date = dates('40Y');
    switch drawn{k}
      case 'grp', o.plot_shock_decomp.use_shock_groups = 'grp';
      case 'detail', o.plot_shock_decomp.detail_plot = true; o.plot_shock_decomp.max_nrows = 1;
      case 'ss', o.plot_shock_decomp.steadystate = true; o.plot_shock_decomp.flip = true;
      case 'partial', o.plot_shock_decomp.use_shock_groups = 'partial';
      case 'full', o.plot_shock_decomp.plot_init_date = []; o.plot_shock_decomp.plot_end_date = [];
    end
    try
      sd_stubs(true);
      plot_shock_decomposition(M_, oo_, o, vl);
      sd_stubs(false);
      r.(['drawn_' drawn{k}]) = sd_drawn();
    catch e
      sd_stubs(false);
      disp(['NOTE drawn ' drawn{k} ': ' e.message]);
      disp(e.stack(1));
    end
  end
  % (d) initial-condition decomposition figure (diff, steadystate)
  try
    o = set_default_initial_condition_decomposition_options(options_);
    o = set_default_plot_shock_decomposition_options(o);
    o.no_graph.plot_shock_decomposition = false;
    o.initial_condition_decomp.plot_init_date = dates('11Y'); o.initial_condition_decomp.plot_end_date = dates('40Y');
    sd_stubs(true);
    initial_condition_decomposition(M_, oo_, o, {'robs'}, bayestopt_, estim_params_);
    sd_stubs(false);
    r.drawn_icd = sd_drawn();
  catch e
    sd_stubs(false);
    disp(['NOTE icd: ' e.message]);
  end
  save('-mat7-binary', 'sd_nk_case1_drawn.mat', '-struct', 'r');
  disp('OK sd_nk case 1');
catch e
  disp(['FAIL sd_nk case 1: ' e.message]); disp(e.stack(1));
end

% ================================================================ sd_nk case 2: maximum likelihood
try
  sd_run('sd_nk', 2);
  r = struct();
  r.names = strjoin(bayestopt_.name', ';');
  r.theta_mode = get_posterior_parameters('mode', M_, estim_params_, oo_, options_, 'mle_');
  r.sd_in_mod = oo_.shock_decomposition;
  r.mle_mode = sd_at('mle_mode', M_, oo_, options_, bayestopt_, estim_params_, true);
  save('-mat7-binary', 'sd_nk_case2.mat', '-struct', 'r');
  disp('OK sd_nk case 2');
catch e
  disp(['FAIL sd_nk case 2: ' e.message]); disp(e.stack(1));
end

% ================================================================ sd_nk case 3: calibration, realtime
try
  sd_run('sd_nk', 3);
  r = sd_collect(oo_, M_, options_);
  r.shock_decomposition = oo_.shock_decomposition;
  rt = oo_.realtime_shock_decomposition; f = fieldnames(rt);
  for k = 1:numel(f), r.(['rt_' f{k}]) = rt.(f{k}); end
  rt = oo_.conditional_shock_decomposition; f = fieldnames(rt);
  for k = 1:numel(f), r.(['cond_' f{k}]) = rt.(f{k}); end
  rt = oo_.realtime_forecast_shock_decomposition; f = fieldnames(rt);
  for k = 1:numel(f), r.(['fcst_' f{k}]) = rt.(f{k}); end
  rt = oo_.realtime_conditional_shock_decomposition; f = fieldnames(rt);
  for k = 1:numel(f), r.(['rtcond_' f{k}]) = rt.(f{k}); end
  % realtime with fast_realtime
  o = options_; o.shock_decomp.fast_realtime = [40 50]; o.shock_decomp.save_realtime = [45];
  oo2 = rmfield(oo_, {'realtime_shock_decomposition', 'conditional_shock_decomposition', ...
                      'realtime_forecast_shock_decomposition', 'realtime_conditional_shock_decomposition'});
  oo2 = realtime_shock_decomposition(M_, oo2, o, {}, bayestopt_, estim_params_);
  r.fast_pool = oo2.realtime_shock_decomposition.pool; r.fast_cond_pool = oo2.conditional_shock_decomposition.pool;
  r.fast_fcst_time_45 = oo2.realtime_forecast_shock_decomposition.time_45;
  r_main = r;
  r = struct();
  base = set_default_plot_shock_decomposition_options(options_);
  base.no_graph.plot_shock_decomposition = false;
  vl = {'y'; 'pi'; 'robs'};
  % expanding a group (what expand_group.m sets up before calling plot_shock_decomposition)
  try
    iv = find(strcmp('pi', M_.endo_names));
    shocks = M_.shock_groups.grp.group2.shocks; Mx = M_;
    for j = 1:length(shocks)
      Mx.shock_groups.demand_and_policy.(['group' int2str(j)]).label = shocks{j};
      Mx.shock_groups.demand_and_policy.(['group' int2str(j)]).shocks = shocks(j);
    end
    o = base; o.plot_shock_decomp.use_shock_groups = 'demand_and_policy'; o.plot_shock_decomp.expand = 1;
    o.plot_shock_decomp.zfull = oo_.shock_decomposition(iv, :, :);
    o.plot_shock_decomp.endo_names = {'pi'}; o.plot_shock_decomp.endo_names_tex = {'pi'};
    r.p_expand = plot_shock_decomposition(Mx, oo_, o, {'pi'}, true);
  catch e
    disp(['NOTE expand: ' e.message]); disp(e.stack(1));
  end
  try
    o = base; o.plot_shock_decomp.use_shock_groups = 'grp';
    o.plot_shock_decomp.plot_end_date = dates('12Y');
    sd_stubs(true);
    plot_shock_decomposition(M_, oo_, o, {'pi'});
    sd_stubs(false);
    r.drawn_first_obs = sd_drawn();
  catch e
    sd_stubs(false);
    disp(['NOTE drawn first_obs: ' e.message]);
  end
  o = base; o.plot_shock_decomp.use_shock_groups = 'grp'; o.plot_shock_decomp.steadystate = true;
  [r.p_grp, r.p_grp_ss] = plot_shock_decomposition(M_, oo_, o, vl, true);
  modes = {'rt_pool', 1, 0; 'rt_vintage', 1, 45; 'cond_pool', 2, 0; 'cond_vintage', 2, 45; 'fcst_pool', 3, 0; 'fcst_vintage', 3, 45};
  for k = 1:rows(modes)
    try
      o = base; o.plot_shock_decomp.realtime = modes{k, 2}; o.plot_shock_decomp.vintage = modes{k, 3};
      o.plot_shock_decomp.use_shock_groups = 'grp';
      r.(['p_' modes{k, 1}]) = plot_shock_decomposition(M_, oo_, o, vl, true);
    catch e
      disp(['NOTE ' modes{k, 1} ': ' e.message]);
    end
  end
  % realtime without forecast and with the default presample
  o = options_; o.shock_decomp.forecast = 0; o.shock_decomp = rmfield(o.shock_decomp, 'presample');
  o.shock_decomp.save_realtime = [10 60];
  oo2 = rmfield(oo_, {'realtime_shock_decomposition', 'conditional_shock_decomposition', ...
                      'realtime_forecast_shock_decomposition', 'realtime_conditional_shock_decomposition'});
  oo2 = realtime_shock_decomposition(M_, oo2, o, {}, bayestopt_, estim_params_);
  r_main.rt0_pool = oo2.realtime_shock_decomposition.pool; r_main.rt0_time_10 = oo2.realtime_shock_decomposition.time_10;
  r_main.rt0_cond_pool = oo2.conditional_shock_decomposition.pool;
  save('-mat7-binary', 'sd_nk_case3.mat', '-struct', 'r_main');
  % squeeze
  oo3 = squeeze_shock_decomposition(M_, oo_, options_, {'pi'; 'y'});
  r.squeezed = oo3.shock_decomposition; r.squeezed_rt_pool = oo3.realtime_shock_decomposition.pool;
  r.squeezed_i_var = oo3.shock_decomposition_info.i_var;
  % screening of shocks: graph_decomp on an artificial decomposition with 21 shocks
  try
    ns = 21; ne = 2; T = 12;
    z = zeros(ne, ns + 2, T);
    for i = 1:ne, for j = 1:ns + 1, for t = 1:T
      z(i, j, t) = sin(0.7*i + 1.3*j*j + 0.37*t) * (1 + mod(j*7, 5));
    end, end, end
    z(:, end, :) = sum(z(:, 1:end-1, :), 2);
    r.screen_z = z;
    Mx = M_; Mx.exo_nbr = ns; Mx.exo_names = cell(ns, 1);
    for j = 1:ns, Mx.exo_names{j} = sprintf('s%d', j); end
    o = base; o.plot_shock_decomp.screen_shocks = true; o.plot_shock_decomp.fig_name = ''; o.plot_shock_decomp.expand = 0;
    o.plot_shock_decomp.colormap = [];
    sd_stubs(true);
    graph_decomp(z, 'calibration', Mx.exo_names, {'v1'; 'v2'}, [1; 2], [], Mx, o);
    sd_stubs(false);
    r.drawn_screen = sd_drawn();
    o.plot_shock_decomp.initval = 0;
    sd_stubs(true);
    graph_decomp_detail(z, 'calibration', Mx.exo_names, {'v1'; 'v2'}, [1; 2], [], Mx, o);
    sd_stubs(false);
    r.drawn_screen_detail = sd_drawn();
  catch e
    sd_stubs(false);
    disp(['NOTE screen: ' e.message]); disp(e.stack(1));
  end
  save('-mat7-binary', 'sd_nk_case3_plot.mat', '-struct', 'r');
  disp('OK sd_nk case 3');
catch e
  disp(['FAIL sd_nk case 3: ' e.message]); disp(e.stack(1));
end

% ================================================================ calibrated smoother cases
cases = {'sd_nk', 4; 'sd_nk', 5; 'sd_ur', 1; 'sd_ur', 2; 'sd_ur', 3; 'sd_ur', 4};
for ic = 1:rows(cases)
  nm = cases{ic, 1}; cs = cases{ic, 2};
  try
    sd_run(nm, cs);
    if strcmp(nm, 'sd_ur') && cs >= 3
      % estimation(mode_compute=0, smoother): the smoother ran at the initial values
      r = struct();
      r.names = strjoin(bayestopt_.name', ';');
      [~, ~, x0] = dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
      r.theta0 = x0;
      r.theta_prior_mode = bayestopt_.p5(:); r.theta_prior_mean = bayestopt_.p1(:);
      r.at_theta0 = sd_collect(oo_, M_, options_);
      r.sd_in_mod = oo_.shock_decomposition;
      if cs == 3, ps = 'prior_mean'; else, ps = 'prior_mode'; end
      r.(ps) = sd_at(ps, M_, oo_, options_, bayestopt_, estim_params_, true);
    else
      r = sd_collect(oo_, M_, options_);
      if isfield(oo_, 'shock_decomposition'), r.shock_decomposition = oo_.shock_decomposition; end
    end
    save('-mat7-binary', sprintf('%s_case%d.mat', nm, cs), '-struct', 'r');
    disp(sprintf('OK %s case %d', nm, cs));
  catch e
    disp(sprintf('FAIL %s case %d: %s', nm, cs, e.message)); disp(e.stack(1));
  end
end
