% Regenerates the reference/smo_*.mat files used by tests/test_smoother.py.
% Run inside a scratch copy of reference/ (Dynare writes many files), then copy the smo_*_case*.mat
% files back:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_smoother.m
% Dynare 7.2, Octave 8.4. Each case is one run of a smo_*.mod file with the macro variable CASE.
addpath /home/claude/dl/src72/matlab
function r = smo_collect(oo_, M_, options_, bayestopt_, estim_params_, with_icd)
  r = struct();
  r.endo_names = strjoin(cellstr(M_.endo_names)', ',');
  r.exo_names = strjoin(cellstr(M_.exo_names)', ',');
  r.varobs = strjoin(options_.varobs(:)', ',');
  r.ys = oo_.dr.ys; r.params = M_.params; r.ghx = oo_.dr.ghx; r.ghu = oo_.dr.ghu;
  r.order_var = oo_.dr.order_var;
  n = M_.endo_nbr; T = length(oo_.SmoothedShocks.(M_.exo_names{1}));
  r.smoothed = zeros(T, n); r.updated = zeros(T, n); r.constant = zeros(1, n); r.init_state = zeros(1, n);
  if isfield(oo_, 'FilteredVariables'), r.filtered = zeros(T, n); end
  for i = 1:n
    nm = M_.endo_names{i};
    r.smoothed(:, i) = oo_.SmoothedVariables.(nm);
    r.updated(:, i) = oo_.UpdatedVariables.(nm);
    r.constant(i) = oo_.Smoother.Constant.(nm)(1);
    r.init_state(i) = oo_.Smoother.Init_State.(nm);
    if isfield(oo_, 'FilteredVariables'), r.filtered(:, i) = oo_.FilteredVariables.(nm); end
  end
  r.shocks = zeros(T, M_.exo_nbr);
  for i = 1:M_.exo_nbr, r.shocks(:, i) = oo_.SmoothedShocks.(M_.exo_names{i}); end
  if isfield(oo_, 'SmoothedMeasurementErrors')
    r.meas_err = zeros(T, length(options_.varobs));
    for i = 1:length(options_.varobs), r.meas_err(:, i) = oo_.SmoothedMeasurementErrors.(options_.varobs{i}); end
  end
  r.init_state_vector = oo_.Smoother.Init_State_Vector;
  r.loglinear = double(oo_.Smoother.loglinear);
  r.steady_state_smoother = oo_.Smoother.SteadyState;
  % large arrays are stored for a few periods only (1-based indices in *_idx)
  idx = unique([1 2 3 4 5 10 round(T/2) T-1 T]);
  r.idx = idx;
  if isfield(oo_, 'FilteredVariablesKStepAhead')
    r.kstep = oo_.FilteredVariablesKStepAhead;
    r.filter_step_ahead = options_.filter_step_ahead;
  end
  if isfield(oo_, 'FilteredVariablesKStepAheadVariances')
    kidx = unique([1 2 3 5 round(T/2) T+1 size(oo_.FilteredVariablesKStepAheadVariances, 4)]);
    r.kidx = kidx;
    r.kstep_var = oo_.FilteredVariablesKStepAheadVariances(:, :, :, kidx);
  end
  if isfield(oo_.Smoother, 'Variance')
    r.variance = oo_.Smoother.Variance(:, :, [idx T+1]);
  end
  if isfield(oo_.Smoother, 'State_uncertainty')
    r.state_uncertainty = oo_.Smoother.State_uncertainty(:, :, idx);
    r.init_state_uncertainty = oo_.Smoother.Init_State_uncertainty;
  end
  if isfield(oo_, 'shock_decomposition')
    r.shock_decomposition = oo_.shock_decomposition;
  end
  if with_icd
    options_ = set_default_initial_condition_decomposition_options(options_);
    oo2 = initial_condition_decomposition(M_, oo_, options_, 0, bayestopt_, estim_params_);
    r.initval_decomposition = oo2.initval_decomposition(:, :, [1:6 T]);
    r.icd_idx = [1:6 T];
  end
end
cases = {'smo_fs', 1, 1; 'smo_fs', 2, 0; 'smo_fs', 3, 0; 'smo_fs', 4, 0; ...
         'smo_me', 1, 1; 'smo_me', 2, 0; 'smo_me', 3, 0; ...
         'smo_ur', 1, 1; 'smo_ur', 2, 0; 'smo_ur', 3, 0; 'smo_ur', 4, 0; 'smo_ur', 5, 0; 'smo_ur', 6, 0; 'smo_ur', 7, 0};
for ic = 1:size(cases, 1)
  nm = cases{ic, 1}; cs = cases{ic, 2};
  try
    eval(sprintf('dynare %s nograph -DCASE=%d', nm, cs));
    r = smo_collect(oo_, M_, options_, bayestopt_, estim_params_, cases{ic, 3});
    save('-mat7-binary', sprintf('%s_case%d.mat', nm, cs), '-struct', 'r');
    disp(sprintf('OK %s case %d', nm, cs));
  catch e
    disp(sprintf('FAIL %s case %d: %s', nm, cs, e.message));
  end
end
