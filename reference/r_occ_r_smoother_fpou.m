% OccBin smoother with smoother.first_period_occbin_update > 1
1;
function c = regimes_to_cell(rh)
  if isempty(rh), c = {}; else, c = struct2cell(rh(:)); c = reshape(c, size(c, 1), []); end
end
addpath([pwd '/dyn/matlab'])
cname = getenv('SM_CASE');
fpous = str2num(getenv('SM_FPOU'));
switch cname
  case 'nkm'
    dynare occest_nkm nograph
    md = load('NKM_mh_mode_saved.mat');
    x0 = md.xparam1(:);
  case 'rbc'
    dynare occest_rbc nograph
    L = load('rbc_data.mat'); c = L.c(:);
    save('datasim.mat', 'c');
    options_.datafile = 'datasim'; options_.nobs = 169; options_.first_obs = 1;
    options_.mode_compute = 0; options_.mh_replic = 0;
    x0 = [0.015; 0.9];
end
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
s = struct(); s.x0 = x0; s.fpous = fpous;
n = M_.orig_endo_nbr;
for j = 1:numel(fpous)
  o = options_;
  o.smoother = true; o.filtered_vars = true; o.filter_step_ahead = 1; o.nk = 1;
  o.occbin.smoother.plot = false; o.nograph = true;
  o.occbin.smoother.first_period_occbin_update = fpous(j);
  oo1 = oo_;
  [oo1, M1] = evaluate_smoother(x0, {}, M_, oo1, o, bayestopt_, estim_params_);
  pre = sprintf('f%d', fpous(j));
  fl = {'SmoothedVariables', 'UpdatedVariables', 'FilteredVariables'};
  for q = 1:3
    T = numel(oo1.(fl{q}).(M1.endo_names{1}));
    A = NaN(T, n);
    for i = 1:n, A(:, i) = oo1.(fl{q}).(M1.endo_names{i}); end
    s.([pre '_' fl{q}]) = A;
  end
  T = numel(oo1.SmoothedShocks.(M1.exo_names{1}));
  A = NaN(T, M1.exo_nbr);
  for i = 1:M1.exo_nbr, A(:, i) = oo1.SmoothedShocks.(M1.exo_names{i}); end
  s.([pre '_SmoothedShocks']) = A;
  sm = oo1.occbin.smoother;
  s.([pre '_regimes']) = regimes_to_cell(sm.regime_history);
  s.([pre '_rt_regimes']) = regimes_to_cell(sm.realtime_regime_history);
  s.([pre '_error_flag']) = sm.error_flag;
  if isfield(sm, 'warning_flag'), s.([pre '_warning_flag']) = sm.warning_flag; end
  printf('fpou %d: error_flag %d\n', fpous(j), sm.error_flag(1));
end
save('-mat7-binary', sprintf('smoother_fpou_%s.mat', cname), '-struct', 's');
