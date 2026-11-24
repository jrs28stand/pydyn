% OccBin findmin (loss-function regime guess) reference: NKM likelihood with loss_function_regime_guess
1;
function patch_filter()
  src = fileread('dyn/matlab/kalman/likelihood/missing_observations_kalman_filter.m');
  key = sprintf('lik(1:s) = .5*lik(1:s);');
  k = strfind(src, key);
  ins = sprintf(['global OCCEST_PKF\nif occbin_.status\n  OCCEST_PKF = struct(''lik'', lik, ''a0'', a0, ''a1'', a1, ' ...
                 '''P0'', P0, ''regimes'', {regimes_}, ''t'', t);\nend\n']);
  src = [src(1:k(1)-1) ins src(k(1):end)];
  fid = fopen('dyn/matlab/kalman/likelihood/missing_observations_kalman_filter.m', 'w'); fwrite(fid, src); fclose(fid);
  rehash;
end
function c = regimes_to_cell(rh)
  if isempty(rh), c = {}; else, c = struct2cell(rh(:)); c = reshape(c, size(c, 1), []); end
end
addpath([pwd '/dyn/matlab'])
patch_filter();
global OCCEST_PKF FINDMIN_LOG
variant = getenv('FM_VARIANT');
dynare occest_nkm nograph
md = load('NKM_mh_mode_saved.mat');
x0 = md.xparam1(:);
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
rel = [0 0.1 -0.1];
n = numel(x0);
X = zeros(n, numel(rel));
for k = 1:numel(rel), X(:, k) = x0 .* (1 + rel(k) * sin(k * (1:n)')); end
o = options_;
switch variant
  case 'base'
  case 'early'      % findmin in the first 20 periods
    o.occbin.likelihood.loss_function_regime_guess = true;
    o.occbin.likelihood.number_of_initial_periods_with_extra_regime_guess = 20;
    o.occbin.likelihood.brute_force_regime_guess = false;
    o.occbin.likelihood.brute_force_extra_regime_guess = false;
  case 'fail'       % regime iteration limited to 1: findmin on the failures
    o.occbin.likelihood.loss_function_regime_guess = true;
    o.occbin.likelihood.max_number_of_iterations = 1;
    o.occbin.likelihood.brute_force_regime_guess = false;
    o.occbin.likelihood.brute_force_extra_regime_guess = false;
  case 'failbase'
    o.occbin.likelihood.max_number_of_iterations = 1;
    o.occbin.likelihood.brute_force_regime_guess = false;
    o.occbin.likelihood.brute_force_extra_regime_guess = false;
end
global OCCEST_PKF FINDMIN_LOG FINDMIN_Y FINDMIN_C FINDMIN_R FINDMIN_A
r = struct(); r.X = X; r.data = dataset_.data;
r.fval = NaN(1,numel(rel)); r.info = zeros(1,numel(rel)); r.nfind = zeros(1,numel(rel));
for k = 1:numel(rel)
  OCCEST_PKF = []; FINDMIN_LOG = zeros(0,2); FINDMIN_Y = {}; FINDMIN_C = []; FINDMIN_R = {}; FINDMIN_A = {};
  [fval, info] = dsge_likelihood(X(:,k), dataset_, dataset_info, o, M_, estim_params_, bayestopt_, bounds, ...
        oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  printf('%s point %d: fval %.10g info %d findmin calls %d\n', variant, k, fval, info(1), size(FINDMIN_LOG,1));
  r.fval(k) = fval; r.info(k) = info(1); r.nfind(k) = size(FINDMIN_LOG,1);
  if ~isempty(FINDMIN_Y)
    r.(sprintf('fm_y_%d', k)) = cell2mat(FINDMIN_Y(:)');
    r.(sprintf('fm_c_%d', k)) = FINDMIN_C;
    r.(sprintf('fm_obs_%d', k)) = cell2mat(FINDMIN_A(:));
    r.(sprintf('fm_reg_%d', k)) = regimes_to_cell([FINDMIN_R{:}]);
  end
  if ~isempty(OCCEST_PKF)
    r.(sprintf('lik_%d', k)) = OCCEST_PKF.lik;
    r.(sprintf('a0_%d', k)) = OCCEST_PKF.a0;
    r.(sprintf('regimes_%d', k)) = regimes_to_cell(OCCEST_PKF.regimes);
  end
end
save('-mat7-binary', sprintf('findmin_%s.mat', variant), '-struct', 'r');
