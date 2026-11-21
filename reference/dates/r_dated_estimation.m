% Regenerates reference/dates/est_dated_v<V>.mat (Dynare 7.2, data command with dates) for tests/test_dates.py.
% Run in a fresh directory holding est_dated.mod and est_nk_dated.csv:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "V=0; r_dated_estimation"
addpath /home/claude/dl/src72/matlab
eval(sprintf('dynare est_dated nograph -DVARIANT=%d', V));
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
n = numel(xparam1); K = 4;
sd = bayestopt_.p2; k = ~isfinite(sd); sd(k) = 0.1*abs(xparam1(k));
X = zeros(n, K);
for k = 0:K-1
  X(:,k+1) = xparam1 + (k>0)*0.2*sin(k*(1:n)').*sd;
end
x = xparam1; x(strcmp(bayestopt_.name, 'corr ea, eu')) = 0; X = [X x];     % diagonal Q
r = struct('X', X);
r.names = strjoin(bayestopt_.name', ';');
r.fval = zeros(1, columns(X)); r.lnprior = r.fval; r.info = r.fval;
for k = 1:columns(X)
  [fval, info] = dsge_likelihood(X(:,k), dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                                 oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  r.fval(k) = fval; r.info(k) = info(1);
  r.lnprior(k) = priordens(X(:,k), bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
end
r.loglik = -(r.fval + r.lnprior);
if options_.heteroskedastic_filter, r.Qvalue = M_.heteroskedastic_shocks.Qvalue; r.Qscale = M_.heteroskedastic_shocks.Qscale; end
% without the option: the likelihood ignores heteroskedastic_shocks
o2 = options_; o2.heteroskedastic_filter = false;
r.fval_plain = dsge_likelihood(X(:,1), dataset_, dataset_info, o2, M_, estim_params_, bayestopt_, bounds, ...
                               oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
if V == 2
  n = M_.orig_endo_nbr;
  r.smoothed = zeros(options_.nobs, n); r.updated = r.smoothed;
  for i = 1:n
    r.smoothed(:,i) = oo_.SmoothedVariables.(M_.endo_names{i});
    r.updated(:,i) = oo_.UpdatedVariables.(M_.endo_names{i});
  end
  r.shocks = zeros(options_.nobs, M_.exo_nbr);
  for i = 1:M_.exo_nbr, r.shocks(:,i) = oo_.SmoothedShocks.(M_.exo_names{i}); end
  if isfield(oo_, 'SmoothedMeasurementErrors'), r.me_yobs = oo_.SmoothedMeasurementErrors.yobs; end
end
save('-mat7-binary', sprintf('est_dated_v%d.mat', V), '-struct', 'r');
disp('DATED_DONE')
