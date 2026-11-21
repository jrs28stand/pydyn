% Regenerates reference/dvar/est_dvar_v<V>.mat (Dynare 7.2, DSGE-VAR) for tests/test_dsge_var.py.
% Run in a fresh directory holding est_dvar.mod and est_nk_data.m:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "V=0; r_dsge_var"
% V=0: dsge_var=0.8, dsge_varlag=2; V=1: estimated weight, dsge_varlag=4, mode_compute=4; V=2: dsge_var=Inf.
addpath /home/claude/dl/src72/matlab
NOBS = [NaN 100 NaN];
eval(sprintf('dynare est_dvar nograph -DVARIANT=%d', V));
options_.nobs = NOBS(V+1);    % the estimation command above replaced it by the length of the dataset with its initial lags
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
dataset_info = var_sample_moments(options_.dsge_varlag, 0, dataset_, dataset_info);
n = numel(xparam1); K = 4;
sd = bayestopt_.p2; k = ~isfinite(sd); sd(k) = 0.1*abs(xparam1(k));
X = zeros(n, K);
for k = 0:K-1
  X(:,k+1) = xparam1 + (k>0)*0.1*sin(k*(1:n)').*sd;
end
r = struct('names', strjoin(bayestopt_.name', ';'));
if V == 1
  x = xparam1; x(strcmp(bayestopt_.name, 'dsge_prior_weight')) = 0.12; X = [X x];   % below (k+n)/T = 16/104: info 51
end
r.X = X;
r.fval = zeros(1, columns(X)); r.info = zeros(4, columns(X)); r.lnprior = r.fval;
for k = 1:columns(X)
  [fval, info] = dsge_var_likelihood(X(:,k), dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                                     oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  r.fval(k) = fval; r.info(:,k) = info(:);
  r.lnprior(k) = priordens(X(:,k), bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, bayestopt_.p3, bayestopt_.p4);
end
r.mYY = dataset_info.mYY; r.mXX = dataset_info.mXX; r.mYX = dataset_info.mYX; r.T = dataset_.nobs;
if V == 1
  r.xmode = oo_.posterior.optimization.mode; r.logpost_mode = oo_.posterior.optimization.log_density;
end
save('-mat7-binary', sprintf('est_dvar_v%d.mat', V), '-struct', 'r');
disp('DVAR_DONE')
