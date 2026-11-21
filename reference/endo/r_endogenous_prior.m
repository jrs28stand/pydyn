% Regenerates reference/endo/est_endo_v<V>.mat (Dynare 7.2, endogenous_prior) for
% tests/test_endogenous_prior.py. Run in a fresh directory holding est_endo.mod, est_nk_data.m, est_nk_miss.m:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "V=0; r_endogenous_prior"
% V = 0, 1, 2, 9: likelihood variants of est_nk.mod evaluated with options_.endogenous_prior = true;
% V = 20: estimation(endogenous_prior, mode_compute=4) in the .mod file (posterior mode).
addpath /home/claude/dl/src72/matlab
eval(sprintf('dynare est_endo nograph -DVARIANT=%d', V));
options_.endogenous_prior = true;
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
n = numel(xparam1); K = 4;
sd = bayestopt_.p2; k = ~isfinite(sd); sd(k) = 0.1*abs(xparam1(k));
X = zeros(n, K);
for k = 0:K-1
  X(:,k+1) = xparam1 + (k>0)*0.1*sin(k*(1:n)').*sd;
end
r = struct('X', X);
r.names = strjoin(bayestopt_.name', ';');
r.fval = zeros(1, columns(X)); r.info = r.fval; r.fval_plain = r.fval;
o2 = options_; o2.endogenous_prior = false;
for k = 1:columns(X)
  [fval, info] = dsge_likelihood(X(:,k), dataset_, dataset_info, options_, M_, estim_params_, bayestopt_, bounds, ...
                                 oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  r.fval(k) = fval; r.info(k) = info(1);
  r.fval_plain(k) = dsge_likelihood(X(:,k), dataset_, dataset_info, o2, M_, estim_params_, bayestopt_, bounds, ...
                                    oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
end
if V == 20
  r.xmode = oo_.posterior.optimization.mode; r.logpost_mode = oo_.posterior.optimization.log_density;
end
save('-mat7-binary', sprintf('est_endo_v%d.mat', V), '-struct', 'r');
disp('ENDO_DONE')
