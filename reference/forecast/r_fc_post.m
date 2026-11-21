% Reference for pydyn.forecast.posterior_forecast (tests/test_forecast.py): Dynare 7.2 under Octave 8.4.
% Runs a short Metropolis-Hastings estimation (fc_post_nk.mod, -DPOST=7 or 8, -DVARIANT=k), then calls
% prior_posterior_statistics('posterior', ...) with the forecast option at seeded random numbers:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "ARGS='-DPOST=7 -DVARIANT=0'; TAG='p7'; HORIZON=6; r_fc_post"
% Saves <TAG>.mat: the sub-draws used (the parameter vectors, in the order Dynare used them) and the statistics of
% oo_.MeanForecast, oo_.PointForecast and oo_.PointForecastME; the random numbers are those of randn('state', 77).
addpath /home/claude/dl/src72/matlab
dynare_config();
for f = {'dyn_figure', 'subplot', 'figure'}
  fid = fopen([f{1} '.m'], 'w'); fprintf(fid, 'function h = %s(varargin)\nh = 1;\nend\n', f{1}); fclose(fid);
end
for f = {'dyn_saveas', 'hold', 'xlim', 'ylim', 'title', 'drawnow', 'box', 'legend', 'xlabel', 'ylabel', 'grid', 'axis'}
  fid = fopen([f{1} '.m'], 'w'); fprintf(fid, 'function %s(varargin)\nend\n', f{1}); fclose(fid);
end
fid = fopen('plot.m', 'w'); fprintf(fid, 'function h = plot(varargin)\nh = 1;\nend\n'); fclose(fid);
rehash;
eval(sprintf('dynare fc_post_nk nograph %s', ARGS));
[dataset_, dataset_info] = makedataset(options_);
if isnan(options_.first_obs), options_.first_obs = 1; end   % dynare_estimation_init does this on its own copy of options_
options_.forecast = HORIZON;
options_.smoother = 0;
options_ = select_qz_criterium_value(options_);   % Dynare sets it inside dynare_estimation_1
randn('state', 77); rand('state', 78);
oo_ = prior_posterior_statistics('posterior', dataset_, dataset_info, M_, oo_, options_, estim_params_, bayestopt_);
r = struct();
r.MeanForecast = oo_.MeanForecast;
r.PointForecast = oo_.PointForecast;
if isfield(oo_, 'PointForecastME'), r.PointForecastME = oo_.PointForecastME; end
for f = {'MeanForecast','PointForecast','PointForecastME'}   % drop the kernel densities (size)
  if isfield(r, f{1}) && isfield(r.(f{1}), 'density'), r.(f{1}) = rmfield(r.(f{1}), 'density'); end
end
q = load(sprintf('%s/metropolis/%s_param1.mat', M_.dname, M_.fname));
r.params = q.stock;
r.names = strjoin(bayestopt_.name', ';');
r.horizon = HORIZON;
r.mh_conf_sig = options_.mh_conf_sig;
r.varlist = strjoin(options_.varlist(:)', ',');
r.nobs = dataset_.nobs;
r.mh_drop = options_.mh_drop;   % the chains are not saved (size limit)
save('-mat7-binary', [TAG '.mat'], '-struct', 'r');
disp('FCPOST_DONE')
