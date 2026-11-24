% OccBin forecasts (occbin.forecast) from the NKM smoothed state of period 42, and qmc_sequence
1;
function c = regimes_to_cell(rh)
  if isempty(rh), c = {}; else, c = struct2cell(rh(:)); c = reshape(c, size(c, 1), []); end
end
function r = pack(r, pre, fc, M_)
  n = M_.endo_nbr;
  fld = {'linear', 'piecewise'};
  for q = 1:2
    st = fieldnames(fc.(fld{q}));
    for k = 1:numel(st)
      A = [];
      for i = 1:n
        A = cat(3, A, reshape(fc.(fld{q}).(st{k}).(M_.endo_names{i}), size(fc.(fld{q}).(st{k}).(M_.endo_names{i}), 1), []));
      end
      r.([pre '_' fld{q} '_' st{k}]) = A;   % horizon x (1 or 9) x n
    end
  end
  if iscell(fc.regimes) || isstruct(fc.regimes)
    if isvector(fc.regimes) && isstruct(fc.regimes)
      r.([pre '_regimes']) = regimes_to_cell(fc.regimes);
    end
  end
end
addpath([pwd '/dyn/matlab'])
dynare occest_nkm nograph
md = load('NKM_mh_mode_saved.mat'); x0 = md.xparam1(:);
[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] = ...
    dynare_estimation_init({}, M_.dname, [], M_, options_, oo_, estim_params_, bayestopt_);
o = options_;
o.smoother = true; o.filtered_vars = true; o.filter_step_ahead = 1; o.nk = 1;
o.occbin.smoother.plot = false; o.nograph = true;
oo1 = oo_;
[oo1, M1] = evaluate_smoother(x0, {}, M_, oo1, o, bayestopt_, estim_params_);
r = struct(); r.x0 = x0;
t0 = 42;
h0 = zeros(M1.endo_nbr, 1);
for i = 1:M1.endo_nbr, h0(i) = oo1.SmoothedVariables.(M1.endo_names{i})(t0); end
M1 = set_all_parameters(x0, estim_params_, M1);
o.noprint = true;
[ys_, params_, info_s] = evaluate_steady_state(oo1.steady_state, [oo1.exo_steady_state; oo1.exo_det_steady_state], M1, o, true); M1.params = params_; oo1.steady_state = ys_;
[dr_, info_r] = resol(0, M1, o, oo1.dr, oo1.steady_state, oo1.exo_steady_state, oo1.exo_det_steady_state); oo1.dr = dr_;
printf('resol info %d, ys vs smoothed diff %g\n', info_r(1), max(abs(oo1.steady_state - (h0 - 0))));
M1.endo_histval = h0;
r.histval = h0;
o.occbin.forecast.waitbar = false;
o.noprint = true;
% A: zero shocks
[f, ef, yf] = occbin.forecast(o, M1, oo1.dr, oo1.steady_state, oo1.exo_steady_state, oo1.exo_det_steady_state, 8);
r = pack(r, 'A', f, M1); r.A_yf = yf; r.A_error = ef;
% B: qmc, 127 replications (tests/occbin/filter/NKM.mod)
o1 = o; o1.occbin.forecast.qmc = true; o1.occbin.forecast.replic = 127;
[f, ef, yf] = occbin.forecast(o1, M1, oo1.dr, oo1.steady_state, oo1.exo_steady_state, oo1.exo_det_steady_state, 8);
r = pack(r, 'B', f, M1); r.B_yf = yf; r.B_error = ef;
% C: Gaussian draws
randn('state', 7);
o2 = o; o2.occbin.forecast.replic = 60;
[f, ef, yf] = occbin.forecast(o2, M1, oo1.dr, oo1.steady_state, oo1.exo_steady_state, oo1.exo_det_steady_state, 8);
r = pack(r, 'C', f, M1); r.C_yf = yf; r.C_error = ef;
r.C_state_after = randn(3,1);
% D: SHOCKS0 matrix (rows = shocks in declaration order of the varexo) and 40 replications
sh = zeros(M1.exo_nbr, 8); sh(2, 1:3) = [-1.5 -1 -0.5]; sh(3, 2) = 0.7;
r.D_shocks = sh;
o3 = o; o3.occbin.forecast.SHOCKS0 = sh;
[f, ef, yf] = occbin.forecast(o3, M1, oo1.dr, oo1.steady_state, oo1.exo_steady_state, oo1.exo_det_steady_state, 8);
r = pack(r, 'D', f, M1); r.D_yf = yf; r.D_error = ef;
randn('state', 3);
o3.occbin.forecast.replic = 40;
[f, ef, yf] = occbin.forecast(o3, M1, oo1.dr, oo1.steady_state, oo1.exo_steady_state, oo1.exo_det_steady_state, 8);
r = pack(r, 'D2', f, M1); r.D2_yf = yf; r.D2_error = ef;
% E: cell form
o4 = o; o4.occbin.forecast.SHOCKS0 = {{'epsi', [-1.5 -1 -0.5 0 0 0 0 0]}};
[f, ef, yf] = occbin.forecast(o4, M1, oo1.dr, oo1.steady_state, oo1.exo_steady_state, oo1.exo_det_steady_state, 8);
r = pack(r, 'E', f, M1); r.E_yf = yf; r.E_error = ef;
r.endo_names = strjoin(M1.endo_names', ';'); r.exo_names = strjoin(M1.exo_names', ';');
r.Sigma_e = M1.Sigma_e;
% qmc_sequence
r.q1 = qmc_sequence(24, int64(1), 1, 127);
r.q2 = qmc_sequence(7, int64(1), 0, 20);
r.q3 = qmc_sequence(1111, int64(1), 0, 3);
[r.q4, s4] = qmc_sequence(300, int64(5), 1, 6); r.q4_seed = double(s4);
r.q5 = qmc_sequence(5, int64(1), 2, 4, 2.5);
L = chol([2 0.5 0; 0.5 1 0.2; 0.1 0.2 3]', 'lower');
r.q6 = qmc_sequence(3, int64(1), 1, 5, L); r.q6_L = L;
save('-mat7-binary', 'forecast_nkm.mat', '-struct', 'r');
