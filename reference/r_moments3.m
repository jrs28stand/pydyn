% Regenerates reference/mom3_*.mat (Dynare 7.2 third-order pruned theoretical moments for
% tests/test_moments3.py). Run in a scratch directory holding mom3_*.mod (copies of mom_*.mod with order=3):
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_moments3.m
addpath /home/claude/dl/src72/matlab
global M_ options_ oo_
models = {'mom3_collard','mom3_aux','mom3_schorfheide'};
subsets = struct('mom3_collard', {{'c','y','a'}}, 'mom3_aux', {{'z','c','AUX_ENDO_LAG_2_1','AUX_EXO_LAG_6_1'}}, ...
                 'mom3_schorfheide', {{'y','R','gy_obs','AUX_ENDO_LEAD_65'}});
clearfields = {'mean','var','contemporaneous_correlation','autocorr','gamma_y', ...
               'variance_decomposition','variance_decomposition_ME','conditional_variance_decomposition', ...
               'conditional_variance_decomposition_ME'};
for im = 1:numel(models)
  nm = models{im};
  eval(['dynare ' nm ' nograph']);
  global M_ options_ oo_
  opts0 = options_;
  r = struct();
  r.endo_names = strjoin(cellstr(M_.endo_names)', ',');
  r.Sigma_e = M_.Sigma_e;
  sub = subsets.(nm);
  cases = {
    {'th3p',     {'order',3,'pruning',true,'ar',3,'contemporaneous_correlation',true}, {}}
    {'th3p_sub', {'order',3,'pruning',true,'ar',2,'conditional_variance_decomposition',[2 5],'contemporaneous_correlation',true}, sub}
    {'th3',      {'order',3,'pruning',false,'ar',2}, {}}
  };
  for ic = 1:numel(cases)
    c = cases{ic}; pre = c{1}; kv = c{2}; vl = c{3};
    o = opts0; o.irf = 0; o.periods = 0; o.nograph = true; o.nodisplay = true;
    o.nodecomposition = false; o.nocorr = false; o.nomoments = false; o.pruning = false;
    o.conditional_variance_decomposition = []; o.contemporaneous_correlation = false;
    for k = 1:2:numel(kv), o.(kv{k}) = kv{k+1}; end
    for k = 1:numel(clearfields)
      if isfield(oo_, clearfields{k}), oo_ = rmfield(oo_, clearfields{k}); end
    end
    try
      txt = evalc('[info, oo_, ~, M_] = stoch_simul(M_, o, oo_, vl);');
      fputs(stdout, txt);
      if strcmp(pre, 'th3p'), r.([pre '_text']) = txt(regexp(txt, 'THEORETICAL MOMENTS', 'once', 'start'):end); end
      r.([pre '_info']) = info;
      names = {'mean','var','contemporaneous_correlation','variance_decomposition','conditional_variance_decomposition'};
      for k = 1:numel(names)
        if isfield(oo_, names{k}) && ~isempty(oo_.(names{k})), r.([pre '_' names{k}]) = oo_.(names{k}); end
      end
      if isfield(oo_, 'autocorr') && ~isempty(oo_.autocorr), r.([pre '_autocorr']) = cat(3, oo_.autocorr{:}); end
      r.([pre '_has_mean']) = isfield(oo_, 'mean');
      if isempty(vl), names = cellstr(M_.endo_names)(1:M_.orig_endo_nbr); else names = vl; end
      [~, iv] = ismember(names, cellstr(M_.endo_names));
      r.([pre '_ivar']) = iv(:);
      printf('CASE OK %s %s\n', nm, pre);
    catch err
      r.([pre '_error']) = err.message;
      printf('CASE FAIL %s %s: %s\n', nm, pre, err.message);
    end
  end
  d = oo_.dr;
  f = {'ghx','ghu','ghxx','ghxu','ghuu','ghs2','ghxxx','ghxxu','ghxuu','ghuuu','ghxss','ghuss','ys','order_var'};
  for k = 1:numel(f), r.(['dr_' f{k}]) = d.(f{k}); end
  save('-mat7-binary', [nm '.mat'], '-struct', 'r');
  printf('SAVED %s\n', nm);
end
disp('MOMENTS3_DONE')
