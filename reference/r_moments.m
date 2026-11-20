% Regenerates reference/mom_*.mat (Dynare 7.2 moments for tests/test_moments.py).
% Run in a scratch directory that holds the mom_*.mod files:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_moments.m
% Each case calls stoch_simul(M_, options_, oo_, var_list) with modified options and stores plain numeric
% copies of the oo_ fields (prefix = case name) in mom_<model>.mat.
addpath /home/claude/dl/src72/matlab
global M_ options_ oo_

function o = setopt(o, name, val)
  p = strsplit(name, '.');
  if numel(p) == 1
    o.(p{1}) = val;
  else
    o.(p{1}).(p{2}) = val;
  end
end

function r = collect(r, pre, oo_, ar)
  names = {'mean','var','skewness','kurtosis','contemporaneous_correlation','variance_decomposition', ...
           'variance_decomposition_ME','conditional_variance_decomposition','conditional_variance_decomposition_ME'};
  for k = 1:numel(names)
    if isfield(oo_, names{k}) && ~isempty(oo_.(names{k}))
      r.([pre '_' names{k}]) = oo_.(names{k});
    end
  end
  if isfield(oo_, 'autocorr') && ~isempty(oo_.autocorr)
    r.([pre '_autocorr']) = cat(3, oo_.autocorr{:});
  end
  if isfield(oo_, 'gamma_y')
    for k = 1:numel(oo_.gamma_y)
      % entries 1..ar+1 duplicate oo_.var and oo_.autocorr; keep the rest (ar+2: variance decomposition, ar+3: mean correction)
      if k > ar + 1 && ~isempty(oo_.gamma_y{k}), r.([pre '_gy' num2str(k)]) = oo_.gamma_y{k}; end
    end
  end
end

models = {'mom_collard','mom_unitroot','mom_aux','mom_schorfheide'};
subsets = struct('mom_collard', {{'c','y','a'}}, 'mom_unitroot', {{'z','d','q','w'}}, ...
                 'mom_aux', {{'z','c','AUX_ENDO_LAG_2_1','AUX_EXO_LAG_6_1'}}, 'mom_schorfheide', {{'y','R','gy_obs','AUX_ENDO_LEAD_65'}});
cvd = [1 2 4 10 40];
clearfields = {'mean','var','skewness','kurtosis','contemporaneous_correlation','autocorr','gamma_y', ...
               'variance_decomposition','variance_decomposition_ME','conditional_variance_decomposition', ...
               'conditional_variance_decomposition_ME','endo_simul','exo_simul'};
for im = 1:numel(models)
  nm = models{im};
  eval(['dynare ' nm ' nograph']);
  global M_ options_ oo_
  opts0 = options_;
  has2 = ~strcmp(nm, 'mom_unitroot');   % linear model: order 1 only
  r = struct();
  r.endo_names = strjoin(cellstr(M_.endo_names)', ',');
  r.exo_names = strjoin(cellstr(M_.exo_names)', ',');
  r.orig_endo_nbr = M_.orig_endo_nbr;
  r.Sigma_e = M_.Sigma_e; r.H = M_.H; r.varobs_id = options_.varobs_id;
  r.nstatic = M_.nstatic; r.nspred = M_.nspred; r.nboth = M_.nboth; r.nfwrd = M_.nfwrd;
  sub = subsets.(nm);
  % case = {name, {option name/value pairs}, var_list, simulate}
  cases = {
    {'th1',      {'order',1,'conditional_variance_decomposition',cvd,'contemporaneous_correlation',true}, {}, false}
    {'th1_hp',   {'order',1,'hp_filter',1600,'ar',4,'conditional_variance_decomposition',[1 5],'contemporaneous_correlation',true}, {}, false}
    {'th1_bp',   {'order',1,'bandpass.indicator',true,'ar',4,'contemporaneous_correlation',true}, {}, false}
    {'th1_hp128',{'order',1,'hp_filter',100,'ar',2,'filtered_theoretical_moments_grid',128}, {}, false}
    {'th1_sub',  {'order',1,'ar',2,'conditional_variance_decomposition',[1 3 8],'contemporaneous_correlation',true}, sub, false}
    {'th1_nodec',{'order',1,'ar',2,'nodecomposition',true,'conditional_variance_decomposition',[2 6]}, {}, false}
    {'s1',       {'order',1,'periods',160,'drop',30,'ar',4,'contemporaneous_correlation',true}, {}, true}
    {'s1_hp',    {'order',1,'periods',160,'drop',30,'ar',4,'hp_filter',1600,'contemporaneous_correlation',true}, {}, true}
    {'s1_os',    {'order',1,'periods',160,'drop',30,'ar',3,'one_sided_hp_filter',1600,'contemporaneous_correlation',true}, {}, true}
    {'s1_bp',    {'order',1,'periods',160,'drop',30,'ar',3,'bandpass.indicator',true,'contemporaneous_correlation',true}, {}, true}
    {'s1_sub',   {'order',1,'periods',160,'drop',30,'ar',2,'contemporaneous_correlation',true}, sub, true}
  };
  if has2
    cases = [cases; {
    {'th2',      {'order',2,'pruning',false,'ar',3,'conditional_variance_decomposition',[1 4],'contemporaneous_correlation',true}, {}, false}
    {'th2p',     {'order',2,'pruning',true,'ar',3,'conditional_variance_decomposition',[1 4 10],'contemporaneous_correlation',true}, {}, false}
    {'th2p_sub', {'order',2,'pruning',true,'ar',2,'conditional_variance_decomposition',[2 5],'contemporaneous_correlation',true}, sub, false}
    {'th2p_nodec',{'order',2,'pruning',true,'ar',2,'nodecomposition',true,'conditional_variance_decomposition',[2 5]}, {}, false}
    {'th2_sub',  {'order',2,'pruning',false,'ar',2,'contemporaneous_correlation',true}, sub, false}
    {'s2p',      {'order',2,'pruning',true,'periods',160,'drop',30,'ar',4,'contemporaneous_correlation',true}, {}, true}
    {'s2p_hp',   {'order',2,'pruning',true,'periods',160,'drop',30,'ar',3,'hp_filter',1600,'contemporaneous_correlation',true}, {}, true}
    }];
  end
  saved_sim = {};
  for ic = 1:numel(cases)
    c = cases{ic}; pre = c{1}; kv = c{2}; vl = c{3}; dosim = c{4};
    o = opts0; o.irf = 0; o.periods = 0; o.nograph = true; o.nodisplay = true;
    o.nodecomposition = false; o.nocorr = false; o.nomoments = false; o.pruning = false;
    o.hp_filter = 0; o.one_sided_hp_filter = 0; o.bandpass.indicator = false;
    o.conditional_variance_decomposition = []; o.contemporaneous_correlation = false;
    for k = 1:2:numel(kv), o = setopt(o, kv{k}, kv{k+1}); end
    for k = 1:numel(clearfields)
      if isfield(oo_, clearfields{k}), oo_ = rmfield(oo_, clearfields{k}); end
    end
    try
      randn('state', 11);
      txt = evalc('[info, oo_, ~, M_] = stoch_simul(M_, o, oo_, vl);');
      fputs(stdout, txt);
      keep = {'th1','th1_hp','th2','th2p','th2p_sub','s1_hp','s2p'};
      if strcmp(nm, 'mom_schorfheide'), keep = {'th1','th2p'}; end
      if any(strcmp(pre, keep)), r.([pre '_text']) = txt(regexp(txt, '(APPROXIMATED )?THEORETICAL MOMENTS|MOMENTS OF SIMULATED', 'once', 'start'):end); end
      r.([pre '_info']) = info;
      r = collect(r, pre, oo_, o.ar);
      if isempty(vl), names = cellstr(M_.endo_names)(1:M_.orig_endo_nbr); else names = vl; end
      [~, iv] = ismember(names, cellstr(M_.endo_names));
      r.([pre '_ivar']) = iv(:);
      if dosim
        key = sprintf('o%d_p%d', o.order, o.pruning);
        r.([pre '_simkey']) = key;
        if ~any(strcmp(saved_sim, key))
          r.(['sim_' key '_endo']) = oo_.endo_simul;
          r.(['sim_' key '_exo']) = oo_.exo_simul;
          saved_sim{end+1} = key;
        else
          assert(isequal(oo_.exo_simul, r.(['sim_' key '_exo'])));   % same shocks in all cases of a group
        end
      end
      if o.order == 2 && ~isfield(r, 'dr_ghxx')
        d = oo_.dr;
        r.dr_ghxx = d.ghxx; r.dr_ghxu = d.ghxu; r.dr_ghuu = d.ghuu; r.dr_ghs2 = d.ghs2;
      end
      r.dr_ghx = oo_.dr.ghx; r.dr_ghu = oo_.dr.ghu; r.dr_ys = oo_.dr.ys; r.dr_order_var = oo_.dr.order_var;
      printf('CASE OK %s %s\n', nm, pre);
    catch err
      r.([pre '_error']) = err.message;
      printf('CASE FAIL %s %s: %s\n', nm, pre, err.message);
    end
  end
  save('-mat7-binary', [nm '.mat'], '-struct', 'r');
  printf('SAVED %s\n', nm);
end
disp('MOMENTS_DONE')
