addpath /home/claude/dl/src72/matlab
for c = 1:4
  try
    eval(sprintf('dynare sw_big nograph -DCASE=%d', c));
    d = oo_.dr; r = struct('ys',d.ys,'params',M_.params,'ghx',d.ghx,'ghu',d.ghu,'order_var',d.order_var,'eig',d.eigval);
    r.eig_abs = abs(d.eigval);
    r.endo_names = strjoin(cellstr(M_.endo_names)',',');
    r.var = oo_.var; r.mean = oo_.mean; r.autocorr = cell2mat(oo_.autocorr); r.vd = oo_.variance_decomposition;
    if isfield(oo_,'conditional_variance_decomposition'), r.cvd = oo_.conditional_variance_decomposition; end
    fn = fieldnames(oo_.irfs); r.irf_names = strjoin(fn',','); r.irfs = cell2mat(struct2cell(oo_.irfs));
    if isfield(oo_,'SmoothedVariables') && c ~= 4
      fn = fieldnames(oo_.SmoothedVariables); r.sm_names = strjoin(fn',',');
      r.smoothed = cell2mat(cellfun(@(f) oo_.SmoothedVariables.(f)(:)', fn, 'UniformOutput', false));
      fn = fieldnames(oo_.SmoothedShocks); r.sh_names = strjoin(fn',',');
      r.shocks = cell2mat(cellfun(@(f) oo_.SmoothedShocks.(f)(:)', fn, 'UniformOutput', false));
    end
    r.ok = 1;
  catch e
    r = struct('ok', 0, 'msg', e.message);
  end
  save('-mat7-binary', sprintf('sw_ref_case%d.mat', c), '-struct', 'r');
  disp(sprintf('CASE %d ok=%d', c, r.ok)); if ~r.ok, disp(r.msg); end
end
addpath /home/claude/dl/src72/matlab
for c = [2 3 4]
 for algo = [3 4]
  try
    eval(sprintf('dynare sw_big nograph -DCASE=%d -DALGO=%d', c, algo));
    r = struct('ok',1);
    fn = fieldnames(oo_.SmoothedVariables); r.sm_names = strjoin(fn',',');
    r.smoothed = cell2mat(cellfun(@(f) oo_.SmoothedVariables.(f)(:)', fn, 'UniformOutput', false));
    fn = fieldnames(oo_.SmoothedShocks); r.sh_names = strjoin(fn',',');
    r.shocks = cell2mat(cellfun(@(f) oo_.SmoothedShocks.(f)(:)', fn, 'UniformOutput', false));
  catch e
    r = struct('ok', 0, 'msg', e.message);
  end
  save('-mat7-binary', sprintf('sw_ref_diffuse_case%d_algo%d.mat', c, algo), '-struct', 'r');
  disp(sprintf('CASE %d ALGO %d ok=%d', c, algo, r.ok)); if ~r.ok, disp(r.msg); end
 end
end
