function run_nk(order, tag, varargin)
  try
    tic; obc_octave( sprintf('nk_o%d.mod', order), 'ShockSequenceFile=shocks.mat', 'SkipQuickPCheck', varargin{:} ); t=toc;
    global oo_ dynareOBC_ M_
    r = struct('endo', oo_.endo_simul, 'ys', oo_.dr.ys, 'names', {cellstr(M_.endo_names)}, 'M', dynareOBC_.MMatrix, 'time', t);
    if isfield(dynareOBC_, 'RootConditionalCovariance'), r.Rcc = dynareOBC_.RootConditionalCovariance; end
    save('-mat7-binary', sprintf('ref_%s.mat', tag), '-struct', 'r');
    printf('OBC %s COMPLETED in %.1f s; endo_simul %s; min R %.6f\n', tag, t, mat2str(size(oo_.endo_simul)), min(oo_.endo_simul(3,:)));
  catch err
    printf('OBC %s FAILED: %s\n', tag, err.message); for k=1:min(6,numel(err.stack)), printf('   at %s line %d\n', err.stack(k).name, err.stack(k).line); end
  end
end
