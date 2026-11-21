names = {'collard_2001_simulated_moments','rbc_occbin','schorfheide_2000'};
for i=1:numel(names)
  S = load([names{i} '_ref.mat']);
  oo_=S.oo_; M_=S.M_;
  r = struct();
  r.ghx=oo_.dr.ghx; r.ghu=oo_.dr.ghu; r.order_var=oo_.dr.order_var; r.ys=oo_.steady_state;
  r.params=M_.params; r.Sigma_e=M_.Sigma_e; r.nstatic=M_.nstatic; r.nspred=M_.nspred;
  r.endo_names=strjoin(cellstr(M_.endo_names)',','); r.exo_names=strjoin(cellstr(M_.exo_names)',',');
  if isfield(oo_,'occbin') && isfield(oo_.occbin,'simul')
    r.occ_piecewise=oo_.occbin.simul.piecewise; r.occ_linear=oo_.occbin.simul.linear;
    r.occ_regime=struct2cell(oo_.occbin.simul.regime_history(:)); 
    fn=fieldnames(oo_.occbin.simul); r.occ_fields=strjoin(fn',',');
    r.occ_shocks=oo_.occbin.simul.shocks_sequence;
    disp(fn'); disp(fieldnames(oo_.occbin.simul.regime_history)');
  end
  if isfield(oo_,'posterior_mode'), r.mode=struct2cell(oo_.posterior_mode.parameters); r.mode_sh=struct2cell(oo_.posterior_mode.shocks_std); end
  save('-mat7-binary',['clean_' names{i} '.mat'],'-struct','r');
end
disp('EXPORT_OK')
