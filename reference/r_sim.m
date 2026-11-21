addpath /home/claude/dl/src72/matlab
dynare schorfheide_2000_sim nograph
r = struct();
r.ghx=oo_.dr.ghx; r.ghu=oo_.dr.ghu; r.order_var=oo_.dr.order_var; r.ys=oo_.steady_state;
r.params=M_.params; r.Sigma_e=M_.Sigma_e;
r.endo_names=strjoin(cellstr(M_.endo_names)',','); r.exo_names=strjoin(cellstr(M_.exo_names)',',');
fn=fieldnames(oo_.irfs); r.irf_names=strjoin(fn',','); r.irfs=cell2mat(struct2cell(oo_.irfs));
save('-mat7-binary','clean_schorfheide_2000_sim.mat','-struct','r');
