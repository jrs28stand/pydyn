addpath /home/claude/dl/src72/matlab
dynare aux_a nograph
d=oo_.dr; r=struct('ys',d.ys,'params',M_.params,'Sigma_e',M_.Sigma_e,'order_var',d.order_var,'ghx',d.ghx,'ghu',d.ghu,'ghxx',d.ghxx,'ghxu',d.ghxu,'ghuu',d.ghuu,'ghs2',d.ghs2);
r.endo_names=strjoin(cellstr(M_.endo_names)',','); r.orig_endo_nbr=M_.orig_endo_nbr;
cs=transpose(chol(M_.Sigma_e)); r.irf1=zeros(12,M_.endo_nbr,M_.exo_nbr);
for i=1:M_.exo_nbr, y=irf(M_,options_,d,cs(:,i),12,0,0,1); r.irf1(:,:,i)=y'; end
randn('state',3); r.ex=randn(60,M_.exo_nbr)*chol(M_.Sigma_e);
options_.pruning=true; y=simult_(M_,options_,d.ys,d,r.ex,2); r.y2_pruned=y';
y=simult_(M_,options_,d.ys,d,r.ex,1); r.y1=y';
save('-mat7-binary','clean_aux_a.mat','-struct','r');
disp(M_.endo_names');
