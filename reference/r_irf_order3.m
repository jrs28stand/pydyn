addpath /home/claude/dl/src72/matlab
dynare collard_o3 nograph
randn('state',11);
nx=M_.exo_nbr; cS=chol(M_.Sigma_e); cs=transpose(cS);
r=struct(); r.ys=oo_.dr.ys; r.params=M_.params; r.Sigma_e=M_.Sigma_e;
long=20; drop=30; replic=4; r.long=long; r.drop=drop;
r.draws = randn(long+drop,nx,replic);
for pr=0:1
  options_.pruning=logical(pr);
  out = zeros(M_.endo_nbr,long,nx);
  for i=1:nx
    e1=cs(:,i); acc=0;
    for j=1:replic
      ex1 = r.draws(:,:,j)*cS; ex2=ex1; ex2(drop+1,:)=ex2(drop+1,:)+e1';
      y1=simult_(M_,options_,oo_.dr.ys,oo_.dr,ex1,3); y2=simult_(M_,options_,oo_.dr.ys,oo_.dr,ex2,3);
      acc = acc + (y2(:,M_.maximum_lag+drop+1:end)-y1(:,M_.maximum_lag+drop+1:end));
    end
    out(:,:,i)=acc/replic;
  end
  if pr, r.irf3_pruned=out; else, r.irf3_unpruned=out; end
end
% Dynare's own irf.m with a fixed seed, to check the port of the algorithm (not only its building block)
options_.pruning=true; options_.replic=6; options_.drop=30;
randn('state',5); r.seed_draw_check = randn(1,3);
randn('state',5);
r.irf_dynare = irf(M_,options_,oo_.dr,cs(:,1),long,options_.drop,options_.replic,3);
save('-mat7-binary','clean_irf_order3.mat','-struct','r');
disp('O3IRF_DONE')
addpath /home/claude/dl/src72/matlab
dynare collard_o3 nograph
r = load('clean_irf_order3.mat'); nx=M_.exo_nbr; cs=transpose(chol(M_.Sigma_e));
randn('state',5); D=zeros(r.long+30,nx,6); for j=1:6, D(:,:,j)=randn(r.long+30,nx); end
r.dynare_draws = D;
options_.pruning=true;
randn('state',5); r.irf_dynare = irf(M_,options_,oo_.dr,cs(:,1),r.long,30,6,3);
randn('state',5); r.irf_dynare_order2 = irf(M_,options_,oo_.dr,cs(:,1),r.long,30,6,2);
options_.pruning=false;
randn('state',5); r.irf_dynare_unpruned = irf(M_,options_,oo_.dr,cs(:,2),r.long,30,6,3);
save('-mat7-binary','clean_irf_order3.mat','-struct','r'); disp('R2_DONE')
