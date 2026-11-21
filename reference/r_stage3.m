addpath /home/claude/dl/src72/matlab
function r = occref(oo_, M_)
  s = oo_.occbin.simul;
  r = struct('piecewise',s.piecewise,'linear',s.linear,'shocks',s.shocks_sequence,'ys',oo_.steady_state,'params',M_.params);
  r.regime = struct2cell(s.regime_history(:));
  r.regime_fields = strjoin(fieldnames(s.regime_history)',',');
  r.endo_names = strjoin(cellstr(M_.endo_names)',',');
end
% ---- one-constraint OccBin
for nm = {'borrcon_sim','borrcon_curb'}
  try
    eval(['dynare ' nm{1} ' nograph']);
    r = occref(oo_, M_); save('-mat7-binary',['clean_' nm{1} '.mat'],'-struct','r'); disp(['OK ' nm{1}]);
  catch e
    disp(['FAIL ' nm{1} ': ' e.message]);
  end
end
% ---- second-order simulation, pruned and unpruned, and IRFs
dynare collard_2001_simulated_moments nograph
randn('state',7);
T=120; nx=M_.exo_nbr; cS=chol(M_.Sigma_e);
r=struct(); r.ys=oo_.dr.ys; r.params=M_.params;
r.ex = randn(T,nx)*cS;
options_.pruning=false; y=simult_(M_,options_,oo_.dr.ys,oo_.dr,r.ex,2); r.y2_unpruned=y';
options_.pruning=true;  y=simult_(M_,options_,oo_.dr.ys,oo_.dr,r.ex,2); r.y2_pruned=y';
y=simult_(M_,options_,oo_.dr.ys,oo_.dr,r.ex,1); r.y1=y';
% IRF at order 2 with saved draws (same algorithm as irf.m)
long=20; drop=30; replic=4; r.long=long; r.drop=drop;
r.draws = randn(long+drop,nx,replic);
cs = transpose(chol(M_.Sigma_e));
for pr=0:1
  options_.pruning=logical(pr);
  out = zeros(M_.endo_nbr,long,nx);
  for i=1:nx
    e1=cs(:,i); acc=0;
    for j=1:replic
      ex1 = r.draws(:,:,j)*cS; ex2=ex1; ex2(drop+1,:)=ex2(drop+1,:)+e1';
      y1=simult_(M_,options_,oo_.dr.ys,oo_.dr,ex1,2); y2=simult_(M_,options_,oo_.dr.ys,oo_.dr,ex2,2);
      acc = acc + (y2(:,M_.maximum_lag+drop+1:end)-y1(:,M_.maximum_lag+drop+1:end));
    end
    out(:,:,i)=acc/replic;
  end
  if pr, r.irf2_pruned=out; else, r.irf2_unpruned=out; end
end
save('-mat7-binary','clean_sim_order2.mat','-struct','r');
% ---- third order
dynare collard_o3 nograph
r3=struct(); r3.ys=oo_.dr.ys; r3.params=M_.params; r3.ex=r.ex;
options_.pruning=true;  y=simult_(M_,options_,oo_.dr.ys,oo_.dr,r3.ex,3); r3.y3_pruned=y';
options_.pruning=false; y=simult_(M_,options_,oo_.dr.ys,oo_.dr,r3.ex,3); r3.y3_unpruned=y';
save('-mat7-binary','clean_sim_order3.mat','-struct','r3');
% ---- external parameter loop (no re-invocation of dynare)
dynare schorfheide_2000_sim nograph
vals_alp=[0.30 0.33 0.36]; vals_rho=[0.5 0.7];
L=struct(); k=0; L.alp=[]; L.rho=[]; L.ghx={}; L.ghu={}; L.ys={}; L.irf={};
for a=vals_alp, for rr=vals_rho
  k=k+1;
  set_param_value('alp',a); set_param_value('rho',rr);
  [oo_.dr, info, M_.params] = resol(0, M_, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
  L.alp(k)=a; L.rho(k)=rr; L.ghx{k}=oo_.dr.ghx; L.ghu{k}=oo_.dr.ghu; L.ys{k}=oo_.dr.ys;
  cs=transpose(chol(M_.Sigma_e)); y=irf(M_,options_,oo_.dr,cs(:,1),20,0,0,1); L.irf{k}=y';
end, end
L.info=info;
save('-mat7-binary','clean_param_loop.mat','-struct','L');
% ---- steady-state file that updates parameters; helper function inside steady_state_model
for nm = {'nk_baseline','collard_2001_analytical_steady_state'}
  eval(['dynare ' nm{1} ' nograph']);
  q=struct('ghx',oo_.dr.ghx,'ghu',oo_.dr.ghu,'ys',oo_.dr.ys,'params',M_.params,'order_var',oo_.dr.order_var,'Sigma_e',M_.Sigma_e);
  q.endo_names = strjoin(cellstr(M_.endo_names)',','); q.param_names = strjoin(cellstr(M_.param_names)',',');
  if strcmp(nm{1},'nk_baseline')
    % loop: change a deep parameter, steady-state file must update dependent parameters
    set_param_value('betta',0.995);
    [oo_.dr, info, M_.params] = resol(0, M_, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
    q.ghx_b=oo_.dr.ghx; q.ghu_b=oo_.dr.ghu; q.ys_b=oo_.dr.ys; q.params_b=M_.params;
  end
  save('-mat7-binary',['clean_' nm{1} '.mat'],'-struct','q');
end
disp('STAGE3_DONE')
