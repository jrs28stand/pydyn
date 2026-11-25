@#include "fs2000_model.inc"

shocks;
var e_a; stderr 0.014;
var e_m; stderr 0.005;
var gp_obs; stderr 0.05;
var gy_obs; stderr 0.03;
corr gy_obs,gp_obs = 0.5;
end;

estimated_params;
alp, 0.356;
gam, 0.0085;
del, 0.01;
stderr e_a, 0.035449;
stderr e_m, 0.008862;
end;


options_.lik_init=2;
estimation(order=1,datafile=fsdat_simul,nobs=192,mode_compute=0,mh_replic=0,kalman_algo=2,nograph,nodisplay) gp_obs gy_obs;
lik=oo_.likelihood_at_initial_parameters;
H=M_.H; CME=M_.Correlation_matrix_ME;
rr=[lik; H(:); CME(:)];
save('res_me_lik2.mat','rr','lik','H','CME');

