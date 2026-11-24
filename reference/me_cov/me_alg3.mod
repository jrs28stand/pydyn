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


options_.lik_init=1;
estimation(order=1,datafile=fsdat_simul,nobs=192,mode_compute=0,mh_replic=0,kalman_algo=3,smoother,filtered_vars,filter_step_ahead=[1,3],nograph,nodisplay) gp_obs gy_obs;
SM=oo_.SmoothedMeasurementErrors; SS=oo_.SmoothedShocks; SV=oo_.SmoothedVariables; FV=oo_.FilteredVariables; UV=oo_.UpdatedVariables;
lik=oo_.likelihood_at_initial_parameters;
H=M_.H; CME=M_.Correlation_matrix_ME;
rr=[lik; H(:); CME(:); SM.gp_obs(1:5); SM.gy_obs(1:5); SS.e_a(1:5); SV.m(1:5); FV.m(1:5); UV.m(1:5); SV.m(end); SS.e_m(end)];
save('res_alg3.mat','rr','lik','H','CME');

