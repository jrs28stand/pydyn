@#include "fs2000_model.inc"

shocks;
var e_a; stderr 0.014;
var e_m; stderr 0.005;
var gp_obs; stderr 0.05;
var gy_obs; stderr 0.03;
corr gy_obs,gp_obs = 0.5;
end;



options_.lik_init=1;
calib_smoother(datafile=fsdat_simul,nobs=192,kalman_algo=1,filtered_vars,filter_step_ahead=[1,3]) gp_obs gy_obs;
SM=oo_.SmoothedMeasurementErrors; SS=oo_.SmoothedShocks; SV=oo_.SmoothedVariables; FV=oo_.FilteredVariables; UV=oo_.UpdatedVariables;
rr=[SM.gp_obs(1:5); SM.gy_obs(1:5); SS.e_a(1:5); SV.m(1:5); FV.m(1:5); UV.m(1:5); SV.m(end); SS.e_m(end)];
save('res_cal1.mat','rr');

