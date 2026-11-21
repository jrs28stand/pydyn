// pydyn test (findings R11): estimated standard deviations of correlated shocks. Dynare's
// set_all_parameters rebuilds Sigma_e from M_.Correlation_matrix, which only 'corr' statements
// set; when correlations are estimated the other calibrated covariances keep their value.
// Variants: -DVARIANT=1 corr e, u calibrated; 2 var e, u (a covariance) calibrated; 3 corr e, u
// calibrated and corr u, w estimated. Maximum likelihood, linear model.
@#ifndef VARIANT
@#define VARIANT = 1
@#endif
var x1 x2 x3 y1 y2;
varexo e u w;
parameters r1 r2 r3;
r1 = 0.7; r2 = 0.5; r3 = 0.3;
model(linear);
x1 = r1*x1(-1) + e;
x2 = r2*x2(-1) + u + 0.5*x1;
x3 = r3*x3(-1) + w;
y1 = x1 + x3;
y2 = x2 - x3;
end;
shocks;
var e; stderr 0.4;
var u; stderr 0.3;
var w; stderr 0.2;
@#if VARIANT == 2
var e, u = 0.06;
@#else
corr e, u = 0.5;
@#endif
end;
estimated_params;
stderr e, 0.4, 0.01, 3;
@#if VARIANT == 3
corr u, w, 0.1, -0.9, 0.9;
@#endif
r1, 0.7, 0, 0.99;
end;
varobs y1 y2;
estimation(datafile=est_corr_data, mode_compute=0, mh_replic=0, nograph);
