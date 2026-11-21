// pydyn test (finding R25): two observed variables driven by one shock, so the forecast-error
// variance F of the Kalman filter is singular in every (stationary) period. Dynare 7.2's
// calib_smoother stops with an error (422), unless use_univariate_smoother_if_singularity_is_detected
// (VARIANT 2) or options_.noprint (set here by an earlier stoch_simul, VARIANT 3), with which it
// switches to the univariate smoother.
@#ifndef VARIANT
@#define VARIANT = 1
@#endif
var x y1 y2;
varexo e;
parameters r;
r = 0.8;
model(linear);
x = r*x(-1) + e;
y1 = x;
y2 = 2*x;
end;
shocks; var e; stderr 0.5; end;
varobs y1 y2;
@#if VARIANT == 3
stoch_simul(order=1, irf=0, noprint, nomoments);
@#endif
@#if VARIANT == 2
calib_smoother(datafile=smo_sing_data, use_univariate_smoother_if_singularity_is_detected);
@#else
calib_smoother(datafile=smo_sing_data);
@#endif
