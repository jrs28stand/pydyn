// Performs a pf_with_expectation_errors simulation with a perfect_foresight_controlled_paths block
// In particular, tests that it’s possible to change the expected value of a given variable (c and y in period 4)

@#define shocks = true
@#define shocks = true
var c k y;
varexo x e_y;

parameters alph gam delt bet aa;
alph=0.5;
gam=0.5;
delt=0.02;
bet=0.05;
aa=0.5;


model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
y = 0.98*y(-1)+e_y;
end;

initval;
x = 1;
k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
c = aa*k^alph-delt*k;
y = 0;
end;

steady;

@#if shocks
perfect_foresight_controlled_paths;
  exogenize c;
  periods 2 4:5;
  values 1.6 1.7;
  endogenize x;

  exogenize k;
  periods 7:9;
  values 13;
  endogenize x;

  exogenize y;
  periods 2003Y:2005Y;
  values 0.5;
  endogenize e_y;
end;

shocks;
  var x;
  periods 1;
  values 1.2;
end;
@#endif


perfect_foresight_controlled_paths(learnt_in=2004Y);
  exogenize c;
  periods 4:6;
  values 1.8;
  endogenize x;

  exogenize y;
  periods 2004Y:2006Y;
  values 0.6;
  endogenize e_y;
end;

perfect_foresight_with_expectation_errors_setup(periods=30, first_simulation_period = 2001Y);
perfect_foresight_with_expectation_errors_solver;

if ~oo_.deterministic_simulation.status
   error('Perfect foresight simulation failed')
end

if oo_.endo_simul(1, 3) ~= 1.6 || any(oo_.endo_simul(1, 5:7) ~= 1.8) || any(oo_.endo_simul(2, 8:10) ~= 13) ...
   || oo_.endo_simul(3, 4) ~= 0.5 || any(oo_.endo_simul(3, 5:7) ~= 0.6) || oo_.exo_simul(2, 1) ~= 1.2
   error('Perfect foresight with expectation errors and controlled paths failed')
end
