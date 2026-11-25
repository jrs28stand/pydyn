@#define bare_first_info_period = false
@#define dates = false
/* Tests perfect_foresight_with_expectation_errors_{setup,solver}
   using the shocks(learnt_in=…), mshocks(learnt_in=…) and endval(learnt_in=…) syntax
   with dates (instead of integer indices) for periods */

var c k;
varexo x;

parameters alph gam delt bet aa;
alph=0.5;
gam=0.5;
delt=0.02;
bet=0.05;
aa=0.5;


model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;

initval;
x = 1;
k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
c = aa*k^alph-delt*k;
end;

steady;

check;

// p contains period indices, either as integer or as dates
@#if dates
@# define p = ["2018Y", "2019Y", "2020Y", "2021Y", "2022Y", "2023Y", "2024Y", "2025Y" ]
@#else
@# define p = 1:8
@#endif

@#if bare_first_info_period
shocks;
@#else
shocks(learnt_in = @{p[1]});
@#endif
  var x;
  periods @{p[1]}:@{p[2]};
  values 1.2;
end;

@#if bare_first_info_period
endval;
@#else
endval(learnt_in = @{p[1]});
@#endif
  x = 1.05;
end;

shocks(learnt_in = @{p[2]});
  var x;
  periods @{p[2]};
  add 0.1;
end;

endval(learnt_in = @{p[2]});
  x = 1.1;
end;

shocks(learnt_in = @{p[3]});
  var x;
  periods @{p[3]};
  values 1.4;
end;

mshocks(learnt_in = @{p[3]});
  var x;
  periods @{p[7]};
  values (1.5/1.2); // 1.2 is the terminal steady as anticipated in period 3
end;

endval(learnt_in = @{p[3]});
  x += 0.1;
end;

// Dummy block, that will be overwritten by the next one
shocks(learnt_in = @{p[6]});
  var x;
  periods @{p[6]}:@{p[8]};
  values 10;
end;

shocks(learnt_in = @{p[6]}, overwrite);
  var x;
  periods @{p[6]}:@{p[7]};
  multiply 0.8;
end;

endval(learnt_in = @{p[6]});
  x *= 0.75;
end;

// Save initial steady state (it will be modified by pfwee)
orig_steady_state = oo_.steady_state;
orig_exo_steady_state = oo_.exo_steady_state;

perfect_foresight_with_expectation_errors_setup(
@#if dates
first_simulation_period = 2018Y, last_simulation_period = 2024Y
@#else
periods = 7
@#endif
);

perfect_foresight_with_expectation_errors_solver;
pfwee_simul = oo_.endo_simul;


zz_endo = oo_.endo_simul; zz_exo = oo_.exo_simul;
