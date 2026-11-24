// pydyn reference case: Dynare tests/pac/var-8e/example1.mod (annealing) with deterministic innovations
// --+ options: json=compute +--

var y x z v;

varexo ex ey ez ;

parameters a_y_1 a_y_2 b_y_1 b_y_2 b_x_1 b_x_2 d_y; // VAR parameters

parameters beta e_c_m c_z_1 c_z_2;               // PAC equation parameters

a_y_1 =  .2;
a_y_2 =  .3;
b_y_1 =  .1;
b_y_2 =  .4;
b_x_1 = -.1;
b_x_2 = -.2;
d_y = .5;


beta  =  .9;
e_c_m =  .1;
c_z_1 =  .7;
c_z_2 = -.3;

var_model(model_name=toto, structural, eqtags=['eq:x', 'eq:y']);

pac_model(auxiliary_model_name=toto, discount=beta, model_name=pacman);

pac_target_info(pacman);
  target v;
  auxname_target_nonstationary vns;

  component y;
  auxname pv_y_;
  kind ll;

  component x;
  auxname pv_dx_;
  kind dd;

end;

model;

  [name='eq:y']
  y = a_y_1*y(-1) + a_y_2*diff(x(-1)) + b_y_1*y(-2) + b_y_2*diff(x(-2)) + ey ;


  [name='eq:x']
  diff(x) = b_x_1*y(-2) + b_x_2*diff(x(-1)) + ex ;

  [name='eq:v']
  v = x + d_y*y ;

  [name='zpac']
  diff(z) = e_c_m*(pac_target_nonstationary(pacman)-z(-1)) + c_z_1*diff(z(-1))  + c_z_2*diff(z(-2)) + pac_expectation(pacman) + ez;

end;

shocks;
  var ex = 1.0;
  var ey = 1.0;
  var ez = 1.0;
end;

// Initialize the PAC model (build the Companion VAR representation for the auxiliary model).
pac.initialize('pacman');

// Update the parameters of the PAC expectation model (h0 and h1 vectors).
pac.update.expectation('pacman');

// Set initial conditions to zero. Please use more sensible values if any...
initialconditions = dseries(zeros(10, M_.endo_nbr+M_.exo_nbr), 2000Q1, vertcat(M_.endo_names,M_.exo_names));

// Simulate the model (deterministic innovations, close to N(0,1))
options_.bnlms.set_dynare_seed_to_default = false;
set_dynare_seed(4);
nsim__ = 800;
tt__ = (1:nsim__)';
E__ = zeros(nsim__, M_.exo_nbr);
for i__ = 1:M_.exo_nbr
    E__(:,i__) = -6;
    for m__ = 1:12
        E__(:,i__) = E__(:,i__) + mod(sin(tt__*12.9898+(i__*12+m__)*78.233)*43758.5453, 1);
    end
end
innov__ = dseries(E__, 2000Q1+10, M_.exo_names);
TrueData = simul_backward_model(initialconditions, nsim__, innov__);

clear eparams
eparams.e_c_m   =  .9;
eparams.c_z_1   =  .5;
eparams.c_z_2   =  .2;

edata = TrueData;
edata.ez = dseries(NaN(TrueData.nobs, 1), 2000Q1, 'ez');
pac.estimate.nls('zpac', eparams, edata, 2005Q1:2005Q1+600, 'annealing', 'verbosity', 0);
e_c_m_sa = M_.params(strmatch('e_c_m', M_.param_names, 'exact'));
c_z_1_sa = M_.params(strmatch('c_z_1', M_.param_names, 'exact'));
c_z_2_sa = M_.params(strmatch('c_z_2', M_.param_names, 'exact'));
