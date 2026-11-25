// solve_algo 14 and 12 (Dynare's block solvers; pydyn's whole-system solvers agree to the solvers' tolerance, 1e-4)
// simul_backward_model: variables in level, in log and log-differentiated (diff(log(z(-1))): auxiliary variables of type 8
// and 9), initial conditions drawn from rand, solve_algo 4, 9, 12, 14
var x1, y1, z1, t1, s1;
parameters pp;
pp = 1;
varexo e_x1, e_y1, e_z1, e_t1, e_s1;
model;
x1 = 0.1*x1(-1) + 0.2*log(y1(-1)) + 0.3*diff(log(z1(-1))) + e_x1;
[name = 'y']
log(y1) = 0.4*x1(-1) + 0.5*log(y1(-1)) + 0.6*diff(log(z1(-1))) + e_y1;
[name = 'z']
diff(log(z1)) = -0.8*(log(z1(-1)) - log(0.5)) + 0.1*x1(-1) + 0.2*log(y1(-1)) - 0.3*diff(log(z1(-1))) + e_z1;
t1 = x1 + 2*log(y1) + 3*diff(log(z1)) + s1 + e_t1;
s1 = 4*x1(-1) + 3*t1 + e_s1;
end;
initialconditions = dseries(rand(2, 5), '2021Q3', {'x1', 'y1', 'z1', 't1', 's1'});
innovations = dseries(randn(25, 5), '2022Q1', {'e_x1', 'e_y1', 'e_z1', 'e_t1', 'e_s1'});
ic_data = initialconditions.data;
inn_data = innovations.data;
sA = simul_backward_model(initialconditions, 25, innovations);
dA = sA.data;
endoA = oo_.endo_simul;
options_.solve_algo = 14;
sC = simul_backward_model(initialconditions, 25, innovations);
dC = sC.data;
options_.solve_algo = 12;
sD = simul_backward_model(initialconditions, 25, innovations);
dD = sD.data;
clear sA sC sD initialconditions innovations;
