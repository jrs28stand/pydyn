// simul_backward_model: nonlinear backward model (Solow, Cobb-Douglas), initial conditions from histval or a dseries,
// innovations drawn from Sigma_e or given
var Efficiency EfficiencyGrowth Population PopulationGrowth Output PhysicalCapitalStock;
varexo e_x e_n;
parameters alpha delta s rho_x rho_n EfficiencyGrowth_ss PopulationGrowth_ss;
alpha = .33; delta = .02; s = .20; rho_x = .90; rho_n = .95;
EfficiencyGrowth_ss = 1.02; PopulationGrowth_ss = 1.02;
model;
    Efficiency = EfficiencyGrowth*Efficiency(-1);
    EfficiencyGrowth/EfficiencyGrowth_ss = (EfficiencyGrowth(-1)/EfficiencyGrowth_ss)^(rho_x)*exp(e_x);
    Population = PopulationGrowth*Population(-1);
    PopulationGrowth/PopulationGrowth_ss = (PopulationGrowth(-1)/PopulationGrowth_ss)^(rho_n)*exp(e_n);
    Output = PhysicalCapitalStock(-1)^alpha*(Efficiency*Population)^(1-alpha);
    PhysicalCapitalStock = (1-delta)*PhysicalCapitalStock(-1) + s*Output;
end;
histval;
    Efficiency(0) = 1; EfficiencyGrowth(0) = 1.02; Population(0) = 1; PopulationGrowth(0) = 1.02; PhysicalCapitalStock(0) = 1;
end;
shocks;
    var e_x = 0.005;
    var e_n = 0.001;
end;
// 1. histval, shocks drawn from Sigma_e
[simulations, flag1] = simul_backward_model([], 40);
sim1 = simulations.data;
names1 = simulations.name;
endo1 = oo_.endo_simul;
exo1 = oo_.exo_simul;
bn1 = oo_.bnlms.shocks;
x1 = randn(2, 2);   // the stream after the call (the draws were made from the default seed)
// 2. histval, given innovations
ein = dseries([0.01*sin((1:20)'), 0.02*cos((1:20)')], '2Y', {'e_x'; 'e_n'});
s2 = simul_backward_model([], 20, ein);
sim2 = s2.data;
endo2 = oo_.endo_simul;
exo2 = oo_.exo_simul;
// 3. a dseries as initial conditions, contiguous innovations, solve_algo 9
ic = dseries(sim1(41,1:6), '2020Q4', M_.endo_names);
ei = dseries([0.01*cos((1:15)'), -0.02*sin((1:15)')], '2021Q1', M_.exo_names);
options_.solve_algo = 9;
s3 = simul_backward_model(ic, 15, ei);
sim3 = s3.data;
last3 = s3.dates.ndat;
// 4. the explicit form, with the structs as arguments: the globals are not touched
oo_.endo_simul = 7;
[s4, oo4, flag4] = simul_backward_nonlinear_model([], 12, options_, M_, oo_);
sim4 = s4.data;
endo4 = oo4.endo_simul;
untouched = oo_.endo_simul;
sim4b = s4.Output.data;
sim4c = s4{'Output','Population'}.data;
ratio = s4.Output/s4.Population;
ratio4 = ratio.data;
sub4 = s4.Output(s4.dates(3:6));
sub4d = sub4.data;
prod4 = (s4.Output*2 + 1).data;
neg4 = (-s4.Output).data;
clear simulations s2 s3 s4 ein ic ei ratio sub4;
