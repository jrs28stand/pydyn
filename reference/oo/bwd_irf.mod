close all

var Efficiency $A$
    EfficiencyGrowth $X$
    Population $L$
    PopulationGrowth $N$
    Output $Y$
    PhysicalCapitalStock $K$ ;

varexo e_x $\varepsilon_x$
       e_n $\varepsilon_n$;

parameters alpha $\alpha$
	   delta $\delta$
	   s $s$
           rho_x $\rho_x$
           rho_n $\rho_n$
           EfficiencyGrowth_ss $X^{\star}$
           PopulationGrowth_ss $N^{\star}$ ;

alpha = .33;
delta = .02;
s     = .20;
rho_x = .90;
rho_n = .95;
EfficiencyGrowth_ss = 1.0;
PopulationGrowth_ss = 1.0;

model;
    Efficiency = EfficiencyGrowth*Efficiency(-1);
    EfficiencyGrowth/EfficiencyGrowth_ss = (EfficiencyGrowth(-1)/EfficiencyGrowth_ss)^(rho_x)*exp(e_x);
    Population = PopulationGrowth*Population(-1);
    PopulationGrowth/PopulationGrowth_ss = (PopulationGrowth(-1)/PopulationGrowth_ss)^(rho_n)*exp(e_n);
    Output = PhysicalCapitalStock(-1)^alpha*(Efficiency*Population)^(1-alpha);
    PhysicalCapitalStock = (1-delta)*PhysicalCapitalStock(-1) + s*Output;
end;


shocks;
    var e_x = 0.005;
    var e_n = 0.001;
end;

histval;
  Efficiency(0) = 1;
  EfficiencyGrowth(0) = .5;
  Population(0) = 1;
  PopulationGrowth(0) = .5;
  PhysicalCapitalStock(0) = 1;
end;


listofshocks = {'e_x', 'e_n'};
listofvariables = {'Efficiency', 'Population', 'Output'};
[irfs1, base1, lev1] = backward_model_irf([], dseries(), listofshocks, listofvariables, 20);
a1 = irfs1.e_x.data; a2 = irfs1.e_n.data; nm = irfs1.e_x.name; b1 = base1.data; l1 = lev1.e_n.data;
x0 = zeros(20,2);
x0_1 = x0; x0_2 = x0;
x0_1(1,1) = sqrt(M_.Sigma_e(1,1));
x0_2(1,2) = 2*sqrt(M_.Sigma_e(2,2));
x0_2(3,1) = 0.01;
simshocks = {dseries(x0_1, 2, M_.exo_names), dseries(x0_2, 2, M_.exo_names)};
irfs2 = backward_model_irf([], dseries(), simshocks, listofvariables, 20);
c1 = irfs2.experiment_1.data; c2 = irfs2.experiment_2.data;
// initial condition from a dseries and set_historical_values
ds = dseries(repmat([1, .5, 1, .5, 0, 1], 10, 1)*1.01, '1990Q1', M_.endo_names);
set_historical_values ds 1991Q1;
eh = M_.endo_histval; xh = M_.exo_histval;
irfs3 = backward_model_irf(ds, dseries(), {'e_x'}, {}, 10);
d1 = irfs3.e_x.data; d1n = irfs3.e_x.name;
