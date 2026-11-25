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
EfficiencyGrowth_ss = 1.02;
PopulationGrowth_ss = 1.02;

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
[simulation__, errorflag] = simul_backward_nonlinear_model([], 10, options_, M_, oo_);
initialcondition = dseries(simulation__.data(10,1:M_.endo_nbr), '2017Q1', M_.endo_names);
listofvariables = {'Efficiency', 'Output'};
forecasts = backward_model_forecast(initialcondition, listofvariables, 16, true);
f_pt = forecasts.pointforecast.data; f_mean = forecasts.meanforecast.data; f_med = forecasts.medianforecast.data;
f_std = forecasts.stdforecast.data; f_lb = forecasts.lb.data; f_ub = forecasts.ub.data;
f_names = forecasts.lb.name;
f2 = backward_model_forecast(initialcondition);
f2_pt = f2.pointforecast.data;
x1 = randn(2,2);
