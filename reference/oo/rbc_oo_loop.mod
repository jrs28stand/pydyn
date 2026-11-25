// MATLAB lines that read and write M_, options_ and oo_ between commands
var y c k a;
varexo e u;
parameters alpha beta delta rho;
alpha = 0.36; beta = 0.99; delta = 0.025; rho = 0.9;

model;
c + k = exp(a)*k(-1)^alpha + (1-delta)*k(-1) + u;
1/c = beta*(1/c(+1))*(alpha*exp(a(+1))*k^(alpha-1)+1-delta);
a = rho*a(-1) + e;
y = exp(a)*k(-1)^alpha;
end;

steady_state_model;
a = 0;
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));
y = k^alpha;
c = y - delta*k;
end;

shocks;
var e; stderr 0.01;
var u; stderr 0.004;
end;

steady;
stoch_simul(order=1, irf=10, nograph, nomoments) y c;
irf0 = oo_.irfs.y_e;
ys0 = oo_.steady_state;
persist = zeros(1, 3);
rhos = [0.5 0.7 0.95];
for i = 1:3
    set_param_value('rho', rhos(i));
    [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, {'y'; 'c'});
    persist(i) = oo_.irfs.y_e(10)/oo_.irfs.y_e(1);
    store(i, :) = oo_.irfs.y_e;
end
M_.params(4) = 0.8;
options_.irf = 5;
options_.order = 1;
stoch_simul(nomoments, nograph) c k;
irf5 = oo_.irfs.c_u;
ghx_end = oo_.dr.ghx;
rho_end = M_.params(4);
eig_end = oo_.dr.eigval;
