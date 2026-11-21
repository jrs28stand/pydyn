// tests/optimal_policy/nk_ramsey.mod with a second instrument (a subsidy s in the Phillips curve,
// penalised in the planner objective): two missing equations, so dyn_ramsey_static's residual of the
// multiplier problem comes from a QR factorisation
var pai, c, n, a, r, s;
varexo u;
parameters beta, rho, epsilon, omega, phi, gamma;
beta=0.99;
gamma=3;
omega=17;
epsilon=8;
phi=1;
rho=0.95;
model;
a=rho*(a(-1))+u;
1/c=beta*(1/(c(+1)))*(r/(pai(+1)));
omega*pai*(pai-1)=beta*omega*(c/(c(+1)))*(pai(+1))*(pai(+1)-1)+epsilon*exp(a)*n*(c/exp(a)*phi*n^gamma-(epsilon-1)/epsilon*(1+s));
(exp(a))*n=c+(omega/2)*((pai-1)^2);
end;
initval;
pai=1;
r=1/beta;
c=0.95;
n=0.95;
a=0;
s=0.01;
end;
shocks;
var u; stderr 0.008;
end;
planner_objective(ln(c)-phi*((n^(1+gamma))/(1+gamma))-0.5*s^2);
ramsey_model(planner_discount=0.99, instruments=(r,s));
stoch_simul(order=1,irf=0);
evaluate_planner_objective;
