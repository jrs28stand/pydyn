/*
OccBin options test, two constraints: the rbc_occbin model of Dynare's examples (copyright Dynare Team,
GPL-3.0-or-later) with parameters and shocks for which the regime guesses cycle, and error_bind / error_relax
expressions in the occbin_constraints block that change which guess the periodic solution picks.
*/

var A           $A$         (long_name='TFP')
    C           $C$         (long_name='consumption')
    Invest      $I$         (long_name='investment')
    K           $K$         (long_name='capital')
    Lambda      $\lambda$   (long_name='Lagrange multiplier')
    log_K       ${\hat K}$  (long_name='log capital')
    log_Invest  ${\hat I}$  (long_name='log investment')
    log_C       ${\hat C}$  (long_name='log consumption')
    ;

varexo epsilon $\varepsilon$        (long_name='TFP shock');

parameters DUMMY, alpha    $\alpha$        (long_name='capital share')
        delta       $\delta$        (long_name='depreciation')
        beta        $\beta$         (long_name='discount factor')
        sigma       $\sigma$        (long_name='risk aversion')
        rho         $\rho$          (long_name='autocorrelation TFP')
        phi         $\phi$          (long_name='irreversibility fraction of steady state investment')
        psi         $\psi$          (long_name='capital adjustment cost')
        ;

beta=0.96;
alpha=0.33;
delta=0.10;
sigma=2;
rho = 0.5028849753522318;
phi = 0.9538230658935829;
psi = 4.038590202884413;
DUMMY = 1;   // unused: lets the reference script force a fresh solve


model;
// 1.
[name='Euler', bind = 'INEG']
-C^(-sigma)*(1+2*psi*(K/K(-1)-1)/K(-1))+ beta*C(+1)^(-sigma)*((1-delta)-2*psi*(K(+1)/K-1)*
  (-K(+1)/K^2)+alpha*exp(A(+1))*K^(alpha-1))= -Lambda+beta*(1-delta)*Lambda(+1);

[name='Euler', relax = 'INEG']
-C^(-sigma) + beta*C(+1)^(-sigma)*(1-delta+alpha*exp(A(+1))*K^(alpha-1))= -Lambda+beta*(1-delta)*Lambda(+1);

// 2.
[name='Budget constraint',bind = 'INEG']
C+K-(1-delta)*K(-1)+psi*(K/K(-1)-1)^2=exp(A)*K(-1)^(alpha);

[name='Budget constraint',relax = 'INEG']
C+K-(1-delta)*K(-1)=exp(A)*K(-1)^(alpha);

// 3.
[name='LOM capital']
Invest = K-(1-delta)*K(-1);

// 4.
[name='investment',bind='IRR,INEG']
(log_Invest - log(phi*steady_state(Invest))) = 0;
[name='investment',relax='IRR']
Lambda=0;
[name='investment',bind='IRR',relax='INEG']
(log_Invest - log(phi*steady_state(Invest))) = 0;

// 5.
[name='LOM TFP']
A = rho*A(-1)+epsilon;

// Definitions
[name='Definition log capital']
log_K=log(K);
[name='Definition log consumption']
log_C=log(C);
[name='Definition log investment']
log_Invest=log(Invest);
end;

occbin_constraints;
name 'IRR'; bind log_Invest-log(steady_state(Invest))<log(phi); relax Lambda<0; error_relax 1+abs(Lambda);
name 'INEG'; bind log_Invest-log(steady_state(Invest))<-0.000001; error_bind abs(log_Invest-log(steady_state(Invest)))*1000;
end;

steady_state_model;
K = ((1/beta-1+delta)/alpha)^(1/(alpha-1));
C = -delta*K +K^alpha;
Invest = delta*K;
log_K = log(K);
log_C = log(C);
log_Invest = log(Invest);
Lambda = 0;
A=0;
end;

shocks;
  var epsilon; stderr 0.015;
end;

steady;


shocks(surprise);
var epsilon;
periods 1, 2, 3;
values 0.278665132869575, -0.4631084100053847, -0.09791875946873171;
end;

occbin_setup(simul_periods=80);
occbin_solver(simul_periodic_solution);
