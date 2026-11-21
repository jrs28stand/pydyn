// Adapted in 2026 for pydyn's validation from Dynare 7.2 tests/occbin/model_irrcap_twoconstraints
// (Copyright Dynare Team, GPL-3.0-or-later).
// OccBin smoother/likelihood reference with TWO constraints: Dynare 7.2 tests/occbin/model_irrcap_twoconstraints
// (dynrbc.mod + dynrbc_common.inc, option 1: data for c simulated with OccBin; Dynare Team, GPL-3.0-or-later).
// variables
var a, c, i, k, lambdak;

// innovations to shock processes
varexo erra;


// parameters
parameters ALPHA, DELTAK, BETA, GAMMAC, RHOA, PHI, PSI;

model;

# zkss = ((1/BETA-1+DELTAK)/ALPHA)^(1/(ALPHA-1));
# zcss = -DELTAK*zkss + zkss^ALPHA;
# ziss = DELTAK*zkss;
# zuss = (zcss^(1-GAMMAC)-1)/(1-GAMMAC);
# zvss = zuss/(1-BETA);

/////////////////////////////////////////////////////////////////
// 1.
[name='Euler', bind = 'INEG']
-exp(c)^(-GAMMAC)*(1+2*PSI*(exp(k)/exp(k(-1))-1)/exp(k(-1)))
+ BETA*exp(c(1))^(-GAMMAC)*((1-DELTAK)-2*PSI*(exp(k(1))/exp(k)-1)*
  (-exp(k(1))/exp(k)^2)+ALPHA*exp(a(1))*exp(k)^(ALPHA-1))=
  -lambdak+BETA*(1-DELTAK)*lambdak(1);

[name='Euler', relax = 'INEG']
-exp(c)^(-GAMMAC) + BETA*exp(c(1))^(-GAMMAC)*(1-DELTAK+ALPHA*exp(a(1))*exp(k)^(ALPHA-1))=
  -lambdak+BETA*(1-DELTAK)*lambdak(1);

// 2.
[name='Budget constraint',bind = 'INEG']
exp(c)+exp(k)-(1-DELTAK)*exp(k(-1))+PSI*(exp(k)/exp(k(-1))-1)^2=exp(a)*exp(k(-1))^(ALPHA);

[name='Budget constraint',relax = 'INEG']
exp(c)+exp(k)-(1-DELTAK)*exp(k(-1))=exp(a)*exp(k(-1))^(ALPHA);

// 3.
exp(i) = exp(k)-(1-DELTAK)*exp(k(-1));

// 4.
[name='investment',bind='IRR,INEG']
(i - log(PHI*ziss)) = 0;
[name='investment',relax='IRR']
lambdak=0;
[name='investment',bind='IRR',relax='INEG']
(i - log(PHI*ziss)) = 0;

// 5.
a = RHOA*a(-1)+erra;
end;

occbin_constraints;
name 'IRR'; bind i-steady_state(i)<log(PHI); relax lambdak<0;
name 'INEG'; bind i-steady_state(i)<-0.000001;
end;
steady_state_model;
kss = ((1/BETA-1+DELTAK)/ALPHA)^(1/(ALPHA-1));
css = -DELTAK*kss +kss^ALPHA;
iss = DELTAK*kss;


k = log(kss);
c = log(css);
i = log(iss);
lambdak = 0;
a=0;
end;

BETA=0.96;
ALPHA=0.33;
DELTAK=0.10;
GAMMAC=2;
RHOA = 0.9;
PHI = 0.975;
PSI = 5;        % adjustment cost for capital if investment is negative

shocks;
  var erra; stderr 0.015;
end;

steady;
shocks(surprise,overwrite);
var erra;
periods 1:9, 10, 50, 90, 130, 131:169;
values -0.0001, -0.01,-0.02, 0.01, 0.02, 0;
end;

occbin_setup;
occbin_solver(simul_periods=200,simul_maxit=200,simul_curb_retrench,simul_check_ahead_periods=200);

varobs c;
estimated_params;
RHOA, beta_pdf, 0.9, 0.05;
stderr erra, inv_gamma_pdf, 0.015, inf;
end;
occbin_setup(smoother_periods=200,smoother_maxit=200,smoother_curb_retrench,smoother_check_ahead_periods=200);
