// Example of optimal simple rule
var y inflation r dummy_var;
varexo y_ inf_;

parameters delta sigma alpha kappa gammax0 gammac0 gamma_y_ gamma_inf_;

delta =  0.44;
kappa =  0.18;
alpha =  0.48;
sigma = -0.06;


model(linear);
y  = delta * y(-1)  + (1-delta)*y(+1)+sigma *(r - inflation(+1)) + y_;
inflation  =   alpha * inflation(-1) + (1-alpha) * inflation(+1) + kappa*y + inf_;
dummy_var=0.9*dummy_var(-1)+0.01*y;
r = gammax0*y(-1)+gammac0*inflation(-1)+gamma_y_*y_+gamma_inf_*inf_;
end;

shocks;
var y_;
stderr 0.63;
var inf_;
stderr 0.4;
end;

options_.nograph=1;
options_.nocorr=1;
osr_params gammax0 gammac0 gamma_y_ gamma_inf_;


optim_weights;
inflation 1;
y 1;
dummy_var 1;
y,inflation 1;
end;
gammax0=1.35533;
gammac0=1.39664;
gamma_y_=16.6667;
gamma_inf_=9.13199;
osr(silent_optimizer,order=1);
