% Model of An and Schorfheide (2007)
% Created by Marco Ratto (@rattoma, marco.ratto@ec.europa.eu)
% =========================================================================
% Copyright © 2010-2020 Dynare Team
%
% This file is part of Dynare.
%
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Dynare is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Dynare.  If not, see <https://www.gnu.org/licenses/>.
% =========================================================================

var pie $\pi$ y R g z YGR INFL INT;
varexo e_R $\varepsilon_R$ e_g e_z;
parameters tau $\tau$ kap $\kappa$ psi1 psi2 rhoR rhog rhoz rr_steady pi_steady gam_steady std_R std_g std_z;

tau        = 2;
kap        = 0.15;

psi1       = 1.5;
psi2       = 1;

rhoR       = 0.96;
rhog       = 0.95;
rhoz       = 0.65;

rr_steady  = 0.4;
pi_steady  = 4.0;
gam_steady = 0.5;
std_R = 0.2;
std_g = 0.8;
std_z = 0.45;

model;
y = y(+1) + g - g(+1) - 1/tau*(R-pie(+1)-z(+1));
pie = 1/(1+rr_steady/400)*pie(+1)+kap*(y-g);
R = rhoR*R(-1)+(1-rhoR)*psi1*pie+(1-rhoR)*psi2*(y-y(-1)+z)+e_R*std_R;
g = rhog*g(-1)+e_g*std_g;
z = rhoz*z(-1)+e_z*std_z;
YGR = gam_steady+100*(y-y(-1)+z);
INFL = pi_steady+400*pie;
INT = pi_steady+rr_steady+4*gam_steady+400*R;
end;

steady_state_model;
pie=0;
y=0;
R=0;
g=0;
z=0;
YGR=gam_steady;
INFL = pi_steady;
INT = pi_steady+rr_steady+4*gam_steady;
end;

estimated_params;
tau, uniform_pdf, , , 0.8, 4;
kap, uniform_pdf, , , 0.05, 0.6;
psi1, uniform_pdf, , , 1.05, 3;
psi2, uniform_pdf, , , 0.1, 1.5;
rhoR, uniform_pdf, , , 0.1, 0.9;
rhog, uniform_pdf, , , 0.3, 0.95;
rhoz, uniform_pdf, , , 0.3, 0.95;
rr_steady, uniform_pdf, , , 0.2, 2;
pi_steady, uniform_pdf, , , 0.2, 2;
gam_steady, uniform_pdf, , , 0.2, 2;
std_R, uniform_pdf, , , 0.01, 0.2;
std_g, uniform_pdf, , , 0.01, 0.2;
std_z, uniform_pdf, , , 0.01, 0.2;
end;

steady;
check;

varobs YGR INFL INT;

shocks;
var e_g;
stderr 1;
var e_R;
stderr 1;
var e_z;
stderr 1;
end;

identification(advanced=0,prior_mc=4,grid_nbr=20,no_identification_strength);
identification(advanced=1,max_dim_cova_group=2,prior_mc=6,no_identification_spectrum);
