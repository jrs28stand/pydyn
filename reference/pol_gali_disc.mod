/*
 * This file implements the baseline New Keynesian model of Jordi Gal� (2008): Monetary Policy, Inflation,
 * and the Business Cycle, Princeton University Press, Chapter 5
 *
 * This implementation was written by Johannes Pfeifer.
 *
 * Please note that the following copyright notice only applies to this Dynare
 * implementation of the model.
 */

/*
 * Copyright © 2013-21 Johannes Pfeifer
 *
 * This is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * It is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * For a copy of the GNU General Public License,
 * see <https://www.gnu.org/licenses/>.
 */


var pi
    y_gap
    r_e
    y_e
    r_nat
    i
    u
    a
    p
    ;

varexo eps_a
       eps_u;

parameters alppha
    betta
    rho_a
    rho_u
    siggma
    phi
    phi_y
    eta
    epsilon
    theta
    ;
%----------------------------------------------------------------
% Parametrization, p. 52
%----------------------------------------------------------------
siggma = 1;
phi=1;
phi_y  = .5/4;
theta=2/3;
rho_u = 0;
rho_a  = 0.9;
betta = 0.99;
eta  =4;
alppha=1/3;
epsilon=6;



%----------------------------------------------------------------
% First Order Conditions
%----------------------------------------------------------------

model(linear);
//Composite parameters
#Omega=(1-alppha)/(1-alppha+alppha*epsilon);  //defined on page 47
#psi_n_ya=(1+phi)/(siggma*(1-alppha)+phi+alppha); //defined on page 48
#lambda=(1-theta)*(1-betta*theta)/theta*Omega; //defined on page 47
#kappa=lambda*(siggma+(phi+alppha)/(1-alppha));  //defined on page 49
#alpha_x=kappa/epsilon; //defined on page 96
#phi_pi=(1-rho_u)*kappa*siggma/(alpha_x)+rho_u; //defined on page 101

r_e=siggma*(y_e(+1)-y_e);
y_e=psi_n_ya*a;
pi=betta*pi(+1)+kappa*y_gap + u;
y_gap=-1/siggma*(i-pi(+1)-r_e)+y_gap(+1);
//3. Interest Rate Rule eq. (25)
% i=r_e+phi_pi*pi;

r_nat=siggma*psi_n_ya*(a(+1)-a);
u=rho_u*u(-1)+eps_u;
a=rho_a*a(-1)+eps_a;

pi=p-p(-1);
end;

%----------------------------------------------------------------
%  define shock variances
%---------------------------------------------------------------


shocks;
var eps_u = 1;
end;

planner_objective pi^2 +(((1-theta)*(1-betta*theta)/theta*((1-alppha)/(1-alppha+alppha*epsilon)))*(siggma+(phi+alppha)/(1-alppha)))/epsilon*y_gap^2;

discretionary_policy(instruments=(i),irf=20,planner_discount=betta,discretionary_tol=1e-12) y_gap pi p u;
