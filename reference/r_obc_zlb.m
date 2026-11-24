%% Reference for the news-shock method on zlb.mod, from Dynare 7.2 under Octave.
% M and q come from Dynare's own first-order solution of the model with news shocks added by hand;
% the LCP is solved with GLPK using the constraint set of DynareOBC's FormOptimizer.m (Holden);
% the OccBin path comes from Dynare's own occbin_solver on the original file.
addpath /home/claude/ref/dynare-7.2/matlab
T = 20; N = 60; shock = -0.05; rbar = -0.01; omega = 1000;
dynare obc_zlb_news nograph
M = zeros(N, T);
for k = 0:T-1
    M(:, k+1) = oo_.irfs.(sprintf('r_w%d', k))(:);
end
q = (oo_.dr.ys(strmatch('r', M_.endo_names, 'exact')) - rbar) + shock * oo_.irfs.r_e(:);
% homogenised MILP: max alpha s.t. 0<=yn<=z, alpha*q + M*yn >= 0, alpha*q_s + M_s*yn <= omega*(1-z)
qn = q / norm(q, Inf);
nv = 2*T + 1;                       % [yn; alpha; z]
c = zeros(nv, 1); c(T+1) = 1;
A = [ eye(T), zeros(T,1), -eye(T);
      M, qn, zeros(N, T);
      M(1:T,:), qn(1:T), omega*eye(T) ];
b = [ zeros(T,1); zeros(N,1); omega*ones(T,1) ];
ctype = [ repmat('U', 1, T), repmat('L', 1, N), repmat('U', 1, T) ];
vartype = [ repmat('C', 1, T+1), repmat('I', 1, T) ];
lb = zeros(nv, 1); ub = [ ones(T,1); 1e8; ones(T,1) ];
param.msglev = 0; param.presol = 0;   % with GLPK's LP presolver on (Octave's default) this problem comes back 'optimal' but infeasible
[x, fval, errnum, extra] = glpk(c, A, b, lb, ub, ctype, vartype, -1, param);
printf('REF glpk status=%d z=%s\n', extra.status, mat2str(round(x(T+2:end))'));
alpha = x(T+1); y = x(1:T) / alpha * norm(q, Inf);
g = q + M*y;
r_news = rbar + g;
lcp_ok = all(y >= -1e-9) && all(g >= -1e-9) && max(abs(min(y, g(1:T)))) < 1e-9;
ys_news = oo_.dr.ys;
clear M_ oo_ options_
dynare obc_zlb nograph
ir = strmatch('r', M_.endo_names, 'exact');
r_occbin = oo_.occbin.simul.piecewise(:, ir);
regimestart = oo_.occbin.simul.regime_history(1).regimestart;
r = struct('M', M, 'q', q, 'y', y, 'alpha', alpha, 'r_news', r_news, 'r_occbin', r_occbin, ...
           'regimestart', regimestart, 'lcp_ok', double(lcp_ok), 'glpk_errnum', errnum, 'T', T, 'N', N, 'shock', shock);
save('-mat7-binary', 'obc_zlb.mat', '-struct', 'r');
printf('REF lcp_ok=%d errnum=%d alpha=%g binding periods=%d max|r_news-r_occbin|=%.3g\n', lcp_ok, errnum, alpha, sum(y>1e-10), max(abs(r_news(1:30)-r_occbin(1:30))));
