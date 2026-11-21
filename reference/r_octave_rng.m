% Regenerates reference/octave_rng.mat: numbers of Octave 8.4's rand and randn after
% rand('state', s) and randn('state', s), for tests/test_nonlinear_estimation.py (pydyn.random_streams).
%   octave --no-gui --quiet r_octave_rng.m
r = struct();
seeds = [0 1 7 123456];
idx = 1:997:200000;
for k = 1:numel(seeds)
  s = seeds(k);
  rand('state', s); u = rand(200000, 1);
  randn('state', s); x = randn(200000, 1);
  r.(sprintf('u_%d', k)) = u(idx); r.(sprintf('x_%d', k)) = x(idx);
  r.(sprintf('u_head_%d', k)) = u(1:50); r.(sprintf('x_head_%d', k)) = x(1:50);
  r.(sprintf('u_sum_%d', k)) = sum(u); r.(sprintf('x_sum_%d', k)) = sum(x);
  r.(sprintf('x_max_%d', k)) = max(abs(x));
end
r.seeds = seeds; r.idx = idx;
% interleaving: separate generators, column-major fill
rand('state', 0); randn('state', 0);
a = rand(2, 3); b = randn(3, 2); c = rand(1, 4); d = randn(5, 1);
r.a = a; r.b = b; r.c = c; r.d = d;
save('-mat7-binary', 'octave_rng.mat', '-struct', 'r');
