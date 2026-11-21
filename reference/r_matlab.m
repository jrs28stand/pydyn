% Reference values for tests/test_matlab.py: every snippet of mat_cases.txt (separated by lines
% holding only %%) is run in Octave and its variable r is saved to mat_semantics.mat.
% Run with:  octave --no-gui --quiet r_matlab.m    (in this directory)
1;  % script file (not a function file)
function r = run_case__(code__)
  r = [];
  eval(code__);
  r = double(r);
end
txt = fileread('mat_cases.txt');
cases = strsplit(txt, sprintf('\n%%%%\n'));
results = cell(numel(cases), 1);
for k = 1:numel(cases)
  try
    results{k} = run_case__(cases{k});
  catch err
    results{k} = ['ERROR: ' err.message];
  end
end
save('-mat7-binary', 'mat_semantics.mat', 'results');
printf('%d cases\n', numel(cases));

% ---- mat_flow.mat: Dynare's calibration after running mat_flow.mod (set DYNARE_MATLAB to Dynare's matlab folder)
if ~isempty(getenv('DYNARE_MATLAB'))
  addpath(getenv('DYNARE_MATLAB'));
  dynare mat_flow nograph
  r = struct('params', M_.params, 'Sigma_e', M_.Sigma_e, 'x1', x1, 'x2', x2, 'x3', x3, 'w', w, 'k0', k0);
  r.param_names = strjoin(cellstr(M_.param_names)', ',');
  save('-mat7-binary', 'mat_flow.mat', '-struct', 'r');
end
% mat_corpus.mat is written by tools/sweep_matlab.py (collect, octave, compare, refs).
