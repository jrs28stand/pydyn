% Octave reference for test_matlab.jl ("extra semantics"): every snippet of matlab_extra_cases.txt
% (separated by lines holding only %%) is run and its variable r saved to matlab_extra.mat; a snippet
% that fails in Octave is stored as 'ERROR: <message>'.
% Run with:  octave --no-gui --quiet r_matlab_extra.m    (in this directory)
1;
function r = run_case__(code__)
  r = [];
  eval(code__);
  r = double(r);
end
txt = fileread('matlab_extra_cases.txt');
cases = strsplit(txt, sprintf('\n%%%%\n'));
results = cell(numel(cases), 1);
for k = 1:numel(cases)
  try
    results{k} = run_case__(cases{k});
  catch err
    results{k} = ['ERROR: ' err.message];
  end
end
save('-mat7-binary', 'matlab_extra.mat', 'results');
printf('%d cases\n', numel(cases));
