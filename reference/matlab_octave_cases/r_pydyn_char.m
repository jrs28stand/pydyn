% Octave reference for pydyn tests ("extra semantics"): every snippet of pydyn_char_cases.txt
% (separated by lines holding only %%) is run and its variable r saved to pydyn_char.mat; a snippet
% that fails in Octave is stored as 'ERROR: <message>'.
% Run with:  octave --no-gui --quiet r_pydyn_char.m    (in this directory)
1;
function r = run_case__(code__)
  r = [];
  eval(code__);
  r = double(r);
end
txt = fileread('pydyn_char_cases.txt');
cases = strsplit(txt, sprintf('\n%%%%\n'));
results = cell(numel(cases), 1);
for k = 1:numel(cases)
  try
    results{k} = run_case__(cases{k});
  catch err
    results{k} = ['ERROR: ' err.message];
  end
end
save('-mat7-binary', 'pydyn_char.mat', 'results');
printf('%d cases\n', numel(cases));
