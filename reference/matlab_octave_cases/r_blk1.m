1;
function r = run_case__(code__)
  r = [];
  eval(code__);
  r = double(r);
end
addpath('/home/claude/dl/src72/matlab');
warning('off','all');
dynare_config;
warning('off','all');
txt = fileread('pydyn_blk1_cases.txt');
cases = strsplit(txt, sprintf('\n%%%%\n'));
results = cell(numel(cases), 1);
for k = 1:numel(cases)
  try
    results{k} = run_case__(cases{k});
  catch err
    results{k} = ['ERROR: ' err.message];
  end
end
save('-mat7-binary', 'pydyn_blk1.mat', 'results');
printf('%d cases\n', numel(cases));
