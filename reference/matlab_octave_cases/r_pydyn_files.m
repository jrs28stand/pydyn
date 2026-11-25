% Octave reference for tests/test_matlab_files.py: the snippets of pydyn_files_cases.txt run in a folder holding
% f.txt (one line) and sub/g.mat (one line).  Run with: octave --no-gui --quiet r_pydyn_files.m   (in this directory)
1;
function r = run_case__(code__)
  r = [];
  eval(code__);
  r = double(r);
end
txt = fileread('pydyn_files_cases.txt');
cases = strsplit(txt, sprintf('\n%%%%\n'));
here = pwd();
d = tempname(); mkdir(d); mkdir([d '/sub']);
fid = fopen([d '/f.txt'], 'w'); fprintf(fid, 'x\n'); fclose(fid);
fid = fopen([d '/sub/g.mat'], 'w'); fprintf(fid, 'y\n'); fclose(fid);
cd(d);
results = cell(numel(cases), 1);
for k = 1:numel(cases)
  try
    results{k} = run_case__(cases{k});
  catch err
    results{k} = ['ERROR: ' err.message];
  end
end
cd(here);
save('-mat7-binary', 'pydyn_files.mat', 'results');
printf('%d cases\n', numel(cases));
