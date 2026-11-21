% Regenerates reference/dates/dates_ops.mat (Dynare 7.2 dseries dates arithmetic) for tests/test_dates.py.
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_dates.m
addpath /home/claude/dl/src72/matlab
dynare_config();
s = {'1990Y','2001a','1990S1','1990H2','1990Q1','1999q4','2000M1','2000m12','1990W1','2015W53','2020W10','1990-01-31','2000-02-29'};
k = [-7 -1 0 1 5 13 60];
r = struct();
r.strings = strjoin(s, ',');
r.freq = zeros(1, numel(s));
r.plus = cell(numel(s), numel(k));
for i = 1:numel(s)
  d = dates(s{i});
  r.freq(i) = d.freq;
  for j = 1:numel(k)
    r.plus{i,j} = char(d + k(j));
  end
end
r.k = k;
% differences between pairs of the same frequency
pairs = {'2000Q1','1990Q3'; '1990Q3','2000Q1'; '2001M3','1999M11'; '2010Y','1990Y'; '2001S1','1995S2'; '2016W1','2015W50'; '2001-03-01','2000-02-27'};
r.pairs = strjoin(pairs(:)', ',');
r.diff = zeros(1, rows(pairs));
for i = 1:rows(pairs), r.diff(i) = dates(pairs{i,1}) - dates(pairs{i,2}); end
r.plus = strjoin(r.plus(:)', ',');
save('-mat7-binary', 'dates_ops.mat', '-struct', 'r');
disp('DATES_DONE')
