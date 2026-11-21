% Regenerates the forecast reference files (reference/forecast/fc_*.mat) with Dynare 7.2 under Octave 8.4, for
% tests/test_forecast.py. Run in a fresh directory holding the .mod file(s) (and data files), one case per
% directory:   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "CASE='fc_ar1'; r_forecast"
% The .mod files save the structs they want (names listed in SAVE) in the workspace; Dynare's graphics are off.
addpath /home/claude/dl/src72/matlab
dynare_config();
for f = {'dyn_figure', 'subplot', 'figure'}
  fid = fopen([f{1} '.m'], 'w'); fprintf(fid, 'function h = %s(varargin)\nh = 1;\nend\n', f{1}); fclose(fid);
end
for f = {'dyn_saveas', 'hold', 'xlim', 'ylim', 'title', 'drawnow', 'box', 'legend', 'xlabel', 'ylabel', 'grid', 'axis'}
  fid = fopen([f{1} '.m'], 'w'); fprintf(fid, 'function %s(varargin)\nend\n', f{1}); fclose(fid);
end
fid = fopen('plot.m', 'w'); fprintf(fid, 'function h = plot(varargin)\nh = 1;\nend\n'); fclose(fid);
rehash;
if ~exist('ARGS', 'var'), ARGS = ''; end
eval(sprintf('dynare %s nograph %s', CASE, ARGS));
r = struct();
names = strsplit(SAVE, ',');
for k = 1:numel(names)
  r.(names{k}) = eval(names{k});
end
save('-mat7-binary', [CASE '.mat'], '-struct', 'r');
disp('FORECAST_DONE')
