% Octave reference of the figures Dynare draws: Dynare 7.2 is run on <case>.mod WITHOUT `nograph` against recording stubs of the
% graphics functions (this Octave has no graphics toolkit).  Every drawing call is stored with the number of the current figure;
% dyn_saveas records the file stem and formats.  Output: <case>_fig.mat (cell array ZZG.calls of {kind, fig, args...}).
%   usage (from a scratch copy of this folder with the case's .mod and data files):
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "addpath('/home/claude/dl/src72/matlab'); ZZ={'case'}; r_fig"
global ZZG
zz_here = pwd;
zz_stubs = [tempname '_stubs']; mkdir(zz_stubs);
zz_rec = {'subplot', 'plot', 'bar', 'barh', 'area', 'fill', 'title', 'legend', 'xlim', 'ylim', 'text', 'xlabel', 'ylabel', 'errorbar'};
for zz_k = 1:numel(zz_rec)
    zz_fid = fopen(fullfile(zz_stubs, [zz_rec{zz_k} '.m']), 'w');
    fprintf(zz_fid, 'function varargout = %s(varargin)\nglobal ZZG\nZZG.calls{end+1} = [{''%s'', ZZG.fig}, varargin];\nfor i = 1:max(nargout,1), varargout{i} = 1; end\nend\n', zz_rec{zz_k}, zz_rec{zz_k});
    fclose(zz_fid);
end
zz_fid = fopen(fullfile(zz_stubs, 'dyn_figure.m'), 'w');
fprintf(zz_fid, 'function h = dyn_figure(nodisplay, varargin)\nglobal ZZG\nZZG.fig = ZZG.fig + 1; nm = ''''; for i=1:2:numel(varargin)-1, if ischar(varargin{i}) && strcmp(varargin{i}, ''Name''), nm = varargin{i+1}; end, end\nZZG.names{ZZG.fig} = nm; h = ZZG.fig;\nend\n');
fclose(zz_fid);
zz_fid = fopen(fullfile(zz_stubs, 'figure.m'), 'w');
fprintf(zz_fid, 'function h = figure(varargin)\nglobal ZZG\nif nargin>0 && isnumeric(varargin{1}), h = varargin{1}; return, end\nZZG.fig = ZZG.fig + 1; nm = ''''; for i=1:2:numel(varargin)-1, if ischar(varargin{i}) && strcmp(varargin{i}, ''Name''), nm = varargin{i+1}; end, end\nZZG.names{ZZG.fig} = nm; h = ZZG.fig;\nend\n');
fclose(zz_fid);
zz_fid = fopen(fullfile(zz_stubs, 'dyn_saveas.m'), 'w');
fprintf(zz_fid, 'function dyn_saveas(h, fname, nodisplay, graph_format)\nglobal ZZG\nZZG.saved{end+1} = {h, fname, strjoin(cellstr(graph_format), ''|'')};\nend\n');
fclose(zz_fid);
zz_noop = {'hold', 'set', 'gca', 'gcf', 'axis', 'grid', 'box', 'colormap', 'drawnow', 'close', 'annotation', 'orient', 'pause', 'uimenu', ...
           'uicontextmenu', 'saveas', 'remove_fractional_xticks', 'linkaxes', 'line', 'patch', 'refresh', 'clf', 'axes', 'groot', 'datetick', ...
           'xticks', 'yticks', 'xticklabels', 'yticklabels', 'colorbar', 'zlabel', 'ishandle', 'isgraphics', 'findobj', 'savefig', 'waitbar'};
for zz_k = 1:numel(zz_noop)
    zz_fid = fopen(fullfile(zz_stubs, [zz_noop{zz_k} '.m']), 'w');
    fprintf(zz_fid, 'function varargout = %s(varargin)\nfor i = 1:max(nargout,1), varargout{i} = 1; end\nend\n', zz_noop{zz_k});
    fclose(zz_fid);
end
zz_fid = fopen(fullfile(zz_stubs, 'get.m'), 'w');
fprintf(zz_fid, 'function varargout = get(varargin)\nfor i = 1:max(nargout,1), varargout{i} = [1 1 1920 1080]; end; if nargin>1 && ischar(varargin{2}) && strcmpi(varargin{2},''color''), varargout{1} = {[0 0.4 0.8];[0.8 0 0]}; end\nend\n');
fclose(zz_fid);
zz_fid = fopen(fullfile(zz_stubs, 'print.m'), 'w');
fprintf(zz_fid, 'function print(varargin)\nglobal ZZG\nZZG.saved{end+1} = {varargin{1}, varargin{end}};\nend\n');
fclose(zz_fid);
addpath(zz_stubs, '-begin');
for zz_i = 1:numel(ZZ)
    zz_c = ZZ{zz_i};
    clear global M_ oo_ options_ estim_params_ bayestopt_ dataset_ dataset_info;
    ZZG = struct('fig', 0, 'calls', {{}}, 'names', {{}}, 'saved', {{}});
    zz_d = tempname; mkdir(zz_d); copyfile([zz_here '/' zz_c '.mod'], zz_d);
    zz_aux = dir([zz_here '/*.m']); for zz_k = 1:numel(zz_aux), if ~strcmp(zz_aux(zz_k).name, 'r_fig.m'), copyfile([zz_here '/' zz_aux(zz_k).name], zz_d); end, end
    zz_aux = dir([zz_here '/' zz_c '_*.mat']); for zz_k = 1:numel(zz_aux), copyfile([zz_here '/' zz_aux(zz_k).name], zz_d); end
    cd(zz_d);
    rehash;
    evalc(['dynare ' zz_c ' noclearall']);
    zz_G = ZZG;
    save('-mat7-binary', [zz_here '/' zz_c '_fig.mat'], 'zz_G');
    cd(zz_here);
end
