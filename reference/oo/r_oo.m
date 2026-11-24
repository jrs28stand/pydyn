% Octave reference runs of the cases of tests/test_results.py (Dynare 7.2 in the path):
%     octave --no-gui --quiet r_oo.m          (from this folder; set `zz_only = {'rbc_oo'}` first to run one case)
% Each case's oo_, M_, options_ and the MATLAB variables of the .mod file are saved to <case>.mat.
% nd_oo and nd_oo2 plot shock decompositions: without a graphics toolkit, stubs of the plotting functions are put
% first in the path (they draw nothing; get returns a screen size).
zz_cases = {'rbc_oo', 'rbc_oo_o2', 'rbc_oo_o3', 'rbc_oo_pf', 'rbc_oo_loop', 'rbc_oo_pfloop', 'est_smoother0', 'est_mode4', 'est_ml', 'cs1', 'fc1', 'occ1', 'ram1', 'osr1', 'est_mh', 'rng_oo', 'nd_oo', 'nd_oo2'};
if exist('zz_only', 'var'), zz_cases = zz_only; end
zz_here = pwd;
if isempty(available_graphics_toolkits())
    zz_stubs = tempname; mkdir(zz_stubs);
    for zz_f = {'figure', 'plot', 'subplot', 'hold', 'title', 'legend', 'xlabel', 'ylabel', 'zlabel', 'close', 'saveas', ...
                'drawnow', 'bar', 'colorbar', 'clf', 'axes', 'gca', 'gcf', 'axis', 'set', 'line', 'text', 'patch', 'fill', ...
                'area', 'xlim', 'ylim', 'grid', 'box', 'annotation', 'colormap', 'print', 'orient', 'savefig', 'pause', ...
                'groot', 'findobj', 'datetick', 'xticks', 'yticks', 'xticklabels', 'yticklabels', 'ishandle', 'isgraphics', ...
                'uimenu', 'uicontextmenu', 'linkaxes', 'refresh', 'barh', 'errorbar'}
        zz_fid = fopen(fullfile(zz_stubs, [zz_f{1} '.m']), 'w');
        fprintf(zz_fid, 'function varargout = %s(varargin)\nfor i = 1:nargout, varargout{i} = 1; end\nend\n', zz_f{1});
        fclose(zz_fid);
    end
    zz_fid = fopen(fullfile(zz_stubs, 'get.m'), 'w');
    fprintf(zz_fid, 'function varargout = get(varargin)\nfor i = 1:nargout, varargout{i} = [0 0 1920 1080]; end\nend\n');
    fclose(zz_fid);
    addpath(zz_stubs);
end
for zz_i = 1:numel(zz_cases)
    zz_c = zz_cases{zz_i};
    clear global M_ oo_ options_ estim_params_ bayestopt_ dataset_ dataset_info;   % noclearall keeps globals
    zz_d = tempname; mkdir(zz_d); copyfile([zz_here '/' zz_c '.mod'], zz_d); copyfile([zz_here '/fsdat_simul.m'], zz_d); cd(zz_d);
    evalc(['dynare ' zz_c ' nograph noclearall']);
    zz_vars = who; zz_keep = {};
    for zz_j = 1:numel(zz_vars)
        if ~strncmp(zz_vars{zz_j}, 'zz_', 3), zz_keep{end+1} = zz_vars{zz_j}; end
    end
    save('-mat7-binary', [zz_here '/' zz_c '.mat'], zz_keep{:});
    cd(zz_here);
    clear(zz_keep{:});
end
