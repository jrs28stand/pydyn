% Octave reference runs of the cases of tests/test_results.py (Dynare 7.2 in the path):
%     octave --no-gui --quiet r_oo.m          (from this folder; set `zz_only = {'rbc_oo'}` first to run one case)
% Each case's oo_, M_, options_ and the MATLAB variables of the .mod file are saved to <case>.mat.
zz_cases = {'rbc_oo', 'rbc_oo_o2', 'rbc_oo_o3', 'rbc_oo_pf', 'rbc_oo_loop', 'rbc_oo_pfloop'};
if exist('zz_only', 'var'), zz_cases = zz_only; end
zz_here = pwd;
for zz_i = 1:numel(zz_cases)
    zz_c = zz_cases{zz_i};
    clear global M_ oo_ options_ estim_params_ bayestopt_ dataset_ dataset_info;   % noclearall keeps globals
    zz_d = tempname; mkdir(zz_d); copyfile([zz_here '/' zz_c '.mod'], zz_d); cd(zz_d);
    evalc(['dynare ' zz_c ' nograph noclearall']);
    zz_vars = who; zz_keep = {};
    for zz_j = 1:numel(zz_vars)
        if ~strncmp(zz_vars{zz_j}, 'zz_', 3), zz_keep{end+1} = zz_vars{zz_j}; end
    end
    save('-mat7-binary', [zz_here '/' zz_c '.mat'], zz_keep{:});
    cd(zz_here);
    clear(zz_keep{:});
end
