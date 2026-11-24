% Octave reference runs of the VAR / trend component / VAR expectation / PAC cases of tests/test_var_models.py
% (Dynare 7.2 in the path; the .mod files are those of Dynare's tests/ folders named in the table):
%     octave --no-gui --quiet r_var.m          (from this folder; DYNARE_TESTS= folder of Dynare's tests, default below)
% Each case saves to <case>.mat: M_.params, the variables of oo_ and M_ that hold the sub-models, the last rows of the
% dseries of the workspace, the numeric variables of the workspace and the .inc files written by pac.print / var_expectation.print.
zz_cases = {
 've1',    'var-expectations/1/example1';
 've3ts',  'var-expectations/3-with-time-shift/example1';
 've5',    'var-expectations/5/example1';
 've6',    'var-expectations/6/example1';
 've7',    'var-expectations/7/example1';
 've9',    'var-expectations/9/example1';
 'pv1',    'pac/var-1/example1';
 'pv4',    'pac/var-4/example1';
 'pv12',   'pac/var-12/example1';
 'pv12c',  'pac/var-12/example3';
 'ptc1',   'pac/trend-component-1/example1';
 'ptc4',   'pac/trend-component-4/example1';
 'ptc13a', 'pac/trend-component-13a/example1';
 'ptc26',  'pac/trend-component-26/example1';
 'mce1',   'pac/trend-component-1-mce/example_det';
 'mce2s',  'pac/trend-component-2-mce/example_sto';
 'tcm1',   'trend-component-and-var-models/tcm1';
 'tcm6',   'trend-component-and-var-models/tcm6';
 'vm1',    'trend-component-and-var-models/vm1';
 'vm4',    'trend-component-and-var-models/vm4';
 'ltcm1',  'trend-component-and-var-models/legacy/tcm1';
 'lvm1',   'trend-component-and-var-models/legacy/vm1'};
zz_tests = getenv('DYNARE_TESTS'); if isempty(zz_tests), zz_tests = '/home/claude/dl/src72/tests'; end
if exist('zz_only', 'var'), zz_cases = zz_cases(ismember(zz_cases(:,1), zz_only), :); end
zz_here = pwd;
zz_stubs = tempname; mkdir(zz_stubs);
for zz_f = {'figure','plot','subplot','hold','title','legend','xlabel','ylabel','close','saveas','drawnow','bar','clf','axes','gca','gcf','axis','line','text','grid','box','print','orient','savefig','pause','xlim','ylim'}
    zz_g = fopen(fullfile(zz_stubs, [zz_f{1} '.m']), 'w');
    fprintf(zz_g, 'function varargout = %s(varargin)\nfor i = 1:nargout, varargout{i} = 1; end\nend\n', zz_f{1}); fclose(zz_g);
end
addpath(zz_stubs);
for zz_i = 1:size(zz_cases, 1)
    clear global M_ oo_ options_ estim_params_ bayestopt_ dataset_ dataset_info;
    clear -x zz_cases zz_tests zz_here zz_stubs zz_i zz_only zz_f zz_g     % nothing of the previous case may leak into the numeric variables
    zz_c = zz_cases{zz_i, 1}; [zz_dir, zz_name] = fileparts(zz_cases{zz_i, 2});
    zz_d = tempname; mkdir(zz_d); copyfile([zz_tests '/' zz_dir '/*'], zz_d);
    copyfile([zz_tests '/' zz_cases{zz_i, 2} '.mod'], [zz_d '/' zz_c '.mod']);     % the case name is the model name
    cd(zz_d);
    evalc(['dynare ' zz_c ' nograph noclearall']);
    zz = struct();
    zz.params = M_.params; zz.param_names = M_.param_names;
    if isfield(oo_, 'var'), zz.oo_var = oo_.var; end
    if isfield(oo_, 'trend_component'), zz.oo_tc = oo_.trend_component; end
    if isfield(M_, 'var'), zz.M_var = M_.var; end
    if isfield(M_, 'trend_component'), zz.M_tc = M_.trend_component; end
    if isfield(M_, 'var_expectation'), zz.M_ve = M_.var_expectation; end
    if isfield(M_, 'pac'), zz.M_pac = M_.pac; end
    zz_inc = dir([zz_c '/model/*-expectations/*.inc']);
    for zz_k = 1:numel(zz_inc)
        zz_t = fileread([zz_inc(zz_k).folder '/' zz_inc(zz_k).name]);
        zz_t = zz_t(find(zz_t == 10, 1) + 1:end);          % without the first line (it holds the time stamp)
        zz.(['inc_' regexprep(zz_inc(zz_k).name, '[^A-Za-z0-9]', '_')]) = zz_t;
    end
    zz_vars = who;
    for zz_j = 1:numel(zz_vars)
        zz_v = zz_vars{zz_j};
        if strncmp(zz_v, 'zz', 2), continue, end
        zz_val = eval(zz_v);
        if isa(zz_val, 'dseries')
            zz_n = zz_val.nobs; zz_r = max(1, zz_n - 11):zz_n;
            zz.(['ds_' zz_v]) = struct('data', zz_val.data(zz_r, :), 'name', {zz_val.name}, 'nobs', zz_n);
        elseif isnumeric(zz_val) && numel(zz_val) < 2000 && ~strcmp(zz_v, 'ans')
            zz.(['v_' zz_v]) = zz_val;
        end
    end
    save('-mat7-binary', [zz_here '/' zz_c '.mat'], '-struct', 'zz');
    cd(zz_here);
end
