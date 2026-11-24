addpath('/home/claude/dl/src72/matlab');
cases = {'chk_a','chk_b','chk_c','chk_d'};
for i=1:numel(cases)
    clear global M_ oo_ options_ estim_params_ bayestopt_ dataset_ dataset_info;
    d = tempname; mkdir(d); copyfile([cases{i} '.mod'], d); copyfile('fsdat_simul.m', d);
    here = pwd; cd(d);
    try
        evalc(['dynare ' cases{i} ' nograph noclearall']);
        printf('%s: NO ERROR\n', cases{i});
    catch err
        printf('%s: [%s]\n', cases{i}, err.message);
    end
    cd(here);
end
