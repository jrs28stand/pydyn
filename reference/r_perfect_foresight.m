% Regenerates reference/pf_*.mat with Dynare 7.2 under Octave. Run in a scratch copy of this folder:
%   GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r_perfect_foresight.m
addpath /home/claude/dl/src72/matlab
dynare pf_backward nograph
pf_save('pf_backward', M_, oo_, options_);
dynare pf_endval nograph
pf_save('pf_endval', M_, oo_, options_);
dynare pf_endval_nosteady nograph
pf_save('pf_endval_nosteady', M_, oo_, options_);
dynare pf_etp nograph
pf_save('pf_etp', M_, oo_, options_);
dynare pf_forward nograph
pf_save('pf_forward', M_, oo_, options_);
dynare pf_histval nograph
pf_save('pf_histval', M_, oo_, options_);
dynare pf_homotopy nograph
pf_save('pf_homotopy', M_, oo_, options_);
dynare pf_homotopy_lin nograph
pf_save('pf_homotopy_lin', M_, oo_, options_);
dynare pf_homotopy_marg nograph
pf_save('pf_homotopy_marg', M_, oo_, options_);
dynare pf_homotopy_opts nograph
pf_save('pf_homotopy_opts', M_, oo_, options_);
dynare pf_initval_nosteady nograph
pf_save('pf_initval_nosteady', M_, oo_, options_);
dynare pf_linapprox nograph
pf_save('pf_linapprox', M_, oo_, options_);
dynare pf_linear nograph
pf_save('pf_linear', M_, oo_, options_);
dynare pf_mshocks nograph
pf_save('pf_mshocks', M_, oo_, options_);
dynare pf_mshocks0 nograph
pf_save('pf_mshocks0', M_, oo_, options_);
dynare pf_nohomotopy nograph
pf_save('pf_nohomotopy', M_, oo_, options_);
dynare pf_pfwee nograph
pf_save('pf_pfwee', M_, oo_, options_);
dynare pf_rbcii nograph
pf_save('pf_rbcii', M_, oo_, options_);
dynare pf_static nograph
pf_save('pf_static', M_, oo_, options_);
dynare pf_surprise nograph
pf_save('pf_surprise', M_, oo_, options_);
dynare pf_temp nograph
pf_save('pf_temp', M_, oo_, options_);
dynare pf_zlb nograph
pf_save('pf_zlb', M_, oo_, options_);
% Added 2026-11-21 (pf_ivfx, pf_hvfx and pf_pfweecsv read the .csv files of the same name)
dynare pf_cp nograph
pf_save('pf_cp', M_, oo_, options_);
dynare pf_cphom nograph
pf_save('pf_cphom', M_, oo_, options_);
dynare pf_cplin nograph
pf_save('pf_cplin', M_, oo_, options_);
dynare pf_hesl nograph
pf_save('pf_hesl', M_, oo_, options_);
dynare pf_hml nograph
pf_save('pf_hml', M_, oo_, options_);
dynare pf_hvfx nograph
pf_save('pf_hvfx', M_, oo_, options_);
dynare pf_irr nograph
pf_save('pf_irr', M_, oo_, options_);
dynare pf_ivf nograph
pf_save('pf_ivf', M_, oo_, options_);
dynare pf_ivfx nograph
pf_save('pf_ivfx', M_, oo_, options_);
dynare pf_pfweecsv nograph
pf_save('pf_pfweecsv', M_, oo_, options_);
dynare pf_pfweeml nograph
pf_save('pf_pfweeml', M_, oo_, options_);
dynare pf_rbcdet nograph
pf_save('pf_rbcdet', M_, oo_, options_);
dynare pf_rbcdet2 nograph
pf_save('pf_rbcdet2', M_, oo_, options_);
dynare pf_solow nograph
pf_save('pf_solow', M_, oo_, options_);
dynare pf_vec nograph
pf_save('pf_vec', M_, oo_, options_);
dynare pf_zlbfwd nograph
pf_save('pf_zlbfwd', M_, oo_, options_);
