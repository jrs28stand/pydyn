# NOTES: work copy /home/claude/work_f (base: main 08663b5), F-list items 1-6

## Done and validated (targeted tests only; the full suite has NOT been run on this copy)
1. osr opt_algo 1, 3, 7, 8, 9: `pydyn/optimal_policy.py::_osr_equivalent` (scipy stand-ins warn, as estimation does;
   7 is the ported fminsearch; active-set bound snap for 1/9; never worse than the start). opt_algo 2 -> NotImplementedError,
   5 -> OSRError. Files now run but several give verdict "wrong" because the optimiser end point differs from Dynare's
   (OSR_planner_objective flat optimum): add tools/accept_known.json "Dynare quirk (optimiser path)" entries once the
   objective is confirmed <= Dynare's. NOT yet added.
2. Graphics: `pydyn/mlgfx.py` (handles as Octave: figures integers, other objects negative fractions; set/get/ishandle/
   close/gcf/gca; draws nothing; never changes results). `print`/`saveas`/`savefig` are REFUSED (clear reason), not rendered.
   Command syntax (hold on, close all) handled in `Interpreter.command()`.
3. Char matrices: `pydyn/mlchar.py` (CharMat; char/strvcat/deblank/strtrim/strmatch/row indexing/padding), hooks in matlab.py.
   `var_list_` is set before each var-list statement (results.py `_VAR_LIST_STATEMENTS`).
4. Fields: `M_.endo_partitions/param_partitions/exo_partitions` (lazy `load_partitions`, second preprocessor pass,
   submodels.PREFIXES), `options_.varobs/varobs_id/varexobs/varexobs_id`, `observed_exo_names`.
5. consider_only_observed: coded (`results_est._est_syms`, results_post `_moments_varendo` var_list); the 3 files
   (slow MH runs) were NOT re-run.
New Octave-checked cases: reference/matlab_octave_cases/pydyn_gfx_cases.txt (+r_pydyn_gfx.m, pydyn_gfx.mat) and
pydyn_char_cases.txt (+r_pydyn_char.m, pydyn_char.mat), registered in tests/test_matlab_octave_cases.py (SETS; `_same`
handles CharMat). New tests: tests/test_mlgfx.py (14), test_optimal_policy.py (test_osr_refusals changed,
test_osr_other_optimisers added).

## Changed files
New: pydyn/mlgfx.py, pydyn/mlchar.py, tests/test_mlgfx.py, reference/matlab_octave_cases/pydyn_{gfx,char}_cases.txt,
r_pydyn_{gfx,char}.m, pydyn_{gfx,char}.mat, NOTES.md.
Edited (shared): pydyn/matlab.py (concat/class/shape_of/transpose hooks, Interpreter.command graphics, imports and
LAZY loop at the end), pydyn/results.py (graphics_enabled, _NOOP_NAMES, var_list_, fn_stoch_simul MatlabError,
varobs_id row, _PARTITION_FIELDS + getfield hook), pydyn/results_oo.py (matlab_message on stoch_simul errors),
pydyn/results_mi.py (removed _MISSING var_list_, varobs/varexobs, load_partitions), pydyn/results_est.py (_est_syms),
pydyn/results_post.py (_moments_varendo var_list), pydyn/submodels.py (PREFIXES), pydyn/results_m.py
(KNOWN_UNSUPPORTED), pydyn/optimal_policy.py, tests/test_optimal_policy.py, tests/test_matlab_octave_cases.py,
docs/REFUSALS.md. Check results.py / results_est.py for conflicts against main before merging.
Option policy: no statement option was moved to "honoured" (check option_policy.py if osr opt_algo needs it).

## Half done: item 6 (nothing coded yet)
Findings and exact next steps:
- dseries readers: new module (e.g. pydyn/dseries_io.py) returning (init Date, data, names), following Dynare's
  dseries/src/read: .csv Octave branch (header split on commas, quotes stripped; second row first field a date -> init and
  first column dropped, else init 1Y; csvread empties -> 0), .m (run script, workspace vectors; INIT__/NAMES__/FREQ__),
  .xlsx via openpyxl (load_xls_file_data: dates in first column if txt has >1 columns, default names Variable_i, init 1Y
  without time; sheet/range from options_.xls_sheet/xls_range as in load_data.m, default sheet 1 whole range); .xls refused
  (xlrd absent). Open question: with xls_range b1:b54 (dates/dseries_interact second call) the txt/names logic of
  xlsread under Octave io must be checked against /home/claude/accept_ref/dates/dseries_interact.mat before coding.
  Wire into results_bwd.py `dseries_read` (path only inside the model folder), estimation.py `load_data` (line ~458),
  dates.py `_resolve`/`data_start_date`, smoother.py load_data, perfect_foresight `_read_datafile`.
- set_time: statement JSON is {"statementName":"set_time","period":"1950Q3"}; it is in results.DECLARATIONS but has no
  handler in Model._exec_statement. Write options_.initial_period (string date, as results_ep reads it). Effect (makedataset.m):
  a data file with default init (1Y) takes options_.initial_period (use in dates.apply_data_command / _dseries_default_start,
  reading the set_time statement from model.statements); if the file has its own init and it is later than set_time, keep
  file init; error if initial_period < dataset init. Needed by fs2000/fs2000_dseries_a and _b.
- Small pickups not done: `dyntable` display noop (Gali_2015_chapter_6_4), occbin.unpack_simulations (model_borrcon x2),
  prior_posterior_function statement, oo_.initval_series, M_.aux_vars struct array, isfield(M_,'exo_partitions') lazy
  trigger, conditional_variance_decomposition/example1 now "wrong" (order-2 simulated draws, uninvestigated),
  Hansen_exo_det_forecast partial (dr.exo_det_length, dr.ghud).
Helpers: /tmp/env.sh (exports DYNARE_PREPROCESSOR, PYDYN_REQUIRE_KORDER=1; needed in every shell), /tmp/runf.sh outdir ids...
