#!/bin/bash
# Writes reference/clean_pol_*.mat (tests/test_optimal_policy.py) with Dynare 7.2 under Octave:
# each case runs in a fresh folder with the .mod file (and its helper files), then pol_export.m.
# usage: reference/r_policy.sh <dynare-7.2-source> <work-folder>
set -e
SRC=${1:-/home/claude/dl/src72}; WORK=${2:-/tmp/pydyn_policy_refs}; HERE=$(cd "$(dirname "$0")" && pwd)
run() {   # run <case> <mod> [dynare args]   (extra files in $FILES)
  local case=$1 mod=$2; shift 2
  local d=$WORK/$case; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  cp "$HERE/$mod.mod" .; for f in $FILES; do cp "$HERE/$f" .; done
  printf "addpath %s/matlab\naddpath %s\nglobal M_ oo_ options_\ndynare %s nograph %s\npol_export('clean_%s.mat');\n" \
    "$SRC" "$HERE" "$mod" "$*" "$case" > r.m
  GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r.m > log.txt 2>&1
  cp "clean_$case.mat" "$HERE/"
}
run pol_nk_ramsey pol_nk_ramsey
run pol_nk_ramsey_o2 pol_nk_ramsey -DORDER=2
run pol_neo_ramsey pol_neo_ramsey
run pol_neo_ramsey_o1 pol_neo_ramsey -DORDER=1
FILES=find_c.m run pol_ramsey_ex pol_ramsey_ex
FILES=find_c.m run pol_ramsey_ex_o2 pol_ramsey_ex -DORDER=2
FILES=find_c.m run pol_ramsey_ex2 pol_ramsey_ex2
FILES=find_c.m run pol_ramsey_aux pol_ramsey_aux
run pol_nk_ramsey_exp pol_nk_ramsey_exp
run pol_ramsey_lin pol_ramsey_lin
FILES=pol_nk_ramsey_sf_steadystate.m run pol_nk_ramsey_sf pol_nk_ramsey_sf
run pol_gali_commit pol_gali_commit
run pol_mult_elim pol_mult_elim
FILES=find_c.m run pol_ramsey_histval pol_ramsey_histval
FILES=find_c.m run pol_ramsey_histval_o2 pol_ramsey_histval -DORDER=2
run pol_nk_ramsey_osr pol_nk_ramsey_osr
run pol_ramsey_policy pol_ramsey_policy
run pol_ramsey_2inst pol_ramsey_2inst
run pol_ramsey_policy_o2 pol_ramsey_policy -DORDER=2
run pol_neo_ramsey_pf pol_neo_ramsey_pf
for c in pol_dennis pol_gali_disc pol_nk_disc_fwd pol_gali15 pol_gali15_nl pol_osr pol_osr_corr pol_osr_cov pol_osr_planner pol_osr_ns; do
  run $c $c
done
# perfect foresight paths: Ramsey constraints (test1 of tests/optimal_policy/RamseyConstraints) and neo_growth_ramsey_foresight
for c in pol_rcons pol_neo_ramsey_pf; do
  d=$WORK/${c}_path; rm -rf "$d"; mkdir -p "$d"; cd "$d"; cp "$HERE/$c.mod" .
  out=clean_${c}.mat; [ $c = pol_neo_ramsey_pf ] && out=clean_${c}_path.mat
  printf "addpath %s/matlab\ndynare %s nograph\nr.endo_simul = oo_.endo_simul; r.steady_state = oo_.steady_state; r.endo_names = strjoin(M_.endo_names', ',');\nsave('-mat7-binary', '%s', '-struct', 'r');\n" "$SRC" "$c" "$out" > r.m
  GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet r.m > log.txt 2>&1
  cp "$out" "$HERE/"
done
# reference/examples/optimal_policy__nk_ramsey_osr.mat is clean_pol_nk_ramsey_osr.mat with ys as dr_ys and ok=1
# (see tests/test_examples.py)
