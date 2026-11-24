# Real DynareOBC under Octave (validation references)

Tom Holden's DynareOBC runs under Octave 8.4 with Dynare 4.5.4 after four small patches. The `.mat` files here are
its output and are what `tests/test_obc_dynareobc.py` compares pydyn against.

## Recipe
1. Dynare 4.5.4 source (`dynare_4.5.4.orig.tar.xz`, Ubuntu archive). Configure with
   `--disable-matlab --disable-octave CXXFLAGS="-O2 -std=gnu++11 -w"` and build only `make -C preprocessor`
   (Dynare's m-file fallbacks replace the MEX files).
2. Sobol MEX (needed for cubature only), in `mex/sources/sobol`:
   `mkoctfile --mex -I. -I.. -I../.. -DOCTAVE_MEX_FILE -o <mexdir>/qmc_sequence.mex qmc_sequence.cc`
3. DynareOBC at the commit in `dynareobc_commit.txt` with submodules; apply `dynareobc_octave.patch`, and
   `yalmip_octave.patch` inside `Extern/YALMIP` (commit in `yalmip_commit.txt`).
4. Edit the paths at the top of `obc_octave.m` (the driver replacing `dynareOBC.m`, whose set-up code needs MATLAB);
   put `shims/` on the path; Octave needs the `statistics` package for more than one cubature region.
5. `GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui --quiet --eval "run_nk(2,'o2')"`.
   The mod file needs `nograph` and `drop >= 1`. `nk_o<k>.mod` is `nk_max.mod` with `order=<k>`.

## Patches
- `ProcessStochSimulCommand.m`: variable-length lookbehind regexes rewritten as tokens (Octave error).
- `ProcessModFileText.m`: `\<`, `\>` inside a character class are word anchors in Octave (silent misparse).
- YALMIP `definesolvers.m`: version detection.
- `ObtainRootConditionalCovariance.m`: Octave's `schur` does not order eigenvalues, so the truncation
  `diagD(1:end-MaxDimension)=0` removed the *largest* one. Sorted first. (Cubature only.)

## Files
- `nk_max.mod` / `nk_occ.mod`: the same model in DynareOBC and OccBin form. `shocks.mat`: 300 draws.
- `ref_o1.mat`, `ref_o2.mat`: orders 1 and 2, no cubature. `ref_o2cub.mat`: order 2, `Cubature`,
  `QuasiMonteCarloLevel=9` (one region), with the conditional covariance root `Rcc`; `sob_o2cub.mat`: Dynare's
  Sobol points for it (Dynare's direction numbers differ from SciPy's beyond dimension 2).
- `plt.mod`, `plt_occ.mod`, `plt_ref.mat`: IRF check.

## Results (pydyn against DynareOBC)
M matrix 9e-16; order-1 and order-2 simulations 2e-16; conditional covariance 5e-19; order-2 cubature 4e-16 with
DynareOBC's root and points, 1.5e-6 with pydyn's own Sobol points (cubature effect itself: 4.8e-5).
With one region and the bound binding today, every point is "relevant" and the rule returns the no-cubature
solution exactly; cubature only matters in periods where the bound binds in expectation but not yet today.
