# OBC news methods in estimation, and equilibrium selection (pydyn, branch `obc-est`)

Started 2026-11-24 from `wip/obc` (0a16a10). Backed up as `wip/obc-est`. Everything below was measured on this machine (2 cores
shared with another worker, runs under `nice`) unless it says otherwise; timings are noisy by 10-20 %, and a wall time of a whole
run is the more exposed to load, so per-evaluation times (solvers called alternately at the same points) are given as well.

## State (keep this section current)

Baseline before any change: `pytest tests/test_obc.py tests/test_obc_perf.py tests/test_occbin.py tests/test_occbin_options.py
tests/test_occbin_estimation.py` = 164 passed, 1 skipped.

| Task | Status |
|---|---|
| 1a proper estimation option `obc_method` / `obc_horizon` / `obc_selection` | done, tested (`tests/test_obc_estimation.py`) |
| 1b news filter solver for several-equation constraints (`RegimeFilterSolver`) | done, tested on bf.mod and bf_ppf.mod |
| 1c fail fast on a failed trial point | done: no integer programme in the filter by default; tested |
| 2 estimation feasibility study | done, tables below; raw results in `bench/results/est_*.json` |
| 3 `obc_selection` | done, tested (`tests/test_obc_selection.py`), cost measured |

Tests at the end: the baseline command above still gives 164 passed, 1 skipped; `tests/test_obc_estimation.py` + `tests/test_obc_selection.py`: 17 passed (new). Whole suite:
1739 passed, 189 skipped, 15 failed; the same 15 fail on the baseline commit 0a16a10 (missing `libpydyn_korder.so`, a few examples), none concerns OBC or estimation.

Nothing is half done. Open points (not done, see the last section): the piecewise Kalman filter with the news method (refused by
design), selection with more than one constraint (refused), a converged posterior mode of the full NKM model (neither solver
gets there with csminwel; the Hessian has one negative eigenvalue at the points reached), a faster selection enumeration.

Scripts: `bench/obc_est_study.py` (Task 2 runs), `bench/make_est_tables.py` (tables), `bench/obc_est_bf.py`,
`bench/obc_est_timing.py`, `bench/obc_est_fallbacks.py`, `bench/obc_knife_edge.py`, `bench/obc_selection_cost.py`.

Found on the way: `Estimation` set `lb`/`ub` to NaN where `estimated_params` has empty bounds (Dynare's NKM files: `phipi,,,,NORMAL_PDF,...`),
so every comparison with the bounds was false and the Metropolis-Hastings sampler rejected every proposal (acceptance 0 on
`occest_nkms.mod`; the existing short-MH test only checked that the draws were finite). Fixed in `pydyn/estimation.py`
(`np.fmax`/`np.fmin`: NaN means "no bound"; the prior's bounds then apply). The posterior-mode and Hessian code do not use these bounds
with csminwel, so earlier mode results are unchanged.

## 1. The estimation option, several equations, failing trial points

### 1a. `obc_method`, `obc_horizon`, `obc_selection` on `OccbinEstimation`
- Constructor keywords, read from the native `.mod` lines `options_.occbin.obc_method = 'news';`, `options_.occbin.obc_horizon = 40;`,
  `options_.occbin.obc_selection = '...';` when not given, and plain attributes (`est.obc_method = 'news'` still works; they are properties
  that validate). Default `'occbin'` / 40 / `'occbin'`; the results of the default are bit for bit what they were (test compares
  `log_posterior` with `==`). Unknown values raise `ValueError`.
- `obc_method='news'` is for the inversion filter only. With the piecewise Kalman filter (Dynare's default for an OccBin model) it raises
  `EstimationError("obc_method='news' is implemented for the inversion filter only ...")` at construction, at assignment, in `run_filter` and in the
  piecewise smoother; the inversion-filter smoother uses the news solver.
- `Occbin.simulate(..., obc_selection=)` and `mod_options()` read the same third option (Task 3). `obc_selection` other than `'occbin'` with
  `obc_method='occbin'` is an error.
- `obc_milp_time_limit` (attribute, default None = the solver's default 0): see 1c.

### 1b. Constraints that switch several equations
`pydyn/obc_filter.py`: `NewsFilterSolver(ob, horizon, check_ahead, selection, milp_time_limit)` returns an `LCPFilterSolver` (one complementary pair per
constraint, as before) or a `RegimeFilterSolver` (`RegimeOBC`; the choice `Occbin.simulate` makes). For a regime sequence the wedges on the binding
positions S are `w = -R[S,S]^-1 (a0 + Am [state; shock])[S]`, so the next state is `[P Q] x + V0[:, S] w` and the first-period rule (T, R, C) is read
off and cached per S.

Test bed: Dynare's `tests/estimation/slice/bf.mod` and `tests/occbin/ppf/bf_ppf.mod` (5 variables, two equations switch), data
`reference/occest_bf_data.mat` (Dynare's `datafile.mat`, which is MATLAB v7.3; converted to v7 with Octave; the two files are identical),
estimated parameters as in the files (5), inversion filter, the two solvers at the file's initial values and at 7 vectors perturbed by about 5 %
(log-normal, seed 1). Both models have the same equations, so the numbers coincide. Achieved: log posterior equal to **2.3e-13** (largest over the
vectors, rounding in the sums; the test asserts 1e-9), identical regime histories, filtered shocks equal to **4.4e-16** (test: 1e-12), states 1e-12;
the first-period rule (T, R, C) of the two solvers equal to 1e-11 in 40 random states (test). One vector makes the filter fail in both
(info 304, in period 50).

bf.mod:

| vector | log posterior OccBin | news | abs. difference | same regimes | binding periods | max shock difference | OccBin s | news s |
|---|---|---|---|---|---|---|---|---|
| 0 | 566.55125595 | 566.55125595 | 1.1e-13 | True | 7 | 1.7e-16 | 0.153 | 0.032 |
| 1 | 521.79732838 | 521.79732838 | 1.1e-13 | True | 6 | 4.4e-16 | 0.159 | 0.030 |
| 2 | 557.40313082 | 557.40313082 | 2.3e-13 | True | 6 | 1.4e-16 | 0.156 | 0.033 |
| 3 | 570.08561695 | 570.08561695 | 0.0e+00 | True | 7 | 6.9e-17 | 0.170 | 0.033 |
| 4 | 567.53429750 | 567.53429750 | 1.1e-13 | True | 7 | 2.0e-16 | 0.157 | 0.031 |
| 5 | -inf (code 304) | -inf (code 304) | | | | | 0.275 | 0.269 |
| 6 | 557.71407769 | 557.71407769 | 1.1e-13 | True | 6 | 1.3e-16 | 0.154 | 0.032 |
| 7 | 512.54045708 | 512.54045708 | 1.1e-13 | True | 6 | 9.4e-17 | 0.156 | 0.030 |

bf_ppf.mod: identical to bf.mod to the digits shown (same equations), times 0.15-0.19 s against 0.030-0.042 s.

Tests: `test_inversion_filter_news_equals_occbin_several_equations`, `test_regime_filter_rule_matches_occbin_rule`.

### 1c. A failed trial point must fail fast
Where the time went (NKM, 4th stored vector, `reference/occest_nkm_lik.mat` column 4): 276 solver calls, 2 of them reach the fallbacks. One is proved
infeasible by a linear programme (3 ms). The other has a feasible linear relaxation, the active-set iteration cycles, and the integer programme
(Holden's formulation, 40 binaries) needs 1.5 s to prove that there is no solution (HiGHS; with a 1 s limit it timed out, with 5 s it returned
"no solution"; a feasibility formulation with y <= U z took 1.0-2.6 s, not faster). OccBin fails at the same state in 6 ms (error 313).

What I did: in the filter the integer programme is **off by default** (`milp_time_limit = 0`), i.e. a trial point on which the cheap checks (active-set
iteration, proof by a linear programme) do not decide is a failed trial point, error 313, which is what OccBin returns there. Same for
`RegimeFilterSolver` (its integer programmes in `_solve_lcp` and `_milp` are skipped when `milp_time_limit <= 0`). The setting is available
(`est.obc_milp_time_limit = 1.0`, solver keyword `milp_time_limit`) and the solver counts what happens (`fallbacks`, `proved_infeasible`, `skipped`,
`timeouts`, `rescued`, `errors`).

Why this does not change the likelihood, as far as I could check (`bench/obc_est_fallbacks.py`: the news filter run at the stored vectors and at
random perturbations; at every solver call that reaches the fallbacks OccBin's solver is called on the same state and shock, and the integer programme
is run with a 1 s limit):

| model | parameter vectors | trial points reaching the fallbacks | news / OccBin flags | integer programme (1 s) found a solution |
|---|---|---|---|---|
| NKM full | 64 | 2 | 313/313 (2) | 0 (1 no solution, 1 time-out) |
| NKM small | 103 | 0 | | |
| bf.mod | 26 | 104 | 313/310 (57), 313/313 (47) | 0 (77 no solution, 27 time-out) |
| bf_ppf.mod | 26 | 104 | same | 0 (69 no solution, 35 time-out) |

So at all 210 such trial points OccBin fails too and the integer programme finds nothing; switching it on or off changes no likelihood. In
addition, at 167 vectors of the NKM models (a first, smaller check) log posterior with a 1 s limit and without it were equal at every vector. This is
evidence, not a proof: a state where the active-set iteration cycles but a solution exists would be turned into a failure. Such states were not met.
(Forced through the integer programme on states where the active set succeeds, the NKM integer programme needs 120 ms median, 1.2 s at the 95th
percentile, 2.3 s at most, over 228 binding states; a small time limit would therefore not be a safe replacement for switching it off.)

Timing, NKM full, 4th stored vector, the three evaluations alternated in one process, 7 repeats:

| | min (s) | median (s) | log posterior |
|---|---|---|---|
| OccBin | 0.332 | 0.367 | 1533.3322164743 |
| news, integer programme with 1 s limit (before) | 1.050 | 1.055 | 1533.3322164743 |
| news, no integer programme (now) | 0.046 | 0.047 | 1533.3322164743 |

(Equal to the stored Dynare value, `-ivf_fval[3]`, to 1e-11 relative.) On the bf models the failing vector now costs the same with both solvers (bf: OccBin 0.275 s, news 0.269 s; bf_ppf: 0.288 and 0.287 s); with the 1 s limit the news solver needed 48 s for it (74 time-outs).

Per-evaluation time and agreement, NKM, the 4 stored vectors and 40 vectors perturbed by 2 % (seed 1), solvers alternated, minimum of 3 per vector
(`bench/obc_est_timing.py`):

| model | vectors | OccBin total (s) | news total (s) | median per evaluation, OccBin / news (s) | slowest news evaluation (s) | likelihoods differing by more than 1e-9 |
|---|---|---|---|---|---|---|
| NKM full | 44 | 13.74 | 1.69 | 0.299 / 0.0359 | 0.075 | 0 (largest difference 9.1e-12) |
| NKM small | 43 | 11.33 | 1.65 | 0.247 / 0.0343 | 0.085 | 0 (largest difference 4.5e-12) |

(The earlier progress log found one of 40 perturbed vectors, with another seed, differing by 0.015: the knife edge below.)

## 2. Is estimation with the news solver feasible?

Set-up. Dynare's NKM OccBin estimation example, full (`reference/occest_nkm.mod`, 9 parameters, 120 observations, 3 observables) and small
(`reference/occest_nkms.mod`, 3 parameters); data and stored vectors from `reference/occest_*_lik.mat`; likelihood: inversion filter
(`occbin={'likelihood_inversion_filter': True}`); the same `OccbinEstimation` object type with `obc_method='occbin'` and `'news'` (horizon 40).
Mode: csminwel (`mode_compute=4`, the settings of `tests/test_occbin_estimation.py`: pydyn's defaults) from the initial values of
`estimated_params` (= prior means, called "prior" below); for the full model also from the first stored vector ("stored"). Hessian
(`Estimation.hessian`, Dynare's finite differences) at each solver's own mode; Laplace approximation `Estimation.laplace`. Metropolis: random walk,
proposal covariance = inverse Hessian at OccBin's mode (full model: the Hessian is not positive definite, so the prior variance), scale from a
300-draw pilot with the news solver (the grid value whose acceptance is closest to 0.30), identical starting points (Dynare's rule: a draw around
the mode with scale 2 x jscale), identical random numbers (per chain seeded from `[20261124, chain]`), the two solvers alternating chain by chain.
"Draws rejected" are evaluations returning minus infinity (the sampler does not evaluate proposals outside the bounds; none occurred).
Dynare's codes: 4 = Blanchard-Kahn indeterminacy (not a filter failure); 304 = the inversion filter failed.

Reading the tables. "First evaluation at which the likelihoods differ": the news solver is run at every parameter vector that OccBin's run visited, both
likelihoods are compared (threshold 1e-9 relative), and the regime histories are compared at the first difference. Agreeing points agree to <= 1e-10.

### 2a/2b. Small model (3 parameters), start = prior mean

#### Posterior mode, nkms, start = prior

| | OccBin | news |
|---|---|---|
| mode | (1.2451, 0.23999, 0.036275) | (1.2451, 0.23998, 0.036276) |
| log posterior at the mode | 1518.4359618 | 1518.4359618 |
| optimiser | csminwel: normal step (17 iterations, 99 evaluations) | csminwel: normal step (19 iterations, 117 evaluations) |
| likelihood evaluations (incl. Hessian) | 115 | 133 |
| evaluations that failed (-inf) | 0 | 0 |
| wall time of the run (s) | 27.6 | 5.1 |
| Hessian (finite differences, in the mode run) positive definite | True | True |
| posterior sd from the Hessian | (0.069911, 0.028592, 0.0035296) | (0.069911, 0.028592, 0.0035296) |
| Laplace approximation | 1508.896403 | 1508.896420 |

Same path: False. Evaluations: OccBin 115, news 133. First evaluation at which the iterates differ (> 1e-9): 5. |mode difference|: (2.94e-06, 5.54e-06, 4.71e-07), as a share of the posterior sd: (4.2e-05, 0.000194, 0.000134).
News evaluated at the 115 points OccBin's run visited: 111 agree (largest difference 8.64e-11), 4 differ (largest difference 0.0265).
First differing evaluation: index 59 of OccBin's run, log posterior 1503.637284 (OccBin) against 1503.610741 (news); regimes differ in period(s) [41]: period 41: OccBin (((0,), (1,)),), news (((1, 0), (1, 2)),).

Time per evaluation (evaluations that ran the filter; the two solvers called alternately at the same points): OccBin min 208 ms, median 237 ms; news min 32 ms, median 35 ms (ratio of medians 6.8). Trial-point failures inside the filter during the mode search: OccBin {}, news {}.

#### Hessian and Laplace approximation, nkms, start = prior

| | OccBin | news |
|---|---|---|
| evaluations / wall time (s) | 13 / 3.4 | 13 / 0.4 |
| positive definite | True | True |
| posterior sd | (0.069911, 0.028592, 0.0035296) | (0.069911, 0.028592, 0.0035296) |
| Laplace approximation | 1508.896403 | 1508.896420 |

Both Hessians at OccBin's mode: largest entrywise difference 2.253e-05 (largest entry 1.879e+05, relative 1.20e-10).


Reading: the two searches follow the same iterates for the first 5 evaluations; at evaluation 6 they differ by 1e-9, because the numerical gradient divides
likelihood differences of about 1e-11 (summation order, about 30 ulp) by the finite-difference step, and the iterates drift apart from there; they stop at points
differing by 3e-6, 6e-6 and 5e-7 (4e-5 to 2e-4 posterior standard deviations), with log posterior equal to 1e-9. The first evaluation where the
**likelihoods** differ is 59 of OccBin's run: a difference of 0.0265 in the log posterior, caused by period 41, which is a knife-edge period (below). That
point is not on the path to the mode; the two modes do not depend on it.

### 2a/2b. Full model (9 parameters)

#### Posterior mode, nkm, start = prior

| | OccBin | news |
|---|---|---|
| mode | (99.973, 1.86, 1.2537, 0.644, 0.88936, 0.91224, 0.0040653, 0.0024004, 0.0027608) | (100, 1.8695, 1.1803, 0.69976, 0.85933, 0.9046, 0.0041296, 0.0032464, 0.0027907) |
| log posterior at the mode | 1693.1305714 | 1692.7512568 |
| optimiser | csminwel: smallest step still improving too slow, reversed gradient (48 iterations, 942 evaluations) | csminwel: smallest step still improving too slow (40 iterations, 896 evaluations) |
| likelihood evaluations (incl. Hessian) | 1035 | 989 |
| evaluations that failed (-inf) | 48 | 60 |
| wall time of the run (s) | 302.4 | 38.4 |
| Hessian (finite differences, in the mode run) positive definite | False | False |
| posterior sd from the Hessian | n/a (Hessian not positive definite) | n/a (Hessian not positive definite) |
| Laplace approximation | nan | nan |

Same path: False. Evaluations: OccBin 1035, news 989. First evaluation at which the iterates differ (> 1e-9): 46. |mode difference|: (0.0321, 0.00955, 0.0734, 0.0558, 0.03, 0.00764, 6.44e-05, 0.000846, 3e-05).
News evaluated at the 1035 points OccBin's run visited: 955 agree (largest difference 9.55e-11), 32 differ (largest difference 0.1240).
First differing evaluation: index 170 of OccBin's run, log posterior 1681.124262 (OccBin) against 1681.248286 (news); regimes differ in period(s) [47]: period 47: OccBin (((1, 0), (1, 9)),), news (((1, 0), (1, 8)),).

Time per evaluation (evaluations that ran the filter; the two solvers called alternately at the same points): OccBin min 0 ms, median 294 ms; news min 0 ms, median 37 ms (ratio of medians 7.9). Trial-point failures inside the filter during the mode search: OccBin {'313': 41, '310': 123}, news {'313': 3}.

#### Hessian and Laplace approximation, nkm, start = prior

| | OccBin | news |
|---|---|---|
| evaluations / wall time (s) | 91 / 28.4 | 91 / 3.6 |
| positive definite | False | False |
| posterior sd | - | - |
| Laplace approximation | - | - |

Both Hessians at OccBin's mode: largest entrywise difference 4.506e-05 (largest entry 4.361e+07, relative 1.03e-12).


#### Posterior mode, nkm, start = stored

| | OccBin | news |
|---|---|---|
| mode | (116.64, 1.9994, 0.70791, 0.68201, 0.90496, 0.88637, 0.0047961, 0.0023115, 0.0030526) | (116.64, 1.9994, 0.7079, 0.68203, 0.90496, 0.88636, 0.0047946, 0.0023123, 0.0030498) |
| log posterior at the mode | 1686.7402668 | 1686.7454082 |
| optimiser | csminwel: smallest step still improving too slow, reversed gradient (16 iterations, 389 evaluations) | csminwel: smallest step still improving too slow, reversed gradient (20 iterations, 530 evaluations) |
| likelihood evaluations (incl. Hessian) | 482 | 623 |
| evaluations that failed (-inf) | 20 | 33 |
| wall time of the run (s) | 152.1 | 25.7 |
| Hessian (finite differences, in the mode run) positive definite | False | False |
| posterior sd from the Hessian | n/a (Hessian not positive definite) | n/a (Hessian not positive definite) |
| Laplace approximation | nan | nan |

Same path: False. Evaluations: OccBin 482, news 623. First evaluation at which the iterates differ (> 1e-9): 29. |mode difference|: (2.17e-08, 2.39e-07, 6.56e-06, 1.66e-05, 9.41e-07, 6.19e-06, 1.53e-06, 7.72e-07, 2.75e-06).
News evaluated at the 482 points OccBin's run visited: 448 agree (largest difference 2.14e-11), 14 differ (largest difference 0.1210).
First differing evaluation: index 182 of OccBin's run, log posterior 1685.416711 (OccBin) against 1685.531919 (news); regimes differ in period(s) [65]: period 65: OccBin (((1, 0), (1, 8)),), news (((1, 0), (1, 7)),).

Time per evaluation (evaluations that ran the filter; the two solvers called alternately at the same points): OccBin min 0 ms, median 323 ms; news min 0 ms, median 40 ms (ratio of medians 8.0). Trial-point failures inside the filter during the mode search: OccBin {}, news {}.

#### Hessian and Laplace approximation, nkm, start = stored

| | OccBin | news |
|---|---|---|
| evaluations / wall time (s) | 91 / 30.9 | 91 / 3.6 |
| positive definite | False | False |
| posterior sd | - | - |
| Laplace approximation | - | - |

Both Hessians at OccBin's mode: largest entrywise difference 6.759e-05 (largest entry 4.801e+07, relative 1.41e-12).


Reading: from the prior mean neither solver converges (csminwel: "smallest step still improving too slow"); they end at different points of a flat
region (log posterior 1693.1306 for OccBin, 1692.7513 for news; parameters differing by (0.032, 0.0096, 0.073, 0.056, 0.030, 0.0076, 6e-5, 8e-4, 3e-5)). The paths differ from evaluation 46 on (iterates), and the first likelihood difference is at index 170: a
spell of 9 periods against 8 (OccBin / news) in period 47. From the stored vector both stop at the same place (parameters equal to 1.7e-5 or better, log posterior
1686.7403 and 1686.7454), again without convergence. The Hessians at all four points are not positive definite: each has **one negative eigenvalue**
(in correlation form -0.65 (OccBin) and -1.25 (news) at the prior-start points, -1.02 and -1.11 at the stored-start points; the other eight between 0.64 and 2.02), i.e. the
point is not a maximum of this likelihood surface; both solvers agree on that, and on the Hessian itself (relative difference 1e-12 at the same point).
No Laplace approximation exists for the full model with either solver at these points. Trial-point failures inside the filter during the two searches from the
prior mean: OccBin 164 (41 x 313, 123 x 310), news 3 (313); from the stored vector: none for either. Failed evaluations in the searches by Dynare's code, from the prior mean: OccBin 41 x 43, 42 x 2, 4 x 2, 304 x 1; news 41 x 52, 42 x 8 (codes 41/42: the optimiser steps outside the prior's support); from the stored vector: OccBin 41 x 20; news 41 x 30, 42 x 3.
Wall time of a mode search: 302 s against 38 s (7.9 x; 1035 against 989 evaluations) and 152 s against 26 s (5.9 x; 482 against 623).

### 2c. Metropolis-Hastings

Small model:

#### Random-walk Metropolis, nkms

2 chains x 2000 draws, seed 20261124, proposal: inverse Hessian at OccBin's mode, jscale 1.3 (pilot: 0.25: 0.83, 0.4: 0.75, 0.6: 0.64, 0.9: 0.51, 1.3: 0.34, 1.8: 0.17, 2.4: 0.09, 3.2: 0.05), same random numbers and starting points for both solvers; first half of each chain dropped.

| | OccBin | news |
|---|---|---|
| acceptance rate per chain | 0.357, 0.338 | 0.357, 0.338 |
| wall time per chain (s) | 501, 465 | 80, 77 |
| total wall time (s) | 966 | 156 |
| median seconds per evaluation | 0.2289 | 0.0352 |
| likelihood evaluations | 4002 | 4002 |
| draws rejected with -inf, by Dynare's code | {'4': 75} | {'4': 75} |
| trial-point failures inside the filter (solver flag: count) | {} | {} |
| posterior mean | (1.2408, 0.23418, 0.037222) | (1.2408, 0.23418, 0.037222) |
| posterior sd | (0.069662, 0.028149, 0.0037301) | (0.069662, 0.028149, 0.0037301) |

Draw for draw:

- chain 0: identical draws: True; largest difference in a draw 0.00e+00; evaluations at the same parameter vector with different likelihood: 5; first at evaluation 56: 1518.334489 (OccBin) against 1518.310119 (news), difference 0.0244, regimes differ in period(s) [66].
- chain 1: identical draws: True; largest difference in a draw 0.00e+00; evaluations at the same parameter vector with different likelihood: 12; first at evaluation 46: 1517.598706 (OccBin) against 1517.574352 (news), difference 0.0244, regimes differ in period(s) [66].


Full model (the mode is not a maximum, so the chains start from draws around OccBin's best point with the prior variance as proposal; this is a test of
the two solvers on identical chains, not an estimate of the posterior: 1500 draws are far too few for 9 parameters, 15 minutes of OccBin compute):

#### Random-walk Metropolis, nkm

2 chains x 1500 draws, seed 20261124, proposal: prior variance (the Hessian at the mode is not positive definite), jscale 0.07 (pilot: 0.02: 0.66, 0.04: 0.43, 0.07: 0.21, 0.1: 0.11, 0.15: 0.01, 0.25: 0.00, 0.4: 0.00, 0.6: 0.00), same random numbers and starting points for both solvers; first half of each chain dropped.

| | OccBin | news |
|---|---|---|
| acceptance rate per chain | 0.190, 0.200 | 0.190, 0.200 |
| wall time per chain (s) | 473, 462 | 64, 64 |
| total wall time (s) | 935 | 128 |
| median seconds per evaluation | 0.2923 | 0.0384 |
| likelihood evaluations | 3002 | 3002 |
| draws rejected with -inf, by Dynare's code | {} | {} |
| trial-point failures inside the filter (solver flag: count) | {} | {} |
| posterior mean | (123.89, 1.8111, 1.2144, 0.70297, 0.87632, 0.90699, 0.0045067, 0.0028771, 0.0027688) | (123.89, 1.8111, 1.2144, 0.70297, 0.87632, 0.90699, 0.0045067, 0.0028771, 0.0027688) |
| posterior sd | (14.761, 0.074266, 0.14678, 0.029275, 0.014398, 0.0093951, 0.00038467, 0.00040993, 0.00017705) | (14.761, 0.074266, 0.14678, 0.029275, 0.014398, 0.0093951, 0.00038467, 0.00040993, 0.00017705) |

Draw for draw:

- chain 0: identical draws: True; largest difference in a draw 0.00e+00; evaluations at the same parameter vector with different likelihood: 10; first at evaluation 83: 1688.093407 (OccBin) against 1688.056837 (news), difference 0.0366, regimes differ in period(s) [41].
- chain 1: identical draws: True; largest difference in a draw 0.00e+00; evaluations at the same parameter vector with different likelihood: 8; first at evaluation 234: 1692.869440 (OccBin) against 1692.896984 (news), difference 0.0275, regimes differ in period(s) [69].


Reading. The chains are identical draw for draw in all four pairs (largest difference in a draw 0; the `accepted` arrays are identical). The
likelihoods are not: at 5 and 12 (small model, chains 0 and 1) and 10 and 8 (full model) of the evaluations the two log posteriors differ by 0.02-0.04, and the
recorded log posterior of an accepted draw then differs too (largest 0.044 small, 0.029 full), but no accept/reject decision changed. So the posterior means and standard deviations are identical to all digits shown; a longer chain, or draws near a decision margin, would
eventually separate the chains at the first such draw. Rejections: none by the filter in either solver, either model; the small model's 75 -inf
evaluations (about 1.9 %) are Blanchard-Kahn indeterminacy (code 4) for both, at the same draws. Trial-point failures inside the filter: none, for both solvers.

### Why the likelihoods differ at a few points: a knife edge (`bench/obc_knife_edge.py`)
This model is the "approximate" class (contract section 3.6): `inom = inomlb*exp(mp)` against `bind inomnot <= inomlb`; the linearised equation and the bind condition differ by a term of
0.6 %. OccBin decides the regime with the condition, the news solver with the linearised gap. When the two are within rounding of each other they decide
differently. Examples, each at the first differing evaluation:
- small model, evaluation 59 of the mode search, period 41: the linearised gap without wedges is -3.4e-6, so the news solver binds for one period (wedge
  3.6e-6, i.e. 1.03 x the gap); OccBin's condition is false by 1.18e-6 on OccBin's solution (OccBin's regime: not binding), and OccBin's own test applied to the
  news path fails by 1.06e-6 in that period. The log likelihood differs by 0.0265.
- full model, mode search from the prior mean, index 170, period 47: OccBin's spell has 9 periods and passes its own test with a margin of 9.6e-9 in period 8;
  the news spell has 8 periods and fails OccBin's test in period 8 by 8.3e-9. Difference 0.124.
- the MCMC examples: period 66 (small, two chains), 41 and 69 (full): the same pattern (regimes differ in one period).
The likelihood's Jacobian term depends on the regime (T, R of the period), hence a jump of 0.02-0.12 although paths and shocks agree to 1e-6 or better (the soak test of docs/OBC_SOAK.md, which looks much further from the modes, found larger jumps: see the note at the end of 2d).
This is a property of the formulation on this model, not an error of either solver: on a model of the exact class (bf, tested above) there is no difference.

### 2d. Verdicts
- **Same surface?** Yes, up to knife-edge points. At the points visited, the two log posteriors agree to 1e-10 except at a few where they differ by
  0.02-0.12 because one regime decision lies within 1e-6 of OccBin's condition: 4 of 115 evaluations on the small model's mode path, 32 of the 987 finite ones on the full
  model's, 17 of 4002 in the small chains, 18 of 3002 in the full chains. Hessians agree (relative 1e-10 and 1e-12), the chains are identical draw for draw, the posterior
  means and sd identical; the small model's modes agree to 6e-6, but the full model's mode searches (which do not converge) end at different points.
- **Size of the knife-edge differences far from the mode (OBC soak test, `docs/OBC_SOAK.md`).** The "0.02-0.12" above holds for the points visited by the mode searches and
  chains. In the soak sweeps (prior draws, perturbations up to 2x, lines between vectors, near-bound vectors; NKM full and small, inversion filter, horizon 40), of
  8,279 pairs with both likelihoods finite, 195 differ by more than 1e-8 (relative): 179 by more than 0.02, 60 by more than 0.12, 25 by more than 0.2, 9 by more than 0.4,
  5 by more than 0.8, the largest 1.54 (at log posterior -1437; one pair at log posterior -8e12 differs by 1.7e6, relative 2e-7). Every one inspected is a one-period shift of a
  regime decision (binding periods differ by one, or the same count with a shifted spell); the differences also occur at a news posterior mode (0.23, equal count, shifted
  sequence) and at MCMC draws (0.21). So the size of a knife-edge jump is better stated as "up to about 1.5 at vectors of ordinary likelihood" than "up to 0.12".
  bf (exact class) showed no such difference above 1e-7 (relative).
- **Faster?** Per evaluation 6.8 to 8.3 x (median 0.034-0.037 s against 0.24-0.30 s); end to end: small model mode search 27.6 s -> 5.1 s (5.4 x, with 15 % more
  evaluations), Hessian 3.4 -> 0.4 s, 2 x 2000 draws 966 s -> 156 s (6.2 x); full model mode search 302 -> 38 s (7.9 x), 2 x 1500 draws 935 -> 128 s (7.3 x).
- **At least as reliable?** In these runs yes: no filter failure in any chain; at all 210 trial points that the cheap checks could not decide (bf, NKM) OccBin failed too;
  in the mode searches from the prior mean the news solver had fewer trial-point failures (3 against 164) and no failed evaluation of code 304 or 4 (OccBin: 1 and 2; the other
  failed evaluations, 48 and 60, are the optimiser leaving the prior's bounds, codes 41 and 42, and the paths differ); the caveat is that the integer programme is
  off, so a state with a solution that only an integer programme finds would count as a failure (none seen), and that
  non-default `obc_selection` values can make the filter fail (Task 3, 'longest').

## 3. Equilibrium selection, `obc_selection`

`obc_method='news'`, one constraint:
- `'occbin'` (default): unchanged (what the guess-and-verify / active-set iteration reaches from the no-binding guess); `n_single_spell` = -1 ("not looked for").
- `'shortest'`, `'longest'`, `'min_wedge'`: in every period in which the default solution binds, enumerate the single-spell solutions (spell {l..l+k-1},
  l <= 8 (`spell_l_max`), any k up to the horizon) and take the one with the fewest / most binding periods, or the smallest sum of squared wedges (all wedges of
  the constraint); ties: the earliest start. If none exists the default's answer stands and the period is flagged. Periods where nothing binds are not enumerated.
- `NewsOBC.single_spells(q)`: solves the square system on each candidate spell and verifies the complementarity problem (wedges positive, gap >= 0 on all rows,
  check-ahead rows included). `RegimeOBC.spells(...)`: the same for the general formulation, each candidate accepted by OccBin's own test (conditions in
  levels), as `RegimeOBC` defines solutions. The older `RegimeOBC.single_spell_solutions` is unchanged.
- `res.news['selection']`, `res.news['n_single_spell']` (per shock period: number found, -1 not looked for), `res.news['selection_fallback']` (True where none was found).
- Plumbing: `Occbin.simulate(obc_selection=)` (so `run()` and `irf_occbin()`), `options_.occbin.obc_selection = '...';` through `mod_options`, `OccbinEstimation(obc_selection=)`
  and the filter solvers. Unknown values: `ValueError`. Non-default with two constraints: `ValueError` ("implemented for models with one constraint only"). Non-default with
  `obc_method='occbin'`: `ValueError`.

Tests (`tests/test_obc_selection.py`, 6; `test_selection_*` in `tests/test_obc_estimation.py`, 2): Smets-Wouters benchmark, -4 shock to `eb`: `occbin` and `shortest` give the 9-period spell, `longest`
the 13-period one, `min_wedge` the 9-period one; every returned path passes OccBin's own test; default and `shortest` equal OccBin's path to 1e-9, `longest` is more than 1 away;
`n_single_spell` = 2. On `obc_zlb.mod` (unique solution) all four equal OccBin's path to 1e-11 over 200 random shock periods with identical regime histories, one single-spell
solution in every binding period (205 of them in the sample used for the cost table; more than 20 in the test's); the default is bit for bit unchanged; the fallback flag is tested by removing the candidates; errors; the native line and `irf_occbin`.

Cost per binding period (`bench/obc_selection_cost.py`; the per-period step on binding states, the four selections interleaved, minimum of 5 per state, median over states):

| model | binding states | default (ms) | shortest | longest | min_wedge | single-spell solutions per state |
|---|---|---|---|---|---|---|
| obc_zlb (NewsOBC, 5 variables) | 205 | 0.057 | 2.996 | 2.986 | 2.997 | 1 in all |
| NKM small (NewsOBC, 24 variables, T = 40) | 31 | 0.056 | 3.095 | 3.106 | 3.109 | 2 in all |
| Smets-Wouters (RegimeOBC, 32 variables, T = 40, check-ahead 60) | 10 | 0.077 | 12.47 | 12.67 | 12.67 | 2 in all |

So a non-default selection costs about 3 ms (NewsOBC) and 12.5 ms (RegimeOBC) per binding period, about 50 and 160 times the default, independent of which of the three is chosen
(the enumeration is the cost: 9 start periods x up to 40 lengths). Not optimised.

A finding on the NKM example: **every binding state of the NKM small model has a second single-spell solution**: a 28-period spell with a first-period wedge of about 1.8 (against 0.003 for
the usual one), a deflationary-trap solution of the LCP. The fourth entry of docs/OBC_PROGRESS.md reports no second solution on the NKM states it checked; its enumeration limits are not stated there, mine goes to spells of 40 periods. With `obc_selection='longest'` the inversion filter on
this model fails (info 304: its trial shocks hop between branches and the root finder does not converge), whereas `'shortest'` and `'min_wedge'` reproduce the default likelihood to 1e-9
(`test_selection_in_the_filter`). Because `NewsOBC` checks candidates on the linearised problem, not on OccBin's levels conditions, that second solution is a solution of the LCP and
not necessarily of OccBin's problem; `RegimeOBC` candidates are checked by OccBin's test.

## Not done, not verified
- The piecewise Kalman filter with the news method: refused with an error (out of scope).
- Selection with two constraints: refused. `spell_l_max = 8`: spells starting later than period 8 are not enumerated (the benchmarks' solutions start at 0).
- A converged posterior mode for the full NKM model with the inversion filter: not obtained with either solver (see 2a); Laplace approximation for it: not available.
- MH runs are short (2 x 2000 and 2 x 1500 draws; one seed); no convergence diagnostics beyond identity of the chains; the full-model chains are not an estimate of the posterior.
- The claim that switching the integer programme off never changes a likelihood is supported by 210 trial points and 167 vectors, not proved.
- Timings of whole runs (mode search, chains) were not measured under identical load for the two solvers except in the chains (alternating chain by chain); the other run times are one run each.
- `bench/obc_est_study.py` runs were made on frozen clones (commits before the last edits to the study scripts and to this file); source files under `pydyn/` were the same.
- JDyn: nothing done here.

## OccBin speed-up (port of JDyn's), verification
`pydyn/occbin.py`: constraint closures built once per model and shared; bind/relax sides evaluated once per condition; rules memoised in a trie keyed from the end of the
guess; allocation-free `path`; `tmax` vectorised. Frozen pre-change copy: `bench/occbin_before.py`. Exact-equality check: `tests/test_occbin_bitwise.py` (simulation on obc_zlb, obc_zlb2
(two constraints), NKM, Smets-Wouters, at two shock scales incl. runs that end in OccbinError: paths, regime histories, iterations, rules, diagnostics, errors compared with
`np.array_equal`; both filters (piecewise Kalman, inversion) on the NKM and two-constraint rbc estimation examples: log posterior, regimes, states `==`), and the full-size
script `bench/occbin_equality.py` (50 parameter sets x 200 periods x 4 models x 2 scales: ALL IDENTICAL). Timing (`bench/obc_bench.py p1`, 1000 periods, median of 5, microseconds per
period, before -> after; raw `bench/results/occbin_speedup_p1.json`): zlb 552 -> 214, zlb2 675 -> 290, nkm 684 -> 222, sw 370 -> 190.

## `minnorm` alias
`obc_method='minnorm'` (JDyn's name) is resolved by `pydyn.occbin.canonical_obc` to `obc_method='news'`, `obc_selection='min_wedge'`: `Occbin.simulate` / `run` / `irf_occbin`,
`OccbinEstimation` (constructor, assignment; `est.obc_method` then reads `'news'`), and the native line `options_.occbin.obc_method = 'minnorm';`. `minnorm` with another explicit
selection: `ValueError`. With two constraints: the existing `ValueError` of non-default selections. Tests: `test_minnorm_is_news_with_min_wedge` (tests/test_obc_selection.py),
`test_minnorm_alias_in_the_estimation` (tests/test_obc_estimation.py). This is the alias only: it is NOT MacroModelling.jl's minimum-norm news-shock programme.

## Singular solves (task 3)
Defect: DynareOBC's `Simple1` (`reference/dynareobc_octave/simple1_max.mod`, OccBin form `simple1_occ.mod`), shock epsilon = -0.02 (r + phi pi < 0 in period 1, no solution; DynareOBC: "Impossible problem"):
`news` returned pi = -1.2e14. Cause: `M[0,0]` = 5.6e-17 (rounding noise: a period-0 wedge does not move the period-0 gap), the 1x1 active-set solve gave a wedge of 1.8e14, which satisfies
`verify_lcp` (g = 0 to rounding, y >= 0), and the same happened in the MILP's polish step and in `single_spells`. Fix (`pydyn/obc.py`, `_solve_checked`): every square solve on a guessed binding set
(`solve_active_set`, the polish in `solve_news`, `NewsOBC.single_spells`, `RegimeOBC._wedges`) rejects a block whose LU has a pivot <= 1e-10 x (largest entry of the whole M); the active set
then fails, the MILP (which finds alpha ~ 0) and its check-ahead test run, and `OBCError('no_solution')` is raised. Slack shocks (-0.01) still solve. `occbin` on this model raises
numpy's `LinAlgError` (singular matrix), not changed. Tests: `test_active_set_rejects_numerically_singular_block`, `test_simple1_impossible_problem_raises` (4 selections). Cost on the fast path
(1000 periods, min of 15, interleaved with the old code, same results bit for bit): NewsOBC zlb 25.4 -> 25.7 us/period, zlb2 33.2 -> 33.8, nkm 33.3 -> 34.0; RegimeOBC zlb 45.0 -> 46.1,
zlb2 70.2 -> 72.8, nkm 58.2 -> 60.3, sw 34.8 -> 35.2 (1 to 4 %).
Not guarded: `obc_filter.py` lines solving on a set found by the guarded solvers (linearisation of an accepted solution), `obc_lk.py`, `RegimeOBC.single_spell_solutions`.

## Orientation test (task 4, contract rule 7)
`_constraint_objects` now requires `b s_y <= 0` and `e s_g >= 0` (tolerance 1e-6 relative to |a| and |c|) after the entry/exit fits; otherwise the constraint is `inconsistent` ("the relax condition points the
wrong way"). Checked on obc_zlb with `relax rn < rbar`, `rn <= rbar`, `rbar > rn` (all now inconsistent; they were `complementarity` with s_y = -1) and with the usual pair and no relax (unchanged).
Test: `test_orientation_rule_7` (5 cases) in tests/test_obc.py. All existing classification tests (Dynare example files) unchanged.

## State / next step
- Done: tasks 0 to 4 (merge, OccBin speed-up verified bit for bit, `minnorm` alias, singular-solve guard, orientation test). OBC + OccBin tests: 221 passed, 2 skipped.
- Next: nothing outstanding from this task list. Not done: JDyn side of any of this; `RegimeOBC.single_spell_solutions`, `obc_lk.py` and filter linearisation solves not guarded against singular blocks.

## Approx-class constraints: LCP, regime and hybrid formulations (study of branch obc-soak)
Question: for constraints that `constraint_objects` classifies as `complementarity` with `approx > 0` (the equation's two versions differ by a term the
condition lacks; accepted up to 5%), the news LCP can land on a different regime sequence than OccBin's iteration. Should the regime formulation
(`RegimeOBC`, one wedge per switched equation, accepted by OccBin's own bind/relax test in levels) be the default for them?  **No default was changed.**

**Option.** `obc_formulation` = `'auto'` (default: LCP when every constraint is one complementary pair, else regime; unchanged behaviour), `'lcp'`, `'regime'`,
`'hybrid'` in `Occbin.simulate(..., obc_method='news', obc_formulation=...)` and `OccbinEstimation.obc_formulation` (also `NewsFilterSolver(formulation=)`).
`lcp` and `hybrid` raise `ValueError` for models with a constraint that is not one complementary pair. `hybrid` = `pydyn.obc.HybridOBC`: solve the LCP by the
active-set solver only (no MILP), take the implied regime sequence, evaluate OccBin's bind/relax test in levels along the **whole anticipated path** (horizon plus
check-ahead rows, not only the current period), accept if it passes, else fall back to the regime solver for that period. Tests: `tests/test_obc_formulation.py` (6).

**Models** (`soak/approx/gen.py`, `models.py`; discrepancy d = relative residual of the entry/exit test from `constraint_objects`). Controls and approx variants
are built from the exact test models zlb / zlb2 / nk_occ by multiplying the rate equation of both regimes by `(1 + km*m)` (a policy shock, as in NKM) while the condition
stays `bind rn < rbar`; the mismatch is (steady-state gap 0.008) x km x m, d = 0.008 km, with km chosen for each d. d = 0 controls use an additive shock `r = rn + km m`.
`zlbm_*` (one constraint, d = 0, 1e-3, 1e-2, 5e-2 -> 0, 0.00100, 0.0100, 0.0499), `zlb2m_*` (same plus a second, exact, constraint), `nkocc_*` (nk_occ; gap and
parameter ranges limited d to 0.00100, 0.00426, 0.00747). Estimated: phi, rho, sige, sigm; observables pi, r (R); 150 periods of OccBin-simulated data (first seed
for which OccBin solves every period). Existing models: NKM, NKMS (`occest_nkm/nkms`, d = 0.0057, `NKM_*` also 0.0057) are the only approx models in the repository;
borrcon, obc_zlb, zlb2, occ_borrcon, nk_occ, plt_occ, simple1_occ, fig_occ1, occ1 are exact `complementarity`; bf, bf_ppf, obc_sw_zlb are `multi_equation`;
rbc_occbin, occ_rbc* , occest_rbc contain an `inconsistent` constraint (relative residual 1).

**Estimation, log posterior at prior draws, perturbations and 13-point line sweeps (`soak/approx/study.py E`; pairs = both OccBin and the method finite).**
Difference = |log posterior(method) - log posterior(OccBin)|; "diff" means above 1e-9 relative.
| model | d | method | pairs | diff | >0.12 | >0.4 | max | only-OccBin-finite | only-method-finite | both fail | median s / eval (method; OccBin) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nkm | .0057 | lcp / regime / hybrid | 457 | 27 (5.9%) / 0 / 0 | 14 / 0 / 0 | 2 / 0 / 0 | 0.73 / 0 / 0 | 1 / 1 / 1 | 0 / 0 / 0 | 142 | .034 / .043 / .047; .135 |
| nkms | .0057 | lcp / regime / hybrid | 563 | 28 (5.0%) / 0 / 0 | 6 / 0 / 0 | 0 | 0.22 / 0 / 0 | 0 | 0 | 37 | same; .131 |
| nkocc_1e-3 | .0010 | lcp / regime / hybrid | 569 | 20 (3.5%) / 0 / 0 | 15 / 0 / 0 | 5 / 0 / 0 | 0.85 / 0 / 0 | 0 | 0 | 31 | .034 / .043 / .046; .119 |
| nkocc_5e-3 | .0043 | lcp / regime / hybrid | 558 | 17 (3.0%) / 0 / 0 | 13 / 0 / 0 | 0 | 0.39 / 0 / 0 | 0 | 0 | 42 | same; .117 |
| nkocc_1.5e-2 | .0075 | lcp / regime / hybrid | 559 | 91 (16.3%) / 0 / 2 (0.4%) | 75 / 0 / 0 | 2 / 0 / 0 | 0.59 / 0 / 5e-4 | 0 | 5 / 0 / 0 | 36 / 41 / 41 | same; .119 |
| zlbm_0 (exact) | 0 | lcp / regime / hybrid | 574 | 0 / 0 / 0 | 0 | 0 | 0 | 0 | 0 | 26 | .035 / .044 / .048; .125 |
| zlbm_1e-3 | .0010 | lcp / regime / hybrid | 418 | 57 (13.6%) / 0 / 0 | 56 | 56 | 4e4 / 0 / 0 | 0 | 146 / 0 / 0 | 36 / 182 / 182 | .034 / .044 / .047; .121 |
| zlbm_1e-2 | .0100 | lcp / regime / hybrid | 439 | 208 (47%) / 0 / 2 (0.5%) | 194 / 0 / 1 | 185 / 0 / 1 | 572 / 0 / 1.44 | 0 / 0 / 1 | 109 / 0 / 3 | 52 / 161 / 158 | same; .122 |
| zlbm_5e-2 | .0499 | lcp / regime / hybrid | 365 | 365 (100%) / 0 / 4 (1.2%) of 338 | 365 / 0 / 1 | 365 / 0 / 1 | 305 / 0 / 0.97 | 0 / 0 / 27 | 207 / 0 / 49 | 28 / 235 / 186 | same; .121 |
| zlb2m_0 (exact) | 0 | all | 557 | 0 | 0 | 0 | 0 | 0 | 0 | 43 | .041 / .053 / .056; .163 |
| zlb2m_1e-3 | .0010 | lcp / regime / hybrid | 298 | 93 (31%) / 0 / 2 (0.7%) | 90 / 0 / 2 | 86 / 0 / 2 | 9e3 / 0 / 1.31 | 0 / 0 / 0 | 266 / 0 / 4 | 36 / 302 / 298 | .039 / .084 / .096; .221 |
| zlb2m_1e-2 | .0100 | lcp / regime / hybrid | 248 | 232 (94%) / 0 / 11 (4.7%) of 233 | 224 / 0 / 6 | 210 / 0 / 3 | 101 / 0 / 1.27 | 0 / 0 / 15 | 308 / 0 / 76 | 44 / 352 / 276 | .038 / .078 / .089; .208 |
| zlb2m_5e-2 | .0499 | lcp / regime / hybrid | 130 | 130 (100%) / 0 / 1 of 29 | 130 / 0 / 1 | 130 / 0 / 0 | 198 / 0 / 0.37 | 0 / 0 / 101 | 447 / 0 / 83 | 23 / 470 / 387 | .040 / .067 / .075; .187 |
Reading it. (i) **Regime formulation: 0 pairs above rounding in all 13 models** (about 5,900 pairs), and it has exactly OccBin's failure set (both fail = same points; where
OccBin fails the regime solver fails too: the 'both fail' column is OccBin's own cycling or non-convergence, which grows from ~5% at d=0 to 50-80% at the perturbed
parameter vectors of the d >= 1e-3 models with a second regime change, so those models yield fewer pairs). (ii) **LCP**: disagreement share grows with d on the zlb models
(0 -> 14% -> 47% -> 100% for d = 0, .1%, 1%, 5%, one constraint) and is 3-6% (NKM, nk_occ-based, d about 0.1-0.7%); most differences are large when they occur (a
different regime sequence moves the log posterior by the Jacobian term: >0.4 in 40-100% of the differing pairs at the zlb models), and the LCP returns finite values where OccBin
fails (only-method-finite; those values are not comparable to OccBin's). The frequency scales with d and with how often the constraint binds near the data, the size does not
(a switch, not a smooth bias). (iii) **Hybrid**: equals OccBin to rounding in the 7 models with d <= 0.0075 except 2 pairs at nkocc_1.5e-2 (5e-4), and disagrees in 0.5-5% of pairs
at the zlb models with d >= 1e-3 (max 1.4), plus loses finite values (only-OccBin-finite: up to 101 of 130 at zlb2m_5e-2): it can be worse than the regime solver.
**NKM 60-point check** (`soak/approx/nkm60.py`; the logged part A points with |occbin - news| > 0.12 on nkm/nkms, 81 vectors since the sweep logs more than the 60 of the first soak
note): LCP differs from OccBin at all 81 (up to 1.54); **the regime formulation and the hybrid bring all 81 to rounding (max 0)**.

**Why the hybrid still disagrees (one example, `zlbm_1e-2`, soak task 693, point 12).** OccBin and the regime solver have the constraint binding in periods [11, 13]; the hybrid
in [10, 11, 13]. At period 10 the hybrid took the LCP regime sequence, it passes OccBin's bind/relax test along the whole path (the check is exactly that
test, 302 of 303 LCP paths were accepted), so it is a valid piecewise-linear solution of that period too, yet not the one OccBin's guess-and-verify iteration lands
on: the inversion filter reproduces the observables with a different shock and regime, and the log posterior moves by 1.44 (the shock Jacobian). So *passing the
levels test does not make the path the one OccBin returns* (multiplicity near regime changes: the test has more than one fixed point, OccBin's iteration picks one by its guess
order, the LCP active set another), and the difference then changes the state history of later periods. Not shown by an independent enumeration of all solutions (only the two found).

**Simulation (`soak/approx/study.py F`, 1000-period segments carried on from the OccBin path, shock scale 0.4-1.2 sd, parameter perturbations 0-3%; per segment).**
| model | d | segments | LCP: segs diff (max, periods with different regime) | regime | hybrid | time per 1000 periods: lcp / regime / hybrid / OccBin |
|---|---|---|---|---|---|---|
| nkm (NKM_0_std_shocks) | .0057 | 90 | 11 (6e-5; 14) | 0 | 0 | .01 / .04 / .04 / .26 |
| nkm_est | .0057 | 120 | n/a: not LCP-class for 30 (guard raises) / 9 (1e-4) | 0 | 0 | same |
| nkocc_1e-3 / 5e-3 / 1.5e-2 | .0010 / .0043 / .0075 | 150 / 148 / 85 | 8 / 11 / 25 (6e-6 / 2e-5 / 6e-5) | 0 / 1 / 1 (1.4e-5, 2.3e-5) | 1 / 2 / 2 | .01-.02 / .04 / .04 / .25 |
| zlbm_0 / zlb2m_0 (exact) | 0 | 150 / 150 | 0 | 0 | 0 | .01 / .03 / .04 / .24 |
| zlbm_1e-3 / 1e-2 / 5e-2 | .001 / .01 / .05 | 149 / 84 / 59 | 19 / 24 / 45 (1e-4 / 1.2e-3 / 4.6e-3) | 0 | 0 / 8 / 7 | .01 / .04 / .04 / .25 |
| zlb2m_1e-3 / 1e-2 / 5e-2 | .001 / .01 / .05 | 142 / 37 / 13 (LCP 7) | 13 / 9 / 5 (1e-4 / 9e-4 / 3e-3) | 0 | 4 / 2 / 2 | .01 / .04-.08 / .05 / .3 |
Simulation differences are small (<= 5e-3 in levels) and rare (a few periods per segment), because in simulation the same shocks are fed to both solvers and a
different regime at a knife edge moves the path only a little; in estimation (inversion) the shock is solved for, so a regime change shifts the log posterior by a Jacobian term.
The regime formulation equals OccBin in simulation except 2 periods in nkocc_5e-3/1.5e-2 (2e-5; the OccBin sequence not reproduced at a knife-edge of 1e-6, as in 2d).

**Cost.** Per evaluation (150 periods): LCP 0.034 s, regime 0.043 s (x1.3; 0.078 at zlb2m, x2), hybrid 0.047 s, OccBin 0.12-0.2 s. Per period in simulation: LCP 0.01 ms-scale,
regime 0.04 s per 1000 periods (x3-4 the LCP), hybrid about the same as regime (the extra LCP solve is repaid by skipping the regime search when accepted). The whole-estimation
cost ratio is therefore 1.3x (estimation, one constraint) to 2x, and 3-4x in simulation. **Exact class (`zlbm_0`, `zlb2m_0`, the Dynare test models): `auto` is unchanged; bit-identical log posterior
(1528.0919683511543) and simulation sum against 3fdb1e7; time 0.976 vs 1.033 s per 30 evaluations (noise), simulation 0.011 vs 0.014 s per 1000 periods.**

**Recommendation (for the user to decide; nothing changed).** For approx-class models use the **regime formulation** in estimation: it is OccBin-identical in every one of
13 models and ~5,900 pairs (including the 81 NKM points), whereas the LCP differs in 3-100% of pairs with a share growing with d, at +30% (to 2x) cost; also in simulation if the
OccBin path matters at the 1e-5 level, at 3-4x cost. Do not make `hybrid` the default: it is never better than regime and loses finite points at d >= 1%. A safe rule: default
`regime` when `approx > 0` (above a small threshold such as 1e-4 given the knife-edge level), `lcp` for exact classes (unchanged).

**Not established.** Pairs are 130-570 per model, not thousands, because OccBin itself fails at 5-80% of the perturbed vectors on the d >= 1e-3 zlb models; no model with
d in (0.0075, 0.05) and a non-zlb structure; constraints of the `multi_equation` classes (only regime applies) not studied; the multiplicity explanation rests on one
replayed example, not on enumerating solutions; "current-period-only" levels test for the hybrid was not measured (the whole path was used); exact-class timing is from one machine, 5 repeats;
whether OccBin's non-convergence at those points means the regime formulation's finiteness is the same event for every cause (the failure sets are identical, causes not tabulated).

## Decision (25 Nov): `obc_formulation='auto'` sends approximate-class models to the regime formulation

Following the approx-class study above (Jacob's decision): under `'auto'` the LCP solver is used only when every
constraint is one exactly complementary pair (`ConstraintObject.approx == 0`); any approximately complementary
pair, like any multi-equation constraint, goes to the regime formulation, in simulation and in estimation. Exact
models are unchanged bit for bit. `'lcp'` remains available on approx-class models for speed.

Consequences seen when the default changed:
- NKM under `obc_method='news'` now uses `RegimeFilterSolver`; values at the stored vectors are unchanged to 1e-11.
- The regime formulation can need a longer horizon than the LCP: at the extreme vector of
  `tests/test_obc_horizon_cause.py` the LCP recovers OccBin's value with horizon 80, the regime formulation needs
  160 (it fails with `horizon_too_short` at period 92 with 80). `obc_horizon_max=160` covers both.
- Higher order and cubature (`HigherOrderNews`) still use the LCP for approx-class models.
