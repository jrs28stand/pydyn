# OBC soak test (pydyn, branch `obc-soak`, from `wip/obc` 0abe8e7)

Brute-force soak test of the occasionally-binding-constraint estimation and simulation code, to find bugs that appear only over many
parameter vectors or long runs. Library code is not changed; anomalies are catalogued with reproducers (`soak/repro/`).

## State / how to resume

(written by hand; see the end of the file for the numbers)

## Coverage so far

| part | run time (h) | tasks | what |
|---|---|---|---|
| A | 0.41 | 969 | evaluations of the log posterior (all configurations): bf 4660, nkm 4903, nkms 4752; tasks by kind: bf/bound 80, bf/line 81, bf/perturb 81, bf/prior 81, nkm/bound 80, nkm/line 81, nkm/perturb 81, nkm/prior 81, nkms/bound 80, nkms/line 81, nkms/perturb 81, nkms/prior 81 |
| B | 0.48 | 196 | 24600 MCMC draws (news) in 4 finished chains + the running one (chain 4, draw 600, replays under occbin of this chain: 60); 31 mode searches (2 skipped: start not finite, 0 timeouts) |
| C | 0.44 | 231 | simulated periods per model (both methods each): bf 233000 (29 runs, 8029 binding), borrcon 293000 (29 runs, 61581 binding), borrcon_d 376000 (29 runs, 37808 binding), nkm 377000 (29 runs, 33692 binding), nkm_est 246000 (28 runs, 15309 binding), sw 150000 (29 runs, 684 binding), zlb 366000 (29 runs, 129978 binding), zlb2 262000 (29 runs, 105223 binding) |
| D | 0.42 | 7715 | runs per model (ok/total): borrcon 616/1102, nk 1071/1103, nkm 586/1102, nkm_est 560/1102, sw_cp 386/1102, zlb 1038/1102, zlb2 1027/1102 |

LCP solutions checked with `verify_lcp` (tolerance 1e-6) at the `solve_active_set` / `solve_news` boundary, all parts: {'verify|solve_active_set': 283102, 'verify|solve_news': 41}.

Determinism checks (same call repeated): A same-process 208 (+48 in a fresh estimation object), C 38, D 1425.

## Anomaly catalogue

| kind | part | count | severity | first reproducer(s) | judgement |
|---|---|---|---|---|---|
| `nonbitwise_ulp:higher_order` | D | 346 | ? | soak/results/anomalies.jsonl (task 0, task 36, task 48) | not yet looked at |
| `path_disagree:nkm` | C | 61 | ? | soak/results/anomalies.jsonl (task 66, task 66, task 82) | not yet looked at |
| `path_disagree:nkm_est` | C | 25 | ? | soak/results/anomalies.jsonl (task 127, task 127, task 127) | not yet looked at |
| `variant_fails_where_news_ok:news-longest` | A | 23 | ? | soak/results/anomalies.jsonl (task 12, task 24, task 96) | not yet looked at |
| `slow_failure` | C | 20 | ? | soak/results/anomalies.jsonl (task 14, task 43, task 43) | not yet looked at |
| `lp_disagree_large:nkm:regime_count_differs` | A | 19 | ? | soak/results/anomalies.jsonl (task 6, task 6, task 162) | not yet looked at |
| `lp_absurd` | A | 9 | ? | soak/results/anomalies.jsonl (task 70, task 70, task 70) | not yet looked at |
| `finite_vs_inf:news_finite_occbin_code304` | A | 8 | ? | soak/results/anomalies.jsonl (task 30, task 193, task 285) | not yet looked at |
| `finite_vs_inf:occbin_finite_news_code304` | A | 6 | ? | soak/results/anomalies.jsonl (task 75, task 155, task 500) | not yet looked at |
| `path_disagree:bf` | C | 6 | ? | soak/results/anomalies.jsonl (task 43, task 83, task 83) | not yet looked at |
| `memory_jump` | A | 5 | ? | soak/results/anomalies.jsonl (task 141, task 273, task 309) | not yet looked at |
| `regime_history_differs_equal_path:nkm` | C | 5 | ? | soak/results/anomalies.jsonl (task 82, task 82, task 130) | not yet looked at |
| `slow_failure` | A | 5 | ? | soak/results/anomalies.jsonl (task 65, task 75, task 105) | not yet looked at |
| `memory_jump` | D | 4 | ? | soak/results/anomalies.jsonl (task 1782, task 2697, task 3313) | not yet looked at |
| `milp_timeout_hit` | A | 3 | ? | soak/results/anomalies.jsonl (task 65, task 75, task 105) | not yet looked at |
| `warning:RuntimeWarning: invalid value encountered in sqrt` | B | 3 | ? | soak/results/anomalies.jsonl (task 167, task 185) | not yet looked at |
| `discontinuity_same_regimes:bf` | A | 2 | ? | soak/results/anomalies.jsonl (task 404, task 404) | not yet looked at |
| `lp_disagree_large:nkms:regime_count_differs` | A | 2 | ? | soak/results/anomalies.jsonl (task 79, task 556) | not yet looked at |
| `lp_disagree_large:nkms:regime_count_differs` | B | 2 | ? | soak/results/anomalies.jsonl (task 103, task 130) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:sw` | C | 2 | ? | soak/results/anomalies.jsonl (task 30, task 86) | not yet looked at |
| `regime_history_differs_equal_path:nkm_est` | C | 2 | ? | soak/results/anomalies.jsonl (task 127, task 215) | not yet looked at |
| `driver_or_unexpected:TypeError:dict() got multiple values for keyword argument 'part'` | B | 1 | ? | soak/results/anomalies.jsonl (task 155) | not yet looked at |
| `lp_disagree_large:nkm:same_regime_count` | B | 1 | ? | soak/results/anomalies.jsonl (task 119) | not yet looked at |
| `lp_disagree_large:nkms:same_regime_count` | A | 1 | ? | soak/results/anomalies.jsonl (task 94) | not yet looked at |
| `memory_jump` | B | 1 | ? | soak/results/anomalies.jsonl (task 35) | not yet looked at |
| `memory_jump` | C | 1 | ? | soak/results/anomalies.jsonl (task 63) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm` | C | 1 | ? | soak/results/anomalies.jsonl (task 82) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm_est` | C | 1 | ? | soak/results/anomalies.jsonl (task 127) | not yet looked at |
| `path_disagree:sw` | C | 1 | ? | soak/results/anomalies.jsonl (task 118) | not yet looked at |
| `timeout` | A | 1 | ? | soak/results/anomalies.jsonl (task 105) | not yet looked at |

## Failure rates, part A (inversion filter, log posterior evaluations)

Distance bin = rms of the relative deviation from the model's initial values: a <0.02, b <0.1, c <0.3, d <1, e >=1; `line` = the 25 points of the straight-line sweeps (-0.1..1.1).

| model | config | n | finite | failed (by code) | by distance bin: finite share |
|---|---|---|---|---|---|
| bf | news | 2267 | 0.824 | c304 269, c41 22, c42 107 | a:1.00 (12) b:1.00 (15) c:0.80 (82) d:0.49 (72) e:0.51 (61) line:0.84 (2025) |
| bf | news-milp | 12 | 0.250 | c304 1, c41 1, c42 7 | a:1.00 (1) d:0.33 (3) e:0.12 (8) |
| bf | news20 | 49 | 0.633 | c304 6, c41 2, c42 10 | a:1.00 (1) b:1.00 (3) c:0.79 (14) d:0.62 (16) e:0.40 (15) |
| bf | news60 | 49 | 0.633 | c304 6, c41 2, c42 10 | a:1.00 (1) b:1.00 (3) c:0.79 (14) d:0.62 (16) e:0.40 (15) |
| bf | occbin | 2267 | 0.826 | c304 265, c41 22, c42 107 | a:1.00 (12) b:1.00 (15) c:0.80 (82) d:0.51 (72) e:0.52 (61) line:0.84 (2025) |
| bf | occbin60 | 16 | 0.812 | c304 3 | c:0.77 (13) d:1.00 (3) |
| nkm | minnorm | 81 | 0.975 | c304 1, c41 1 | c:0.97 (39) d:0.97 (35) e:1.00 (7) |
| nkm | news | 2267 | 0.890 | c3 2, c304 40, c4 9, c41 24, c42 171, c5 3 | a:1.00 (2) b:0.67 (3) c:0.87 (84) d:0.70 (82) e:0.37 (71) line:0.92 (2025) |
| nkm | news-longest | 81 | 0.691 | c304 24, c41 1 | c:0.69 (39) d:0.69 (35) e:0.71 (7) |
| nkm | news-milp | 11 | 0.455 | c0 1, c304 1, c4 1, c41 2, c42 1 | c:0.50 (2) d:0.25 (4) e:0.60 (5) |
| nkm | news-shortest | 81 | 0.975 | c304 1, c41 1 | c:0.97 (39) d:0.97 (35) e:1.00 (7) |
| nkm | news20 | 49 | 0.592 | c304 6, c4 1, c41 3, c42 9, c5 1 | b:0.00 (1) c:0.78 (18) d:0.53 (15) e:0.47 (15) |
| nkm | news60 | 49 | 0.592 | c304 6, c4 1, c41 3, c42 9, c5 1 | b:0.00 (1) c:0.78 (18) d:0.53 (15) e:0.47 (15) |
| nkm | occbin | 2267 | 0.888 | c3 2, c304 45, c4 9, c41 24, c42 171, c5 3 | a:1.00 (2) b:0.67 (3) c:0.87 (84) d:0.71 (82) e:0.32 (71) line:0.92 (2025) |
| nkm | occbin60 | 17 | 0.941 | c304 1 | c:0.88 (8) d:1.00 (7) e:1.00 (2) |
| nkms | minnorm | 81 | 0.741 | c4 14, c42 7 | a:1.00 (5) b:1.00 (6) c:1.00 (1) d:1.00 (6) e:0.67 (63) |
| nkms | news | 2267 | 0.942 | c304 10, c306 1, c4 89, c41 17, c42 12, c5 3 | a:1.00 (5) b:1.00 (14) c:0.96 (50) d:0.70 (50) e:0.51 (123) line:0.97 (2025) |
| nkms | news-milp | 9 | 1.000 |  | b:1.00 (1) c:1.00 (5) d:1.00 (2) e:1.00 (1) |
| nkms | news20 | 48 | 0.729 | c304 1, c306 1, c4 6, c41 2, c42 3 | a:1.00 (1) b:1.00 (6) c:0.89 (9) d:0.60 (10) e:0.64 (22) |
| nkms | news60 | 48 | 0.729 | c304 1, c306 1, c4 6, c41 2, c42 3 | a:1.00 (1) b:1.00 (6) c:0.89 (9) d:0.60 (10) e:0.64 (22) |
| nkms | occbin | 2267 | 0.941 | c304 11, c306 1, c4 89, c41 17, c42 12, c5 3 | a:1.00 (5) b:1.00 (14) c:0.94 (50) d:0.70 (50) e:0.51 (123) line:0.97 (2025) |
| nkms | occbin60 | 32 | 0.594 | c304 1, c306 1, c4 6, c41 2, c42 3 | a:1.00 (1) b:1.00 (2) c:0.00 (1) d:0.43 (7) e:0.62 (21) |

occbin vs news, both finite: |difference in log posterior|

| model | pairs | <=1e-9 (rel) | 1e-9..0.15 | >0.15 | max |
|---|---|---|---|---|---|
| bf | 1869 | 1864 | 3 | 2 | 1 |
| nkm | 2012 | 1950 | 43 | 19 | 0.874 |
| nkms | 2133 | 2103 | 27 | 3 | 0.404 |

## Timing distributions

Part A, seconds per finite log-posterior evaluation:

| model | config | n | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 1869 | 0.0307 | 0.2003 | 0.4317 |
| bf | news-milp | 3 | 0.0288 | 0.04379 | 0.0441 |
| bf | news20 | 31 | 0.0322 | 0.04953 | 0.051 |
| bf | news60 | 31 | 0.0327 | 0.0541 | 0.0544 |
| bf | occbin | 1873 | 0.0862 | 0.1677 | 0.2514 |
| bf | occbin60 | 13 | 0.0851 | 0.1177 | 0.1198 |
| nkm | minnorm | 79 | 0.0358 | 0.06 | 0.0607 |
| nkm | news | 2018 | 0.0358 | 0.07218 | 0.1147 |
| nkm | news-longest | 56 | 0.2705 | 0.3843 | 0.3928 |
| nkm | news-milp | 5 | 0.0389 | 0.1213 | 0.1242 |
| nkm | news-shortest | 79 | 0.286 | 0.4963 | 0.5325 |
| nkm | news20 | 29 | 0.0367 | 0.1373 | 0.1598 |
| nkm | news60 | 29 | 0.0371 | 0.1102 | 0.112 |
| nkm | occbin | 2013 | 0.1369 | 0.2592 | 0.5461 |
| nkm | occbin60 | 16 | 0.1396 | 0.1744 | 0.1765 |
| nkms | minnorm | 60 | 0.03345 | 0.06278 | 0.077 |
| nkms | news | 2135 | 0.0335 | 0.05966 | 0.121 |
| nkms | news-milp | 9 | 0.039 | 0.06636 | 0.0683 |
| nkms | news20 | 35 | 0.0358 | 0.07174 | 0.0764 |
| nkms | news60 | 35 | 0.0378 | 0.06868 | 0.073 |
| nkms | occbin | 2134 | 0.1237 | 0.2084 | 0.6148 |
| nkms | occbin60 | 19 | 0.126 | 0.1968 | 0.1968 |

Part C, seconds per simulated period (per 1000-period segment, averaged over a task):

| model | method | tasks | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 9 | 3.4e-05 | 5e-05 | 5e-05 |
| bf | occbin | 9 | 0.0002457 | 0.0002968 | 0.0003 |
| borrcon | news | 25 | 3e-05 | 7.076e-05 | 7.1e-05 |
| borrcon | occbin | 25 | 0.000337 | 0.0004891 | 0.000492 |
| borrcon_d | news | 19 | 2.333e-05 | 9.94e-05 | 0.0001 |
| borrcon_d | occbin | 19 | 0.00031 | 0.0008344 | 0.00088 |
| nkm | news | 19 | 2.333e-05 | 4.88e-05 | 5e-05 |
| nkm | occbin | 19 | 0.00029 | 0.0004181 | 0.0004233 |
| nkm_est | news | 17 | 1.8e-05 | 4.28e-05 | 4.333e-05 |
| nkm_est | occbin | 17 | 0.00028 | 0.0003818 | 0.0003867 |
| sw | news | 8 | 3.85e-05 | 4.372e-05 | 4.4e-05 |
| sw | occbin | 8 | 0.0002717 | 0.0003657 | 0.00037 |
| zlb | news | 28 | 3.75e-05 | 6.55e-05 | 7e-05 |
| zlb | occbin | 28 | 0.0003608 | 0.0005015 | 0.00052 |
| zlb2 | news | 29 | 5e-05 | 7.907e-05 | 8e-05 |
| zlb2 | occbin | 29 | 0.0005 | 0.00083 | 0.0009 |

Part D, seconds per period of `HigherOrderNews.simulate` (successful runs):

| model | config | runs | median | p99 | max |
|---|---|---|---|---|---|
| borrcon | cub_all11 | 24 | 0.0025 | 0.03585 | 0.03675 |
| borrcon | cub_all7_tol3 | 78 | 0.0835 | 0.2871 | 0.3025 |
| borrcon | cub_all9_tol05 | 49 | 0.0035 | 0.1459 | 0.1605 |
| borrcon | cub_r1_l9 | 100 | 0.0005 | 0.000835 | 0.001 |
| borrcon | cub_r1_l9_tol1 | 92 | 0.0005 | 0.0008333 | 0.0008333 |
| borrcon | cub_r4_l9 | 79 | 0.005 | 0.007885 | 0.00925 |
| borrcon | order1 | 99 | 0 | 0.0001683 | 0.00025 |
| borrcon | order2 | 95 | 0 | 1e-05 | 0.0001667 |
| nk | cub_all11 | 129 | 0.016 | 0.03136 | 0.0325 |
| nk | cub_all7_tol3 | 133 | 0.0025 | 0.004167 | 0.043 |
| nk | cub_all9_tol05 | 132 | 0.005833 | 0.0282 | 0.0365 |
| nk | cub_r1_l9 | 137 | 0.0005 | 0.00094 | 0.001 |
| nk | cub_r1_l9_tol1 | 136 | 0.0005 | 0.0009417 | 0.001 |
| nk | cub_r4_l9 | 132 | 0.0035 | 0.00569 | 0.006667 |
| nk | order1 | 137 | 0 | 0 | 0 |
| nk | order2 | 135 | 0 | 0 | 0.00025 |
| nkm | cub_all11 | 65 | 0.0035 | 0.01837 | 0.02583 |
| nkm | cub_all7_tol3 | 69 | 0.001 | 0.5969 | 0.7442 |
| nkm | cub_all9_tol05 | 70 | 0.0015 | 0.005 | 0.005 |
| nkm | cub_r1_l9 | 76 | 0.00025 | 0.0008333 | 0.0008333 |
| nkm | cub_r1_l9_tol1 | 79 | 0.0001667 | 0.0008333 | 0.0008333 |
| nkm | cub_r4_l9 | 81 | 0.002 | 0.004033 | 0.004167 |
| nkm | order1 | 79 | 0 | 0 | 0 |
| nkm | order2 | 67 | 0 | 5.667e-05 | 0.0001667 |
| nkm_est | cub_all11 | 71 | 0.00425 | 0.02492 | 0.0325 |
| nkm_est | cub_all7_tol3 | 73 | 0.0008333 | 0.00176 | 0.002 |
| nkm_est | cub_all9_tol05 | 69 | 0.001667 | 0.005267 | 0.005833 |
| nkm_est | cub_r1_l9 | 71 | 0.0001667 | 0.0007167 | 0.0008333 |
| nkm_est | cub_r1_l9_tol1 | 72 | 0.00025 | 0.0005967 | 0.0008333 |
| nkm_est | cub_r4_l9 | 68 | 0.002 | 0.004055 | 0.004167 |
| nkm_est | order1 | 70 | 0 | 0 | 0 |
| nkm_est | order2 | 66 | 0 | 0 | 0 |
| sw_cp | cub_all7_tol3 | 4 | 0.01825 | 0.04953 | 0.05 |
| sw_cp | cub_all9_tol05 | 4 | 0.004583 | 0.01632 | 0.01667 |
| sw_cp | cub_r1_l9 | 89 | 0.0005 | 0.0008533 | 0.001 |
| sw_cp | cub_r1_l9_tol1 | 84 | 0.0005 | 0.0008333 | 0.0008333 |
| sw_cp | cub_r4_l9 | 1 | 0.0015 | 0.0015 | 0.0015 |
| sw_cp | order1 | 103 | 0 | 0 | 0 |
| sw_cp | order2 | 101 | 0 | 0 | 0 |
| zlb | cub_all11 | 98 | 0.01937 | 0.03419 | 0.035 |
| zlb | cub_all7_tol3 | 136 | 0.0025 | 0.005 | 0.074 |
| zlb | cub_all9_tol05 | 131 | 0.0065 | 0.01167 | 0.0395 |
| zlb | cub_r1_l9 | 135 | 0.0005 | 0.0008333 | 0.0008333 |
| zlb | cub_r1_l9_tol1 | 135 | 0.0005 | 0.0009433 | 0.001 |
| zlb | cub_r4_l9 | 134 | 0.0035 | 0.005833 | 0.006 |
| zlb | order1 | 134 | 0 | 0 | 0.0001667 |
| zlb | order2 | 135 | 0 | 0 | 0.0001667 |
| zlb2 | cub_all11 | 95 | 0.07375 | 0.1039 | 0.1333 |
| zlb2 | cub_all7_tol3 | 131 | 0.007333 | 0.02205 | 0.0245 |
| zlb2 | cub_all9_tol05 | 131 | 0.02083 | 0.09232 | 0.1325 |
| zlb2 | cub_r1_l9 | 133 | 0.00075 | 0.001 | 0.001 |
| zlb2 | cub_r1_l9_tol1 | 132 | 0.00075 | 0.001115 | 0.00125 |
| zlb2 | cub_r4_l9 | 136 | 0.0105 | 0.015 | 0.016 |
| zlb2 | order1 | 133 | 0 | 0.0001667 | 0.0001667 |
| zlb2 | order2 | 136 | 0 | 0.0001667 | 0.00025 |

## Failures, parts C and D (expected error types)

Part C: per model, segments of 1000 periods that failed, by method and error (count of segments):


Part D: runs per model and configuration (ok / error types):

| model | config | outcomes |
|---|---|---|
| borrcon | cub_all11 | expected:CubatureFailure:horizon_too_short 73, expected:OBCClassError: 41, ok 24 |
| borrcon | cub_all7_tol3 | expected:CubatureFailure:horizon_too_short 17, expected:OBCClassError: 41, expected:SteadyStateError: 1, ok 78 |
| borrcon | cub_all9_tol05 | expected:CubatureFailure:horizon_too_short 60, expected:OBCClassError: 28, expected:SteadyStateError: 1, ok 49 |
| borrcon | cub_r1_l9 | expected:OBCClassError: 38, ok 100 |
| borrcon | cub_r1_l9_tol1 | expected:CubatureFailure:horizon_too_short 2, expected:OBCClassError: 42, expected:SteadyStateError: 1, ok 92 |
| borrcon | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 21, expected:OBCClassError: 38, ok 79 |
| borrcon | order1 | expected:OBCClassError: 39, ok 99 |
| borrcon | order2 | expected:OBCClassError: 40, expected:SteadyStateError: 3, ok 95 |
| nk | cub_all11 | expected:BlanchardKahnError: 2, expected:CubatureFailure:horizon_too_short 7, ok 129 |
| nk | cub_all7_tol3 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 1, ok 133 |
| nk | cub_all9_tol05 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 2, ok 132 |
| nk | cub_r1_l9 | expected:CubatureFailure:horizon_too_short 1, ok 137 |
| nk | cub_r1_l9_tol1 | expected:BlanchardKahnError: 1, ok 136 |
| nk | cub_r4_l9 | expected:BlanchardKahnError: 2, expected:CubatureFailure:horizon_too_short 4, ok 132 |
| nk | order1 | expected:BlanchardKahnError: 1, ok 137 |
| nk | order2 | expected:BlanchardKahnError: 3, ok 135 |
| nkm | cub_all11 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 1, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 43, expected:SteadyStateError: 23, ok 65 |
| nkm | cub_all7_tol3 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 43, expected:SteadyStateError: 22, ok 69 |
| nkm | cub_all9_tol05 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 3, expected:OBCClassError: 35, expected:SteadyStateError: 28, ok 70 |
| nkm | cub_r1_l9 | expected:BlanchardKahnError: 1, expected:OBCClassError: 37, expected:SteadyStateError: 24, ok 76 |
| nkm | cub_r1_l9_tol1 | expected:BlanchardKahnError: 2, expected:OBCClassError: 31, expected:SteadyStateError: 25, ok 79 |
| nkm | cub_r4_l9 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 1, expected:OBCClassError: 29, expected:SteadyStateError: 26, ok 81 |
| nkm | order1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 31, expected:SteadyStateError: 27, ok 79 |
| nkm | order2 | expected:BlanchardKahnError: 2, expected:OBCClassError: 39, expected:OBCError:no_solution 1, expected:SteadyStateError: 29, ok 67 |
| nkm_est | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 31, expected:SteadyStateError: 33, ok 71 |
| nkm_est | cub_all7_tol3 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 38, expected:SteadyStateError: 22, ok 73 |
| nkm_est | cub_all9_tol05 | expected:CubatureFailure:no_solution 2, expected:OBCClassError: 41, expected:SteadyStateError: 26, ok 69 |
| nkm_est | cub_r1_l9 | expected:BlanchardKahnError: 1, expected:OBCClassError: 38, expected:SteadyStateError: 28, ok 71 |
| nkm_est | cub_r1_l9_tol1 | expected:BlanchardKahnError: 3, expected:OBCClassError: 38, expected:SteadyStateError: 24, ok 72 |
| nkm_est | cub_r4_l9 | expected:BlanchardKahnError: 1, expected:OBCClassError: 45, expected:SteadyStateError: 24, ok 68 |
| nkm_est | order1 | expected:OBCClassError: 41, expected:SteadyStateError: 27, ok 70 |
| nkm_est | order2 | expected:BlanchardKahnError: 1, expected:OBCClassError: 46, expected:SteadyStateError: 25, ok 66 |
| sw_cp | cub_all7_tol3 | expected:BlanchardKahnError: 4, expected:CubatureFailure:no_solution 111, expected:OBCClassError: 18, ok 4 |
| sw_cp | cub_all9_tol05 | expected:BlanchardKahnError: 9, expected:CubatureFailure:no_solution 227, expected:OBCClassError: 36, ok 4 |
| sw_cp | cub_r1_l9 | expected:BlanchardKahnError: 6, expected:CubatureFailure:no_solution 26, expected:OBCClassError: 17, ok 89 |
| sw_cp | cub_r1_l9_tol1 | expected:BlanchardKahnError: 4, expected:CubatureFailure:no_solution 25, expected:OBCClassError: 24, ok 84 |
| sw_cp | cub_r4_l9 | expected:BlanchardKahnError: 5, expected:CubatureFailure:no_solution 111, expected:OBCClassError: 21, ok 1 |
| sw_cp | order1 | expected:BlanchardKahnError: 3, expected:OBCClassError: 12, expected:OBCError:no_solution 20, ok 103 |
| sw_cp | order2 | expected:BlanchardKahnError: 2, expected:OBCClassError: 10, expected:OBCError:no_solution 25, ok 101 |
| zlb | cub_all11 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 37, ok 98 |
| zlb | cub_all7_tol3 | expected:CubatureFailure:horizon_too_short 1, ok 136 |
| zlb | cub_all9_tol05 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 6, ok 131 |
| zlb | cub_r1_l9 | expected:BlanchardKahnError: 3, ok 135 |
| zlb | cub_r1_l9_tol1 | expected:BlanchardKahnError: 2, ok 135 |
| zlb | cub_r4_l9 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 1, ok 134 |
| zlb | order1 | expected:BlanchardKahnError: 4, ok 134 |
| zlb | order2 | expected:BlanchardKahnError: 3, ok 135 |
| zlb2 | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 42, ok 95 |
| zlb2 | cub_all7_tol3 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 1, ok 131 |
| zlb2 | cub_all9_tol05 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 4, ok 131 |
| zlb2 | cub_r1_l9 | expected:BlanchardKahnError: 5, ok 133 |
| zlb2 | cub_r1_l9_tol1 | expected:BlanchardKahnError: 5, ok 132 |
| zlb2 | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 2, ok 136 |
| zlb2 | order1 | expected:BlanchardKahnError: 2, expected:OBCError:no_solution 3, ok 133 |
| zlb2 | order2 | expected:BlanchardKahnError: 2, ok 136 |

## Warnings (distinct messages, numbers normalised, counts over all parts)

- 222 x `UserWarning: M is not an S-matrix: some paths can have no solution; cubature points may fail (see Highe`
- 164 x `UserWarning: cubature_failure_tolerance=#: # of # relevant perturbed points (#%) were dropped in # of #`
- 3 x `RuntimeWarning: invalid value encountered in sqrt`

## Memory (resident MB at the end of each task; one process per ~8 minute chunk, so growth is only meaningful within a chunk)

- A: median 168, max 1446
- B: median 305, max 1409
- C: median 314, max 1412
- D: median 415, max 1415
