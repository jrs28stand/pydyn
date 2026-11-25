# OBC soak test (pydyn, branch `obc-soak`, from `wip/obc` 0abe8e7)

Brute-force soak test of the occasionally-binding-constraint estimation and simulation code, to find bugs that appear only over many
parameter vectors or long runs. Library code is not changed; anomalies are catalogued with reproducers (`soak/repro/`).

## State / how to resume

(written by hand; see the end of the file for the numbers)

## Coverage so far

| part | run time (h) | tasks | what |
|---|---|---|---|
| A | 0.55 | 1386 | evaluations of the log posterior (all configurations): bf 6622, nkm 6971, nkms 6756; tasks by kind: bf/bound 115, bf/line 115, bf/perturb 116, bf/prior 116, nkm/bound 115, nkm/line 115, nkm/perturb 116, nkm/prior 116, nkms/bound 115, nkms/line 115, nkms/perturb 116, nkms/prior 116 |
| B | 0.68 | 258 | 32250 MCMC draws (news) in 5 finished chains + the running one (chain 5, draw 2250, replays under occbin of this chain: 225); 42 mode searches (5 skipped: start not finite, 1 timeouts) |
| C | 0.60 | 333 | simulated periods per model (both methods each): bf 274000 (42 runs, 10854 binding), borrcon 387000 (41 runs, 86417 binding), borrcon_d 453000 (42 runs, 61100 binding), nkm 493000 (42 runs, 49730 binding), nkm_est 365000 (41 runs, 36672 binding), sw 220000 (41 runs, 792 binding), zlb 451000 (42 runs, 157939 binding), zlb2 348000 (42 runs, 135046 binding) |
| D | 0.55 | 10741 | runs per model (ok/total): borrcon 879/1534, nk 1490/1535, nkm 814/1534, nkm_est 793/1535, sw_cp 517/1534, zlb 1446/1534, zlb2 1434/1535 |

LCP solutions checked with `verify_lcp` (tolerance 1e-6) at the `solve_active_set` / `solve_news` boundary, all parts: {'verify|solve_active_set': 407829, 'verify|solve_news': 47}.

Determinism checks (same call repeated): A same-process 298 (+68 in a fresh estimation object), C 55, D 2002.

## Anomaly catalogue

| kind | part | count | severity | first reproducer(s) | judgement |
|---|---|---|---|---|---|
| `nonbitwise_ulp:higher_order` | D | 493 | ? | soak/results/anomalies.jsonl (task 0, task 36, task 48) | not yet looked at |
| `path_disagree:nkm` | C | 88 | ? | soak/results/anomalies.jsonl (task 66, task 66, task 82) | not yet looked at |
| `path_disagree:nkm_est` | C | 58 | ? | soak/results/anomalies.jsonl (task 127, task 127, task 127) | not yet looked at |
| `variant_fails_where_news_ok:news-longest` | A | 38 | ? | soak/results/anomalies.jsonl (task 12, task 24, task 96) | not yet looked at |
| `lp_disagree_large:nkm:regime_count_differs` | A | 29 | ? | soak/results/anomalies.jsonl (task 6, task 6, task 162) | not yet looked at |
| `slow_failure` | C | 29 | ? | soak/results/anomalies.jsonl (task 14, task 43, task 43) | not yet looked at |
| `finite_vs_inf:news_finite_occbin_code304` | A | 10 | ? | soak/results/anomalies.jsonl (task 30, task 193, task 285) | not yet looked at |
| `lp_absurd` | A | 9 | ? | soak/results/anomalies.jsonl (task 70, task 70, task 70) | not yet looked at |
| `memory_jump` | A | 9 | ? | soak/results/anomalies.jsonl (task 141, task 273, task 309) | not yet looked at |
| `path_disagree:bf` | C | 8 | ? | soak/results/anomalies.jsonl (task 43, task 83, task 83) | not yet looked at |
| `regime_history_differs_equal_path:nkm` | C | 7 | ? | soak/results/anomalies.jsonl (task 82, task 82, task 130) | not yet looked at |
| `slow_failure` | A | 7 | ? | soak/results/anomalies.jsonl (task 65, task 75, task 105) | not yet looked at |
| `finite_vs_inf:occbin_finite_news_code304` | A | 6 | ? | soak/results/anomalies.jsonl (task 75, task 155, task 500) | not yet looked at |
| `memory_jump` | D | 6 | ? | soak/results/anomalies.jsonl (task 1782, task 2697, task 3313) | not yet looked at |
| `lp_disagree_large:nkms:regime_count_differs` | A | 3 | ? | soak/results/anomalies.jsonl (task 79, task 556, task 1156) | not yet looked at |
| `milp_timeout_hit` | A | 3 | ? | soak/results/anomalies.jsonl (task 65, task 75, task 105) | not yet looked at |
| `regime_history_differs_equal_path:nkm_est` | C | 3 | ? | soak/results/anomalies.jsonl (task 127, task 215, task 247) | not yet looked at |
| `warning:RuntimeWarning: invalid value encountered in sqrt` | B | 3 | ? | soak/results/anomalies.jsonl (task 167, task 185) | not yet looked at |
| `discontinuity_same_regimes:bf` | A | 2 | ? | soak/results/anomalies.jsonl (task 404, task 404) | not yet looked at |
| `lp_disagree_large:nkms:regime_count_differs` | B | 2 | ? | soak/results/anomalies.jsonl (task 103, task 130) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:sw` | C | 2 | ? | soak/results/anomalies.jsonl (task 30, task 86) | not yet looked at |
| `driver_or_unexpected:TypeError:dict() got multiple values for keyword argument 'part'` | B | 1 | ? | soak/results/anomalies.jsonl (task 155) | not yet looked at |
| `lp_disagree_large:nkm:same_regime_count` | B | 1 | ? | soak/results/anomalies.jsonl (task 119) | not yet looked at |
| `lp_disagree_large:nkms:same_regime_count` | A | 1 | ? | soak/results/anomalies.jsonl (task 94) | not yet looked at |
| `memory_jump` | B | 1 | ? | soak/results/anomalies.jsonl (task 35) | not yet looked at |
| `memory_jump` | C | 1 | ? | soak/results/anomalies.jsonl (task 63) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm` | C | 1 | ? | soak/results/anomalies.jsonl (task 82) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm_est` | C | 1 | ? | soak/results/anomalies.jsonl (task 127) | not yet looked at |
| `path_disagree:sw` | C | 1 | ? | soak/results/anomalies.jsonl (task 118) | not yet looked at |
| `regime_seq_differs_equal_lp` | A | 1 | ? | soak/results/anomalies.jsonl (task 1342) | not yet looked at |
| `timeout` | A | 1 | ? | soak/results/anomalies.jsonl (task 105) | not yet looked at |
| `timeout` | B | 1 | ? | soak/results/anomalies.jsonl (task 203) | not yet looked at |

## Failure rates, part A (inversion filter, log posterior evaluations)

Distance bin = rms of the relative deviation from the model's initial values: a <0.02, b <0.1, c <0.3, d <1, e >=1; `line` = the 25 points of the straight-line sweeps (-0.1..1.1).

| model | config | n | finite | failed (by code) | by distance bin: finite share |
|---|---|---|---|---|---|
| bf | news | 3222 | 0.814 | c304 392, c41 34, c42 174 | a:1.00 (18) b:0.90 (20) c:0.82 (114) d:0.50 (104) e:0.51 (91) line:0.83 (2875) |
| bf | news-milp | 15 | 0.267 | c304 1, c41 2, c42 8 | a:1.00 (2) c:0.00 (1) d:0.25 (4) e:0.12 (8) |
| bf | news20 | 70 | 0.629 | c304 8, c41 4, c42 14 | a:1.00 (2) b:1.00 (4) c:0.74 (19) d:0.59 (27) e:0.44 (18) |
| bf | news60 | 70 | 0.629 | c304 8, c41 4, c42 14 | a:1.00 (2) b:1.00 (4) c:0.74 (19) d:0.59 (27) e:0.44 (18) |
| bf | occbin | 3222 | 0.815 | c304 388, c41 34, c42 174 | a:1.00 (18) b:0.90 (20) c:0.82 (114) d:0.52 (104) e:0.52 (91) line:0.83 (2875) |
| bf | occbin60 | 23 | 0.783 | c304 5 | c:0.76 (17) d:0.83 (6) |
| nkm | minnorm | 116 | 0.966 | c304 2, c41 2 | c:0.98 (57) d:0.94 (49) e:1.00 (10) |
| nkm | news | 3222 | 0.876 | c3 2, c304 70, c4 14, c41 32, c42 278, c5 4 | a:1.00 (2) b:0.75 (4) c:0.86 (126) d:0.64 (119) e:0.35 (96) line:0.90 (2875) |
| nkm | news-longest | 116 | 0.638 | c304 40, c41 2 | c:0.63 (57) d:0.63 (49) e:0.70 (10) |
| nkm | news-milp | 15 | 0.533 | c0 1, c304 1, c4 1, c41 2, c42 1, c5 1 | c:0.75 (4) d:0.33 (6) e:0.60 (5) |
| nkm | news-shortest | 116 | 0.966 | c304 2, c41 2 | c:0.98 (57) d:0.94 (49) e:1.00 (10) |
| nkm | news20 | 70 | 0.614 | c304 6, c4 1, c41 4, c42 14, c5 2 | b:0.00 (1) c:0.81 (27) d:0.61 (23) e:0.37 (19) |
| nkm | news60 | 70 | 0.614 | c304 6, c4 1, c41 4, c42 14, c5 2 | b:0.00 (1) c:0.81 (27) d:0.61 (23) e:0.37 (19) |
| nkm | occbin | 3222 | 0.874 | c3 2, c304 76, c4 14, c41 32, c42 278, c5 4 | a:1.00 (2) b:0.75 (4) c:0.86 (126) d:0.65 (119) e:0.32 (96) line:0.90 (2875) |
| nkm | occbin60 | 24 | 0.958 | c304 1 | c:0.91 (11) d:1.00 (11) e:1.00 (2) |
| nkms | minnorm | 116 | 0.759 | c4 18, c42 10 | a:1.00 (5) b:1.00 (9) c:1.00 (4) d:0.82 (11) e:0.70 (87) |
| nkms | news | 3222 | 0.944 | c304 16, c306 1, c4 112, c41 29, c42 17, c5 6 | a:1.00 (5) b:1.00 (23) c:0.97 (72) d:0.67 (76) e:0.53 (171) line:0.97 (2875) |
| nkms | news-milp | 12 | 1.000 |  | b:1.00 (2) c:1.00 (6) d:1.00 (3) e:1.00 (1) |
| nkms | news20 | 69 | 0.710 | c304 2, c306 1, c4 6, c41 6, c42 5 | a:1.00 (1) b:1.00 (7) c:0.93 (15) d:0.57 (14) e:0.59 (32) |
| nkms | news60 | 69 | 0.710 | c304 2, c306 1, c4 6, c41 6, c42 5 | a:1.00 (1) b:1.00 (7) c:0.93 (15) d:0.57 (14) e:0.59 (32) |
| nkms | occbin | 3222 | 0.943 | c304 18, c306 1, c4 112, c41 29, c42 17, c5 6 | a:1.00 (5) b:1.00 (23) c:0.96 (72) d:0.67 (76) e:0.53 (171) line:0.97 (2875) |
| nkms | occbin60 | 46 | 0.565 | c304 2, c306 1, c4 6, c41 6, c42 5 | a:1.00 (1) b:1.00 (2) c:0.50 (2) d:0.40 (10) e:0.58 (31) |

occbin vs news, both finite: |difference in log posterior|

| model | pairs | <=1e-9 (rel) | 1e-9..0.15 | >0.15 | max |
|---|---|---|---|---|---|
| bf | 2622 | 2614 | 5 | 3 | 1 |
| nkm | 2815 | 2711 | 74 | 30 | 1.88 |
| nkms | 3038 | 2989 | 44 | 5 | 2.12 |

## Timing distributions

Part A, seconds per finite log-posterior evaluation:

| model | config | n | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 2622 | 0.0306 | 0.196 | 0.4317 |
| bf | news-milp | 4 | 0.02875 | 0.04364 | 0.0441 |
| bf | news20 | 44 | 0.03065 | 0.0498 | 0.051 |
| bf | news60 | 44 | 0.03205 | 0.05397 | 0.0544 |
| bf | occbin | 2626 | 0.0857 | 0.1665 | 0.2591 |
| bf | occbin60 | 18 | 0.0843 | 0.1168 | 0.1198 |
| nkm | minnorm | 112 | 0.03635 | 0.0606 | 0.1042 |
| nkm | news | 2822 | 0.0357 | 0.07492 | 0.1428 |
| nkm | news-longest | 74 | 0.2746 | 0.3823 | 0.3928 |
| nkm | news-milp | 8 | 0.03655 | 0.1202 | 0.1242 |
| nkm | news-shortest | 112 | 0.2872 | 0.5007 | 0.5325 |
| nkm | news20 | 43 | 0.0371 | 0.1261 | 0.1598 |
| nkm | news60 | 43 | 0.0371 | 0.1104 | 0.112 |
| nkm | occbin | 2816 | 0.1361 | 0.2525 | 0.5461 |
| nkm | occbin60 | 23 | 0.1414 | 0.224 | 0.2374 |
| nkms | minnorm | 88 | 0.0332 | 0.05603 | 0.077 |
| nkms | news | 3041 | 0.0334 | 0.05898 | 0.121 |
| nkms | news-milp | 12 | 0.03675 | 0.06563 | 0.0683 |
| nkms | news20 | 49 | 0.0351 | 0.06982 | 0.0764 |
| nkms | news60 | 49 | 0.0359 | 0.0669 | 0.073 |
| nkms | occbin | 3039 | 0.1228 | 0.2084 | 0.6148 |
| nkms | occbin60 | 26 | 0.1258 | 0.1968 | 0.1968 |

Part C, seconds per simulated period (per 1000-period segment, averaged over a task):

| model | method | tasks | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 16 | 3.367e-05 | 5.85e-05 | 6e-05 |
| bf | occbin | 16 | 0.000248 | 0.0003028 | 0.0003033 |
| borrcon | news | 35 | 3e-05 | 7.694e-05 | 8e-05 |
| borrcon | occbin | 35 | 0.000337 | 0.0005118 | 0.00052 |
| borrcon_d | news | 30 | 2.333e-05 | 9.903e-05 | 0.0001 |
| borrcon_d | occbin | 30 | 0.00031 | 0.0008065 | 0.00088 |
| nkm | news | 26 | 2.333e-05 | 4.833e-05 | 5e-05 |
| nkm | occbin | 26 | 0.0002935 | 0.000416 | 0.0004233 |
| nkm_est | news | 28 | 2.1e-05 | 4.946e-05 | 5e-05 |
| nkm_est | occbin | 28 | 0.00028 | 0.0004037 | 0.00041 |
| sw | news | 11 | 3.967e-05 | 4.36e-05 | 4.4e-05 |
| sw | occbin | 11 | 0.00027 | 0.0003638 | 0.00037 |
| zlb | news | 40 | 3.683e-05 | 6.61e-05 | 7e-05 |
| zlb | occbin | 40 | 0.0003512 | 0.0004932 | 0.00052 |
| zlb2 | news | 42 | 5.183e-05 | 9.18e-05 | 0.0001 |
| zlb2 | occbin | 42 | 0.0004833 | 0.0007975 | 0.0009 |

Part D, seconds per period of `HigherOrderNews.simulate` (successful runs):

| model | config | runs | median | p99 | max |
|---|---|---|---|---|---|
| borrcon | cub_all11 | 38 | 0.00575 | 0.03926 | 0.04 |
| borrcon | cub_all7_tol3 | 112 | 0.08042 | 0.2816 | 0.3025 |
| borrcon | cub_all9_tol05 | 63 | 0.0035 | 0.167 | 0.1775 |
| borrcon | cub_r1_l9 | 141 | 0.0005 | 0.0009333 | 0.001 |
| borrcon | cub_r1_l9_tol1 | 137 | 0.0005 | 0.0008333 | 0.0008333 |
| borrcon | cub_r4_l9 | 110 | 0.005 | 0.00744 | 0.00925 |
| borrcon | order1 | 138 | 0 | 0.000105 | 0.00025 |
| borrcon | order2 | 140 | 0 | 0.0001667 | 0.0001667 |
| nk | cub_all11 | 180 | 0.016 | 0.03111 | 0.0325 |
| nk | cub_all7_tol3 | 186 | 0.0025 | 0.004942 | 0.043 |
| nk | cub_all9_tol05 | 185 | 0.0055 | 0.02862 | 0.0365 |
| nk | cub_r1_l9 | 190 | 0.0005 | 0.0008517 | 0.001 |
| nk | cub_r1_l9_tol1 | 189 | 0.0005 | 0.0008533 | 0.001 |
| nk | cub_r4_l9 | 183 | 0.0035 | 0.00612 | 0.00675 |
| nk | order1 | 190 | 0 | 0 | 0 |
| nk | order2 | 187 | 0 | 0 | 0.00025 |
| nkm | cub_all11 | 89 | 0.0035 | 0.01557 | 0.02583 |
| nkm | cub_all7_tol3 | 96 | 0.0008333 | 0.5385 | 0.7442 |
| nkm | cub_all9_tol05 | 97 | 0.0015 | 0.005 | 0.005 |
| nkm | cub_r1_l9 | 109 | 0.0001667 | 0.0008333 | 0.0008333 |
| nkm | cub_r1_l9_tol1 | 109 | 0.0001667 | 0.0008333 | 0.0008333 |
| nkm | cub_r4_l9 | 111 | 0.002 | 0.004 | 0.004167 |
| nkm | order1 | 109 | 0 | 0 | 0 |
| nkm | order2 | 94 | 0 | 0.0001667 | 0.0001667 |
| nkm_est | cub_all11 | 103 | 0.00375 | 0.02162 | 0.0325 |
| nkm_est | cub_all7_tol3 | 111 | 0.0008333 | 0.001967 | 0.002 |
| nkm_est | cub_all9_tol05 | 99 | 0.001667 | 0.005017 | 0.005833 |
| nkm_est | cub_r1_l9 | 102 | 0.0001667 | 0.000665 | 0.0008333 |
| nkm_est | cub_r1_l9_tol1 | 99 | 0.0001667 | 0.0005067 | 0.0008333 |
| nkm_est | cub_r4_l9 | 95 | 0.002 | 0.00401 | 0.004167 |
| nkm_est | order1 | 90 | 0 | 0 | 0 |
| nkm_est | order2 | 94 | 0 | 0 | 0 |
| sw_cp | cub_all7_tol3 | 4 | 0.01825 | 0.04953 | 0.05 |
| sw_cp | cub_all9_tol05 | 4 | 0.004583 | 0.01632 | 0.01667 |
| sw_cp | cub_r1_l9 | 109 | 0.0005 | 0.0008333 | 0.001 |
| sw_cp | cub_r1_l9_tol1 | 118 | 0.0005 | 0.0008333 | 0.0008333 |
| sw_cp | cub_r4_l9 | 1 | 0.0015 | 0.0015 | 0.0015 |
| sw_cp | order1 | 141 | 0 | 0 | 0.001 |
| sw_cp | order2 | 140 | 0 | 0 | 0 |
| zlb | cub_all11 | 135 | 0.01917 | 0.03292 | 0.035 |
| zlb | cub_all7_tol3 | 189 | 0.0025 | 0.005 | 0.074 |
| zlb | cub_all9_tol05 | 183 | 0.0065 | 0.038 | 0.05417 |
| zlb | cub_r1_l9 | 189 | 0.0005 | 0.0008333 | 0.0008333 |
| zlb | cub_r1_l9_tol1 | 187 | 0.0005 | 0.0008567 | 0.001 |
| zlb | cub_r4_l9 | 187 | 0.0035 | 0.005833 | 0.006 |
| zlb | order1 | 187 | 0 | 2.333e-05 | 0.0001667 |
| zlb | order2 | 189 | 0 | 2e-05 | 0.0001667 |
| zlb2 | cub_all11 | 131 | 0.0735 | 0.1035 | 0.1333 |
| zlb2 | cub_all7_tol3 | 185 | 0.007 | 0.02345 | 0.02667 |
| zlb2 | cub_all9_tol05 | 181 | 0.0205 | 0.08675 | 0.1325 |
| zlb2 | cub_r1_l9 | 187 | 0.00075 | 0.001 | 0.003333 |
| zlb2 | cub_r1_l9_tol1 | 185 | 0.00075 | 0.001167 | 0.00125 |
| zlb2 | cub_r4_l9 | 189 | 0.01033 | 0.015 | 0.016 |
| zlb2 | order1 | 186 | 0 | 0.0001667 | 0.0001667 |
| zlb2 | order2 | 190 | 0 | 0.0001667 | 0.00025 |

## Failures, parts C and D (expected error types)

Part C: per model, segments of 1000 periods that failed, by method and error (count of segments):


Part D: runs per model and configuration (ok / error types):

| model | config | outcomes |
|---|---|---|
| borrcon | cub_all11 | expected:CubatureFailure:horizon_too_short 100, expected:OBCClassError: 54, ok 38 |
| borrcon | cub_all7_tol3 | expected:CubatureFailure:horizon_too_short 20, expected:OBCClassError: 58, expected:SteadyStateError: 1, ok 112 |
| borrcon | cub_all9_tol05 | expected:CubatureFailure:horizon_too_short 88, expected:OBCClassError: 40, expected:SteadyStateError: 1, ok 63 |
| borrcon | cub_r1_l9 | expected:OBCClassError: 51, ok 141 |
| borrcon | cub_r1_l9_tol1 | expected:CubatureFailure:horizon_too_short 2, expected:OBCClassError: 51, expected:SteadyStateError: 1, ok 137 |
| borrcon | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 28, expected:OBCClassError: 54, ok 110 |
| borrcon | order1 | expected:OBCClassError: 54, ok 138 |
| borrcon | order2 | expected:OBCClassError: 49, expected:SteadyStateError: 3, ok 140 |
| nk | cub_all11 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 9, ok 180 |
| nk | cub_all7_tol3 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 1, ok 186 |
| nk | cub_all9_tol05 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 3, ok 185 |
| nk | cub_r1_l9 | expected:CubatureFailure:horizon_too_short 2, ok 190 |
| nk | cub_r1_l9_tol1 | expected:BlanchardKahnError: 2, ok 189 |
| nk | cub_r4_l9 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 5, ok 183 |
| nk | order1 | expected:BlanchardKahnError: 2, ok 190 |
| nk | order2 | expected:BlanchardKahnError: 4, expected:OBCError:no_solution 1, ok 187 |
| nkm | cub_all11 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 1, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 61, expected:SteadyStateError: 35, ok 89 |
| nkm | cub_all7_tol3 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 61, expected:SteadyStateError: 31, ok 96 |
| nkm | cub_all9_tol05 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 5, expected:OBCClassError: 51, expected:SteadyStateError: 37, ok 97 |
| nkm | cub_r1_l9 | expected:BlanchardKahnError: 2, expected:OBCClassError: 50, expected:SteadyStateError: 31, ok 109 |
| nkm | cub_r1_l9_tol1 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 41, expected:SteadyStateError: 38, ok 109 |
| nkm | cub_r4_l9 | expected:BlanchardKahnError: 2, expected:CubatureFailure:horizon_too_short 1, expected:OBCClassError: 47, expected:SteadyStateError: 31, ok 111 |
| nkm | order1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 46, expected:SteadyStateError: 36, ok 109 |
| nkm | order2 | expected:BlanchardKahnError: 2, expected:OBCClassError: 58, expected:OBCError:no_solution 1, expected:SteadyStateError: 37, ok 94 |
| nkm_est | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 49, expected:SteadyStateError: 37, ok 103 |
| nkm_est | cub_all7_tol3 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 46, expected:SteadyStateError: 31, ok 111 |
| nkm_est | cub_all9_tol05 | expected:BlanchardKahnError: 3, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 58, expected:SteadyStateError: 30, ok 99 |
| nkm_est | cub_r1_l9 | expected:BlanchardKahnError: 1, expected:OBCClassError: 52, expected:SteadyStateError: 37, ok 102 |
| nkm_est | cub_r1_l9_tol1 | expected:BlanchardKahnError: 4, expected:OBCClassError: 53, expected:SteadyStateError: 35, ok 99 |
| nkm_est | cub_r4_l9 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 61, expected:SteadyStateError: 34, ok 95 |
| nkm_est | order1 | expected:OBCClassError: 67, expected:SteadyStateError: 35, ok 90 |
| nkm_est | order2 | expected:BlanchardKahnError: 2, expected:OBCClassError: 63, expected:SteadyStateError: 33, ok 94 |
| sw_cp | cub_all7_tol3 | expected:BlanchardKahnError: 5, expected:CubatureFailure:no_solution 151, expected:OBCClassError: 31, ok 4 |
| sw_cp | cub_all9_tol05 | expected:BlanchardKahnError: 13, expected:CubatureFailure:no_solution 314, expected:OBCClassError: 53, ok 4 |
| sw_cp | cub_r1_l9 | expected:BlanchardKahnError: 10, expected:CubatureFailure:no_solution 47, expected:OBCClassError: 26, ok 109 |
| sw_cp | cub_r1_l9_tol1 | expected:BlanchardKahnError: 5, expected:CubatureFailure:no_solution 33, expected:OBCClassError: 35, ok 118 |
| sw_cp | cub_r4_l9 | expected:BlanchardKahnError: 5, expected:CubatureFailure:no_solution 159, expected:OBCClassError: 27, ok 1 |
| sw_cp | order1 | expected:BlanchardKahnError: 3, expected:OBCClassError: 19, expected:OBCError:no_solution 29, ok 141 |
| sw_cp | order2 | expected:BlanchardKahnError: 3, expected:OBCClassError: 17, expected:OBCError:no_solution 32, ok 140 |
| zlb | cub_all11 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 53, ok 135 |
| zlb | cub_all7_tol3 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 1, ok 189 |
| zlb | cub_all9_tol05 | expected:BlanchardKahnError: 2, expected:CubatureFailure:horizon_too_short 7, ok 183 |
| zlb | cub_r1_l9 | expected:BlanchardKahnError: 3, ok 189 |
| zlb | cub_r1_l9_tol1 | expected:BlanchardKahnError: 4, ok 187 |
| zlb | cub_r4_l9 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 1, ok 187 |
| zlb | order1 | expected:BlanchardKahnError: 5, ok 187 |
| zlb | order2 | expected:BlanchardKahnError: 3, ok 189 |
| zlb2 | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 60, ok 131 |
| zlb2 | cub_all7_tol3 | expected:BlanchardKahnError: 6, expected:CubatureFailure:horizon_too_short 1, ok 185 |
| zlb2 | cub_all9_tol05 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 7, ok 181 |
| zlb2 | cub_r1_l9 | expected:BlanchardKahnError: 5, ok 187 |
| zlb2 | cub_r1_l9_tol1 | expected:BlanchardKahnError: 6, ok 185 |
| zlb2 | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 3, ok 189 |
| zlb2 | order1 | expected:BlanchardKahnError: 3, expected:OBCError:no_solution 3, ok 186 |
| zlb2 | order2 | expected:BlanchardKahnError: 2, ok 190 |

## Warnings (distinct messages, numbers normalised, counts over all parts)

- 291 x `UserWarning: M is not an S-matrix: some paths can have no solution; cubature points may fail (see Highe`
- 224 x `UserWarning: cubature_failure_tolerance=#: # of # relevant perturbed points (#%) were dropped in # of #`
- 3 x `RuntimeWarning: invalid value encountered in sqrt`

## Memory (resident MB at the end of each task; one process per ~8 minute chunk, so growth is only meaningful within a chunk)

- A: median 157, max 1446
- B: median 305, max 1409
- C: median 314, max 1412
- D: median 415, max 1415
