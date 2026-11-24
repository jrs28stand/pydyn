# OBC soak test (pydyn, branch `obc-soak`, from `wip/obc` 0abe8e7)

Brute-force soak test of the occasionally-binding-constraint estimation and simulation code, to find bugs that appear only over many
parameter vectors or long runs. Library code is not changed; anomalies are catalogued with reproducers (`soak/repro/`).

## State / how to resume

(written by hand; see the end of the file for the numbers)

## Coverage so far

| part | run time (h) | tasks | what |
|---|---|---|---|
| A | 0.14 | 200 | evaluations of the log posterior (all configurations): bf 928, nkm 1032, nkms 996; tasks by kind: bf/bound 16, bf/line 16, bf/perturb 17, bf/prior 17, nkm/bound 16, nkm/line 17, nkm/perturb 17, nkm/prior 17, nkms/bound 16, nkms/line 17, nkms/perturb 17, nkms/prior 17 |
| B | 0.14 | 47 | 6000 MCMC draws (news) in 0 finished chains + the running one (chain 0, draw 6000, replays under occbin of this chain: 600); 7 mode searches (0 skipped: start not finite, 0 timeouts) |
| C | 0.15 | 84 | simulated periods per model (both methods each): bf 80000 (11 runs, 2310 binding), borrcon 141000 (10 runs, 30587 binding), borrcon_d 121000 (10 runs, 10007 binding), nkm 139000 (11 runs, 9842 binding), nkm_est 39000 (10 runs, 135 binding), sw 13000 (10 runs, 3 binding), zlb 148000 (11 runs, 59894 binding), zlb2 100000 (11 runs, 40238 binding) |
| D | 0.15 | 2337 | runs per model (ok/total): borrcon 191/334, nk 322/334, nkm 173/333, nkm_est 179/334, sw_cp 122/334, zlb 318/334, zlb2 314/334 |

LCP solutions checked with `verify_lcp` (tolerance 1e-6) at the `solve_active_set` / `solve_news` boundary, all parts: {'verify|solve_active_set': 70788, 'verify|solve_news': 10}.

Determinism checks (same call repeated): A same-process 44 (+12 in a fresh estimation object), C 15, D 444.

## Anomaly catalogue

| kind | part | count | severity | first reproducer(s) | judgement |
|---|---|---|---|---|---|
| `nonbitwise_ulp:higher_order` | D | 110 | ? | soak/results/anomalies.jsonl (task 0, task 36, task 48) | not yet looked at |
| `path_disagree:nkm` | C | 18 | ? | soak/results/anomalies.jsonl (task 66, task 66, task 82) | not yet looked at |
| `lp_absurd` | A | 9 | ? | soak/results/anomalies.jsonl (task 70, task 70, task 70) | not yet looked at |
| `slow_failure` | C | 8 | ? | soak/results/anomalies.jsonl (task 14, task 43, task 43) | not yet looked at |
| `slow_failure` | A | 5 | ? | soak/results/anomalies.jsonl (task 65, task 75, task 105) | not yet looked at |
| `lp_disagree_large:nkm:regime_count_differs` | A | 4 | ? | soak/results/anomalies.jsonl (task 6, task 6, task 162) | not yet looked at |
| `variant_fails_where_news_ok:news-longest` | A | 4 | ? | soak/results/anomalies.jsonl (task 12, task 24, task 96) | not yet looked at |
| `milp_timeout_hit` | A | 3 | ? | soak/results/anomalies.jsonl (task 65, task 75, task 105) | not yet looked at |
| `path_disagree:bf` | C | 3 | ? | soak/results/anomalies.jsonl (task 43, task 83, task 83) | not yet looked at |
| `finite_vs_inf:news_finite_occbin_code304` | A | 2 | ? | soak/results/anomalies.jsonl (task 30, task 193) | not yet looked at |
| `finite_vs_inf:occbin_finite_news_code304` | A | 2 | ? | soak/results/anomalies.jsonl (task 75, task 155) | not yet looked at |
| `regime_history_differs_equal_path:nkm` | C | 2 | ? | soak/results/anomalies.jsonl (task 82, task 82) | not yet looked at |
| `lp_disagree_large:nkms:regime_count_differs` | A | 1 | ? | soak/results/anomalies.jsonl (task 79) | not yet looked at |
| `lp_disagree_large:nkms:same_regime_count` | A | 1 | ? | soak/results/anomalies.jsonl (task 94) | not yet looked at |
| `memory_jump` | A | 1 | ? | soak/results/anomalies.jsonl (task 141) | not yet looked at |
| `memory_jump` | B | 1 | ? | soak/results/anomalies.jsonl (task 35) | not yet looked at |
| `memory_jump` | C | 1 | ? | soak/results/anomalies.jsonl (task 63) | not yet looked at |
| `memory_jump` | D | 1 | ? | soak/results/anomalies.jsonl (task 1782) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm` | C | 1 | ? | soak/results/anomalies.jsonl (task 82) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:sw` | C | 1 | ? | soak/results/anomalies.jsonl (task 30) | not yet looked at |
| `timeout` | A | 1 | ? | soak/results/anomalies.jsonl (task 105) | not yet looked at |

## Failure rates, part A (inversion filter, log posterior evaluations)

Distance bin = rms of the relative deviation from the model's initial values: a <0.02, b <0.1, c <0.3, d <1, e >=1; `line` = the 25 points of the straight-line sweeps (-0.1..1.1).

| model | config | n | finite | failed (by code) | by distance bin: finite share |
|---|---|---|---|---|---|
| bf | news | 450 | 0.800 | c304 66, c41 3, c42 21 | a:1.00 (2) b:1.00 (3) c:0.80 (20) d:0.42 (12) e:0.46 (13) line:0.82 (400) |
| bf | news-milp | 5 | 0.000 | c304 1, c41 1, c42 3 | d:0.00 (2) e:0.00 (3) |
| bf | news20 | 10 | 0.400 | c304 2, c41 1, c42 3 | c:1.00 (3) d:0.25 (4) e:0.00 (3) |
| bf | news60 | 10 | 0.400 | c304 2, c41 1, c42 3 | c:1.00 (3) d:0.25 (4) e:0.00 (3) |
| bf | occbin | 450 | 0.802 | c304 65, c41 3, c42 21 | a:1.00 (2) b:1.00 (3) c:0.80 (20) d:0.50 (12) e:0.46 (13) line:0.82 (400) |
| bf | occbin60 | 3 | 1.000 |  | c:1.00 (3) |
| nkm | minnorm | 17 | 1.000 |  | c:1.00 (8) d:1.00 (7) e:1.00 (2) |
| nkm | news | 475 | 0.865 | c304 5, c4 1, c41 12, c42 45, c5 1 | a:1.00 (1) c:1.00 (19) d:0.64 (22) e:0.38 (8) line:0.88 (425) |
| nkm | news-longest | 17 | 0.765 | c304 4 | c:0.75 (8) d:0.71 (7) e:1.00 (2) |
| nkm | news-milp | 5 | 0.400 | c0 1, c304 1, c41 1 | c:1.00 (1) d:0.33 (3) e:0.00 (1) |
| nkm | news-shortest | 17 | 1.000 |  | c:1.00 (8) d:1.00 (7) e:1.00 (2) |
| nkm | news20 | 11 | 0.636 | c304 2, c41 1, c42 1 | c:1.00 (3) d:0.50 (6) e:0.50 (2) |
| nkm | news60 | 11 | 0.636 | c304 2, c41 1, c42 1 | c:1.00 (3) d:0.50 (6) e:0.50 (2) |
| nkm | occbin | 475 | 0.865 | c304 5, c4 1, c41 12, c42 45, c5 1 | a:1.00 (1) c:1.00 (19) d:0.68 (22) e:0.38 (8) line:0.88 (425) |
| nkm | occbin60 | 4 | 1.000 |  | c:1.00 (1) d:1.00 (2) e:1.00 (1) |
| nkms | minnorm | 17 | 0.647 | c4 5, c42 1 | a:1.00 (2) b:1.00 (1) e:0.57 (14) |
| nkms | news | 475 | 0.937 | c304 2, c4 23, c41 4, c42 1 | a:1.00 (2) b:1.00 (3) c:1.00 (9) d:0.78 (9) e:0.48 (27) line:0.97 (425) |
| nkms | news-milp | 2 | 1.000 |  | b:1.00 (1) d:1.00 (1) |
| nkms | news20 | 10 | 0.800 | c4 2 | a:1.00 (1) b:1.00 (2) c:1.00 (1) d:1.00 (1) e:0.60 (5) |
| nkms | news60 | 10 | 0.800 | c4 2 | a:1.00 (1) b:1.00 (2) c:1.00 (1) d:1.00 (1) e:0.60 (5) |
| nkms | occbin | 475 | 0.935 | c304 3, c4 23, c41 4, c42 1 | a:1.00 (2) b:1.00 (3) c:0.89 (9) d:0.78 (9) e:0.48 (27) line:0.97 (425) |
| nkms | occbin60 | 7 | 0.714 | c4 2 | a:1.00 (1) b:1.00 (1) e:0.60 (5) |

occbin vs news, both finite: |difference in log posterior|

| model | pairs | <=1e-9 (rel) | 1e-9..0.15 | >0.15 | max |
|---|---|---|---|---|---|
| bf | 360 | 360 | 0 | 0 | 1.86e-09 |
| nkm | 410 | 396 | 10 | 4 | 0.461 |
| nkms | 444 | 438 | 4 | 2 | 0.404 |

## Timing distributions

Part A, seconds per finite log-posterior evaluation:

| model | config | n | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 360 | 0.03225 | 0.2212 | 0.3128 |
| bf | news20 | 4 | 0.03335 | 0.04315 | 0.0434 |
| bf | news60 | 4 | 0.0354 | 0.04365 | 0.0438 |
| bf | occbin | 361 | 0.0906 | 0.1744 | 0.1937 |
| bf | occbin60 | 3 | 0.0866 | 0.1019 | 0.1022 |
| nkm | minnorm | 17 | 0.0368 | 0.06027 | 0.0607 |
| nkm | news | 411 | 0.0366 | 0.07479 | 0.0885 |
| nkm | news-longest | 13 | 0.27 | 0.3712 | 0.3773 |
| nkm | news-milp | 2 | 0.03655 | 0.03885 | 0.0389 |
| nkm | news-shortest | 17 | 0.2964 | 0.3898 | 0.3927 |
| nkm | news20 | 7 | 0.0357 | 0.0447 | 0.0452 |
| nkm | news60 | 7 | 0.0365 | 0.04348 | 0.0437 |
| nkm | occbin | 411 | 0.1398 | 0.2755 | 0.5461 |
| nkm | occbin60 | 4 | 0.1416 | 0.1618 | 0.1622 |
| nkms | minnorm | 11 | 0.0354 | 0.07381 | 0.077 |
| nkms | news | 445 | 0.0342 | 0.06068 | 0.0978 |
| nkms | news-milp | 2 | 0.05115 | 0.06796 | 0.0683 |
| nkms | news20 | 8 | 0.03545 | 0.07399 | 0.0764 |
| nkms | news60 | 8 | 0.0411 | 0.07131 | 0.073 |
| nkms | occbin | 444 | 0.1262 | 0.2019 | 0.4155 |
| nkms | occbin60 | 5 | 0.1612 | 0.1968 | 0.1968 |

Part C, seconds per simulated period (per 1000-period segment, averaged over a task):

| model | method | tasks | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 4 | 3.15e-05 | 3.494e-05 | 3.5e-05 |
| bf | occbin | 4 | 0.000243 | 0.0002596 | 0.00026 |
| borrcon | news | 9 | 2.333e-05 | 6.912e-05 | 7e-05 |
| borrcon | occbin | 9 | 0.000345 | 0.0004789 | 0.00048 |
| borrcon_d | news | 5 | 2.667e-05 | 9.74e-05 | 0.0001 |
| borrcon_d | occbin | 5 | 0.0003733 | 0.000862 | 0.00088 |
| nkm | news | 8 | 3.167e-05 | 4.31e-05 | 4.333e-05 |
| nkm | occbin | 8 | 0.0002937 | 0.0003903 | 0.0003933 |
| nkm_est | news | 5 | 1.667e-05 | 3.907e-05 | 4e-05 |
| nkm_est | occbin | 5 | 0.00027 | 0.00028 | 0.00028 |
| sw | news | 1 | 3.667e-05 | 3.667e-05 | 3.667e-05 |
| sw | occbin | 1 | 0.0002733 | 0.0002733 | 0.0002733 |
| zlb | news | 10 | 4e-05 | 6.85e-05 | 7e-05 |
| zlb | occbin | 10 | 0.000384 | 0.0005138 | 0.00052 |
| zlb2 | news | 11 | 5.267e-05 | 7.81e-05 | 8e-05 |
| zlb2 | occbin | 11 | 0.0005 | 0.000875 | 0.0009 |

Part D, seconds per period of `HigherOrderNews.simulate` (successful runs):

| model | config | runs | median | p99 | max |
|---|---|---|---|---|---|
| borrcon | cub_all11 | 6 | 0 | 0.01145 | 0.012 |
| borrcon | cub_all7_tol3 | 25 | 0.088 | 0.2756 | 0.2825 |
| borrcon | cub_all9_tol05 | 17 | 0.013 | 0.151 | 0.1605 |
| borrcon | cub_r1_l9 | 30 | 0.0005 | 0.0008333 | 0.0008333 |
| borrcon | cub_r1_l9_tol1 | 27 | 0.0005 | 0.0008333 | 0.0008333 |
| borrcon | cub_r4_l9 | 26 | 0.005 | 0.007333 | 0.0075 |
| borrcon | order1 | 29 | 0 | 0 | 0 |
| borrcon | order2 | 31 | 0 | 0.0001167 | 0.0001667 |
| nk | cub_all11 | 39 | 0.0175 | 0.03131 | 0.0315 |
| nk | cub_all7_tol3 | 39 | 0.0025 | 0.02824 | 0.043 |
| nk | cub_all9_tol05 | 39 | 0.006 | 0.01322 | 0.01417 |
| nk | cub_r1_l9 | 41 | 0.0005 | 0.0009333 | 0.001 |
| nk | cub_r1_l9_tol1 | 41 | 0.0005 | 0.001 | 0.001 |
| nk | cub_r4_l9 | 40 | 0.0035 | 0.00561 | 0.006 |
| nk | order1 | 41 | 0 | 0 | 0 |
| nk | order2 | 42 | 0 | 0 | 0 |
| nkm | cub_all11 | 14 | 0.003333 | 0.01106 | 0.01167 |
| nkm | cub_all7_tol3 | 26 | 0.001 | 0.3965 | 0.5277 |
| nkm | cub_all9_tol05 | 24 | 0.001667 | 0.00454 | 0.005 |
| nkm | cub_r1_l9 | 20 | 0.0003333 | 0.00077 | 0.0008333 |
| nkm | cub_r1_l9_tol1 | 27 | 0.0001667 | 0.0008333 | 0.0008333 |
| nkm | cub_r4_l9 | 22 | 0.001667 | 0.003162 | 0.00325 |
| nkm | order1 | 25 | 0 | 0 | 0 |
| nkm | order2 | 15 | 0 | 0 | 0 |
| nkm_est | cub_all11 | 22 | 0.005 | 0.02777 | 0.0325 |
| nkm_est | cub_all7_tol3 | 22 | 0.001 | 0.001632 | 0.001667 |
| nkm_est | cub_all9_tol05 | 24 | 0.00175 | 0.004828 | 0.005 |
| nkm_est | cub_r1_l9 | 27 | 0.00025 | 0.00079 | 0.0008333 |
| nkm_est | cub_r1_l9_tol1 | 24 | 0 | 0.0005 | 0.0005 |
| nkm_est | cub_r4_l9 | 20 | 0.002292 | 0.00327 | 0.003333 |
| nkm_est | order1 | 21 | 0 | 0 | 0 |
| nkm_est | order2 | 19 | 0 | 0 | 0 |
| sw_cp | cub_all7_tol3 | 1 | 0.002 | 0.002 | 0.002 |
| sw_cp | cub_all9_tol05 | 2 | 0.004583 | 0.004992 | 0.005 |
| sw_cp | cub_r1_l9 | 29 | 0.0005 | 0.0009533 | 0.001 |
| sw_cp | cub_r1_l9_tol1 | 25 | 0.0005 | 0.0008333 | 0.0008333 |
| sw_cp | order1 | 33 | 0 | 0 | 0 |
| sw_cp | order2 | 32 | 0 | 0 | 0 |
| zlb | cub_all11 | 30 | 0.02033 | 0.03476 | 0.035 |
| zlb | cub_all7_tol3 | 41 | 0.00275 | 0.004167 | 0.004167 |
| zlb | cub_all9_tol05 | 40 | 0.006875 | 0.01134 | 0.01167 |
| zlb | cub_r1_l9 | 42 | 0.0005 | 0.0008333 | 0.0008333 |
| zlb | cub_r1_l9_tol1 | 40 | 0.0005 | 0.000935 | 0.001 |
| zlb | cub_r4_l9 | 42 | 0.003792 | 0.005492 | 0.005833 |
| zlb | order1 | 41 | 0 | 0 | 0 |
| zlb | order2 | 42 | 0 | 0 | 0 |
| zlb2 | cub_all11 | 30 | 0.08125 | 0.1242 | 0.1333 |
| zlb2 | cub_all7_tol3 | 40 | 0.0075 | 0.01727 | 0.01925 |
| zlb2 | cub_all9_tol05 | 41 | 0.02083 | 0.0544 | 0.065 |
| zlb2 | cub_r1_l9 | 41 | 0.0008333 | 0.001 | 0.001 |
| zlb2 | cub_r1_l9_tol1 | 39 | 0.0006667 | 0.001218 | 0.00125 |
| zlb2 | cub_r4_l9 | 41 | 0.01075 | 0.0132 | 0.0135 |
| zlb2 | order1 | 40 | 0 | 0.0001667 | 0.0001667 |
| zlb2 | order2 | 42 | 0 | 0 | 0 |

## Failures, parts C and D (expected error types)

Part C: per model, segments of 1000 periods that failed, by method and error (count of segments):


Part D: runs per model and configuration (ok / error types):

| model | config | outcomes |
|---|---|---|
| borrcon | cub_all11 | expected:CubatureFailure:horizon_too_short 28, expected:OBCClassError: 8, ok 6 |
| borrcon | cub_all7_tol3 | expected:CubatureFailure:horizon_too_short 5, expected:OBCClassError: 11, ok 25 |
| borrcon | cub_all9_tol05 | expected:CubatureFailure:horizon_too_short 18, expected:OBCClassError: 7, ok 17 |
| borrcon | cub_r1_l9 | expected:OBCClassError: 12, ok 30 |
| borrcon | cub_r1_l9_tol1 | expected:CubatureFailure:horizon_too_short 2, expected:OBCClassError: 12, ok 27 |
| borrcon | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 4, expected:OBCClassError: 12, ok 26 |
| borrcon | order1 | expected:OBCClassError: 13, ok 29 |
| borrcon | order2 | expected:OBCClassError: 10, expected:SteadyStateError: 1, ok 31 |
| nk | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 2, ok 39 |
| nk | cub_all7_tol3 | expected:BlanchardKahnError: 2, ok 39 |
| nk | cub_all9_tol05 | expected:BlanchardKahnError: 3, ok 39 |
| nk | cub_r1_l9 | expected:CubatureFailure:horizon_too_short 1, ok 41 |
| nk | cub_r1_l9_tol1 | ok 41 |
| nk | cub_r4_l9 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 1, ok 40 |
| nk | order1 | expected:BlanchardKahnError: 1, ok 41 |
| nk | order2 | ok 42 |
| nkm | cub_all11 | expected:CubatureFailure:horizon_too_short 1, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 15, expected:SteadyStateError: 11, ok 14 |
| nkm | cub_all7_tol3 | expected:CubatureFailure:no_solution 1, expected:OBCClassError: 8, expected:SteadyStateError: 6, ok 26 |
| nkm | cub_all9_tol05 | expected:CubatureFailure:no_solution 1, expected:OBCClassError: 8, expected:SteadyStateError: 8, ok 24 |
| nkm | cub_r1_l9 | expected:OBCClassError: 13, expected:SteadyStateError: 9, ok 20 |
| nkm | cub_r1_l9_tol1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 8, expected:SteadyStateError: 5, ok 27 |
| nkm | cub_r4_l9 | expected:BlanchardKahnError: 1, expected:OBCClassError: 10, expected:SteadyStateError: 9, ok 22 |
| nkm | order1 | expected:OBCClassError: 12, expected:SteadyStateError: 5, ok 25 |
| nkm | order2 | expected:BlanchardKahnError: 1, expected:OBCClassError: 18, expected:SteadyStateError: 8, ok 15 |
| nkm_est | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 10, expected:SteadyStateError: 7, ok 22 |
| nkm_est | cub_all7_tol3 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 12, expected:SteadyStateError: 5, ok 22 |
| nkm_est | cub_all9_tol05 | expected:CubatureFailure:no_solution 2, expected:OBCClassError: 9, expected:SteadyStateError: 7, ok 24 |
| nkm_est | cub_r1_l9 | expected:OBCClassError: 10, expected:SteadyStateError: 5, ok 27 |
| nkm_est | cub_r1_l9_tol1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 11, expected:SteadyStateError: 5, ok 24 |
| nkm_est | cub_r4_l9 | expected:OBCClassError: 16, expected:SteadyStateError: 6, ok 20 |
| nkm_est | order1 | expected:OBCClassError: 11, expected:SteadyStateError: 10, ok 21 |
| nkm_est | order2 | expected:OBCClassError: 14, expected:SteadyStateError: 9, ok 19 |
| sw_cp | cub_all7_tol3 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 35, expected:OBCClassError: 4, ok 1 |
| sw_cp | cub_all9_tol05 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 70, expected:OBCClassError: 11, ok 2 |
| sw_cp | cub_r1_l9 | expected:CubatureFailure:no_solution 8, expected:OBCClassError: 5, ok 29 |
| sw_cp | cub_r1_l9_tol1 | expected:BlanchardKahnError: 3, expected:CubatureFailure:no_solution 9, expected:OBCClassError: 4, ok 25 |
| sw_cp | cub_r4_l9 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 32, expected:OBCClassError: 8 |
| sw_cp | order1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 1, expected:OBCError:no_solution 7, ok 33 |
| sw_cp | order2 | expected:OBCClassError: 1, expected:OBCError:no_solution 9, ok 32 |
| zlb | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 11, ok 30 |
| zlb | cub_all7_tol3 | ok 41 |
| zlb | cub_all9_tol05 | expected:CubatureFailure:horizon_too_short 2, ok 40 |
| zlb | cub_r1_l9 | ok 42 |
| zlb | cub_r1_l9_tol1 | expected:BlanchardKahnError: 1, ok 40 |
| zlb | cub_r4_l9 | ok 42 |
| zlb | order1 | expected:BlanchardKahnError: 1, ok 41 |
| zlb | order2 | ok 42 |
| zlb2 | cub_all11 | expected:CubatureFailure:horizon_too_short 12, ok 30 |
| zlb2 | cub_all7_tol3 | expected:CubatureFailure:horizon_too_short 1, ok 40 |
| zlb2 | cub_all9_tol05 | expected:BlanchardKahnError: 1, ok 41 |
| zlb2 | cub_r1_l9 | expected:BlanchardKahnError: 1, ok 41 |
| zlb2 | cub_r1_l9_tol1 | expected:BlanchardKahnError: 2, ok 39 |
| zlb2 | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 1, ok 41 |
| zlb2 | order1 | expected:BlanchardKahnError: 1, expected:OBCError:no_solution 1, ok 40 |
| zlb2 | order2 | ok 42 |

## Warnings (distinct messages, numbers normalised, counts over all parts)

- 69 x `UserWarning: M is not an S-matrix: some paths can have no solution; cubature points may fail (see Highe`
- 52 x `UserWarning: cubature_failure_tolerance=#: # of # relevant perturbed points (#%) were dropped in # of #`

## Memory (resident MB at the end of each task; one process per ~8 minute chunk, so growth is only meaningful within a chunk)

- A: median 131, max 1446
- B: median 372, max 1409
- C: median 371, max 1412
- D: median 397, max 1415
