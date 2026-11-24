# OBC soak test (pydyn, branch `obc-soak`, from `wip/obc` 0abe8e7)

Brute-force soak test of the occasionally-binding-constraint estimation and simulation code, to find bugs that appear only over many
parameter vectors or long runs. Library code is not changed; anomalies are catalogued with reproducers (`soak/repro/`).

## State / how to resume

(written by hand; see the end of the file for the numbers)

## Coverage so far

| part | run time (h) | tasks | what |
|---|---|---|---|
| A | 0.03 | 67 | evaluations of the log posterior (all configurations): bf 296, nkm 364, nkms 299; tasks by kind: bf/bound 5, bf/line 5, bf/perturb 6, bf/prior 6, nkm/bound 5, nkm/line 6, nkm/perturb 6, nkm/prior 6, nkms/bound 5, nkms/line 5, nkms/perturb 6, nkms/prior 6 |
| B | 0.04 | 12 | 1500 MCMC draws (news) in 0 finished chains + the running one (chain 0, draw 1500, replays under occbin of this chain: 150); 2 mode searches (0 skipped: start not finite, 0 timeouts) |
| C | 0.04 | 31 | simulated periods per model (both methods each): bf 13000 (4 runs, 2 binding), borrcon 64000 (4 runs, 25040 binding), borrcon_d 26000 (4 runs, 838 binding), nkm 13000 (4 runs, 455 binding), nkm_est 14000 (3 runs, 0 binding), sw 3000 (4 runs, 0 binding), zlb 7000 (4 runs, 2534 binding), zlb2 51000 (4 runs, 19923 binding) |
| D | 0.05 | 609 | runs per model (ok/total): borrcon 53/87, nk 85/87, nkm 39/87, nkm_est 40/87, sw_cp 32/87, zlb 82/87, zlb2 81/87 |

LCP solutions checked with `verify_lcp` (tolerance 1e-6) at the `solve_active_set` / `solve_news` boundary, all parts: {}.

Determinism checks (same call repeated): A same-process 14 (+4 in a fresh estimation object), C 6, D 109.

## Anomaly catalogue

| kind | part | count | severity | first reproducer(s) | judgement |
|---|---|---|---|---|---|
| `nonbitwise_ulp:higher_order` | D | 25 | ? | soak/results/anomalies.jsonl (task 0, task 36, task 48) | not yet looked at |
| `lp_disagree_large:nkm:regime_count_differs` | A | 2 | ? | soak/results/anomalies.jsonl (task 6, task 6) | not yet looked at |
| `variant_fails_where_news_ok:news-longest` | A | 2 | ? | soak/results/anomalies.jsonl (task 12, task 24) | not yet looked at |
| `finite_vs_inf:news_finite_occbin_code304` | A | 1 | ? | soak/results/anomalies.jsonl (task 30) | not yet looked at |
| `milp_timeout_hit` | A | 1 | ? | soak/results/anomalies.jsonl (task 65) | not yet looked at |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:sw` | C | 1 | ? | soak/results/anomalies.jsonl (task 30) | not yet looked at |
| `slow_failure` | A | 1 | ? | soak/results/anomalies.jsonl (task 65) | not yet looked at |
| `slow_failure` | C | 1 | ? | soak/results/anomalies.jsonl (task 14) | not yet looked at |

## Failure rates, part A (inversion filter, log posterior evaluations)

Distance bin = rms of the relative deviation from the model's initial values: a <0.02, b <0.1, c <0.3, d <1, e >=1; `line` = the 25 points of the straight-line sweeps (-0.1..1.1).

| model | config | n | finite | failed (by code) | by distance bin: finite share |
|---|---|---|---|---|---|
| bf | news | 142 | 0.704 | c304 21, c41 3, c42 18 | c:0.71 (7) d:0.00 (4) e:0.17 (6) line:0.75 (125) |
| bf | news-milp | 3 | 0.000 | c304 1, c41 1, c42 1 | d:0.00 (1) e:0.00 (2) |
| bf | news20 | 4 | 0.250 | c304 1, c41 1, c42 1 | c:1.00 (1) d:0.00 (1) e:0.00 (2) |
| bf | news60 | 4 | 0.250 | c304 1, c41 1, c42 1 | c:1.00 (1) d:0.00 (1) e:0.00 (2) |
| bf | occbin | 142 | 0.704 | c304 21, c41 3, c42 18 | c:0.71 (7) d:0.00 (4) e:0.17 (6) line:0.75 (125) |
| bf | occbin60 | 1 | 1.000 |  | c:1.00 (1) |
| nkm | minnorm | 6 | 1.000 |  | c:1.00 (3) d:1.00 (3) |
| nkm | news | 167 | 0.898 | c304 2, c41 2, c42 13 | c:1.00 (9) d:0.67 (6) e:0.50 (2) line:0.91 (150) |
| nkm | news-longest | 6 | 0.667 | c304 2 | c:1.00 (3) d:0.33 (3) |
| nkm | news-milp | 2 | 1.000 |  | c:1.00 (1) d:1.00 (1) |
| nkm | news-shortest | 6 | 1.000 |  | c:1.00 (3) d:1.00 (3) |
| nkm | news20 | 4 | 1.000 |  | c:1.00 (2) d:1.00 (2) |
| nkm | news60 | 4 | 1.000 |  | c:1.00 (2) d:1.00 (2) |
| nkm | occbin | 167 | 0.892 | c304 3, c41 2, c42 13 | c:1.00 (9) d:0.67 (6) e:0.50 (2) line:0.90 (150) |
| nkm | occbin60 | 2 | 1.000 |  | c:1.00 (1) d:1.00 (1) |
| nkms | minnorm | 6 | 0.500 | c4 3 | a:1.00 (1) e:0.40 (5) |
| nkms | news | 142 | 0.873 | c304 1, c4 16, c41 1 | a:1.00 (1) b:1.00 (1) c:1.00 (1) d:0.83 (6) e:0.38 (8) line:0.90 (125) |
| nkms | news-milp | 1 | 1.000 |  | d:1.00 (1) |
| nkms | news20 | 3 | 0.667 | c4 1 | a:1.00 (1) d:1.00 (1) e:0.00 (1) |
| nkms | news60 | 3 | 0.667 | c4 1 | a:1.00 (1) d:1.00 (1) e:0.00 (1) |
| nkms | occbin | 142 | 0.873 | c304 1, c4 16, c41 1 | a:1.00 (1) b:1.00 (1) c:1.00 (1) d:0.83 (6) e:0.38 (8) line:0.90 (125) |
| nkms | occbin60 | 2 | 0.500 | c4 1 | a:1.00 (1) e:0.00 (1) |

occbin vs news, both finite: |difference in log posterior|

| model | pairs | <=1e-9 (rel) | 1e-9..0.15 | >0.15 | max |
|---|---|---|---|---|---|
| bf | 100 | 100 | 0 | 0 | 0 |
| nkm | 149 | 146 | 1 | 2 | 0.279 |
| nkms | 124 | 124 | 0 | 0 | 4.07e-10 |

## Timing distributions

Part A, seconds per finite log-posterior evaluation:

| model | config | n | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 100 | 0.03135 | 0.2493 | 0.3128 |
| bf | news20 | 1 | 0.0316 | 0.0316 | 0.0316 |
| bf | news60 | 1 | 0.032 | 0.032 | 0.032 |
| bf | occbin | 100 | 0.0893 | 0.1782 | 0.1854 |
| bf | occbin60 | 1 | 0.0866 | 0.0866 | 0.0866 |
| nkm | minnorm | 6 | 0.0355 | 0.04044 | 0.0406 |
| nkm | news | 150 | 0.0335 | 0.0636 | 0.0805 |
| nkm | news-longest | 4 | 0.265 | 0.2699 | 0.27 |
| nkm | news-milp | 2 | 0.03655 | 0.03885 | 0.0389 |
| nkm | news-shortest | 6 | 0.2626 | 0.3238 | 0.3252 |
| nkm | news20 | 4 | 0.03385 | 0.03579 | 0.0358 |
| nkm | news60 | 4 | 0.03565 | 0.03706 | 0.0371 |
| nkm | occbin | 149 | 0.1278 | 0.2487 | 0.2852 |
| nkm | occbin60 | 2 | 0.1329 | 0.1355 | 0.1356 |
| nkms | minnorm | 3 | 0.0322 | 0.0324 | 0.0324 |
| nkms | news | 124 | 0.0321 | 0.04493 | 0.0476 |
| nkms | news-milp | 1 | 0.034 | 0.034 | 0.034 |
| nkms | news20 | 2 | 0.03335 | 0.03369 | 0.0337 |
| nkms | news60 | 2 | 0.03505 | 0.03775 | 0.0378 |
| nkms | occbin | 124 | 0.12 | 0.1751 | 0.2388 |
| nkms | occbin60 | 1 | 0.1194 | 0.1194 | 0.1194 |

Part C, seconds per simulated period (per 1000-period segment, averaged over a task):

| model | method | tasks | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 3 | 3e-05 | 3.294e-05 | 3.3e-05 |
| bf | occbin | 3 | 0.00024 | 0.0002459 | 0.000246 |
| borrcon | news | 4 | 2.7e-05 | 5.825e-05 | 5.9e-05 |
| borrcon | occbin | 4 | 0.0003077 | 0.0004517 | 0.0004547 |
| borrcon_d | news | 1 | 1.9e-05 | 1.9e-05 | 1.9e-05 |
| borrcon_d | occbin | 1 | 0.000278 | 0.000278 | 0.000278 |
| nkm | news | 3 | 4e-05 | 4e-05 | 4e-05 |
| nkm | occbin | 3 | 0.00029 | 0.0003488 | 0.00035 |
| nkm_est | news | 2 | 2.833e-05 | 3.977e-05 | 4e-05 |
| nkm_est | occbin | 2 | 0.0002533 | 0.0002566 | 0.0002567 |
| zlb | news | 3 | 3.333e-05 | 3.987e-05 | 4e-05 |
| zlb | occbin | 3 | 0.00032 | 0.0003984 | 0.0004 |
| zlb2 | news | 4 | 4.617e-05 | 5.992e-05 | 6e-05 |
| zlb2 | occbin | 4 | 0.0004628 | 0.0006461 | 0.00065 |

Part D, seconds per period of `HigherOrderNews.simulate` (successful runs):

| model | config | runs | median | p99 | max |
|---|---|---|---|---|---|
| borrcon | cub_all11 | 1 | 0 | 0 | 0 |
| borrcon | cub_all7_tol3 | 8 | 0.04267 | 0.1087 | 0.1092 |
| borrcon | cub_all9_tol05 | 6 | 0.0095 | 0.09888 | 0.1008 |
| borrcon | cub_r1_l9 | 10 | 0.0005 | 0.0008033 | 0.0008333 |
| borrcon | cub_r1_l9_tol1 | 9 | 0.0005 | 0.0008333 | 0.0008333 |
| borrcon | cub_r4_l9 | 4 | 0.00175 | 0.004682 | 0.00475 |
| borrcon | order1 | 8 | 0 | 0 | 0 |
| borrcon | order2 | 7 | 0 | 0.0001567 | 0.0001667 |
| nk | cub_all11 | 10 | 0.01375 | 0.02198 | 0.0225 |
| nk | cub_all7_tol3 | 11 | 0.0025 | 0.00385 | 0.004 |
| nk | cub_all9_tol05 | 11 | 0.0055 | 0.006967 | 0.007 |
| nk | cub_r1_l9 | 11 | 0.0005 | 0.0008 | 0.0008333 |
| nk | cub_r1_l9_tol1 | 10 | 0.0005 | 0.0008333 | 0.0008333 |
| nk | cub_r4_l9 | 10 | 0.003333 | 0.0035 | 0.0035 |
| nk | order1 | 11 | 0 | 0 | 0 |
| nk | order2 | 11 | 0 | 0 | 0 |
| nkm | cub_all11 | 2 | 0.001 | 0.00198 | 0.002 |
| nkm | cub_all7_tol3 | 6 | 0.001 | 0.5014 | 0.5277 |
| nkm | cub_all9_tol05 | 5 | 0.0015 | 0.00249 | 0.0025 |
| nkm | cub_r1_l9 | 4 | 0.000375 | 0.0005 | 0.0005 |
| nkm | cub_r1_l9_tol1 | 6 | 0.00025 | 0.0005 | 0.0005 |
| nkm | cub_r4_l9 | 6 | 0.001833 | 0.003229 | 0.00325 |
| nkm | order1 | 7 | 0 | 0 | 0 |
| nkm | order2 | 3 | 0 | 0 | 0 |
| nkm_est | cub_all11 | 5 | 0.00375 | 0.00901 | 0.009167 |
| nkm_est | cub_all7_tol3 | 5 | 0.0005 | 0.00132 | 0.001333 |
| nkm_est | cub_all9_tol05 | 4 | 0.0004167 | 0.002692 | 0.00275 |
| nkm_est | cub_r1_l9 | 6 | 0.00025 | 0.00025 | 0.00025 |
| nkm_est | cub_r1_l9_tol1 | 5 | 0.00025 | 0.00025 | 0.00025 |
| nkm_est | cub_r4_l9 | 7 | 0.00175 | 0.00296 | 0.003 |
| nkm_est | order1 | 4 | 0 | 0 | 0 |
| nkm_est | order2 | 4 | 0 | 0 | 0 |
| sw_cp | cub_r1_l9 | 8 | 0.0005 | 0.00081 | 0.0008333 |
| sw_cp | cub_r1_l9_tol1 | 8 | 0.0005 | 0.0005 | 0.0005 |
| sw_cp | order1 | 10 | 0 | 0 | 0 |
| sw_cp | order2 | 6 | 0 | 0 | 0 |
| zlb | cub_all11 | 7 | 0.01733 | 0.02089 | 0.021 |
| zlb | cub_all7_tol3 | 11 | 0.0025 | 0.003483 | 0.0035 |
| zlb | cub_all9_tol05 | 11 | 0.006 | 0.009383 | 0.0095 |
| zlb | cub_r1_l9 | 11 | 0.0005 | 0.0008333 | 0.0008333 |
| zlb | cub_r1_l9_tol1 | 10 | 0.0004167 | 0.000955 | 0.001 |
| zlb | cub_r4_l9 | 11 | 0.003333 | 0.00415 | 0.004167 |
| zlb | order1 | 10 | 0 | 0 | 0 |
| zlb | order2 | 11 | 0 | 0 | 0 |
| zlb2 | cub_all11 | 7 | 0.07375 | 0.09905 | 0.1 |
| zlb2 | cub_all7_tol3 | 11 | 0.005833 | 0.009833 | 0.01 |
| zlb2 | cub_all9_tol05 | 10 | 0.01942 | 0.02705 | 0.02717 |
| zlb2 | cub_r1_l9 | 10 | 0.0007083 | 0.000985 | 0.001 |
| zlb2 | cub_r1_l9_tol1 | 10 | 0.0005 | 0.0008258 | 0.0008333 |
| zlb2 | cub_r4_l9 | 11 | 0.00975 | 0.0133 | 0.0135 |
| zlb2 | order1 | 11 | 0 | 0 | 0 |
| zlb2 | order2 | 11 | 0 | 0 | 0 |

## Failures, parts C and D (expected error types)

Part C: per model, segments of 1000 periods that failed, by method and error (count of segments):


Part D: runs per model and configuration (ok / error types):

| model | config | outcomes |
|---|---|---|
| borrcon | cub_all11 | expected:CubatureFailure:horizon_too_short 6, expected:OBCClassError: 4, ok 1 |
| borrcon | cub_all7_tol3 | expected:CubatureFailure:horizon_too_short 1, expected:OBCClassError: 2, ok 8 |
| borrcon | cub_all9_tol05 | expected:CubatureFailure:horizon_too_short 5, ok 6 |
| borrcon | cub_r1_l9 | expected:OBCClassError: 1, ok 10 |
| borrcon | cub_r1_l9_tol1 | expected:OBCClassError: 1, ok 9 |
| borrcon | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 2, expected:OBCClassError: 5, ok 4 |
| borrcon | order1 | expected:OBCClassError: 3, ok 8 |
| borrcon | order2 | expected:OBCClassError: 4, ok 7 |
| nk | cub_all11 | expected:BlanchardKahnError: 1, ok 10 |
| nk | cub_all7_tol3 | ok 11 |
| nk | cub_all9_tol05 | ok 11 |
| nk | cub_r1_l9 | ok 11 |
| nk | cub_r1_l9_tol1 | ok 10 |
| nk | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 1, ok 10 |
| nk | order1 | ok 11 |
| nk | order2 | ok 11 |
| nkm | cub_all11 | expected:OBCClassError: 5, expected:SteadyStateError: 4, ok 2 |
| nkm | cub_all7_tol3 | expected:OBCClassError: 2, expected:SteadyStateError: 3, ok 6 |
| nkm | cub_all9_tol05 | expected:CubatureFailure:no_solution 1, expected:OBCClassError: 4, expected:SteadyStateError: 1, ok 5 |
| nkm | cub_r1_l9 | expected:OBCClassError: 5, expected:SteadyStateError: 2, ok 4 |
| nkm | cub_r1_l9_tol1 | expected:OBCClassError: 3, expected:SteadyStateError: 1, ok 6 |
| nkm | cub_r4_l9 | expected:OBCClassError: 2, expected:SteadyStateError: 3, ok 6 |
| nkm | order1 | expected:OBCClassError: 3, expected:SteadyStateError: 1, ok 7 |
| nkm | order2 | expected:OBCClassError: 6, expected:SteadyStateError: 2, ok 3 |
| nkm_est | cub_all11 | expected:BlanchardKahnError: 1, expected:OBCClassError: 3, expected:SteadyStateError: 2, ok 5 |
| nkm_est | cub_all7_tol3 | expected:OBCClassError: 5, expected:SteadyStateError: 1, ok 5 |
| nkm_est | cub_all9_tol05 | expected:OBCClassError: 5, expected:SteadyStateError: 2, ok 4 |
| nkm_est | cub_r1_l9 | expected:OBCClassError: 4, expected:SteadyStateError: 1, ok 6 |
| nkm_est | cub_r1_l9_tol1 | expected:OBCClassError: 1, expected:SteadyStateError: 4, ok 5 |
| nkm_est | cub_r4_l9 | expected:OBCClassError: 3, expected:SteadyStateError: 1, ok 7 |
| nkm_est | order1 | expected:OBCClassError: 5, expected:SteadyStateError: 2, ok 4 |
| nkm_est | order2 | expected:OBCClassError: 3, expected:SteadyStateError: 4, ok 4 |
| sw_cp | cub_all7_tol3 | expected:CubatureFailure:no_solution 9, expected:OBCClassError: 2 |
| sw_cp | cub_all9_tol05 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 16, expected:OBCClassError: 5 |
| sw_cp | cub_r1_l9 | expected:CubatureFailure:no_solution 2, expected:OBCClassError: 1, ok 8 |
| sw_cp | cub_r1_l9_tol1 | expected:CubatureFailure:no_solution 2, ok 8 |
| sw_cp | cub_r4_l9 | expected:CubatureFailure:no_solution 8, expected:OBCClassError: 3 |
| sw_cp | order1 | expected:OBCClassError: 1, ok 10 |
| sw_cp | order2 | expected:OBCClassError: 1, expected:OBCError:no_solution 4, ok 6 |
| zlb | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 3, ok 7 |
| zlb | cub_all7_tol3 | ok 11 |
| zlb | cub_all9_tol05 | ok 11 |
| zlb | cub_r1_l9 | ok 11 |
| zlb | cub_r1_l9_tol1 | ok 10 |
| zlb | cub_r4_l9 | ok 11 |
| zlb | order1 | expected:BlanchardKahnError: 1, ok 10 |
| zlb | order2 | ok 11 |
| zlb2 | cub_all11 | expected:CubatureFailure:horizon_too_short 4, ok 7 |
| zlb2 | cub_all7_tol3 | ok 11 |
| zlb2 | cub_all9_tol05 | expected:BlanchardKahnError: 1, ok 10 |
| zlb2 | cub_r1_l9 | expected:BlanchardKahnError: 1, ok 10 |
| zlb2 | cub_r1_l9_tol1 | ok 10 |
| zlb2 | cub_r4_l9 | ok 11 |
| zlb2 | order1 | ok 11 |
| zlb2 | order2 | ok 11 |

## Warnings (distinct messages, numbers normalised, counts over all parts)

- 20 x `UserWarning: M is not an S-matrix: some paths can have no solution; cubature points may fail (see Highe`
- 14 x `UserWarning: cubature_failure_tolerance=#: # of # relevant perturbed points (#%) were dropped in # of #`

## Memory (resident MB at the end of each task; one process per ~8 minute chunk, so growth is only meaningful within a chunk)

- A: median 131, max 372
- B: median 373, max 373
- C: median 372, max 374
- D: median 464, max 464
