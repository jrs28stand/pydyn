# OBC soak test (pydyn, branch `obc-soak`, from `wip/obc` 0abe8e7)

Brute-force soak test of the occasionally-binding-constraint estimation and simulation code, to find bugs that appear only over many
parameter vectors or long runs. Library code is not changed; anomalies are catalogued with reproducers (`soak/repro/`).

## State / how to resume

Everything lives under `soak/` of branch `obc-soak` (pushed to `wip/obc-soak`): `soak/run.py` (driver), `soak/lib.py`, `soak/part_a.py` (parts A and B),
`soak/part_cd.py` (parts C and D), `soak/report.py` (regenerates this file from `soak/notes.md`, `soak/notes.json`, `soak/state/*.json`,
`soak/results/*.jsonl`), `soak/repro/` (reproducers).

Resume (one process at a time, 2 shared cores, always `nice`):
```
git clone /home/claude/pydyn /home/claude/work/pydyn-soak; cd /home/claude/work/pydyn-soak   # or: git fetch origin wip/obc-soak && git checkout obc-soak
cp /home/claude/pydyn/korder/libpydyn_korder.so korder/
soak/chunk.sh            # ~8 minutes: all four parts in rotation, resumes from soak/state/*.json, appends to soak/results/*.jsonl
python3 soak/report.py   # regenerate docs/OBC_SOAK.md; then git add -A soak docs; commit; git push origin obc-soak:wip/obc-soak
python3 soak/run.py A --replay 141     # re-run one task (any part) by index, nothing is written; anomalies are printed
```
Design: task i of a part is a pure function of (part, i) (`rng_for`), the state is only the next index (+ the MCMC chain, with generator state, in `state/B.json`),
so a stop loses at most one task.  Parts: A = log-posterior sweeps (`i%3`: nkm, nkms, bf; `(i//3)%4`: prior draw, perturbation of x0/stored vectors at scales 0.01-2,
25-point line sweep -0.1..1.1, near-bound), configurations occbin/news always, +minnorm (i%4==0), +shortest/longest (i%6==0, not bf), +horizons 20/60 (i%5==0), +MILP 1 s
(i%40==5); B = adaptive-Metropolis chains of 6,000 news draws in segments of 150 (every 10th filter-reaching proposal replayed under occbin) and, for every sixth task, a
csminwel mode search from a random start (90 s limit); C = occbin vs news simulation in 1000-period segments up to 30,000 periods or 30 s on zlb, zlb2, nkm
(NKM_0_std_shocks), bf, borrcon (Dynare tests), occ_borrcon, sw, nkm_est, random parameter draws (scale 0/0.02/0.1/0.3) and shock scales 0.5-5 x a per-model base;
D = `HigherOrderNews` order 2 / 1, cubature all level 11 / 9 / 7 (with tolerance), regions 1 and 4 at level 9, on nk, nkm_est, nkm, zlb, zlb2, borrcon, sw_cp.
Every returned LCP solution is checked with `verify_lcp` (tolerance 1e-6) by wrapping `solve_active_set` and `solve_news`.
Per-evaluation timer 40 s (SIGALRM, `Alarm` is a BaseException). Expected exceptions: OBCError, OBCClassError, CubatureFailure, OccbinError (and subclasses),
BlanchardKahnError, SteadyStateError, LinAlgError; anything else is an anomaly with traceback.
Known limitations of the checks: wedge sign is checked only for LCP (one-equation) models; `lp_disagree` tolerance 0.15 (+1e-10 |lp|) for the approx-class NKM models and
1e-7 relative for bf; part C counts per 1000-period segment, so failure/anomaly counts of C are segments, not events.

## Coverage so far

| part | run time (h) | tasks | what |
|---|---|---|---|
| A | 0.76 | 1964 | evaluations of the log posterior (all configurations): bf 9380, nkm 9922, nkms 9627; tasks by kind: bf/bound 163, bf/line 163, bf/perturb 164, bf/prior 164, nkm/bound 163, nkm/line 164, nkm/perturb 164, nkm/prior 164, nkms/bound 163, nkms/line 164, nkms/perturb 164, nkms/prior 164 |
| B | 0.90 | 356 | 44550 MCMC draws (news) in 7 finished chains + the running one (chain 7, draw 2550, replays under occbin of this chain: 255); 58 mode searches (9 skipped: start not finite, 1 timeouts) |
| C | 0.82 | 455 | simulated periods per model (both methods each): bf 429000 (57 runs, 16301 binding), borrcon 627000 (57 runs, 133947 binding), borrcon_d 671000 (57 runs, 88677 binding), nkm 596000 (57 runs, 70399 binding), nkm_est 504000 (56 runs, 46430 binding), sw 314000 (57 runs, 1154 binding), zlb 602000 (57 runs, 216944 binding), zlb2 446000 (57 runs, 175419 binding) |
| D | 0.73 | 14794 | runs per model (ok/total): borrcon 1213/2113, nk 2049/2114, nkm 1104/2113, nkm_est 1083/2114, sw_cp 707/2113, zlb 1998/2113, zlb2 1975/2114 |

LCP solutions checked with `verify_lcp` (tolerance 1e-6) at the `solve_active_set` / `solve_news` boundary, all parts: {'verify|solve_active_set': 567484, 'verify|solve_news': 55}.

Determinism checks (same call repeated): A same-process 422 (+96 in a fresh estimation object), C 72, D 2746.

## Anomaly catalogue

| kind | part | count | severity | first reproducer(s) | judgement |
|---|---|---|---|---|---|
| `nonbitwise_ulp:higher_order` | D | 674 | cosmetic | soak/results/anomalies.jsonl (task 0, task 36, task 48) | two constructions of HigherOrderNews on identical inputs in one process give results differing at 1e-16..1e-18, not bitwise (rounding order, probably BLAS alignment / threading-free kernels); all differences < 1e-9. No nondeterminism of the order-1 or the filter calls (see determinism counts) |
| `path_disagree:nkm` | C | 119 | cosmetic (known cause) | soak/results/anomalies.jsonl (task 66, task 66, task 82) | the 'approx' constraint class: paths differ by 1e-5 relative in a long run once a knife-edge period flips (shock scale >= 2 x sd, 1000-period segments); regimes differ in that period |
| `path_disagree:nkm_est` | C | 87 | cosmetic (known cause) | soak/results/anomalies.jsonl (task 127, task 127, task 127) | same as nkm (occest_nkm.mod) |
| `variant_fails_where_news_ok:news-longest` | A | 52 | cosmetic | soak/results/anomalies.jsonl (task 12, task 24, task 96) | obc_selection='longest' (and 'shortest') can fail the filter (304) where the default selection succeeds: a different (longer) spell makes the root search fail; not a bug as such |
| `slow_failure` | C | 39 | perf | soak/repro/occbin_rule_trie_memory.py (task 14, task 43, task 43) | evaluations/simulations that fail after > 5 s (A) or > 10 s (C): OccBin inversion-filter failures with huge regime-guess tries (see memory_jump), the opt-in MILP configuration, and RegimeOBC's 30 s integer-programme limit (sw model: 30 s for one failing period) |
| `lp_disagree_large:nkm:regime_count_differs` | A | 37 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 6, task 6, task 162) | knife edge of the 'approx' class: one binding period starts/ends one period apart (regime count differs by 1, or same count with shifted sequence). Size 0.15-0.4, i.e. larger than the documented ~0.12. Period 21 of the sample at line t=0.1 (A#6) differs; filtered shock differs by 0.04. |
| `finite_vs_inf:news_finite_occbin_code304` | A | 14 | wrong-result (occbin refuses a point news evaluates) | soak/results/anomalies.jsonl (task 30, task 193, task 285) | OccBin's failure codes (304 via 310/311/313) at points where the news filter finds a solution; the counterpart of the previous kind; expected at cycling/periodic guess-and-verify failures |
| `path_disagree:bf` | C | 14 | probably not a bug (multiplicity) | soak/repro/bf_path_disagree.py (task 43, task 83, task 83) | bf (two equations switch): OccBin binds for one period with rnot - rlb = -2e-6, news does not bind (rnot - rlb = +2e-5): each satisfies its own regime condition; the paths differ by 0.006 although the two sit at the boundary: a jump across the constraint, i.e. two solutions |
| `memory_jump` | A | 12 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 141, task 273, task 309) | Occbin.rules' memo trie (pydyn/occbin.py:260) is unbounded: at extreme parameter vectors OccBin's filter builds 232,740 rule nodes in one Occbin instance (1.3 GB peak, 10 s, then fails with 304). Seen 6+ times in A, C and D (the events listed under B, C, D with 1.0-1.1 GB are the same process memory after an A event: an artefact of the first version of the detector). The news solver needs 0.5 s at the same vector. Python does not return the memory to the OS: the process stays at ~1.4 GB |
| `finite_vs_inf:occbin_finite_news_code304` | A | 10 | wrong-result (news refuses a point OccBin evaluates) | soak/repro/finite_vs_inf_news_304.py (task 75, task 155, task 500) | news horizon 40 too short at extreme vectors (filtered shocks of size 100+, spells longer than 40): horizon 80 gives the OccBin value to 1e-11. The filter reports the generic code 304; the message does not say that the horizon is the cause (trial points fail with 313, 168 of them at the first failing period) |
| `lp_absurd` | A | 9 | cosmetic | soak/results/anomalies.jsonl (task 70, task 70, task 70) | FALSE POSITIVE of an early threshold (|lp|>1e6): legitimate huge negative values from tiny shock standard deviations (-3e10 ... -2e12). Threshold changed to lp>1e5; huge negative values are counted separately |
| `memory_jump` | D | 9 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 1782, task 2697, task 3313) | Occbin.rules' memo trie (pydyn/occbin.py:260) is unbounded: at extreme parameter vectors OccBin's filter builds 232,740 rule nodes in one Occbin instance (1.3 GB peak, 10 s, then fails with 304). Seen 6+ times in A, C and D (the events listed under B, C, D with 1.0-1.1 GB are the same process memory after an A event: an artefact of the first version of the detector). The news solver needs 0.5 s at the same vector. Python does not return the memory to the OS: the process stays at ~1.4 GB |
| `regime_history_differs_equal_path:nkm` | C | 9 | cosmetic | soak/results/anomalies.jsonl (task 82, task 82, task 130) | regime histories differ in the knife-edge period while paths agree to 1e-6 |
| `slow_failure` | A | 8 | perf | soak/repro/occbin_rule_trie_memory.py (task 65, task 75, task 105) | evaluations/simulations that fail after > 5 s (A) or > 10 s (C): OccBin inversion-filter failures with huge regime-guess tries (see memory_jump), the opt-in MILP configuration, and RegimeOBC's 30 s integer-programme limit (sw model: 30 s for one failing period) |
| `milp_timeout_hit` | A | 6 | perf | soak/results/anomalies.jsonl (task 65, task 75, task 105) | only in the opt-in configuration est.obc_milp_time_limit=1.0 (default 0): up to 96 integer programmes per evaluation, each running to its 1 s limit -> one evaluation 45 s (bf) or > 40 s (nkm); 'rescued' stays 0. Documented cost of the option |
| `lp_disagree_large:nkms:regime_count_differs` | A | 3 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 79, task 556, task 1156) | same knife edge as for nkm (3-parameter model); also seen at news MCMC draws (replay under occbin, part B) and at a news posterior mode (0.23 with equal binding-period count but a shifted sequence) |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm` | C | 3 | cosmetic | soak/results/anomalies.jsonl (task 82, task 354, task 370) | OccBin guess-and-verify cycles (error 310) or news reports horizon_too_short/no_solution on one side only; counted per 1000-period segment (counts are segments, not events) |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:sw` | C | 3 | cosmetic | soak/results/anomalies.jsonl (task 30, task 86, task 430) | OccBin guess-and-verify cycles (error 310) or news reports horizon_too_short/no_solution on one side only; counted per 1000-period segment (counts are segments, not events) |
| `regime_history_differs_equal_path:nkm_est` | C | 3 | cosmetic | soak/results/anomalies.jsonl (task 127, task 215, task 247) | regime histories differ in the knife-edge period while paths agree to 1e-6 |
| `warning:RuntimeWarning: invalid value encountered in sqrt` | B | 3 | cosmetic | soak/results/anomalies.jsonl (task 167, task 185) | Estimation.laplace takes sqrt of the diagonal of an inverse Hessian that is not positive definite at the mode csminwel reached (estimation.py:1754, outside the OBC code; the result is NaN, handled) |
| `discontinuity_same_regimes:bf` | A | 2 | false positive | soak/repro/bf_line_jump.py (task 404, task 404) | a steep but smooth slope far from the mode (bf, -1283 -> -840 within 0.05 of the line parameter, 11 sub-steps all smooth, same regimes). Detector changed to compare with neighbouring steps |
| `lp_disagree_large:nkms:regime_count_differs` | B | 2 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 103, task 130) | same knife edge as for nkm (3-parameter model); also seen at news MCMC draws (replay under occbin, part B) and at a news posterior mode (0.23 with equal binding-period count but a shifted sequence) |
| `path_disagree:sw` | C | 2 | probably not a bug (multiplicity) | soak/results/anomalies.jsonl (task 118, task 350) | SW model has several equilibria (tests document it); paths 0.08 apart (0.5%) with different regimes. Not minimised |
| `timeout` | A | 2 | see judgement | soak/results/anomalies.jsonl (task 105, task 1445) | per-evaluation 40-60 s timer fired: the opt-in MILP configuration (A#105: 80 MILP time-outs)  |
| `driver_or_unexpected:TypeError:dict() got multiple values for keyword argument 'part'` | B | 1 | ? | soak/results/anomalies.jsonl (task 155) | not yet looked at |
| `lp_disagree_large:nkm:same_regime_count` | B | 1 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 119) | knife edge of the 'approx' class: one binding period starts/ends one period apart (regime count differs by 1, or same count with shifted sequence). Size 0.15-0.4, i.e. larger than the documented ~0.12. Period 21 of the sample at line t=0.1 (A#6) differs; filtered shock differs by 0.04. |
| `lp_disagree_large:nkms:same_regime_count` | A | 1 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 94) | same knife edge as for nkm (3-parameter model); also seen at news MCMC draws (replay under occbin, part B) and at a news posterior mode (0.23 with equal binding-period count but a shifted sequence) |
| `memory_jump` | B | 1 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 35) | Occbin.rules' memo trie (pydyn/occbin.py:260) is unbounded: at extreme parameter vectors OccBin's filter builds 232,740 rule nodes in one Occbin instance (1.3 GB peak, 10 s, then fails with 304). Seen 6+ times in A, C and D (the events listed under B, C, D with 1.0-1.1 GB are the same process memory after an A event: an artefact of the first version of the detector). The news solver needs 0.5 s at the same vector. Python does not return the memory to the OS: the process stays at ~1.4 GB |
| `memory_jump` | C | 1 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 63) | Occbin.rules' memo trie (pydyn/occbin.py:260) is unbounded: at extreme parameter vectors OccBin's filter builds 232,740 rule nodes in one Occbin instance (1.3 GB peak, 10 s, then fails with 304). Seen 6+ times in A, C and D (the events listed under B, C, D with 1.0-1.1 GB are the same process memory after an A event: an artefact of the first version of the detector). The news solver needs 0.5 s at the same vector. Python does not return the memory to the OS: the process stays at ~1.4 GB |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:bf` | C | 1 | cosmetic | soak/results/anomalies.jsonl (task 403) | OccBin guess-and-verify cycles (error 310) or news reports horizon_too_short/no_solution on one side only; counted per 1000-period segment (counts are segments, not events) |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm_est` | C | 1 | cosmetic | soak/results/anomalies.jsonl (task 127) | OccBin guess-and-verify cycles (error 310) or news reports horizon_too_short/no_solution on one side only; counted per 1000-period segment (counts are segments, not events) |
| `perf_cliff` | A | 1 | ? | soak/results/anomalies.jsonl (task 1894) | not yet looked at |
| `regime_seq_differs_equal_lp` | A | 1 | cosmetic | soak/repro/regime_seq_differs_equal_lp.py (task 1342) | equal log posterior to 1e-15 relative (lp = -8.5e14 at an absurd vector); one period flips: knife edge |
| `timeout` | B | 1 | see judgement | soak/results/anomalies.jsonl (task 203) | per-evaluation 40-60 s timer fired: the opt-in MILP configuration (A#105: 80 MILP time-outs)  |

## Failure rates, part A (inversion filter, log posterior evaluations)

Distance bin = rms of the relative deviation from the model's initial values: a <0.02, b <0.1, c <0.3, d <1, e >=1; `line` = the 25 points of the straight-line sweeps (-0.1..1.1).

| model | config | n | finite | failed (by code) | by distance bin: finite share |
|---|---|---|---|---|---|
| bf | news | 4566 | 0.819 | c304 546, c41 44, c42 237 | a:1.00 (30) b:0.91 (33) c:0.82 (160) d:0.54 (143) e:0.44 (125) line:0.84 (4075) |
| bf | news-milp | 20 | 0.300 | c0 1, c304 1, c41 2, c42 10 | a:1.00 (2) b:1.00 (1) c:0.50 (2) d:0.17 (6) e:0.11 (9) |
| bf | news20 | 98 | 0.643 | c304 10, c41 7, c42 18 | a:1.00 (2) b:1.00 (6) c:0.82 (28) d:0.57 (37) e:0.44 (25) |
| bf | news60 | 98 | 0.643 | c304 10, c41 7, c42 18 | a:1.00 (2) b:1.00 (6) c:0.82 (28) d:0.57 (37) e:0.44 (25) |
| bf | occbin | 4566 | 0.819 | c304 544, c41 44, c42 237 | a:1.00 (30) b:0.91 (33) c:0.82 (160) d:0.55 (143) e:0.44 (125) line:0.84 (4075) |
| bf | occbin60 | 32 | 0.844 | c304 5 | b:1.00 (1) c:0.82 (22) d:0.89 (9) |
| nkm | minnorm | 164 | 0.957 | c304 4, c41 3 | c:0.99 (76) d:0.92 (77) e:1.00 (11) |
| nkm | news | 4591 | 0.859 | c3 2, c304 98, c4 24, c41 39, c42 481, c5 5 | a:1.00 (4) b:0.75 (4) c:0.88 (170) d:0.60 (181) e:0.35 (132) line:0.89 (4100) |
| nkm | news-longest | 164 | 0.640 | c304 56, c41 3 | c:0.66 (76) d:0.62 (77) e:0.64 (11) |
| nkm | news-milp | 19 | 0.579 | c0 1, c304 1, c4 2, c41 2, c42 1, c5 1 | c:0.75 (4) d:0.38 (8) e:0.71 (7) |
| nkm | news-shortest | 164 | 0.957 | c304 4, c41 3 | c:0.99 (76) d:0.92 (77) e:1.00 (11) |
| nkm | news20 | 98 | 0.592 | c304 6, c4 3, c41 8, c42 21, c5 2 | b:0.00 (1) c:0.85 (34) d:0.51 (39) e:0.38 (24) |
| nkm | news60 | 98 | 0.592 | c304 6, c4 3, c41 8, c42 21, c5 2 | b:0.00 (1) c:0.85 (34) d:0.51 (39) e:0.38 (24) |
| nkm | occbin | 4591 | 0.858 | c3 2, c304 103, c4 24, c41 39, c42 481, c5 5 | a:1.00 (4) b:0.75 (4) c:0.88 (170) d:0.61 (181) e:0.33 (132) line:0.88 (4100) |
| nkm | occbin60 | 33 | 0.939 | c304 1, c41 1 | c:0.93 (15) d:0.94 (16) e:1.00 (2) |
| nkms | minnorm | 164 | 0.780 | c4 23, c42 13 | a:1.00 (8) b:1.00 (10) c:0.86 (7) d:0.67 (15) e:0.76 (124) |
| nkms | news | 4591 | 0.947 | c304 22, c306 1, c4 135, c41 46, c42 31, c5 7 | a:1.00 (8) b:1.00 (29) c:0.95 (103) d:0.64 (110) e:0.59 (241) line:0.98 (4100) |
| nkms | news-milp | 17 | 1.000 |  | b:1.00 (2) c:1.00 (10) d:1.00 (4) e:1.00 (1) |
| nkms | news20 | 99 | 0.697 | c304 3, c306 1, c4 9, c41 8, c42 9 | a:1.00 (1) b:1.00 (8) c:0.88 (24) d:0.48 (21) e:0.64 (45) |
| nkms | news60 | 99 | 0.697 | c304 3, c306 1, c4 9, c41 8, c42 9 | a:1.00 (1) b:1.00 (8) c:0.88 (24) d:0.48 (21) e:0.64 (45) |
| nkms | occbin | 4591 | 0.947 | c304 23, c306 1, c4 135, c41 46, c42 31, c5 7 | a:1.00 (8) b:1.00 (29) c:0.94 (103) d:0.63 (110) e:0.59 (241) line:0.98 (4100) |
| nkms | occbin60 | 66 | 0.561 | c304 2, c306 1, c4 9, c41 8, c42 9 | a:1.00 (1) b:1.00 (2) c:0.25 (4) d:0.27 (15) e:0.66 (44) |

occbin vs news, both finite: |difference in log posterior|

| model | pairs | <=1e-9 (rel) | 1e-9..0.15 | >0.15 | max |
|---|---|---|---|---|---|
| bf | 3737 | 3724 | 9 | 4 | 4 |
| nkm | 3934 | 3794 | 100 | 40 | 1.69e+06 |
| nkms | 4345 | 4276 | 64 | 5 | 2.12 |

## Timing distributions

Part A, seconds per finite log-posterior evaluation:

| model | config | n | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 3739 | 0.0303 | 0.188 | 0.4317 |
| bf | news-milp | 6 | 0.02875 | 0.04344 | 0.0441 |
| bf | news20 | 63 | 0.0305 | 0.05702 | 0.06 |
| bf | news60 | 63 | 0.0315 | 0.0544 | 0.0544 |
| bf | occbin | 3741 | 0.0843 | 0.1646 | 0.2645 |
| bf | occbin60 | 27 | 0.0829 | 0.1187 | 0.1198 |
| nkm | minnorm | 157 | 0.0361 | 0.06378 | 0.1042 |
| nkm | news | 3942 | 0.0357 | 0.07566 | 0.2644 |
| nkm | news-longest | 105 | 0.2736 | 0.3922 | 0.4262 |
| nkm | news-milp | 11 | 0.0389 | 13.28 | 13.3 |
| nkm | news-shortest | 157 | 0.286 | 0.4933 | 0.5325 |
| nkm | news20 | 58 | 0.0368 | 0.1776 | 0.2012 |
| nkm | news60 | 58 | 0.037 | 0.1921 | 0.2962 |
| nkm | occbin | 3937 | 0.1347 | 0.2525 | 1.108 |
| nkm | occbin60 | 31 | 0.1383 | 0.2191 | 0.2374 |
| nkms | minnorm | 128 | 0.0331 | 0.0633 | 0.077 |
| nkms | news | 4349 | 0.0334 | 0.06086 | 0.6264 |
| nkms | news-milp | 17 | 0.0355 | 0.06441 | 0.0683 |
| nkms | news20 | 69 | 0.0344 | 0.06708 | 0.0764 |
| nkms | news60 | 69 | 0.035 | 0.06436 | 0.073 |
| nkms | occbin | 4348 | 0.1213 | 0.2139 | 2.547 |
| nkms | occbin60 | 37 | 0.1244 | 1.359 | 1.978 |

Part C, seconds per simulated period (per 1000-period segment, averaged over a task):

| model | method | tasks | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 20 | 3.45e-05 | 5.81e-05 | 6e-05 |
| bf | occbin | 20 | 0.000248 | 0.0003027 | 0.0003033 |
| borrcon | news | 48 | 3e-05 | 7.843e-05 | 8e-05 |
| borrcon | occbin | 48 | 0.0003217 | 0.0005447 | 0.0005667 |
| borrcon_d | news | 40 | 2.667e-05 | 9.87e-05 | 0.0001 |
| borrcon_d | occbin | 40 | 0.000311 | 0.0007812 | 0.00088 |
| nkm | news | 33 | 2.333e-05 | 4.787e-05 | 5e-05 |
| nkm | occbin | 33 | 0.00029 | 0.0004755 | 0.0005 |
| nkm_est | news | 37 | 2.2e-05 | 5.64e-05 | 6e-05 |
| nkm_est | occbin | 37 | 0.00028 | 0.0004761 | 0.0005133 |
| sw | news | 15 | 4e-05 | 9.822e-05 | 0.0001067 |
| sw | occbin | 15 | 0.00027 | 0.0004675 | 0.0004833 |
| zlb | news | 55 | 3.7e-05 | 6.46e-05 | 7e-05 |
| zlb | occbin | 55 | 0.000341 | 0.0005092 | 0.00052 |
| zlb2 | news | 56 | 5.183e-05 | 0.0001207 | 0.0001459 |
| zlb2 | occbin | 56 | 0.0004657 | 0.0009382 | 0.0009848 |

Part D, seconds per period of `HigherOrderNews.simulate` (successful runs):

| model | config | runs | median | p99 | max |
|---|---|---|---|---|---|
| borrcon | cub_all11 | 52 | 0.006833 | 0.03898 | 0.04 |
| borrcon | cub_all7_tol3 | 154 | 0.07883 | 0.288 | 0.3025 |
| borrcon | cub_all9_tol05 | 80 | 0.003625 | 0.1641 | 0.1775 |
| borrcon | cub_r1_l9 | 198 | 0.0005 | 0.001 | 0.001 |
| borrcon | cub_r1_l9_tol1 | 194 | 0.0005 | 0.0008333 | 0.0008333 |
| borrcon | cub_r4_l9 | 151 | 0.005 | 0.007917 | 0.00925 |
| borrcon | order1 | 188 | 0 | 2.167e-05 | 0.00025 |
| borrcon | order2 | 196 | 0 | 0.0001667 | 0.00025 |
| nk | cub_all11 | 247 | 0.0165 | 0.03092 | 0.0325 |
| nk | cub_all7_tol3 | 256 | 0.0025 | 0.01661 | 0.1197 |
| nk | cub_all9_tol05 | 256 | 0.005583 | 0.03065 | 0.07 |
| nk | cub_r1_l9 | 259 | 0.0005 | 0.0008333 | 0.001 |
| nk | cub_r1_l9_tol1 | 259 | 0.0005 | 0.0009033 | 0.001 |
| nk | cub_r4_l9 | 254 | 0.0035 | 0.005912 | 0.00675 |
| nk | order1 | 260 | 0 | 0 | 0 |
| nk | order2 | 258 | 0 | 0 | 0.00025 |
| nkm | cub_all11 | 130 | 0.0035 | 0.01547 | 0.02583 |
| nkm | cub_all7_tol3 | 129 | 0.001 | 0.4302 | 0.7442 |
| nkm | cub_all9_tol05 | 138 | 0.0015 | 0.005 | 0.005833 |
| nkm | cub_r1_l9 | 150 | 0.0001667 | 0.0008333 | 0.0008333 |
| nkm | cub_r1_l9_tol1 | 153 | 0.0001667 | 0.0008333 | 0.0008333 |
| nkm | cub_r4_l9 | 138 | 0.002 | 0.004 | 0.004167 |
| nkm | order1 | 138 | 0 | 0 | 0 |
| nkm | order2 | 128 | 0 | 0.0001217 | 0.0001667 |
| nkm_est | cub_all11 | 137 | 0.0035 | 0.02188 | 0.0325 |
| nkm_est | cub_all7_tol3 | 148 | 0.0008333 | 0.002 | 0.0025 |
| nkm_est | cub_all9_tol05 | 133 | 0.001667 | 0.005567 | 0.005833 |
| nkm_est | cub_r1_l9 | 134 | 0.0001667 | 0.0007783 | 0.0008333 |
| nkm_est | cub_r1_l9_tol1 | 138 | 0.0001667 | 0.00071 | 0.0008333 |
| nkm_est | cub_r4_l9 | 130 | 0.002 | 0.003807 | 0.004167 |
| nkm_est | order1 | 128 | 0 | 0 | 0 |
| nkm_est | order2 | 135 | 0 | 0 | 0 |
| sw_cp | cub_all7_tol3 | 4 | 0.01825 | 0.04953 | 0.05 |
| sw_cp | cub_all9_tol05 | 4 | 0.004583 | 0.01632 | 0.01667 |
| sw_cp | cub_r1_l9 | 151 | 0.0005 | 0.0009167 | 0.001 |
| sw_cp | cub_r1_l9_tol1 | 166 | 0.0005 | 0.0008333 | 0.001 |
| sw_cp | cub_r4_l9 | 2 | 0.002333 | 0.00315 | 0.003167 |
| sw_cp | order1 | 190 | 0 | 0 | 0.001 |
| sw_cp | order2 | 190 | 0 | 0 | 0 |
| zlb | cub_all11 | 186 | 0.01917 | 0.03105 | 0.035 |
| zlb | cub_all7_tol3 | 259 | 0.0025 | 0.00521 | 0.074 |
| zlb | cub_all9_tol05 | 255 | 0.006667 | 0.03851 | 0.05417 |
| zlb | cub_r1_l9 | 260 | 0.0005 | 0.0008333 | 0.001 |
| zlb | cub_r1_l9_tol1 | 259 | 0.0005 | 0.0009033 | 0.001 |
| zlb | cub_r4_l9 | 259 | 0.0035 | 0.005833 | 0.0065 |
| zlb | order1 | 258 | 0 | 0 | 0.0001667 |
| zlb | order2 | 262 | 0 | 6.5e-05 | 0.0001667 |
| zlb2 | cub_all11 | 183 | 0.07375 | 0.1105 | 0.1333 |
| zlb2 | cub_all7_tol3 | 256 | 0.007 | 0.02381 | 0.0915 |
| zlb2 | cub_all9_tol05 | 250 | 0.0205 | 0.07933 | 0.1325 |
| zlb2 | cub_r1_l9 | 257 | 0.00075 | 0.001073 | 0.003333 |
| zlb2 | cub_r1_l9_tol1 | 253 | 0.00075 | 0.001207 | 0.0015 |
| zlb2 | cub_r4_l9 | 259 | 0.01 | 0.015 | 0.01667 |
| zlb2 | order1 | 256 | 0 | 0.0001667 | 0.0001667 |
| zlb2 | order2 | 261 | 0 | 0.0001667 | 0.00025 |

## Failures, parts C and D (expected error types)

Part C: per model, segments of 1000 periods that failed, by method and error (count of segments):


Part D: runs per model and configuration (ok / error types):

| model | config | outcomes |
|---|---|---|
| borrcon | cub_all11 | expected:CubatureFailure:horizon_too_short 132, expected:OBCClassError: 78, expected:SteadyStateError: 2, ok 52 |
| borrcon | cub_all7_tol3 | expected:CubatureFailure:horizon_too_short 27, expected:OBCClassError: 82, expected:SteadyStateError: 1, ok 154 |
| borrcon | cub_all9_tol05 | expected:CubatureFailure:horizon_too_short 122, expected:OBCClassError: 60, expected:SteadyStateError: 2, ok 80 |
| borrcon | cub_r1_l9 | expected:OBCClassError: 66, ok 198 |
| borrcon | cub_r1_l9_tol1 | expected:CubatureFailure:horizon_too_short 3, expected:OBCClassError: 66, expected:SteadyStateError: 1, ok 194 |
| borrcon | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 41, expected:OBCClassError: 71, expected:SteadyStateError: 1, ok 151 |
| borrcon | order1 | expected:OBCClassError: 76, ok 188 |
| borrcon | order2 | expected:OBCClassError: 66, expected:SteadyStateError: 3, ok 196 |
| nk | cub_all11 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 13, ok 247 |
| nk | cub_all7_tol3 | expected:BlanchardKahnError: 6, expected:CubatureFailure:horizon_too_short 2, ok 256 |
| nk | cub_all9_tol05 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 3, ok 256 |
| nk | cub_r1_l9 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 2, ok 259 |
| nk | cub_r1_l9_tol1 | expected:BlanchardKahnError: 5, ok 259 |
| nk | cub_r4_l9 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 5, ok 254 |
| nk | order1 | expected:BlanchardKahnError: 4, ok 260 |
| nk | order2 | expected:BlanchardKahnError: 6, expected:OBCError:no_solution 1, ok 258 |
| nkm | cub_all11 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 1, expected:CubatureFailure:no_solution 3, expected:CubatureFailure:not_converged 1, expected:OBCClassError: 80, expected:SteadyStateError: 44, ok 130 |
| nkm | cub_all7_tol3 | expected:BlanchardKahnError: 3, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 82, expected:SteadyStateError: 49, ok 129 |
| nkm | cub_all9_tol05 | expected:BlanchardKahnError: 3, expected:CubatureFailure:no_solution 5, expected:OBCClassError: 73, expected:SteadyStateError: 45, ok 138 |
| nkm | cub_r1_l9 | expected:BlanchardKahnError: 2, expected:OBCClassError: 74, expected:SteadyStateError: 38, ok 150 |
| nkm | cub_r1_l9_tol1 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 57, expected:SteadyStateError: 50, ok 153 |
| nkm | cub_r4_l9 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 1, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 70, expected:SteadyStateError: 50, ok 138 |
| nkm | order1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 73, expected:SteadyStateError: 52, ok 138 |
| nkm | order2 | expected:BlanchardKahnError: 5, expected:OBCClassError: 80, expected:OBCError:no_solution 1, expected:SteadyStateError: 51, ok 128 |
| nkm_est | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 3, expected:OBCClassError: 71, expected:SteadyStateError: 53, ok 137 |
| nkm_est | cub_all7_tol3 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 66, expected:SteadyStateError: 46, ok 148 |
| nkm_est | cub_all9_tol05 | expected:BlanchardKahnError: 4, expected:CubatureFailure:no_solution 3, expected:OBCClassError: 77, expected:SteadyStateError: 47, ok 133 |
| nkm_est | cub_r1_l9 | expected:BlanchardKahnError: 2, expected:OBCClassError: 81, expected:SteadyStateError: 47, ok 134 |
| nkm_est | cub_r1_l9_tol1 | expected:BlanchardKahnError: 4, expected:OBCClassError: 73, expected:SteadyStateError: 49, ok 138 |
| nkm_est | cub_r4_l9 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 90, expected:SteadyStateError: 41, ok 130 |
| nkm_est | order1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 86, expected:SteadyStateError: 49, ok 128 |
| nkm_est | order2 | expected:BlanchardKahnError: 3, expected:OBCClassError: 79, expected:SteadyStateError: 48, ok 135 |
| sw_cp | cub_all7_tol3 | expected:BlanchardKahnError: 10, expected:CubatureFailure:no_solution 212, expected:OBCClassError: 38, ok 4 |
| sw_cp | cub_all9_tol05 | expected:BlanchardKahnError: 19, expected:CubatureFailure:no_solution 433, expected:OBCClassError: 72, ok 4 |
| sw_cp | cub_r1_l9 | expected:BlanchardKahnError: 11, expected:CubatureFailure:no_solution 56, expected:OBCClassError: 46, ok 151 |
| sw_cp | cub_r1_l9_tol1 | expected:BlanchardKahnError: 8, expected:CubatureFailure:no_solution 48, expected:OBCClassError: 42, ok 166 |
| sw_cp | cub_r4_l9 | expected:BlanchardKahnError: 8, expected:CubatureFailure:no_solution 221, expected:OBCClassError: 33, ok 2 |
| sw_cp | order1 | expected:BlanchardKahnError: 4, expected:OBCClassError: 33, expected:OBCError:no_solution 37, ok 190 |
| sw_cp | order2 | expected:BlanchardKahnError: 4, expected:OBCClassError: 31, expected:OBCError:no_solution 40, ok 190 |
| zlb | cub_all11 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 73, ok 186 |
| zlb | cub_all7_tol3 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 2, ok 259 |
| zlb | cub_all9_tol05 | expected:BlanchardKahnError: 2, expected:CubatureFailure:horizon_too_short 7, ok 255 |
| zlb | cub_r1_l9 | expected:BlanchardKahnError: 4, ok 260 |
| zlb | cub_r1_l9_tol1 | expected:BlanchardKahnError: 5, ok 259 |
| zlb | cub_r4_l9 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 1, ok 259 |
| zlb | order1 | expected:BlanchardKahnError: 5, expected:OBCError:no_solution 1, ok 258 |
| zlb | order2 | expected:BlanchardKahnError: 3, ok 262 |
| zlb2 | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 81, ok 183 |
| zlb2 | cub_all7_tol3 | expected:BlanchardKahnError: 7, expected:CubatureFailure:horizon_too_short 1, ok 256 |
| zlb2 | cub_all9_tol05 | expected:BlanchardKahnError: 6, expected:CubatureFailure:horizon_too_short 8, ok 250 |
| zlb2 | cub_r1_l9 | expected:BlanchardKahnError: 6, expected:CubatureFailure:horizon_too_short 1, ok 257 |
| zlb2 | cub_r1_l9_tol1 | expected:BlanchardKahnError: 11, ok 253 |
| zlb2 | cub_r4_l9 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 4, ok 259 |
| zlb2 | order1 | expected:BlanchardKahnError: 5, expected:OBCError:no_solution 3, ok 256 |
| zlb2 | order2 | expected:BlanchardKahnError: 4, ok 261 |

## Warnings (distinct messages, numbers normalised, counts over all parts)

- 412 x `UserWarning: M is not an S-matrix: some paths can have no solution; cubature points may fail (see Highe`
- 310 x `UserWarning: cubature_failure_tolerance=#: # of # relevant perturbed points (#%) were dropped in # of #`
- 3 x `RuntimeWarning: invalid value encountered in sqrt`

## Memory (resident MB at the end of each task; one process per ~8 minute chunk, so growth is only meaningful within a chunk)

- A: median 142, max 1446
- B: median 247, max 1409
- C: median 259, max 1412
- D: median 395, max 1415
