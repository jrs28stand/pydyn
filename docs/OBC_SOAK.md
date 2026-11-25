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
| A | 0.72 | 1849 | evaluations of the log posterior (all configurations): bf 8857, nkm 9327, nkms 9042; tasks by kind: bf/bound 154, bf/line 154, bf/perturb 154, bf/prior 154, nkm/bound 154, nkm/line 154, nkm/perturb 154, nkm/prior 155, nkms/bound 154, nkms/line 154, nkms/perturb 154, nkms/prior 154 |
| B | 0.87 | 343 | 42900 MCMC draws (news) in 7 finished chains + the running one (chain 7, draw 900, replays under occbin of this chain: 90); 56 mode searches (8 skipped: start not finite, 1 timeouts) |
| C | 0.79 | 439 | simulated periods per model (both methods each): bf 398000 (55 runs, 16041 binding), borrcon 587000 (55 runs, 129449 binding), borrcon_d 611000 (55 runs, 88677 binding), nkm 593000 (55 runs, 70184 binding), nkm_est 473000 (54 runs, 46099 binding), sw 312000 (55 runs, 1154 binding), zlb 562000 (55 runs, 201766 binding), zlb2 435000 (55 runs, 170268 binding) |
| D | 0.70 | 14214 | runs per model (ok/total): borrcon 1166/2031, nk 1967/2031, nkm 1057/2030, nkm_est 1044/2031, sw_cp 681/2030, zlb 1919/2030, zlb2 1900/2031 |

LCP solutions checked with `verify_lcp` (tolerance 1e-6) at the `solve_active_set` / `solve_news` boundary, all parts: {'verify|solve_active_set': 543437, 'verify|solve_news': 52}.

Determinism checks (same call repeated): A same-process 398 (+90 in a fresh estimation object), C 69, D 2638.

## Anomaly catalogue

| kind | part | count | severity | first reproducer(s) | judgement |
|---|---|---|---|---|---|
| `nonbitwise_ulp:higher_order` | D | 647 | cosmetic | soak/results/anomalies.jsonl (task 0, task 36, task 48) | two constructions of HigherOrderNews on identical inputs in one process give results differing at 1e-16..1e-18, not bitwise (rounding order, probably BLAS alignment / threading-free kernels); all differences < 1e-9. No nondeterminism of the order-1 or the filter calls (see determinism counts) |
| `path_disagree:nkm` | C | 117 | cosmetic (known cause) | soak/results/anomalies.jsonl (task 66, task 66, task 82) | the 'approx' constraint class: paths differ by 1e-5 relative in a long run once a knife-edge period flips (shock scale >= 2 x sd, 1000-period segments); regimes differ in that period |
| `path_disagree:nkm_est` | C | 87 | cosmetic (known cause) | soak/results/anomalies.jsonl (task 127, task 127, task 127) | same as nkm (occest_nkm.mod) |
| `variant_fails_where_news_ok:news-longest` | A | 51 | cosmetic | soak/results/anomalies.jsonl (task 12, task 24, task 96) | obc_selection='longest' (and 'shortest') can fail the filter (304) where the default selection succeeds: a different (longer) spell makes the root search fail; not a bug as such |
| `lp_disagree_large:nkm:regime_count_differs` | A | 37 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 6, task 6, task 162) | knife edge of the 'approx' class: one binding period starts/ends one period apart (regime count differs by 1, or same count with shifted sequence). Size 0.15-0.4, i.e. larger than the documented ~0.12. Period 21 of the sample at line t=0.1 (A#6) differs; filtered shock differs by 0.04. |
| `slow_failure` | C | 37 | perf | soak/repro/occbin_rule_trie_memory.py (task 14, task 43, task 43) | evaluations/simulations that fail after > 5 s (A) or > 10 s (C): OccBin inversion-filter failures with huge regime-guess tries (see memory_jump), the opt-in MILP configuration, and RegimeOBC's 30 s integer-programme limit (sw model: 30 s for one failing period) |
| `finite_vs_inf:news_finite_occbin_code304` | A | 14 | wrong-result (occbin refuses a point news evaluates) | soak/results/anomalies.jsonl (task 30, task 193, task 285) | OccBin's failure codes (304 via 310/311/313) at points where the news filter finds a solution; the counterpart of the previous kind; expected at cycling/periodic guess-and-verify failures |
| `path_disagree:bf` | C | 13 | probably not a bug (multiplicity) | soak/repro/bf_path_disagree.py (task 43, task 83, task 83) | bf (two equations switch): OccBin binds for one period with rnot - rlb = -2e-6, news does not bind (rnot - rlb = +2e-5): each satisfies its own regime condition; the paths differ by 0.006 although the two sit at the boundary: a jump across the constraint, i.e. two solutions |
| `memory_jump` | A | 11 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 141, task 273, task 309) | Occbin.rules' memo trie (pydyn/occbin.py:260) is unbounded: at extreme parameter vectors OccBin's filter builds 232,740 rule nodes in one Occbin instance (1.3 GB peak, 10 s, then fails with 304). Seen 6+ times in A, C and D (the events listed under B, C, D with 1.0-1.1 GB are the same process memory after an A event: an artefact of the first version of the detector). The news solver needs 0.5 s at the same vector. Python does not return the memory to the OS: the process stays at ~1.4 GB |
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
| `regime_seq_differs_equal_lp` | A | 1 | cosmetic | soak/repro/regime_seq_differs_equal_lp.py (task 1342) | equal log posterior to 1e-15 relative (lp = -8.5e14 at an absurd vector); one period flips: knife edge |
| `timeout` | B | 1 | see judgement | soak/results/anomalies.jsonl (task 203) | per-evaluation 40-60 s timer fired: the opt-in MILP configuration (A#105: 80 MILP time-outs)  |

## Failure rates, part A (inversion filter, log posterior evaluations)

Distance bin = rms of the relative deviation from the model's initial values: a <0.02, b <0.1, c <0.3, d <1, e >=1; `line` = the 25 points of the straight-line sweeps (-0.1..1.1).

| model | config | n | finite | failed (by code) | by distance bin: finite share |
|---|---|---|---|---|---|
| bf | news | 4312 | 0.818 | c304 511, c41 43, c42 230 | a:1.00 (28) b:0.90 (30) c:0.82 (152) d:0.52 (133) e:0.46 (119) line:0.84 (3850) |
| bf | news-milp | 19 | 0.316 | c0 1, c304 1, c41 2, c42 9 | a:1.00 (2) b:1.00 (1) c:0.50 (2) d:0.17 (6) e:0.12 (8) |
| bf | news20 | 92 | 0.652 | c304 9, c41 7, c42 16 | a:1.00 (2) b:1.00 (5) c:0.81 (26) d:0.58 (36) e:0.48 (23) |
| bf | news60 | 92 | 0.652 | c304 9, c41 7, c42 16 | a:1.00 (2) b:1.00 (5) c:0.81 (26) d:0.58 (36) e:0.48 (23) |
| bf | occbin | 4312 | 0.819 | c304 509, c41 43, c42 230 | a:1.00 (28) b:0.90 (30) c:0.82 (152) d:0.53 (133) e:0.46 (119) line:0.84 (3850) |
| bf | occbin60 | 30 | 0.833 | c304 5 | c:0.81 (21) d:0.89 (9) |
| nkm | minnorm | 155 | 0.961 | c304 4, c41 2 | c:0.99 (72) d:0.93 (72) e:1.00 (11) |
| nkm | news | 4313 | 0.859 | c3 2, c304 88, c4 22, c41 38, c42 452, c5 5 | a:1.00 (4) b:0.75 (4) c:0.88 (160) d:0.61 (168) e:0.36 (127) line:0.89 (3850) |
| nkm | news-longest | 155 | 0.632 | c304 55, c41 2 | c:0.64 (72) d:0.62 (72) e:0.64 (11) |
| nkm | news-milp | 19 | 0.579 | c0 1, c304 1, c4 2, c41 2, c42 1, c5 1 | c:0.75 (4) d:0.38 (8) e:0.71 (7) |
| nkm | news-shortest | 155 | 0.961 | c304 4, c41 2 | c:0.99 (72) d:0.93 (72) e:1.00 (11) |
| nkm | news20 | 93 | 0.591 | c304 6, c4 3, c41 7, c42 20, c5 2 | b:0.00 (1) c:0.84 (32) d:0.51 (37) e:0.39 (23) |
| nkm | news60 | 93 | 0.591 | c304 6, c4 3, c41 7, c42 20, c5 2 | b:0.00 (1) c:0.84 (32) d:0.51 (37) e:0.39 (23) |
| nkm | occbin | 4313 | 0.858 | c3 2, c304 93, c4 22, c41 38, c42 452, c5 5 | a:1.00 (4) b:0.75 (4) c:0.88 (160) d:0.61 (168) e:0.34 (127) line:0.89 (3850) |
| nkm | occbin60 | 31 | 0.968 | c304 1 | c:0.93 (14) d:1.00 (15) e:1.00 (2) |
| nkms | minnorm | 154 | 0.786 | c4 21, c42 12 | a:1.00 (8) b:1.00 (9) c:0.86 (7) d:0.71 (14) e:0.76 (116) |
| nkms | news | 4312 | 0.947 | c304 22, c306 1, c4 128, c41 43, c42 28, c5 6 | a:1.00 (8) b:1.00 (26) c:0.96 (95) d:0.66 (105) e:0.58 (228) line:0.98 (3850) |
| nkms | news-milp | 16 | 1.000 |  | b:1.00 (2) c:1.00 (9) d:1.00 (4) e:1.00 (1) |
| nkms | news20 | 93 | 0.710 | c304 3, c306 1, c4 7, c41 8, c42 8 | a:1.00 (1) b:1.00 (7) c:0.91 (22) d:0.53 (19) e:0.64 (44) |
| nkms | news60 | 93 | 0.710 | c304 3, c306 1, c4 7, c41 8, c42 8 | a:1.00 (1) b:1.00 (7) c:0.91 (22) d:0.53 (19) e:0.64 (44) |
| nkms | occbin | 4312 | 0.947 | c304 23, c306 1, c4 128, c41 43, c42 28, c5 6 | a:1.00 (8) b:1.00 (26) c:0.95 (95) d:0.65 (105) e:0.58 (228) line:0.98 (3850) |
| nkms | occbin60 | 62 | 0.581 | c304 2, c306 1, c4 7, c41 8, c42 8 | a:1.00 (1) b:1.00 (2) c:0.33 (3) d:0.31 (13) e:0.65 (43) |

occbin vs news, both finite: |difference in log posterior|

| model | pairs | <=1e-9 (rel) | 1e-9..0.15 | >0.15 | max |
|---|---|---|---|---|---|
| bf | 3526 | 3514 | 8 | 4 | 4 |
| nkm | 3698 | 3566 | 92 | 40 | 1.69e+06 |
| nkms | 4080 | 4014 | 61 | 5 | 2.12 |

## Timing distributions

Part A, seconds per finite log-posterior evaluation:

| model | config | n | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 3528 | 0.0304 | 0.1919 | 0.4317 |
| bf | news-milp | 6 | 0.02875 | 0.04344 | 0.0441 |
| bf | news20 | 60 | 0.0305 | 0.05717 | 0.06 |
| bf | news60 | 60 | 0.0316 | 0.0544 | 0.0544 |
| bf | occbin | 3530 | 0.0849 | 0.1663 | 0.2645 |
| bf | occbin60 | 25 | 0.0837 | 0.1188 | 0.1198 |
| nkm | minnorm | 149 | 0.0362 | 0.06434 | 0.1042 |
| nkm | news | 3706 | 0.0357 | 0.07617 | 0.2644 |
| nkm | news-longest | 98 | 0.2734 | 0.3938 | 0.4262 |
| nkm | news-milp | 11 | 0.0389 | 13.28 | 13.3 |
| nkm | news-shortest | 149 | 0.2868 | 0.4946 | 0.5325 |
| nkm | news20 | 55 | 0.0369 | 0.1788 | 0.2012 |
| nkm | news60 | 55 | 0.0371 | 0.1976 | 0.2962 |
| nkm | occbin | 3701 | 0.1355 | 0.2551 | 1.108 |
| nkm | occbin60 | 30 | 0.1386 | 0.2197 | 0.2374 |
| nkms | minnorm | 121 | 0.0331 | 0.06408 | 0.077 |
| nkms | news | 4084 | 0.0334 | 0.06103 | 0.2877 |
| nkms | news-milp | 16 | 0.03535 | 0.06465 | 0.0683 |
| nkms | news20 | 66 | 0.0346 | 0.06749 | 0.0764 |
| nkms | news60 | 66 | 0.03535 | 0.06474 | 0.073 |
| nkms | occbin | 4083 | 0.1221 | 0.2146 | 2.019 |
| nkms | occbin60 | 36 | 0.125 | 1.376 | 1.978 |

Part C, seconds per simulated period (per 1000-period segment, averaged over a task):

| model | method | tasks | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 19 | 3.4e-05 | 5.82e-05 | 6e-05 |
| bf | occbin | 19 | 0.00025 | 0.0003027 | 0.0003033 |
| borrcon | news | 46 | 3e-05 | 7.85e-05 | 8e-05 |
| borrcon | occbin | 46 | 0.0003217 | 0.0005457 | 0.0005667 |
| borrcon_d | news | 40 | 2.667e-05 | 9.87e-05 | 0.0001 |
| borrcon_d | occbin | 40 | 0.000311 | 0.0007812 | 0.00088 |
| nkm | news | 33 | 2.333e-05 | 4.787e-05 | 5e-05 |
| nkm | occbin | 33 | 0.00029 | 0.0004755 | 0.0005 |
| nkm_est | news | 36 | 2.1e-05 | 5.65e-05 | 6e-05 |
| nkm_est | occbin | 36 | 0.00028 | 0.0004772 | 0.0005133 |
| sw | news | 15 | 4e-05 | 9.822e-05 | 0.0001067 |
| sw | occbin | 15 | 0.00027 | 0.0004675 | 0.0004833 |
| zlb | news | 53 | 3.667e-05 | 6.48e-05 | 7e-05 |
| zlb | occbin | 53 | 0.0003467 | 0.0005096 | 0.00052 |
| zlb2 | news | 54 | 5.133e-05 | 0.0001216 | 0.0001459 |
| zlb2 | occbin | 54 | 0.0004657 | 0.0009399 | 0.0009848 |

Part D, seconds per period of `HigherOrderNews.simulate` (successful runs):

| model | config | runs | median | p99 | max |
|---|---|---|---|---|---|
| borrcon | cub_all11 | 49 | 0.006667 | 0.03904 | 0.04 |
| borrcon | cub_all7_tol3 | 147 | 0.07675 | 0.2888 | 0.3025 |
| borrcon | cub_all9_tol05 | 76 | 0.0035 | 0.1648 | 0.1775 |
| borrcon | cub_r1_l9 | 194 | 0.0005 | 0.001 | 0.001 |
| borrcon | cub_r1_l9_tol1 | 186 | 0.0005 | 0.0008333 | 0.0008333 |
| borrcon | cub_r4_l9 | 147 | 0.005 | 0.00795 | 0.00925 |
| borrcon | order1 | 181 | 0 | 3.333e-05 | 0.00025 |
| borrcon | order2 | 186 | 0 | 0.0001667 | 0.00025 |
| nk | cub_all11 | 237 | 0.016 | 0.03094 | 0.0325 |
| nk | cub_all7_tol3 | 246 | 0.0025 | 0.007008 | 0.1197 |
| nk | cub_all9_tol05 | 246 | 0.0055 | 0.0227 | 0.0365 |
| nk | cub_r1_l9 | 249 | 0.0005 | 0.0008333 | 0.001 |
| nk | cub_r1_l9_tol1 | 248 | 0.0005 | 0.0009217 | 0.001 |
| nk | cub_r4_l9 | 244 | 0.0035 | 0.005928 | 0.00675 |
| nk | order1 | 250 | 0 | 0 | 0 |
| nk | order2 | 247 | 0 | 0 | 0.00025 |
| nkm | cub_all11 | 123 | 0.0035 | 0.0156 | 0.02583 |
| nkm | cub_all7_tol3 | 125 | 0.001 | 0.4442 | 0.7442 |
| nkm | cub_all9_tol05 | 132 | 0.0015 | 0.005 | 0.005 |
| nkm | cub_r1_l9 | 142 | 0.0001667 | 0.0008333 | 0.0008333 |
| nkm | cub_r1_l9_tol1 | 145 | 0.0001667 | 0.0008333 | 0.0008333 |
| nkm | cub_r4_l9 | 134 | 0.002 | 0.004 | 0.004167 |
| nkm | order1 | 133 | 0 | 0 | 0 |
| nkm | order2 | 123 | 0 | 0.00013 | 0.0001667 |
| nkm_est | cub_all11 | 133 | 0.0035 | 0.02189 | 0.0325 |
| nkm_est | cub_all7_tol3 | 146 | 0.0008333 | 0.002 | 0.0025 |
| nkm_est | cub_all9_tol05 | 125 | 0.001667 | 0.005 | 0.005833 |
| nkm_est | cub_r1_l9 | 130 | 0.0001667 | 0.000785 | 0.0008333 |
| nkm_est | cub_r1_l9_tol1 | 135 | 0.0001667 | 0.00072 | 0.0008333 |
| nkm_est | cub_r4_l9 | 127 | 0.002 | 0.003827 | 0.004167 |
| nkm_est | order1 | 122 | 0 | 0 | 0 |
| nkm_est | order2 | 126 | 0 | 0 | 0 |
| sw_cp | cub_all7_tol3 | 4 | 0.01825 | 0.04953 | 0.05 |
| sw_cp | cub_all9_tol05 | 4 | 0.004583 | 0.01632 | 0.01667 |
| sw_cp | cub_r1_l9 | 146 | 0.0005 | 0.000925 | 0.001 |
| sw_cp | cub_r1_l9_tol1 | 160 | 0.0005 | 0.0008333 | 0.001 |
| sw_cp | cub_r4_l9 | 2 | 0.002333 | 0.00315 | 0.003167 |
| sw_cp | order1 | 184 | 0 | 0 | 0.001 |
| sw_cp | order2 | 181 | 0 | 0 | 0 |
| zlb | cub_all11 | 179 | 0.01917 | 0.03131 | 0.035 |
| zlb | cub_all7_tol3 | 249 | 0.0025 | 0.00526 | 0.074 |
| zlb | cub_all9_tol05 | 245 | 0.0065 | 0.03869 | 0.05417 |
| zlb | cub_r1_l9 | 250 | 0.0005 | 0.0008333 | 0.001 |
| zlb | cub_r1_l9_tol1 | 248 | 0.0005 | 0.0009217 | 0.001 |
| zlb | cub_r4_l9 | 249 | 0.0035 | 0.005833 | 0.0065 |
| zlb | order1 | 248 | 0 | 0 | 0.0001667 |
| zlb | order2 | 251 | 0 | 8.333e-05 | 0.0001667 |
| zlb2 | cub_all11 | 176 | 0.07342 | 0.1108 | 0.1333 |
| zlb2 | cub_all7_tol3 | 246 | 0.007 | 0.02394 | 0.0915 |
| zlb2 | cub_all9_tol05 | 241 | 0.0205 | 0.0801 | 0.1325 |
| zlb2 | cub_r1_l9 | 248 | 0.00075 | 0.001088 | 0.003333 |
| zlb2 | cub_r1_l9_tol1 | 242 | 0.00075 | 0.001216 | 0.0015 |
| zlb2 | cub_r4_l9 | 250 | 0.01 | 0.015 | 0.01667 |
| zlb2 | order1 | 247 | 0 | 0.0001667 | 0.0001667 |
| zlb2 | order2 | 250 | 0 | 0.0001667 | 0.00025 |

## Failures, parts C and D (expected error types)

Part C: per model, segments of 1000 periods that failed, by method and error (count of segments):


Part D: runs per model and configuration (ok / error types):

| model | config | outcomes |
|---|---|---|
| borrcon | cub_all11 | expected:CubatureFailure:horizon_too_short 129, expected:OBCClassError: 74, expected:SteadyStateError: 2, ok 49 |
| borrcon | cub_all7_tol3 | expected:CubatureFailure:horizon_too_short 26, expected:OBCClassError: 80, expected:SteadyStateError: 1, ok 147 |
| borrcon | cub_all9_tol05 | expected:CubatureFailure:horizon_too_short 120, expected:OBCClassError: 56, expected:SteadyStateError: 2, ok 76 |
| borrcon | cub_r1_l9 | expected:OBCClassError: 60, ok 194 |
| borrcon | cub_r1_l9_tol1 | expected:CubatureFailure:horizon_too_short 2, expected:OBCClassError: 64, expected:SteadyStateError: 1, ok 186 |
| borrcon | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 39, expected:OBCClassError: 67, expected:SteadyStateError: 1, ok 147 |
| borrcon | order1 | expected:OBCClassError: 73, ok 181 |
| borrcon | order2 | expected:OBCClassError: 65, expected:SteadyStateError: 3, ok 186 |
| nk | cub_all11 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 13, ok 237 |
| nk | cub_all7_tol3 | expected:BlanchardKahnError: 6, expected:CubatureFailure:horizon_too_short 2, ok 246 |
| nk | cub_all9_tol05 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 3, ok 246 |
| nk | cub_r1_l9 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 2, ok 249 |
| nk | cub_r1_l9_tol1 | expected:BlanchardKahnError: 5, ok 248 |
| nk | cub_r4_l9 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 5, ok 244 |
| nk | order1 | expected:BlanchardKahnError: 4, ok 250 |
| nk | order2 | expected:BlanchardKahnError: 6, expected:OBCError:no_solution 1, ok 247 |
| nkm | cub_all11 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 1, expected:CubatureFailure:no_solution 3, expected:CubatureFailure:not_converged 1, expected:OBCClassError: 78, expected:SteadyStateError: 43, ok 123 |
| nkm | cub_all7_tol3 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 79, expected:SteadyStateError: 46, ok 125 |
| nkm | cub_all9_tol05 | expected:BlanchardKahnError: 3, expected:CubatureFailure:no_solution 5, expected:OBCClassError: 69, expected:SteadyStateError: 45, ok 132 |
| nkm | cub_r1_l9 | expected:BlanchardKahnError: 2, expected:OBCClassError: 72, expected:SteadyStateError: 38, ok 142 |
| nkm | cub_r1_l9_tol1 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 56, expected:SteadyStateError: 48, ok 145 |
| nkm | cub_r4_l9 | expected:BlanchardKahnError: 3, expected:CubatureFailure:horizon_too_short 1, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 68, expected:SteadyStateError: 46, ok 134 |
| nkm | order1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 70, expected:SteadyStateError: 50, ok 133 |
| nkm | order2 | expected:BlanchardKahnError: 5, expected:OBCClassError: 76, expected:OBCError:no_solution 1, expected:SteadyStateError: 49, ok 123 |
| nkm_est | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 3, expected:OBCClassError: 66, expected:SteadyStateError: 51, ok 133 |
| nkm_est | cub_all7_tol3 | expected:BlanchardKahnError: 2, expected:CubatureFailure:no_solution 2, expected:OBCClassError: 59, expected:SteadyStateError: 45, ok 146 |
| nkm_est | cub_all9_tol05 | expected:BlanchardKahnError: 4, expected:CubatureFailure:no_solution 3, expected:OBCClassError: 75, expected:SteadyStateError: 47, ok 125 |
| nkm_est | cub_r1_l9 | expected:BlanchardKahnError: 2, expected:OBCClassError: 76, expected:SteadyStateError: 46, ok 130 |
| nkm_est | cub_r1_l9_tol1 | expected:BlanchardKahnError: 4, expected:OBCClassError: 69, expected:SteadyStateError: 45, ok 135 |
| nkm_est | cub_r4_l9 | expected:BlanchardKahnError: 1, expected:CubatureFailure:no_solution 1, expected:OBCClassError: 85, expected:SteadyStateError: 40, ok 127 |
| nkm_est | order1 | expected:BlanchardKahnError: 1, expected:OBCClassError: 84, expected:SteadyStateError: 47, ok 122 |
| nkm_est | order2 | expected:BlanchardKahnError: 3, expected:OBCClassError: 78, expected:SteadyStateError: 47, ok 126 |
| sw_cp | cub_all7_tol3 | expected:BlanchardKahnError: 9, expected:CubatureFailure:no_solution 202, expected:OBCClassError: 38, ok 4 |
| sw_cp | cub_all9_tol05 | expected:BlanchardKahnError: 18, expected:CubatureFailure:no_solution 417, expected:OBCClassError: 69, ok 4 |
| sw_cp | cub_r1_l9 | expected:BlanchardKahnError: 11, expected:CubatureFailure:no_solution 54, expected:OBCClassError: 43, ok 146 |
| sw_cp | cub_r1_l9_tol1 | expected:BlanchardKahnError: 7, expected:CubatureFailure:no_solution 45, expected:OBCClassError: 41, ok 160 |
| sw_cp | cub_r4_l9 | expected:BlanchardKahnError: 8, expected:CubatureFailure:no_solution 212, expected:OBCClassError: 32, ok 2 |
| sw_cp | order1 | expected:BlanchardKahnError: 4, expected:OBCClassError: 29, expected:OBCError:no_solution 37, ok 184 |
| sw_cp | order2 | expected:BlanchardKahnError: 4, expected:OBCClassError: 30, expected:OBCError:no_solution 39, ok 181 |
| zlb | cub_all11 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 71, ok 179 |
| zlb | cub_all7_tol3 | expected:BlanchardKahnError: 2, expected:CubatureFailure:horizon_too_short 2, ok 249 |
| zlb | cub_all9_tol05 | expected:BlanchardKahnError: 2, expected:CubatureFailure:horizon_too_short 7, ok 245 |
| zlb | cub_r1_l9 | expected:BlanchardKahnError: 4, ok 250 |
| zlb | cub_r1_l9_tol1 | expected:BlanchardKahnError: 5, ok 248 |
| zlb | cub_r4_l9 | expected:BlanchardKahnError: 4, expected:CubatureFailure:horizon_too_short 1, ok 249 |
| zlb | order1 | expected:BlanchardKahnError: 5, expected:OBCError:no_solution 1, ok 248 |
| zlb | order2 | expected:BlanchardKahnError: 3, ok 251 |
| zlb2 | cub_all11 | expected:BlanchardKahnError: 1, expected:CubatureFailure:horizon_too_short 77, ok 176 |
| zlb2 | cub_all7_tol3 | expected:BlanchardKahnError: 7, expected:CubatureFailure:horizon_too_short 1, ok 246 |
| zlb2 | cub_all9_tol05 | expected:BlanchardKahnError: 5, expected:CubatureFailure:horizon_too_short 8, ok 241 |
| zlb2 | cub_r1_l9 | expected:BlanchardKahnError: 6, ok 248 |
| zlb2 | cub_r1_l9_tol1 | expected:BlanchardKahnError: 11, ok 242 |
| zlb2 | cub_r4_l9 | expected:CubatureFailure:horizon_too_short 4, ok 250 |
| zlb2 | order1 | expected:BlanchardKahnError: 4, expected:OBCError:no_solution 3, ok 247 |
| zlb2 | order2 | expected:BlanchardKahnError: 4, ok 250 |

## Warnings (distinct messages, numbers normalised, counts over all parts)

- 398 x `UserWarning: M is not an S-matrix: some paths can have no solution; cubature points may fail (see Highe`
- 295 x `UserWarning: cubature_failure_tolerance=#: # of # relevant perturbed points (#%) were dropped in # of #`
- 3 x `RuntimeWarning: invalid value encountered in sqrt`

## Memory (resident MB at the end of each task; one process per ~8 minute chunk, so growth is only meaningful within a chunk)

- A: median 149, max 1446
- B: median 218, max 1409
- C: median 259, max 1412
- D: median 395, max 1415
