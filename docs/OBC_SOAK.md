# OBC soak test (pydyn, branch `obc-soak`, from `wip/obc` 0abe8e7)

Brute-force soak test of the occasionally-binding-constraint estimation and simulation code, to find bugs that appear only over many
parameter vectors or long runs. Library code is not changed; anomalies are catalogued with reproducers (`soak/repro/`).

## Fixes made after the soak (branch obc-soak, commits after 0454d5f)
1. `pydyn/occbin.py`: `Occbin.rule_memo_bytes` (default 384 MiB per Occbin object) bounds the rule memo; beyond it rules are computed as before without being stored (bit for bit the same). Tests: `tests/test_occbin_memo_cap.py`; `tests/test_occbin_bitwise.py` and `bench/occbin_equality.py` unchanged and passing.
2. `pydyn/obc_filter.py`, `occbin_filter.py`, `occbin_estimation.py`: failure causes of news trial points (`solver.causes`, `failure_info()`), `est.last_failure`, extra text in `est.last_info` (code stays 304/313), opt-in `est.obc_horizon_max`. Tests: `tests/test_obc_horizon_cause.py`.
3. docs/OBC_ESTIMATION.md 2d: size of knife-edge differences. 4. `driver_or_unexpected:TypeError ... 'part'` was a driver bug (`record_warnings` passed `part` twice), fixed.

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
| A | 1.11 | 2791 | evaluations of the log posterior (all configurations): bf 13347, nkm 14098, nkms 13626; tasks by kind: bf/bound 232, bf/line 232, bf/perturb 233, bf/prior 233, nkm/bound 232, nkm/line 233, nkm/perturb 233, nkm/prior 233, nkms/bound 232, nkms/line 232, nkms/perturb 233, nkms/prior 233 |
| B | 1.04 | 408 | 51000 MCMC draws (news) in 8 finished chains + the running one (chain 8, draw 3000, replays under occbin of this chain: 300); 67 mode searches (10 skipped: start not finite, 2 timeouts) |
| C | 0.96 | 533 | simulated periods per model (both methods each): bf 490000 (67 runs, 17572 binding), borrcon 734000 (66 runs, 138235 binding), borrcon_d 808000 (67 runs, 114745 binding), nkm 676000 (67 runs, 81963 binding), nkm_est 583000 (66 runs, 57417 binding), sw 329000 (66 runs, 1442 binding), zlb 703000 (67 runs, 249909 binding), zlb2 576000 (67 runs, 232139 binding) |
| D | 0.73 | 14794 | runs per model (ok/total): borrcon 1213/2113, nk 2049/2114, nkm 1104/2113, nkm_est 1083/2114, sw_cp 707/2113, zlb 1998/2113, zlb2 1975/2114 |

LCP solutions checked with `verify_lcp` (tolerance 1e-6) at the `solve_active_set` / `solve_news` boundary, all parts: {'verify|solve_active_set': 567484, 'verify|solve_news': 55}.

Determinism checks (same call repeated): A same-process 598 (+134 in a fresh estimation object), C 83, D 2746.

## Anomaly catalogue

| kind | part | count | severity | first reproducer(s) | judgement |
|---|---|---|---|---|---|
| `nonbitwise_ulp:higher_order` | D | 674 | cosmetic | soak/results/anomalies.jsonl (task 0, task 36, task 48) | two constructions of HigherOrderNews on identical inputs in one process give results differing at 1e-16..1e-18, not bitwise (rounding order, probably BLAS alignment / threading-free kernels); all differences < 1e-9. No nondeterminism of the order-1 or the filter calls (see determinism counts) |
| `path_disagree:nkm` | C | 138 | cosmetic (known cause) | soak/results/anomalies.jsonl (task 66, task 66, task 82) | the 'approx' constraint class: paths differ by 1e-5 relative in a long run once a knife-edge period flips (shock scale >= 2 x sd, 1000-period segments); regimes differ in that period |
| `path_disagree:nkm_est` | C | 110 | cosmetic (known cause) | soak/results/anomalies.jsonl (task 127, task 127, task 127) | same as nkm (occest_nkm.mod) |
| `variant_fails_where_news_ok:news-longest` | A | 76 | cosmetic | soak/results/anomalies.jsonl (task 12, task 24, task 96) | obc_selection='longest' (and 'shortest') can fail the filter (304) where the default selection succeeds: a different (longer) spell makes the root search fail; not a bug as such |
| `lp_disagree_large:nkm:regime_count_differs` | A | 53 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 6, task 6, task 162) | knife edge of the approx class: one regime decision shifts by one period. Full counts from the sweeps: 195 of 8,279 finite NKM pairs differ by > 1e-8, 60 by > 0.12, 9 by > 0.4, max 1.54 (docs/OBC_ESTIMATION.md, 2d). The first logged lines (20 per kind) understated the size. |
| `slow_failure` | C | 45 | perf | soak/repro/occbin_rule_trie_memory.py (task 14, task 43, task 43) | evaluations/simulations that fail after > 5 s (A) or > 10 s (C): OccBin inversion-filter failures with huge regime-guess tries (see memory_jump), the opt-in MILP configuration, and RegimeOBC's 30 s integer-programme limit (sw model: 30 s for one failing period) |
| `finite_vs_inf:news_finite_occbin_code304` | A | 18 | wrong-result (occbin refuses a point news evaluates) | soak/results/anomalies.jsonl (task 30, task 193, task 285) | OccBin's failure codes (304 via 310/311/313) at points where the news filter finds a solution; the counterpart of the previous kind; expected at cycling/periodic guess-and-verify failures |
| `finite_vs_inf:occbin_finite_news_code304` | A | 15 | wrong-result (news refuses a point OccBin evaluates) | soak/repro/finite_vs_inf_news_304.py (task 75, task 155, task 500) | news fails (304) where OccBin has a value: the news solver's trial points are undecided / horizon_too_short at horizon 40. Now reported with a cause (est.last_failure) and recoverable with the opt-in est.obc_horizon_max: 9 of the 15 logged vectors are recovered at obc_horizon_max=160 (equal to OccBin up to 5e-12, knife-edge 0.02/0.2, or 4.6 at lp -1.4e11), 4 still fail with horizon_too_short at 160 (bf 155, 2483, 2531, nkms 1630) and 2 are root-search failures unrelated to the horizon (nkms 874, 1642) |
| `memory_jump` | A | 15 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 141, task 273, task 309) | FIXED in obc-soak (Occbin.rule_memo_bytes, 384 MiB per Occbin object): the memo trie of Occbin.rules was unbounded (232,740 nodes, 1.3 GB at extreme vectors). After the fix the peak is ~550 MB and results are bit for bit unchanged. Remaining jumps of 100-440 MB are the bounded memo filling. (Events listed for B/C/D at 1.0-1.1 GB were artefacts of the first detector.) |
| `path_disagree:bf` | C | 15 | probably not a bug (multiplicity) | soak/repro/bf_path_disagree.py (task 43, task 83, task 83) | bf (two equations switch): OccBin binds for one period with rnot - rlb = -2e-6, news does not bind (rnot - rlb = +2e-5): each satisfies its own regime condition; the paths differ by 0.006 although the two sit at the boundary: a jump across the constraint, i.e. two solutions |
| `regime_history_differs_equal_path:nkm` | C | 10 | cosmetic | soak/results/anomalies.jsonl (task 82, task 82, task 130) | regime histories differ in the knife-edge period while paths agree to 1e-6 |
| `lp_absurd` | A | 9 | cosmetic | soak/results/anomalies.jsonl (task 70, task 70, task 70) | FALSE POSITIVE of an early threshold (|lp|>1e6): legitimate huge negative values from tiny shock standard deviations (-3e10 ... -2e12). Threshold changed to lp>1e5; huge negative values are counted separately |
| `memory_jump` | D | 9 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 1782, task 2697, task 3313) | FIXED in obc-soak (Occbin.rule_memo_bytes, 384 MiB per Occbin object): the memo trie of Occbin.rules was unbounded (232,740 nodes, 1.3 GB at extreme vectors). After the fix the peak is ~550 MB and results are bit for bit unchanged. Remaining jumps of 100-440 MB are the bounded memo filling. (Events listed for B/C/D at 1.0-1.1 GB were artefacts of the first detector.) |
| `slow_failure` | A | 9 | perf | soak/repro/occbin_rule_trie_memory.py (task 65, task 75, task 105) | evaluations/simulations that fail after > 5 s (A) or > 10 s (C): OccBin inversion-filter failures with huge regime-guess tries (see memory_jump), the opt-in MILP configuration, and RegimeOBC's 30 s integer-programme limit (sw model: 30 s for one failing period) |
| `lp_disagree_large:nkms:regime_count_differs` | A | 6 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 79, task 556, task 1156) | same knife edge as for nkm (3-parameter model); also seen at news MCMC draws (replay under occbin, part B) and at a news posterior mode (0.23 with equal binding-period count but a shifted sequence) |
| `milp_timeout_hit` | A | 6 | perf | soak/results/anomalies.jsonl (task 65, task 75, task 105) | only in the opt-in configuration est.obc_milp_time_limit=1.0 (default 0): up to 96 integer programmes per evaluation, each running to its 1 s limit -> one evaluation 45 s (bf) or > 40 s (nkm); 'rescued' stays 0. Documented cost of the option |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:sw` | C | 4 | cosmetic | soak/results/anomalies.jsonl (task 30, task 86, task 430) | OccBin guess-and-verify cycles (error 310) or news reports horizon_too_short/no_solution on one side only; counted per 1000-period segment (counts are segments, not events) |
| `regime_history_differs_equal_path:nkm_est` | C | 4 | cosmetic | soak/results/anomalies.jsonl (task 127, task 215, task 247) | regime histories differ in the knife-edge period while paths agree to 1e-6 |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm` | C | 3 | cosmetic | soak/results/anomalies.jsonl (task 82, task 354, task 370) | OccBin guess-and-verify cycles (error 310) or news reports horizon_too_short/no_solution on one side only; counted per 1000-period segment (counts are segments, not events) |
| `path_disagree:sw` | C | 3 | probably not a bug (multiplicity) | soak/results/anomalies.jsonl (task 118, task 350, task 478) | SW model has several equilibria (tests document it); paths 0.08 apart (0.5%) with different regimes. Not minimised |
| `warning:RuntimeWarning: invalid value encountered in sqrt` | B | 3 | cosmetic | soak/results/anomalies.jsonl (task 167, task 185) | Estimation.laplace takes sqrt of the diagonal of an inverse Hessian that is not positive definite at the mode csminwel reached (estimation.py:1754, outside the OBC code; the result is NaN, handled) |
| `discontinuity_same_regimes:bf` | A | 2 | false positive | soak/repro/bf_line_jump.py (task 404, task 404) | a steep but smooth slope far from the mode (bf, -1283 -> -840 within 0.05 of the line parameter, 11 sub-steps all smooth, same regimes). Detector changed to compare with neighbouring steps |
| `lp_disagree_large:nkms:regime_count_differs` | B | 2 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 103, task 130) | same knife edge as for nkm (3-parameter model); also seen at news MCMC draws (replay under occbin, part B) and at a news posterior mode (0.23 with equal binding-period count but a shifted sequence) |
| `timeout` | A | 2 | see judgement | soak/results/anomalies.jsonl (task 105, task 1445) | per-evaluation 40-60 s timer fired: the opt-in MILP configuration (A#105: 80 MILP time-outs)  |
| `driver_or_unexpected:TypeError:dict() got multiple values for keyword argument 'part'` | B | 1 | ? | soak/results/anomalies.jsonl (task 155) | not yet looked at |
| `lp_disagree_large:nkm:same_regime_count` | B | 1 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 119) | knife edge of the approx class: one regime decision shifts by one period. Full counts from the sweeps: 195 of 8,279 finite NKM pairs differ by > 1e-8, 60 by > 0.12, 9 by > 0.4, max 1.54 (docs/OBC_ESTIMATION.md, 2d). The first logged lines (20 per kind) understated the size. |
| `lp_disagree_large:nkms:same_regime_count` | A | 1 | wrong-result (known cause) | soak/repro/lp_disagree_nkm.py (task 94) | same knife edge as for nkm (3-parameter model); also seen at news MCMC draws (replay under occbin, part B) and at a news posterior mode (0.23 with equal binding-period count but a shifted sequence) |
| `memory_jump` | B | 1 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 35) | FIXED in obc-soak (Occbin.rule_memo_bytes, 384 MiB per Occbin object): the memo trie of Occbin.rules was unbounded (232,740 nodes, 1.3 GB at extreme vectors). After the fix the peak is ~550 MB and results are bit for bit unchanged. Remaining jumps of 100-440 MB are the bounded memo filling. (Events listed for B/C/D at 1.0-1.1 GB were artefacts of the first detector.) |
| `memory_jump` | C | 1 | resource (real) | soak/repro/occbin_rule_trie_memory.py (task 63) | FIXED in obc-soak (Occbin.rule_memo_bytes, 384 MiB per Occbin object): the memo trie of Occbin.rules was unbounded (232,740 nodes, 1.3 GB at extreme vectors). After the fix the peak is ~550 MB and results are bit for bit unchanged. Remaining jumps of 100-440 MB are the bounded memo filling. (Events listed for B/C/D at 1.0-1.1 GB were artefacts of the first detector.) |
| `mode_search_unfinished_90s` | B | 1 | cosmetic | soak/repro/mode_search_slow.py (task 407) | csminwel with the news likelihood from random starts does not stop within 90 s: 2,700 evaluations at 33 ms each, still improving (lp 1679.9); not a hang (soak/repro/mode_search_slow.py). The known non-convergence of csminwel on the NKM posterior |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:bf` | C | 1 | cosmetic | soak/results/anomalies.jsonl (task 403) | OccBin guess-and-verify cycles (error 310) or news reports horizon_too_short/no_solution on one side only; counted per 1000-period segment (counts are segments, not events) |
| `one_method_fails:occbin_fails:expected:OccbinPeriodicError:310:nkm_est` | C | 1 | cosmetic | soak/results/anomalies.jsonl (task 127) | OccBin guess-and-verify cycles (error 310) or news reports horizon_too_short/no_solution on one side only; counted per 1000-period segment (counts are segments, not events) |
| `perf_cliff` | A | 1 | ? | soak/results/anomalies.jsonl (task 1894) | not yet looked at |
| `regime_seq_differs_equal_lp` | A | 1 | cosmetic | soak/repro/regime_seq_differs_equal_lp.py (task 1342) | equal log posterior to 1e-15 relative (lp = -8.5e14 at an absurd vector); one period flips: knife edge |
| `timeout` | B | 1 | see judgement | soak/results/anomalies.jsonl (task 203) | per-evaluation 40-60 s timer fired: the opt-in MILP configuration (A#105: 80 MILP time-outs)  |

## Failure rates, part A (inversion filter, log posterior evaluations)

Distance bin = rms of the relative deviation from the model's initial values: a <0.02, b <0.1, c <0.3, d <1, e >=1; `line` = the 25 points of the straight-line sweeps (-0.1..1.1).

| model | config | n | finite | failed (by code) | by distance bin: finite share |
|---|---|---|---|---|---|
| bf | news | 6498 | 0.837 | c304 714, c41 56, c42 288 | a:1.00 (46) b:0.89 (47) c:0.81 (223) d:0.57 (209) e:0.45 (173) line:0.86 (5800) |
| bf | news-milp | 27 | 0.407 | c0 1, c304 1, c41 3, c42 11 | a:1.00 (3) b:1.00 (1) c:0.75 (4) d:0.33 (9) e:0.10 (10) |
| bf | news20 | 139 | 0.676 | c304 14, c41 9, c42 22 | a:1.00 (5) b:1.00 (8) c:0.80 (45) d:0.60 (48) e:0.48 (33) |
| bf | news60 | 139 | 0.676 | c304 14, c41 9, c42 22 | a:1.00 (5) b:1.00 (8) c:0.80 (45) d:0.60 (48) e:0.48 (33) |
| bf | occbin | 6498 | 0.838 | c304 709, c41 56, c42 288 | a:1.00 (46) b:0.89 (47) c:0.81 (223) d:0.60 (209) e:0.45 (173) line:0.86 (5800) |
| bf | occbin60 | 46 | 0.826 | c304 8 | b:1.00 (2) c:0.79 (34) d:0.90 (10) |
| nkm | minnorm | 233 | 0.953 | c304 6, c41 5 | c:0.97 (105) d:0.95 (112) e:0.88 (16) |
| nkm | news | 6523 | 0.849 | c3 4, c304 152, c4 36, c41 62, c42 727, c5 6 | a:1.00 (4) b:0.75 (4) c:0.88 (232) d:0.61 (266) e:0.33 (192) line:0.88 (5825) |
| nkm | news-longest | 233 | 0.627 | c304 82, c41 5 | c:0.63 (105) d:0.65 (112) e:0.44 (16) |
| nkm | news-milp | 26 | 0.577 | c0 1, c304 1, c4 3, c41 4, c42 1, c5 1 | c:0.75 (4) d:0.36 (11) e:0.73 (11) |
| nkm | news-shortest | 233 | 0.953 | c304 6, c41 5 | c:0.97 (105) d:0.95 (112) e:0.88 (16) |
| nkm | news20 | 140 | 0.614 | c304 7, c4 4, c41 11, c42 30, c5 2 | b:0.00 (1) c:0.87 (47) d:0.53 (57) e:0.43 (35) |
| nkm | news60 | 140 | 0.614 | c304 7, c4 4, c41 11, c42 30, c5 2 | b:0.00 (1) c:0.87 (47) d:0.53 (57) e:0.43 (35) |
| nkm | occbin | 6523 | 0.848 | c3 4, c304 157, c4 36, c41 62, c42 727, c5 6 | a:1.00 (4) b:0.75 (4) c:0.88 (232) d:0.61 (266) e:0.32 (192) line:0.88 (5825) |
| nkm | occbin60 | 47 | 0.957 | c304 1, c41 1 | c:0.95 (21) d:0.96 (24) e:1.00 (2) |
| nkms | minnorm | 233 | 0.747 | c4 38, c42 21 | a:1.00 (12) b:1.00 (15) c:0.85 (13) d:0.55 (20) e:0.72 (173) |
| nkms | news | 6498 | 0.948 | c304 31, c306 1, c4 192, c41 57, c42 46, c5 8 | a:1.00 (12) b:1.00 (42) c:0.96 (156) d:0.63 (144) e:0.56 (344) line:0.98 (5800) |
| nkms | news-milp | 24 | 1.000 |  | b:1.00 (3) c:1.00 (14) d:1.00 (6) e:1.00 (1) |
| nkms | news20 | 140 | 0.679 | c304 6, c306 1, c4 17, c41 10, c42 11 | a:1.00 (2) b:1.00 (12) c:0.88 (34) d:0.54 (26) e:0.56 (66) |
| nkms | news60 | 140 | 0.679 | c304 6, c306 1, c4 17, c41 10, c42 11 | a:1.00 (2) b:1.00 (12) c:0.88 (34) d:0.54 (26) e:0.56 (66) |
| nkms | occbin | 6498 | 0.948 | c304 34, c306 1, c4 192, c41 57, c42 46, c5 8 | a:1.00 (12) b:1.00 (42) c:0.95 (156) d:0.62 (144) e:0.56 (344) line:0.98 (5800) |
| nkms | occbin60 | 93 | 0.527 | c304 5, c306 1, c4 17, c41 10, c42 11 | a:1.00 (2) b:1.00 (4) c:0.20 (5) d:0.29 (17) e:0.57 (65) |

occbin vs news, both finite: |difference in log posterior|

| model | pairs | <=1e-9 (rel) | 1e-9..0.15 | >0.15 | max |
|---|---|---|---|---|---|
| bf | 5438 | 5418 | 16 | 4 | 4 |
| nkm | 5526 | 5334 | 135 | 57 | 1.69e+06 |
| nkms | 6157 | 6056 | 92 | 9 | 2.31 |

## Timing distributions

Part A, seconds per finite log-posterior evaluation:

| model | config | n | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 5440 | 0.0304 | 0.1942 | 0.4941 |
| bf | news-milp | 11 | 0.0288 | 0.0711 | 0.0714 |
| bf | news20 | 94 | 0.0309 | 0.08347 | 0.2584 |
| bf | news60 | 94 | 0.03205 | 0.09285 | 0.4018 |
| bf | occbin | 5445 | 0.0858 | 0.1928 | 0.3352 |
| bf | occbin60 | 38 | 0.08475 | 0.1739 | 0.1752 |
| nkm | minnorm | 222 | 0.0367 | 0.09551 | 0.1083 |
| nkm | news | 5536 | 0.0359 | 0.08801 | 0.2644 |
| nkm | news-longest | 146 | 0.2803 | 0.5125 | 0.7029 |
| nkm | news-milp | 15 | 0.0528 | 13.27 | 13.3 |
| nkm | news-shortest | 222 | 0.2942 | 0.5372 | 0.7295 |
| nkm | news20 | 86 | 0.03715 | 0.166 | 0.2012 |
| nkm | news60 | 86 | 0.03755 | 0.141 | 0.2962 |
| nkm | occbin | 5531 | 0.1371 | 0.2914 | 1.108 |
| nkm | occbin60 | 45 | 0.1388 | 0.2375 | 0.2375 |
| nkms | minnorm | 174 | 0.03335 | 0.08783 | 0.1346 |
| nkms | news | 6163 | 0.0335 | 0.08124 | 0.6264 |
| nkms | news-milp | 24 | 0.0348 | 0.07438 | 0.0762 |
| nkms | news20 | 95 | 0.0342 | 0.07642 | 0.0767 |
| nkms | news60 | 95 | 0.0354 | 0.07927 | 0.0803 |
| nkms | occbin | 6160 | 0.1234 | 0.2691 | 2.547 |
| nkms | occbin60 | 49 | 0.1255 | 1.152 | 1.978 |

Part C, seconds per simulated period (per 1000-period segment, averaged over a task):

| model | method | tasks | median | p99 | max |
|---|---|---|---|---|---|
| bf | news | 24 | 3.4e-05 | 5.77e-05 | 6e-05 |
| bf | occbin | 24 | 0.0002458 | 0.0003026 | 0.0003033 |
| borrcon | news | 53 | 3e-05 | 7.827e-05 | 8e-05 |
| borrcon | occbin | 53 | 0.00032 | 0.0005424 | 0.0005667 |
| borrcon_d | news | 47 | 2.667e-05 | 9.847e-05 | 0.0001 |
| borrcon_d | occbin | 47 | 0.00031 | 0.0007635 | 0.00088 |
| nkm | news | 39 | 2.333e-05 | 4.747e-05 | 5e-05 |
| nkm | occbin | 39 | 0.000289 | 0.0004709 | 0.0005 |
| nkm_est | news | 44 | 3e-05 | 6.19e-05 | 6.333e-05 |
| nkm_est | occbin | 44 | 0.0002822 | 0.0005171 | 0.00052 |
| sw | news | 17 | 4e-05 | 0.0001013 | 0.0001067 |
| sw | occbin | 17 | 0.00027 | 0.0004668 | 0.0004833 |
| zlb | news | 65 | 3.667e-05 | 6.36e-05 | 7e-05 |
| zlb | occbin | 65 | 0.000332 | 0.0005072 | 0.00052 |
| zlb2 | news | 66 | 5.183e-05 | 0.0001161 | 0.0001459 |
| zlb2 | occbin | 66 | 0.000472 | 0.0009297 | 0.0009848 |

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

- A: median 187, max 1446
- B: median 216, max 1409
- C: median 226, max 1412
- D: median 395, max 1415
