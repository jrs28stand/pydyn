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

{{AUTO}}
