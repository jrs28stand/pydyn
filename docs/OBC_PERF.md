# OBC news methods: performance and reliability (pydyn)

Protocol: BENCH_PROTOCOL.md of the main session. Status: IN PROGRESS (full write-up not yet done).

## State at interruption (2026-11-24)

Done and measured (results in bench/results/*.json, tables via bench/make_perf_tables.py {p1,p2,p3,p4} and `r1 FILE...`):
- Harness bench/obc_bench.py (P1-P4, R1, warm), frozen baseline bench/obc_before.py (original obc.py, same-process before/after).
- BEFORE numbers: *_before.json. Final before/after interleaved runs: p1/p2/p3/p4/r1_final.json, r1_sw.json (sw, after only), warm_D.json.
- Code changes in pydyn/obc.py (all tests pass, 155 passed 1 skipped in the four occbin/obc test files plus tests/test_obc_perf.py):
  1. next state only per period, full path once after the last shock; 2. q and next state from state variables only;
  4. O(T) build of M and wedge responses (diagonal shift recursion), LU instead of inverse; 5. constraint objects,
  affine conditions and compiled conditions cached; 6. LAPACK solves, cheaper regime checks and histories, linear path on states.
  3. warm start implemented as option (warm_start=False), measured slower, results identical.
- Headline (min, same process): P1 us/period zlb news 228->28, zlb2 253->38, nkm 262->37, sw regime 212->38;
  P2 ms news zlb 9.9->1.4, nkm 12.3->2.3, sw regime 15.9->2.1; P3 s zlb news 3.2->0.38, sw regime 3.3->0.56.
- R1: zlb, zlb2, nkm: 0 failures at s 0.5-2, regimes identical to OccBin except nkm news (approximate class, 1 of 2000 at s>=1.5).
  sw: OccBin and RegimeOBC fail on exactly the same states (9-18% of 2000), never one without the other; regimes agree elsewhere.
- sw cannot reach 30% binding without failures (s=0.25 gives 7.5%).

Half done: bench/sw_timeout_check.py (running; re-runs sw time-outs with 30 s limit; output bench/results/sw_timeout_check.json).
This document: only this section so far. Missing: full tables, profiles (cProfile top entries before/after), what each change bought,
"Lessons for the Julia port", "Not measured / open".

Rerun: `export DYNARE_PREPROCESSOR=/home/claude/ref/pp-standalone/src/dynare-preprocessor PYDYN_EXAMPLES=/home/claude/ref/dynare-7.2/examples; nice python3 bench/obc_bench.py {p1,p2,p3,p4,r1} --tag X`
Tests: `python3 -m pytest tests/test_obc.py tests/test_obc_perf.py tests/test_occbin.py tests/test_occbin_options.py tests/test_occbin_estimation.py -q -p no:warnings`

Next three steps: (1) collect sw_timeout_check result; (2) write full doc from the JSON tables; (3) final report.
