# OBC methods: performance and reliability protocol (both ports)

Written 2026-11-24 by the main session. Both ports measure the same things on the same models so the
numbers can be put side by side. Contract: /home/claude/work/OBC_CONTRACT.md.

## Models
| ID | File | Size | Class |
|---|---|---|---|
| zlb | pydyn reference/obc_zlb.mod | 5 variables | complementarity |
| zlb2 | pydyn reference/obc_zlb2.mod | 7 variables, two constraints | complementarity |
| nkm | Dynare tests/occbin/filter/NKM_0_std_shocks.mod | 24 variables | complementarity (approximate) |
| sw | /home/claude/work/bench/obc_sw_zlb.mod | 32 variables, 7 shocks | multi_equation |

`sw` is Smets-Wouters with a lower bound (from Dynare's swpkfppf.mod, data and estimation lines removed).
Known baseline behaviour in pydyn before any speed work: a -4 shock to `eb` binds for 9 periods and both
methods agree; a -6 shock makes OccBin cycle (error 313) and the general regime formulation fail too.

Shocks: draws are standard normal times the model's shock standard deviations (0.01 where a standard
deviation is zero) times a scale s. Use a fixed seed and record it. If the two ports cannot share draws
bit for bit, say so; the statistics are what is compared.

## What to measure
For each model and each method available in the port (occbin, and news / minnorm in each formulation that
applies), with horizon 40 unless stated:

- **P1 per period.** Simulation of 1,000 shock periods at fixed parameters, s chosen so the constraint binds in
  30-60% of periods and no method fails (report s and the share). Microseconds per period. Also report the
  split between periods where nothing binds and periods where something binds.
- **P2 per parameter set.** Time from "parameters changed" to "ready to solve the first period": steady state,
  first-order solution, and everything the method precomputes. Report the parts separately. Milliseconds.
- **P3 likelihood-like loop.** 50 times: perturb two parameters slightly, redo P2, simulate 200 periods.
  Seconds in total and per iteration. This is the cost profile of estimation (the inversion filter solves
  one such path per likelihood evaluation).
- **P4 scaling.** P1 and P2 at horizons 20, 40, 80 on `nkm` and `sw`.
- **R1 reliability.** Take 2,000 states from a long unconstrained (linear) simulation at s = 1. From each
  state solve ONE period with a fresh shock at scale s in {0.5, 1, 1.5, 2}. Count per method: solved,
  failed (by error type), and among those both methods solve, how many give different regime sequences.
  Use a time limit per solve where a method has one and count time-outs as failures.

## Rules for timing on this machine
- Two cores shared by three workers, so wall-clock numbers are noisy. Time before and after in the same
  process, interleaved, at least 5 repeats, report the minimum and the median. Never compare a number from
  one run with a number from another run.
- Exclude compilation (Julia: one warm-up call first) and exclude the preprocessor.
- Every timing comes with a correctness check from the same run: maximum path difference against the
  port's OccBin and whether regime histories agree.
- Record results in the repo as docs/OBC_PERF.md: a before/after table per measurement, the profile's top
  entries before and after, and what each change bought. State what was not measured.
