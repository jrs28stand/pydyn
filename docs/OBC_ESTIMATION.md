# OBC news methods in estimation, and equilibrium selection (pydyn, branch `obc-est`)

Started 2026-11-24 from `wip/obc` (0a16a10). Backed up as `wip/obc-est`.

## State (keep this section current)

Baseline before any change: `pytest tests/test_obc.py tests/test_obc_perf.py tests/test_occbin.py tests/test_occbin_options.py
tests/test_occbin_estimation.py` = 164 passed, 1 skipped.

| Task | Status |
|---|---|
| 1a proper estimation option `obc_method` / `obc_horizon` / `obc_selection` | done, tested (`tests/test_obc_estimation.py`) |
| 1b news filter solver for several-equation constraints (`RegimeFilterSolver`) | done, tested on bf.mod and bf_ppf.mod |
| 1c fail fast on a failed trial point | done (integer programme off by default in the filter), tested |
| 3 `obc_selection` | done, tested (`tests/test_obc_selection.py`); cost per binding period still to be measured (`bench/obc_selection_cost.py`) |
| 2 estimation feasibility study | driver `bench/obc_est_study.py` written; runs in progress on a frozen clone (/home/claude/work/pydyn-t2, logs /home/claude/work/t2_*.log); results go to `bench/results/est_*.json`; tables below not yet written |

Next steps: when the runs finish, read `bench/results/est_<case>_<start>_{mode,hessian,mh}.json`, write the tables of Task 2 below,
run `bench/obc_selection_cost.py` (alone on the machine), run the full test suite, push, final report.

Found on the way: `Estimation` set `lb`/`ub` to NaN where `estimated_params` has empty bounds (Dynare's NKM files), so every comparison with
the bounds was false and the Metropolis-Hastings sampler rejected every proposal (acceptance 0 on `occest_nkms.mod`). Fixed in
`pydyn/estimation.py` (`np.fmax`/`np.fmin`, NaN means "no bound"); the existing short-MH test only checked that the draws were finite.

