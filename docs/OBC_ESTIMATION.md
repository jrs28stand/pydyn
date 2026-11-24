# OBC news methods in estimation, and equilibrium selection (pydyn, branch `obc-est`)

Started 2026-11-24 from `wip/obc` (0a16a10). Backed up as `wip/obc-est`.

## State (keep this section current)

Baseline before any change: `pytest tests/test_obc.py tests/test_obc_perf.py tests/test_occbin.py tests/test_occbin_options.py
tests/test_occbin_estimation.py` = 164 passed, 1 skipped.

| Task | Status |
|---|---|
| 1a proper estimation option `obc_method` / `obc_horizon` / `obc_selection` | code written (`OccbinEstimation`), tests not yet written |
| 1b news filter solver for several-equation constraints (`RegimeFilterSolver`) | code written, verified by script on bf and bf_ppf (1e-13); tests not yet written |
| 1c fail fast on a failed trial point | investigating (NKM parameter vector 4: MILP proves "no solution" after 1.5 s) |
| 2 estimation feasibility study | not started |
| 3 `obc_selection` | not started |

Next steps: see the bottom of this section when updated.

