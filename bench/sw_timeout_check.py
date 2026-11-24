"""R1 follow-up on sw: cross-tabulate OccBin and RegimeOBC failures at s = 1 and re-run a sample of the states on which
RegimeOBC stopped at the 0.5 s limit of the integer programme with the class default of 30 s.
Usage: sw_timeout_check.py [n_sample]  -> bench/results/sw_timeout_check.json"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import obc_bench as B  # noqa: E402
from pydyn.occbin import Occbin  # noqa: E402

n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 12
m = B.load_model("sw")
ob = Occbin(m)
states = B.linear_states(ob, m, 2000, B.SEED)
z = np.random.default_rng(B.SEED + 3).standard_normal((2000, m.n_exo)) * B.shock_sd(m)
rb = B.make_obj("after", "regime", ob, 40)
rb.milp_time_limit = 0.5
cross, failing = Counter(), []
for i in range(2000):
    sh = (z[i] * 1.0)[None, :]
    e1 = e2 = None
    try:
        ob.simulate(sh, periods=1, init=states[i], check_ahead_periods=B.OCCBIN_CHECK_AHEAD)
    except Exception as err:  # noqa: BLE001
        e1 = B.err_type(err)
    try:
        rb.simulate(sh, periods=1, init=states[i])
    except Exception as err:  # noqa: BLE001
        e2 = B.err_type(err)
        if e2 == "OBCError:not_converged":
            failing.append(i)
    cross[(e1, e2)] += 1
print("cross-tab (OccBin error, RegimeOBC error): count")
for k, v in sorted(cross.items(), key=lambda kv: -kv[1]):
    print(k, v)
rb.milp_time_limit = 30.0
outcome = Counter()
detail = []
import time
for i in failing[:n_sample]:
    sh = (z[i] * 1.0)[None, :]
    t0 = time.perf_counter()
    try:
        rb.simulate(sh, periods=1, init=states[i])
        outcome["solved with 30 s limit"] += 1
        tag = "solved"
    except Exception as err:  # noqa: BLE001
        tag = B.err_type(err)
        outcome[tag] += 1
    detail.append((i, tag, round(time.perf_counter() - t0, 1)))
print(dict(outcome), detail)
Path(__file__).resolve().parent.joinpath("results", "sw_timeout_check.json").write_text(json.dumps(dict(
    cross={f"{a} | {b}": v for (a, b), v in cross.items()}, resolved_with_30s=dict(outcome), detail=detail,
    n_failing_not_converged=len(failing)), indent=1))
