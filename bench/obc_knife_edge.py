"""Where the OccBin and news likelihoods differ at a parameter vector, show that the first differing period is a knife edge:
the distance of each solver's solution to OccBin's bind/relax condition (docs/OBC_ESTIMATION.md).

    python3 bench/obc_knife_edge.py CASE THETA_JSON
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import obc_est_study as S  # noqa: E402
from pydyn import obc  # noqa: E402


def knife_edge(case, theta):
    res = {}
    ests = {}
    for meth in S.METHODS:
        _, ests[meth], _ = S.load(case, meth)
        ests[meth].log_posterior(theta)
        res[meth] = ests[meth].last_result
    ro, rn = res["occbin"], res["news"]
    per = [t for t in range(min(len(ro.regime_history), len(rn.regime_history))) if ro.regime_history[t] != rn.regime_history[t]]
    out = dict(lp_occbin=-ests["occbin"].last_result.LIK, lp_news=-ests["news"].last_result.LIK, periods_different=[t + 1 for t in per])
    if not per:
        return out
    t0 = per[0]
    ob = ests["occbin"].occbin_at(theta)
    nb = obc.NewsOBC(ob)
    out["period"] = t0 + 1
    out["regime_occbin"], out["regime_news"] = str(ro.regime_history[t0]), str(rn.regime_history[t0])
    for meth, r in (("occbin", ro), ("news", rn)):
        init = np.zeros(ob.n) if t0 == 0 else r.stateval[t0 - 1]
        u = np.zeros(ob.n_exo)
        u[ests[meth].ivf_mapping] = r.filtered_errs[t0]
        d = {}
        # OccBin's solution at this state and shock, and OccBin's own test (bind/relax conditions, in levels) on each path
        occ = ob.simulate(u[None, :], periods=3, init=init, check_ahead_periods=200)
        y, path, info = nb.solve_period(init, u)
        for label, z, hist in (("occbin_solution", occ.piecewise, occ.regime_history[0]),
                               ("news_solution", path[:3] + ob.ys, nb_regime(y, nb))):
            binding, relax, eb, er = ob.check(z)
            d[label] = dict(regime=str(hist), bind_condition_true_in_period1=bool(binding[0, 0]), relax_condition_true_in_period1=bool(relax[0, 0]),
                            bind_margin=float(eb[0, 0]), relax_margin=float(er[0, 0]))
        q = nb.q0 + nb.Qm @ np.r_[init, u]
        d["gap_without_wedges_period1"] = float(q[0])
        d["wedge_period1"] = float(y[0])
        d["ratio_wedge_to_gap"] = float(y[0] / q[0]) if q[0] else None
        out[f"at_{meth}_state_and_shock"] = d
    return out


def nb_regime(y, nb):
    from pydyn.obc import _regime_history
    return _regime_history(y, nb.T, nb.n_c)[0]


if __name__ == "__main__":
    case, theta = sys.argv[1], np.array(json.loads(sys.argv[2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(json.dumps(knife_edge(case, theta), indent=1))
