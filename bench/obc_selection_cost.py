"""Cost of obc_selection per binding period (docs/OBC_ESTIMATION.md): the per-period step of NewsOBC / RegimeOBC on
binding states, default against 'shortest', 'longest', 'min_wedge', interleaved, min over repeats per state.

    python3 bench/obc_selection_cost.py            # writes bench/results/selection_cost.json
"""
import json
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pydyn  # noqa: E402
from pydyn.obc import NewsOBC, RegimeOBC  # noqa: E402
from pydyn.occbin_estimation import OccbinEstimation  # noqa: E402

REF = ROOT / "reference"
SELS = ("occbin", "shortest", "longest", "min_wedge")


def load(name):
    d = Path(tempfile.mkdtemp())
    shutil.copy(REF / name, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(d / name))


def binding_states(obc, shocks):
    """(state, shock) of the shock periods in which something binds, along a default simulation."""
    y, out = np.zeros(obc.ob.n), []
    for sh in shocks:
        if isinstance(obc, NewsOBC):
            wedge, nxt, _info = obc._step(y, sh)
        else:
            _z, wedge, nxt, _info = obc._step(y, sh)
        if wedge.any():
            out.append((y.copy(), sh.copy()))
        y = nxt
    return out


def timeit(obc, states, repeats):
    best = {s: np.full(len(states), np.inf) for s in SELS}
    nsp = {s: [] for s in SELS}
    for rep in range(repeats):
        for i, (y, sh) in enumerate(states):
            for sel in SELS:
                t = time.perf_counter()
                r = obc._step(y, sh, selection=sel)
                dt = time.perf_counter() - t
                best[sel][i] = min(best[sel][i], dt)
                if rep == 0:
                    nsp[sel].append(r[-1].get("n_single_spell", -1))
    return best, nsp


def main():
    out = {}
    cases = []
    m = load("obc_zlb.mod")
    ob = pydyn.Occbin(m)
    nb = NewsOBC(ob)
    rng = np.random.default_rng(1)
    cases.append(("zlb (NewsOBC, 5 variables)", nb, rng.normal(0, 0.02, (400, 1))))
    # NKM small model at Dynare's first stored parameter vector
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(REF / "occest_nkms.mod", tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2 = pydyn.Model(pydyn.run_preprocessor(tmp / "occest_nkms.mod"))
        r = loadmat(REF / "occest_nkms_lik.mat", squeeze_me=True)
        data = np.atleast_2d(r["data"].T).T
        est = OccbinEstimation(m2, data={v: data[:, j] for j, v in enumerate(m2.varobs)}, nobs=data.shape[0], first_obs=1,
                               occbin={"likelihood_inversion_filter": True})
    X = np.atleast_2d(r["X"])
    X = X if X.shape[0] == est.n else X.T
    ob2 = est.occbin_at(X[:, 0])
    nb2 = NewsOBC(ob2)
    sd = np.sqrt(np.diag(m2.Sigma_e))
    sd[sd == 0] = 0.01
    cases.append(("NKM small (NewsOBC, 24 variables)", nb2, np.random.default_rng(2).normal(size=(400, ob2.n_exo)) * sd * 3))
    m3 = load("obc_sw_zlb.mod")
    ob3 = pydyn.Occbin(m3)
    rb = RegimeOBC(ob3, horizon=40, check_ahead=60)
    sd3 = np.sqrt(np.diag(m3.Sigma_e))
    S3 = np.random.default_rng(7).normal(size=(120, m3.n_exo)) * sd3 * 0.25
    ext = np.zeros((1, m3.n_exo))
    ext[0, m3._ix["eb"]] = -4.0
    cases.append(("Smets-Wouters (RegimeOBC, 32 variables)", rb, np.vstack([S3, ext, ext * 0.5])))
    for name, obc, shocks in cases:
        states = binding_states(obc, shocks)
        best, nsp = timeit(obc, states, repeats=5)
        row = dict(n_binding_states=len(states))
        for sel in SELS:
            row[sel] = dict(min_ms_median_over_states=1e3 * float(np.median(best[sel])), mean_ms=1e3 * float(best[sel].mean()),
                            max_ms=1e3 * float(best[sel].max()))
        row["extra_ms_per_binding_period_median"] = {s: 1e3 * float(np.median(best[s] - best["occbin"])) for s in SELS[1:]}
        row["single_spell_solutions_found"] = {s: {str(k): int(v) for k, v in zip(*np.unique(nsp[s], return_counts=True))} for s in SELS[1:]}
        out[name] = row
        print(name, json.dumps(row, indent=1), flush=True)
    (ROOT / "bench" / "results" / "selection_cost.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
