"""Likelihood evaluation time and agreement, OccBin vs news solver, on Dynare's NKM example (full and small): the stored
parameter vectors and 40 vectors perturbed by 2% (log-normal, seed 1) around the first.  The two solvers are called alternately
at each vector, 3 repeats, minimum per vector.   python3 bench/obc_est_timing.py -> bench/results/est_timing.json"""
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
from pydyn.occbin_estimation import OccbinEstimation  # noqa: E402

REF = ROOT / "reference"


def main():
    out = {}
    for case in ("nkm", "nkms"):
        tmp = Path(tempfile.mkdtemp())
        shutil.copy(REF / f"occest_{case}.mod", tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pydyn.Model(pydyn.run_preprocessor(tmp / f"occest_{case}.mod"))
            r = loadmat(REF / f"occest_{case}_lik.mat", squeeze_me=True)
            data = np.atleast_2d(r["data"].T).T
            mk = lambda **k: OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)}, nobs=data.shape[0],  # noqa: E731
                                              first_obs=1, occbin={"likelihood_inversion_filter": True}, **k)
            eo, en = mk(), mk(obc_method="news")
        X = np.atleast_2d(r["X"])
        X = X if X.shape[0] == eo.n else X.T
        rng = np.random.default_rng(1)
        pts = [("stored", X[:, k]) for k in range(X.shape[1])] + [("perturbed", X[:, 0] * np.exp(0.02 * rng.standard_normal(eo.n))) for _ in range(40)]
        rows = []
        for kind, th in pts:
            to, tn = [], []
            for _ in range(3):
                t = time.perf_counter(); a = eo.log_posterior(th); to.append(time.perf_counter() - t)
                t = time.perf_counter(); b = en.log_posterior(th); tn.append(time.perf_counter() - t)
            rows.append(dict(kind=kind, lp_occbin=a, lp_news=b, occbin_s=min(to), news_s=min(tn)))
        d = np.array([abs(x["lp_occbin"] - x["lp_news"]) if np.isfinite(x["lp_occbin"]) else 0.0 for x in rows])
        to, tn = np.array([x["occbin_s"] for x in rows]), np.array([x["news_s"] for x in rows])
        out[case] = dict(n=len(rows), occbin_total_s=float(to.sum()), news_total_s=float(tn.sum()),
                         occbin_median_s=float(np.median(to)), news_median_s=float(np.median(tn)),
                         occbin_min_s=float(to.min()), news_min_s=float(tn.min()),
                         n_lp_differ_over_1e9=int((d > 1e-9).sum()), max_lp_difference=float(d.max()),
                         lp_differences=[float(x) for x in d[d > 1e-9]], slowest_news_s=float(tn.max()),
                         rows=rows)
        print(case, {k: v for k, v in out[case].items() if k != "rows"}, flush=True)
    (ROOT / "bench" / "results" / "est_timing.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
