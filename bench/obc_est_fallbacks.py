"""At trial points where the news solver's cheap checks (active-set iteration, linear-programme proof) do not decide, does
OccBin's own solver succeed?  And does an integer programme (1 s limit) find a solution?  (docs/OBC_ESTIMATION.md, 1c)

The inversion filter is run with the news solver at the stored parameter vectors and at random perturbations of the first
(log-normal, given sd); at every solver call that reaches the fallbacks, OccBin's solver is called on the same state and
shock and the integer programme is tried (1 s).   python3 bench/obc_est_fallbacks.py   -> bench/results/est_fallbacks.json
"""
import collections
import json
import os
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
from pydyn import obc, obc_filter  # noqa: E402
from pydyn.occbin_estimation import OccbinEstimation  # noqa: E402
from pydyn.occbin_filter import _Solver  # noqa: E402

REF = ROOT / "reference"
T = Path(os.environ.get("PYDYN_DYNARE_TESTS", "/home/claude/ref/dynare-7.2/tests"))


def nkm(case):
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(REF / f"occest_{case}.mod", tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp / f"occest_{case}.mod"))
        r = loadmat(REF / f"occest_{case}_lik.mat", squeeze_me=True)
        data = np.atleast_2d(r["data"].T).T
        est = OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)}, nobs=data.shape[0], first_obs=1,
                               occbin={"likelihood_inversion_filter": True}, obc_method="news")
    X = np.atleast_2d(r["X"])
    return est, (X if X.shape[0] == est.n else X.T)


def bf(rel):
    src = T / rel
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(src, tmp / src.name)
    d = loadmat(REF / "occest_bf_data.mat", squeeze_me=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp / src.name))
        est = OccbinEstimation(m, data={"q": d["q"], "r": d["r"]}, nobs=100, first_obs=1, occbin={"likelihood_inversion_filter": True},
                               posterior_sampling_method="random_walk_metropolis_hastings", posterior_sampler_options=(),
                               obc_method="news")
    return est, np.array(est.x0)[:, None]


def run(name, est, X, npts, sd, seed):
    rows = []
    rng = np.random.default_rng(seed)
    # spy on the fallbacks: both solver classes
    ctr = collections.Counter()
    for cls in (obc_filter.LCPFilterSolver, obc_filter.RegimeFilterSolver):
        orig = cls.solve

        def spy(self, shocks, init, sim, init_regime=None, exo_pos=None, __orig=orig, **kw):
            f0 = self.fallbacks
            out = __orig(self, shocks, init, sim, init_regime, exo_pos, **kw)
            if self.fallbacks > f0:
                flag, _ = _Solver(self.ob).solve(shocks, init, sim, exo_pos=exo_pos)
                rows.append(dict(news=out[0], occbin=flag))
                if isinstance(self, obc_filter.LCPFilterSolver):
                    u, x = self._x(shocks, init, exo_pos)
                    nb = self.news
                    q = nb.q0 + nb.Qm @ x
                    y, st = obc.solve_news(nb.Mo, q, nb.n_c * nb.T, tol=1e-8, n_c=nb.n_c, horizons=[nb.T], time_limit=1.0)
                    ctr["milp_found_solution" if y is not None else "milp_" + st] += 1
                else:
                    self.rb.milp_time_limit = 1.0
                    try:
                        self.rb._step(np.asarray(init, float).ravel(), u_full(self, shocks, exo_pos))
                        ctr["milp_found_solution"] += 1
                    except obc.OBCError as e:
                        ctr["milp_" + e.status] += 1
                    self.rb.milp_time_limit = 0.0
            return out
        cls.solve = spy
    pts = [X[:, k] for k in range(X.shape[1])] + [X[:, 0] * np.exp(sd * rng.standard_normal(X.shape[0])) for _ in range(npts)]
    t = time.perf_counter()
    nfin = 0
    for th in pts:
        nfin += bool(np.isfinite(est.log_posterior(th)))
    c = collections.Counter((r["news"], r["occbin"]) for r in rows)
    return dict(model=name, parameter_vectors=len(pts), finite_likelihood=nfin, trial_points_reaching_the_fallbacks=len(rows),
                news_flag_occbin_flag={f"{a}/{b}": v for (a, b), v in sorted(c.items())}, integer_programme_1s=dict(ctr),
                wall_s=time.perf_counter() - t)


def u_full(self, shocks, exo_pos):
    u = np.zeros(self.ob.n_exo)
    sh = np.asarray(shocks, float).ravel()
    if exo_pos is None:
        u[:sh.size] = sh
    else:
        u[np.asarray(exo_pos, int)] = sh
    return u


def main():
    out = []
    for case, npts, sd in (("nkm", 60, 0.05), ("nkms", 100, 0.15)):
        est, X = nkm(case)
        out.append(run(f"NKM {case}", est, X, npts, sd, 5))
        print(out[-1], flush=True)
    for rel in ("estimation/slice/bf.mod", "occbin/ppf/bf_ppf.mod"):
        est, X = bf(rel)
        out.append(run(rel, est, X, 25, 0.05, 11))
        print(out[-1], flush=True)
    (ROOT / "bench" / "results" / "est_fallbacks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
