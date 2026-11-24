"""Estimation with the news solver vs OccBin's on Dynare's NKM OccBin estimation example (docs/OBC_ESTIMATION.md).

    python3 bench/obc_est_study.py CASE STAGES [--replic N] [--chains K] [--seed S]
CASE: nkm (9 parameters) or nkms (3 parameters).  STAGES: any of mode, hessian, mh, joined by commas.
Everything is run with the inversion-filter likelihood; results go to bench/results/est_<case>_*.json (+ .npz traces).
Needs DYNARE_PREPROCESSOR; reads reference/occest_*.mod, *_lik.mat (data, stored parameter vectors).
"""
import argparse
import collections
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
from pydyn import obc_filter  # noqa: E402
from pydyn import occbin_filter as _of  # noqa: E402
from pydyn.occbin_estimation import OccbinEstimation  # noqa: E402

REF = ROOT / "reference"
OUT = ROOT / "bench" / "results"
METHODS = ("occbin", "news")


class Trace:
    """Every evaluation of the log posterior: theta, value, Dynare's code, time, trial-point failures of the solver."""

    def __init__(self, est):
        self.est, self.rows, self.flags = est, [], collections.Counter()
        self.orig = est.log_posterior
        est.log_posterior = self

    def __call__(self, theta, bounds=None):
        f0 = dict(FLAGS)
        t = time.perf_counter()
        lp = self.orig(theta) if bounds is None else self.orig(theta, bounds=bounds)
        dt = time.perf_counter() - t
        fl = {k: FLAGS[k] - f0.get(k, 0) for k in FLAGS if FLAGS[k] != f0.get(k, 0)}
        self.rows.append((np.array(theta, float), float(lp), int(self.est.last_code), dt, fl))
        return lp

    def close(self):
        self.est.log_posterior = self.orig

    def arrays(self):
        return dict(theta=np.array([r[0] for r in self.rows]), lp=np.array([r[1] for r in self.rows]),
                    code=np.array([r[2] for r in self.rows]), time=np.array([r[3] for r in self.rows]))


FLAGS = collections.Counter()


def _count_flags():
    """Count the error flags that the solvers return to the inversion filter (OccBin 310-313, news 313/311)."""
    def wrap(cls):
        f = cls.solve

        def solve(self, *a, **k):
            flag, res = f(self, *a, **k)
            if flag:
                FLAGS[int(flag)] += 1
            return flag, res
        cls.solve = solve
    for cls in (_of._Solver, obc_filter.LCPFilterSolver, obc_filter.RegimeFilterSolver):
        if "solve" in cls.__dict__:
            wrap(cls)


def load(case, method, **kw):
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(REF / f"occest_{case}.mod", tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp / f"occest_{case}.mod"))
        r = loadmat(REF / f"occest_{case}_lik.mat", squeeze_me=True)
        data = np.atleast_2d(r["data"].T).T
        est = OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)}, nobs=data.shape[0], first_obs=1,
                               occbin={"likelihood_inversion_filter": True}, obc_method=method, **kw)
    X = np.atleast_2d(r["X"])
    X = X if X.shape[0] == est.n else X.T
    return m, est, X


def jl(o):
    return json.loads(json.dumps(o, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def regime_diff(m, theta, methods_est):
    """Compare the two solvers' regime histories at theta: periods that differ, and the likelihoods."""
    out = {}
    for meth, e in methods_est.items():
        lp = e.log_posterior(theta) if not isinstance(e.log_posterior, Trace) else e.log_posterior.orig(theta)
        out[meth] = (lp, e.last_result.regime_history if e.last_result is not None else None)
    a, b = out["occbin"][1], out["news"][1]
    per = []
    if a is not None and b is not None:
        per = [t + 1 for t in range(min(len(a), len(b))) if a[t] != b[t]]
    return dict(lp_occbin=out["occbin"][0], lp_news=out["news"][0], periods_with_different_regime=per,
                regimes={t: dict(occbin=str(a[t - 1]), news=str(b[t - 1])) for t in per[:4]})


def stage_mode(case, ests, X, tag, x0, log):
    res = {}
    traces = {}
    for meth in METHODS:
        e = ests[meth]
        tr = Trace(e)
        FLAGS.clear()
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = e.mode(x0=x0, mode_compute=4)
        wall = time.perf_counter() - t
        tr.close()
        A = tr.arrays()
        traces[meth] = A
        res[meth] = dict(x=r.x, log_posterior=r.log_posterior, success=r.success, message=r.message, nfev_optimizer=r.nfev,
                         evaluations=len(tr.rows), wall_s=wall, std=r.std, laplace=r.laplace,
                         hessian_pd=bool(r.inv_hessian is not None and np.all(np.isfinite(r.inv_hessian))),
                         failed_evaluations=int(np.sum(~np.isfinite(A["lp"]))), trial_failures=dict(FLAGS),
                         hessian=r.hessian, inv_hessian=r.inv_hessian)
        log(f"[{case} {tag}] mode {meth}: lp {r.log_posterior:.6f} evals {len(tr.rows)} wall {wall:.1f}s {r.message}")
    np.savez(OUT / f"est_{case}_{tag}_mode_traces.npz", **{f"{m}_{k}": v for m, A in traces.items() for k, v in A.items()})
    # path comparison and the first evaluation at which the likelihoods differ
    A, B = traces["occbin"], traces["news"]
    n = min(len(A["lp"]), len(B["lp"]))
    dth = np.abs(A["theta"][:n] - B["theta"][:n]).max(axis=1)
    first_theta = int(np.argmax(dth > 1e-9)) if np.any(dth > 1e-9) else None
    same_len = len(A["lp"]) == len(B["lp"])
    cmp = dict(identical_path=bool(first_theta is None and same_len), first_evaluation_with_different_theta=first_theta,
               n_occbin=len(A["lp"]), n_news=len(B["lp"]))
    # evaluate news at OccBin's thetas (and OccBin at news' thetas where they differ): first likelihood difference
    en = ests["news"]
    eo = ests["occbin"]
    diffs, agree = [], []
    tocc, tnews = [], []
    for i, th in enumerate(A["theta"]):
        t = time.perf_counter(); lb = en.log_posterior(th); tnews.append(time.perf_counter() - t)
        t = time.perf_counter(); la = eo.log_posterior(th); tocc.append(time.perf_counter() - t)
        if not ((np.isinf(la) and np.isinf(lb) and la == lb) or abs(la - lb) <= 1e-9 * max(1.0, abs(la))):
            diffs.append((i, la, lb, la - lb))
        elif np.isfinite(la):
            agree.append(abs(la - lb))
    cmp["occbin_trace_points_where_likelihoods_differ"] = [dict(index=i, occbin=a, news=b, diff=d) for i, a, b, d in diffs]
    cmp["n_points_checked"] = len(A["theta"])
    cmp["max_abs_lp_difference_where_equal"] = float(max(agree)) if agree else None
    cmp["n_equal"] = len(agree)
    if diffs:
        i = diffs[0][0]
        cmp["first_difference_explained"] = regime_diff(en.model, A["theta"][i], ests)
    cmp["per_evaluation_seconds_interleaved"] = dict(occbin_min=min(tocc), occbin_median=float(np.median(tocc)),
                                                     news_min=min(tnews), news_median=float(np.median(tnews)),
                                                     ratio_of_medians=float(np.median(tocc) / np.median(tnews)))
    res["comparison"] = cmp
    return res


def stage_hessian(case, ests, res_mode, log):
    out = {}
    pts = {m: res_mode[m]["x"] for m in METHODS}
    for meth in METHODS:
        e = ests[meth]
        tr = Trace(e)
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H = e.hessian(np.asarray(pts[meth]))
        wall = time.perf_counter() - t
        tr.close()
        ok = bool(np.all(np.isfinite(H)) and np.all(np.linalg.eigvalsh((H + H.T) / 2) > 0))
        d = dict(hessian=H, evaluations=len(tr.rows), wall_s=wall, positive_definite=ok)
        if ok:
            Hi = e.inverse_hessian(H)
            d.update(inv_hessian=Hi, std=np.sqrt(np.diag(Hi)), laplace=e.laplace(np.asarray(pts[meth]), Hi),
                     log_posterior=-e.log_posterior(np.asarray(pts[meth])) * -1)
        out[meth] = d
        log(f"[{case}] hessian {meth}: {len(tr.rows)} evals, {wall:.1f}s, pd={ok}, laplace={d.get('laplace')}")
    # both at OccBin's mode
    x = np.asarray(pts["occbin"])
    for meth in METHODS:
        e = ests[meth]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H = e.hessian(x)
        out[f"{meth}_at_occbin_mode"] = dict(hessian=H, max_abs_diff_to_occbin=float(np.abs(H - out["occbin"]["hessian"]).max()))
    return out


def stage_mh(case, ests, mode_x, cov, jscale, nchains, replic, seed, log, starts):
    """Chains one at a time, the two solvers alternating chain by chain (so that machine load affects both alike)."""
    n = len(mode_x)
    res = {m: dict(draws=[], lp=[], acc=[], rows=[], wall=[]) for m in METHODS}
    for c in range(nchains):
        rng = np.random.default_rng([seed, c])
        innov = rng.standard_normal((1, replic, n))
        unif = rng.random((1, replic))
        for meth in (METHODS if c % 2 == 0 else METHODS[::-1]):
            e = ests[meth]
            tr = Trace(e)
            FLAGS.clear()
            t = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mc = e.sample(replic=replic, nblocks=1, mode=mode_x, cov=cov, jscale=jscale, x0=starts[c][None, :],
                              innovations=innov, uniforms=unif, drop=0.5)
            wall = time.perf_counter() - t
            tr.close()
            res[meth]["draws"].append(mc.draws[0]); res[meth]["lp"].append(mc.log_posterior[0])
            res[meth]["acc"].append(mc.accepted[0]); res[meth]["wall"].append(wall)
            res[meth]["rows"].append(dict(codes=np.array([r[2] for r in tr.rows]), lp=np.array([r[1] for r in tr.rows]),
                                          theta=np.array([r[0] for r in tr.rows]), time=np.array([r[3] for r in tr.rows]),
                                          flags=dict(FLAGS)))
            log(f"[{case}] chain {c} {meth}: acceptance {mc.acceptance_rate[0]:.3f}, wall {wall:.0f}s, "
                f"failed evaluations {int(np.sum(~np.isfinite([r[1] for r in tr.rows])))}")
    out = {}
    for meth in METHODS:
        D = np.array(res[meth]["draws"]); L = np.array(res[meth]["lp"]); A = np.array(res[meth]["acc"])
        kept = D[:, replic // 2:, :].reshape(-1, n)
        codes = collections.Counter(int(x) for r in res[meth]["rows"] for x in r["codes"] if x != 0)
        trial = collections.Counter()
        for r in res[meth]["rows"]:
            trial.update(r["flags"])
        nfail = int(sum(np.sum(~np.isfinite(r["lp"])) for r in res[meth]["rows"]))
        out[meth] = dict(acceptance_rate=A.mean(axis=1), wall_s=res[meth]["wall"], total_wall_s=float(sum(res[meth]["wall"])),
                         mean=kept.mean(axis=0), sd=kept.std(axis=0, ddof=1), n_kept=int(kept.shape[0]),
                         evaluations=int(sum(len(r["lp"]) for r in res[meth]["rows"])), rejected_by_error_code=dict(codes),
                         evaluations_with_infinite_lp=nfail, trial_point_failures_inside_filter=dict(trial),
                         median_seconds_per_evaluation=float(np.median(np.concatenate([r["time"] for r in res[meth]["rows"]]))))
        np.savez(OUT / f"est_{case}_mh_{meth}.npz", draws=D, lp=L, accepted=A)
    # draw-for-draw comparison
    Do, Dn = np.array(res["occbin"]["draws"]), np.array(res["news"]["draws"])
    Lo, Ln = np.array(res["occbin"]["lp"]), np.array(res["news"]["lp"])
    per_chain = []
    for c in range(nchains):
        dd = np.abs(Do[c] - Dn[c]).max(axis=1)
        first = int(np.argmax(dd > 1e-8)) if np.any(dd > 1e-8) else None
        ent = dict(chain=c, identical=first is None, first_different_draw=first,
                   max_abs_draw_difference=float(dd.max()), max_abs_lp_difference_where_same_draws=float(np.abs(Lo[c] - Ln[c])[dd <= 1e-8].max(initial=0.0)),
                   accepted_differs_at=int(np.argmax(res["occbin"]["acc"][c] != res["news"]["acc"][c])) if np.any(res["occbin"]["acc"][c] != res["news"]["acc"][c]) else None)
        # first evaluation whose likelihood differs between the two traces (the proposals are the same until then)
        ro, rn = res["occbin"]["rows"][c], res["news"]["rows"][c]
        k = min(len(ro["lp"]), len(rn["lp"]))
        same_th = np.abs(ro["theta"][:k] - rn["theta"][:k]).max(axis=1) < 1e-8
        d = np.abs(np.where(np.isinf(ro["lp"][:k]) & np.isinf(rn["lp"][:k]), 0.0, ro["lp"][:k] - rn["lp"][:k]))
        bad = np.flatnonzero(same_th & (d > 1e-8))
        ent["first_evaluation_with_different_likelihood"] = None if bad.size == 0 else dict(
            index=int(bad[0]), occbin=float(ro["lp"][bad[0]]), news=float(rn["lp"][bad[0]]), diff=float(d[bad[0]]),
            theta=ro["theta"][bad[0]], explained=regime_diff(None, ro["theta"][bad[0]], ests))
        ent["n_evaluations_with_different_likelihood_at_same_theta"] = int(bad.size)
        per_chain.append(ent)
    out["comparison"] = per_chain
    return out


def pilot_jscale(e, xm, cov, seed, log, case, grid=(0.25, 0.4, 0.6, 0.9, 1.3, 1.8, 2.4, 3.2), draws=300, target=0.30):
    """Scale of the jumping distribution: the grid value whose acceptance rate over a 300-draw pilot chain (news solver,
    which is the fast one; started at the mode) is closest to ``target``."""
    d = np.linalg.cholesky(cov).T
    rng = np.random.default_rng([seed, 77])
    innov, unif = rng.standard_normal((1, draws, len(xm))), rng.random((1, draws))
    best, rows = None, []
    for js in grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mc = e.sample(replic=draws, nblocks=1, mode=xm, cov=cov, jscale=js, x0=xm[None, :], innovations=innov,
                          uniforms=unif)
        ar = float(mc.acceptance_rate[0])
        rows.append((js, ar))
        log(f"[{case}] pilot jscale {js}: acceptance {ar:.3f}")
        if best is None or abs(ar - target) < abs(best[1] - target):
            best = (js, ar)
    return best[0], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("stages")
    ap.add_argument("--replic", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--start", default="prior", help="mode search start: prior (the initial values) or stored (first stored vector)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    _count_flags()
    ests = {}
    for meth in METHODS:
        m, ests[meth], X = load(a.case, meth)
    est = ests["occbin"]

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)

    x0 = np.array(est.x0) if a.start == "prior" else X[:, 0]
    tag = a.start
    stages = a.stages.split(",")
    fm = OUT / f"est_{a.case}_{tag}_mode.json"
    if "mode" in stages:
        res = stage_mode(a.case, ests, X, tag, x0, log)
        fm.write_text(json.dumps(jl(res), indent=1))
    res_mode = json.loads(fm.read_text())
    if "hessian" in stages:
        h = stage_hessian(a.case, ests, res_mode, log)
        (OUT / f"est_{a.case}_{tag}_hessian.json").write_text(json.dumps(jl(h), indent=1))
    if "mh" in stages:
        hs = json.loads((OUT / f"est_{a.case}_{tag}_hessian.json").read_text())
        xm = np.array(res_mode["occbin"]["x"])
        cov = np.array(hs["occbin"]["inv_hessian"])
        from pydyn import posterior as _post
        # same proposal for both: the inverse Hessian at OccBin's mode, scale tuned (news solver, fast) to about 1/3 acceptance
        tuned, pilot = pilot_jscale(ests["news"], xm, cov, a.seed, log, a.case)
        d = np.linalg.cholesky(cov).T
        lb, ub = est.prior.bounds(est.options["prior_trunc"])
        rng = np.random.default_rng([a.seed, 99])
        starts = []
        while len(starts) < a.chains:                       # Dynare's rule: draw around the mode until finite
            cand = xm + rng.standard_normal(est.n) @ (d * 2.0 * tuned)
            if np.all(cand >= lb) and np.all(cand <= ub) and np.isfinite(est.log_posterior(cand)):
                starts.append(cand)
        mh = stage_mh(a.case, ests, xm, cov, tuned, a.chains, a.replic, a.seed, log, starts)
        mh["settings"] = dict(replic=a.replic, chains=a.chains, seed=a.seed, jscale=tuned, pilot=pilot, starts=starts,
                              cov="inverse Hessian at OccBin's mode")
        (OUT / f"est_{a.case}_{tag}_mh.json").write_text(json.dumps(jl(mh), indent=1))
        log("mh done")


if __name__ == "__main__":
    main()
