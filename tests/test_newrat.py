"""newrat (mode_compute=5, pydyn.newrat) against Dynare 7.2 under Octave 8.4 (reference/r_newrat.m).

Evaluation-by-evaluation comparisons. newrat's path is decided, after a while, by rounding noise: mr_gstep.m takes the
curvature of every coordinate from the second difference f(x+h) + f(x-h) - 2 f(x) with steps h chosen so that
f(x+h) - f(x) is about 1e-5 (TolGstep); at the step sizes it ends up with (1e-9 relative on the test function) that
second difference is of the order of the rounding error of f, so last-bit differences in the objective or in the
linear algebra (Octave's and numpy's BLAS) change the step taken. The paths are therefore compared up to that point
(measured: 45 evaluations on the test function, 38 on post_nk, 29 on post_schorfheide), and the optima by the quality
newrat promises (no improvement larger than its tolerance, 1e-5).
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation, newrat

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
pytestmark = pytest.mark.filterwarnings("ignore:statements not handled", "ignore:matrix singular",
                                        "ignore:Estimation does not act")


def testfun():
    """The objective of nr_testfun.m (r_newrat.m), with the same floating-point operations."""
    X, F = [], []

    def f(x):
        x = np.asarray(x, float)
        X.append(x.copy())
        if x[3] <= 0.05:
            F.append(np.inf)
            return np.inf, 42, (0.05 - x[3]) * (0.05 - x[3]), None
        mu, sd, c0 = [0.5, 1, 0, 1], [2, 2, 2, 1], 0.0
        for j in range(4):
            d = (x[j] - mu[j]) / sd[j]
            c0 = c0 + 0.5 * d * d
        c, fval = np.zeros(60), c0
        for t in range(1, 61):
            s = t / 60
            y = 1 + 2 * s - 0.5 * s * s / (1 + 0.8 * s) + 0.1 * (((t * 7919) % 13) / 13 - 0.5)
            r = y - (x[0] + x[1] * s + x[2] * s * s / (1 + x[3] * x[3] * s))
            c[t - 1] = 0.5 * r * r / 0.01
            fval = fval + c[t - 1]
        F.append(fval)
        return fval, 0, 0.0, np.concatenate([[c0], c])
    return f, X, F


def prefix(X, F, DX, DF, rtol):
    """Number of leading evaluations at the same point (to rtol) with the same value."""
    n = min(len(F), DF.size)
    for k in range(n):
        same_x = np.allclose(X[k], DX[:, k], rtol=rtol, atol=1e-300)
        same_f = (np.isinf(F[k]) and np.isinf(DF[k])) or np.isclose(F[k], DF[k], rtol=rtol, atol=0)
        if not (same_x and same_f):
            return k
    return n


@pytest.mark.parametrize("v", [1, 2, 3])
def test_newrat_on_test_function_matches_dynare(v):
    """Three settings: default; robust with TolGstep 1e-6; MaxIter 3. The first evaluations (the step-size search of
    mr_hessian.m, the outer-product Hessian, the first line searches; a penalised point included) coincide."""
    T = loadmat(REF / "nr_testfun.mat", squeeze_me=True)
    p, st = f"s{v}_", T["settings"][v - 1]
    f, X, F = testfun()
    hi = newrat.default_hess_info(4, 1e-6, crit=1e-5, tol_gstep=st[1], robust=bool(st[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = newrat.newrat(f, T["x0"], T["bounds"], 1e-5, int(st[2]), 0, hi, T["prior_std"], 1e-6)
    k = prefix(X, F, T[p + "X"], T[p + "F"], 1e-11)
    assert k >= {1: 45, 2: 100, 3: 45}[v], k
    assert r["exitflag"] == T[p + "exitflag"]
    if v < 3:                                      # converged: both optima are as good, within newrat's tolerance
        assert abs(r["f"] - T[p + "fval"]) < 1e-5
    assert np.all(np.isfinite(r["x"])) and np.any(T[p + "info"][0] != 0)   # the run met the penalised region


@pytest.mark.parametrize("mod, n_same", [("post_nk", 38), ("post_schorfheide", 29)])
def test_newrat_on_dsge_likelihood_matches_dynare(tmp_path, mod, n_same):
    """mode_compute=5 with Dynare's default options from the initial values. post_nk: pydyn's optimum is as good as
    Dynare's (minus log posterior within 1e-5; pydyn 280.3025920, Dynare 280.3025950) and within 0.002 posterior
    standard deviations of it. post_schorfheide: the beta prior of rho has an unbounded density at 0 and both runs drive
    rho to its bound (Dynare's ends at 3e-49), so the optima are not compared."""
    R = loadmat(REF / f"nr_{mod}.mat", squeeze_me=True)
    sub = tmp_path / mod
    sub.mkdir()
    for f in (f"{mod}.mod", "est_nk_data.m", "est_schorfheide_data.m", f"{mod}_mode.mat"):
        shutil.copy(REF / f, sub)
    m = pydyn.Model(pydyn.run_preprocessor(sub / f"{mod}.mod", extra_args=("-DPOST=0",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimation.Estimation(m)
    assert est.names == R["names"].split(";")
    np.testing.assert_array_equal(est.x0, R["x0"])
    X, F = [], []
    base = newrat.estimation_objective(est)

    def f(x):
        out = base(x)
        X.append(np.array(x))
        F.append(out[0])
        return out
    o = est.options
    hi = newrat.default_hess_info(est.n, o["gradient_epsilon"], o["gstep"], crit=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = newrat.newrat(f, est.x0, np.column_stack([est.lb, est.ub]), 1e-5, 1000, 0, hi, est.prior.p2,
                          o["gradient_epsilon"])
    idx = R["Xidx"].astype(int) - 1
    DX = np.full((est.n, int(idx.max()) + 1), np.nan)
    DX[:, idx] = R["X"]
    k = prefix(X, F, DX[:, :n_same + 5], R["F"][:n_same + 5], 1e-10)
    assert k >= n_same, k
    assert r["exitflag"] == R["exitflag"] == 1
    i = est.names.index("rho") if "rho" in est.names else None
    if mod == "post_nk":
        assert abs(r["f"] - R["fh"]) < 1e-5
        sd = np.sqrt(np.diag(np.linalg.inv(loadmat(REF / f"{mod}_mode.mat", squeeze_me=True)["hh"])))
        assert np.max(np.abs(r["x"] - R["xh"]) / sd) < 0.002
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mode = est.mode(mode_compute=5)                    # through Estimation.mode
        np.testing.assert_allclose(mode.x, r["x"], rtol=0, atol=0)
        assert "newrat" in mode.message and np.all(np.isfinite(mode.std))
    else:
        assert r["x"][i] < 1e-6 and R["xh"][i] < 1e-6
