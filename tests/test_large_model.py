"""Smets-Wouters (2007) scale check: 40 variables, 20 states, roots at 0.9977 in the baseline and
pushed towards one in the stress cases; data with gaps at the start of the sample.

Cases (macro variable CASE): 1 baseline; 2 all shock persistences 0.9999; 3 three persistences
0.99999; 4 two persistences 0.9999995, which Dynare treats as unit roots (cut-off 1 - 1e-6).
References: Dynare 7.2 under Octave (reference/r_sw.m)."""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import moments, smoother

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("sw")
    for f in ("sw_big.mod", "sw_data.m"):
        shutil.copy(REF / f, d)
    return d


def load(work, case, algo=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(work / "sw_big.mod", extra_args=(f"-DCASE={case}", f"-DALGO={algo}")))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float).reshape(np.shape(a))
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = np.isfinite(b)
    return 0.0 if not ok.any() else np.max(np.abs(a[ok] - b[ok])) / np.max(np.abs(b[ok]))


@pytest.mark.parametrize("case", [1, 2, 3, 4])
def test_rules_irfs_and_moments(work, case):
    m, r = load(work, case), loadmat(REF / f"sw_ref_case{case}.mat", squeeze_me=True)
    assert m.n_endo == 40 and m.n_exo == 7
    sol = pydyn.first_order(m)
    assert len(sol.state_var) == 20
    assert rel(sol.ghx, r["ghx"]) < 1e-10 and rel(sol.ghu, r["ghu"]) < 1e-10
    eig = np.abs(sol.eigenvalues[np.isfinite(sol.eigenvalues)])
    np.testing.assert_allclose(eig[eig < 1.000001].max(), [0.9977, 0.9999, 0.99999, 0.9999995][case - 1], rtol=1e-9)
    chol = np.linalg.cholesky(m.Sigma_e)
    for label, series in zip(r["irf_names"].split(","), np.atleast_2d(r["irfs"])):
        shock = next(x for x in sorted(m.exo_names, key=len, reverse=True) if label.endswith("_" + x))
        path = sol.irf(chol[:, m.exo_names.index(shock)], periods=len(series))
        np.testing.assert_allclose(path[:, m.endo_names.index(label[: -len(shock) - 1])], series, rtol=1e-9, atol=1e-11)
    mo = moments.stoch_simul_moments(m, m.endo_names, sol.as_dr(), order=1)
    tol = 1e-10 if case in (1, 4) else 1e-7          # the Lyapunov equation is ill-conditioned near a unit root
    assert rel(mo.var, r["var"]) < tol
    assert rel(np.hstack(mo.autocorr), r["autocorr"]) < tol
    assert rel(mo.variance_decomposition, r["vd"]) < tol
    assert rel(mo.conditional_variance_decomposition, r["cvd"]) < 1e-10
    assert int((~mo.stationary).sum()) == (33 if case == 4 else 0)      # as Dynare: NaN moments beyond the cut-off


@pytest.mark.parametrize("case,tol", [(1, 1e-9), (2, 1e-6), (3, 1e-6)])
def test_smoother_with_gaps_at_the_start(work, case, tol):
    """Stationary initialisation; two observables missing for the first 32 periods."""
    m, r = load(work, case), loadmat(REF / f"sw_ref_case{case}.mat", squeeze_me=True)
    res = smoother.smooth(m)
    sm = np.array([res.SmoothedVariables[v] for v in r["sm_names"].split(",")])
    sh = np.array([res.SmoothedShocks[v] for v in r["sh_names"].split(",")])
    assert rel(sm, r["smoothed"]) < tol and rel(sh, r["shocks"]) < tol


@pytest.mark.parametrize("case,atol", [(2, 1e-5), (3, 1e-5), (4, 1e-8)])
def test_diffuse_smoother_matches_both_dynare_algorithms(work, case, atol):
    """With diffuse_filter. Roots below Dynare's cut-off are initialised as stationary (cases 2, 3);
    in case 4 Dynare switches to the univariate filter by itself. Smoothed values are of order 100."""
    m = load(work, case, algo=3)
    res = smoother.smooth(m, diffuse_filter=True, kalman_algo=3)
    for algo in (3, 4):
        r = loadmat(REF / f"sw_ref_diffuse_case{case}_algo{algo}.mat", squeeze_me=True)
        sm = np.array([res.SmoothedVariables[v] for v in r["sm_names"].split(",")])
        np.testing.assert_allclose(sm, r["smoothed"], rtol=0, atol=atol)
