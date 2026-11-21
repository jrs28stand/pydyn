"""endogenous_prior (Christiano, Trabandt and Walentin 2011) against Dynare 7.2: reference/endo/est_endo.mod
(reference/est_nk.mod plus variant 20, estimation(endogenous_prior, mode_compute=4)); reference/endo/
est_endo_v<V>.mat from reference/endo/r_endogenous_prior.m: dsge_likelihood with and without the option at
four parameter vectors (complete data, first_obs/nobs/presample, missing observations, correlated measurement
errors), and the posterior mode of variant 20."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation
from pydyn.endogenous_prior import lnpriormom

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
_models = {}


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    def _load(v):
        if v not in _models:
            d = tmp_path_factory.mktemp(f"endo{v}")
            for f in ("endo/est_endo.mod", "est_nk_data.m", "est_nk_miss.m"):
                shutil.copy(REF / f, d)
            _models[v] = pydyn.Model(pydyn.run_preprocessor(d / "est_endo.mod", extra_args=(f"-DVARIANT={v}",)))
        return _models[v]
    return _load


@pytest.mark.parametrize("v", [0, 1, 2, 9])
def test_posterior_kernel_matches_dynare(load, v):
    r = loadmat(REF / "endo" / f"est_endo_v{v}.mat", squeeze_me=True)
    m = load(v)
    est = estimation.Estimation(m, endogenous_prior=True)
    plain = estimation.Estimation(m)
    for k in range(r["X"].shape[1]):
        x = r["X"][:, k]
        assert r["info"][k] == 0
        np.testing.assert_allclose(-est.log_posterior(x), r["fval"][k], rtol=1e-12)
        np.testing.assert_allclose(-plain.log_posterior(x), r["fval_plain"][k], rtol=1e-12)
        assert abs(r["fval"][k] - r["fval_plain"][k]) > 1          # the endogenous prior matters here


def test_posterior_mode_matches_dynare(load):
    """endogenous_prior and mode_compute=4 read from the .mod file. csminwel follows Dynare's iterates up to
    rounding (see tests/test_estimation.py::test_csminwel_follows_dynare_on_the_posterior); the optima agree
    within the convergence criterion (1e-7) in the objective. Measured: 8e-10 and 1.2e-5 in the parameters."""
    r = loadmat(REF / "endo" / "est_endo_v20.mat", squeeze_me=True)
    est = estimation.Estimation(load(20))
    assert est.options["endogenous_prior"] and est.options["mode_compute"] == 4
    np.testing.assert_allclose(-est.log_posterior(r["X"][:, 0]), r["fval"][0], rtol=1e-12)
    mode = est.mode()
    assert abs(est.log_posterior(mode.x) - r["logpost_mode"]) < 1e-7
    assert np.max(np.abs(mode.x - r["xmode"])) < 1e-4


def test_formula_and_refusals(load):
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((2, 50)) * np.array([[1.0], [2.0]])
    P = np.diag([1.0, 4.0, 0.5])
    v = lnpriormom(Y, P, [0, 1], np.zeros((2, 2)))
    assert np.isfinite(v)
    assert lnpriormom(Y, P, [0, 1], np.zeros((2, 2))) > lnpriormom(Y, 4 * P, [0, 1], np.zeros((2, 2)))
    est = estimation.Estimation(load(0), endogenous_prior=True, lik_init=2)
    with pytest.raises(estimation.EstimationError, match="non-stationary"):
        est.log_posterior(est.x0)
