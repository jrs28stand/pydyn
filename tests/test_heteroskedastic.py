"""heteroskedastic_shocks with the heteroskedastic_filter option against Dynare 7.2 (reference/het/est_het.mod,
reference/est_nk.mod with a heteroskedastic_shocks block; reference/het/est_het_v<V>.mat from
reference/het/r_heteroskedastic.m: dsge_likelihood at five parameter vectors, one of them with a diagonal
Q, and the smoother at the initial values)."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation, heteroskedastic, smoother

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
# 0: complete data, 1: first_obs=11, nobs=80, presample=4 and an extra values block, 2: missing observations,
# 4: univariate filter (kalman_algo=2), 5: lik_init=2, 6/7: smoother (complete / missing data)
VARIANTS = [0, 1, 2, 4, 5, 6, 7]
_models = {}


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    def _load(v):
        if v not in _models:
            d = tmp_path_factory.mktemp(f"het{v}")
            for f in ("het/est_het.mod", "est_nk_data.m", "est_nk_miss.m"):
                shutil.copy(REF / f, d)
            _models[v] = pydyn.Model(pydyn.run_preprocessor(d / "est_het.mod", extra_args=(f"-DVARIANT={v}",)))
        return _models[v]
    return _load


def ref(v):
    return loadmat(REF / "het" / f"est_het_v{v}.mat", squeeze_me=True)


@pytest.mark.parametrize("v", VARIANTS)
def test_likelihood_matches_dynare(load, v):
    m, r = load(v), ref(v)
    est = estimation.Estimation(m)
    assert est.options["heteroskedastic_filter"]
    Qvalue, Qscale = heteroskedastic.q_arrays(m, est.nobs, int(est.options["first_obs"]))
    np.testing.assert_array_equal(np.isnan(Qvalue), np.isnan(r["Qvalue"]))
    np.testing.assert_array_equal(np.isnan(Qscale), np.isnan(r["Qscale"]))
    np.testing.assert_allclose(Qvalue[~np.isnan(Qvalue)], r["Qvalue"][~np.isnan(r["Qvalue"])], rtol=1e-15)
    np.testing.assert_allclose(Qscale[~np.isnan(Qscale)], r["Qscale"][~np.isnan(r["Qscale"])], rtol=1e-15)
    X = r["X"]
    for k in range(X.shape[1]):
        ll = est.log_likelihood(X[:, k])
        if r["info"][k] != 0:          # 41: outside the prior bounds, a posterior (not likelihood) matter
            assert r["info"][k] == 41
            continue
        np.testing.assert_allclose(ll, r["loglik"][k], rtol=1e-12, err_msg=f"point {k}")
    # without the option the blocks are ignored, as in Dynare
    plain = estimation.Estimation(m, heteroskedastic_filter=False).log_likelihood(X[:, 0])
    np.testing.assert_allclose(plain, -r["fval_plain"] - r["lnprior"][0], rtol=1e-12)
    assert abs(plain - r["loglik"][0]) > 100


@pytest.mark.parametrize("v", [6, 7])
def test_smoother_matches_dynare(load, v):
    m, r = load(v), ref(v)
    est = estimation.Estimation(m)
    res = smoother.smooth(m, estimation=est, theta=est.x0)
    for i, name in enumerate(m.endo_names[:r["smoothed"].shape[1]]):
        np.testing.assert_allclose(res.SmoothedVariables[name], r["smoothed"][:, i], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(res.UpdatedVariables[name], r["updated"][:, i], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(res.shocks, r["shocks"], rtol=1e-9, atol=1e-10)
    plain = smoother.smooth(m, estimation=est, theta=est.x0, heteroskedastic_filter=False)
    assert np.max(np.abs(plain.shocks - r["shocks"])) > 0.1
    with pytest.raises(NotImplementedError, match="heteroskedastic"):
        est.smoother(est.x0)
    with pytest.raises(NotImplementedError, match="multivariate smoother"):
        smoother.smooth(m, estimation=est, theta=est.x0, kalman_algo=2)


def test_qvec_follows_dynare_rules():
    """Values replace a variance, scales multiply the standard deviation; with a non-diagonal Q the
    correlations are kept."""
    Q = np.array([[4.0, 1.0], [1.0, 9.0]])
    Qvalue = np.array([[np.nan, 1.0], [np.nan, np.nan]])
    Qscale = np.array([[np.nan, np.nan], [0.25, 4.0]])
    Qv = heteroskedastic.qvec(Q, Qvalue, Qscale)
    np.testing.assert_allclose(Qv[:, :, 0], [[4.0, 0.5], [0.5, 2.25]])
    np.testing.assert_allclose(Qv[:, :, 1], [[1.0, 1.0], [1.0, 36.0]])
    Qd = heteroskedastic.qvec(np.diag([4.0, 9.0]), Qvalue, Qscale)
    np.testing.assert_allclose(Qd[:, :, 1], np.diag([1.0, 36.0]))
