"""DSGE-VAR (dsge_var, dsge_varlag) against Dynare 7.2: reference/dvar/est_dvar.mod (reference/est_nk.mod without
measurement errors or correlated shocks); reference/dvar/est_dvar_v<V>.mat from reference/dvar/r_dsge_var.m:
dsge_var_likelihood at several parameter vectors and the sample moments; variant 1 also the posterior mode
(estimated weight, mode_compute=4). 0: dsge_var=0.8, dsge_varlag=2; 1: estimated weight, dsge_varlag=4,
nobs=100; 2: dsge_var=Inf, dsge_varlag=1."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
_models = {}


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    def _load(v):
        if v not in _models:
            d = tmp_path_factory.mktemp(f"dvar{v}")
            for f in ("dvar/est_dvar.mod", "est_nk_data.m"):
                shutil.copy(REF / f, d)
            _models[v] = pydyn.Model(pydyn.run_preprocessor(d / "est_dvar.mod", extra_args=(f"-DVARIANT={v}",)))
        return _models[v]
    return _load


def ref(v):
    return loadmat(REF / "dvar" / f"est_dvar_v{v}.mat", squeeze_me=True)


@pytest.mark.parametrize("v", [0, 1, 2])
def test_posterior_kernel_matches_dynare(load, v):
    r = ref(v)
    est = estimation.Estimation(load(v))
    assert est.names == r["names"].split(";")
    for k in range(r["X"].shape[1]):
        lp = est.log_posterior(r["X"][:, k])
        if r["info"][0, k]:
            assert lp == -np.inf and est.last_code == r["info"][0, k]
            np.testing.assert_allclose(est.last_penalty, r["info"][3, k], rtol=1e-12)
            continue
        # dsge_var=Inf: the likelihood is T * trace(inv(SIGMA_u*) (mYY - mYX PHI - (mYX PHI)' + PHI' mXX PHI)) / 2
        # with sums of cross products of the whole sample, which cancel to about 1e-4 of their size far
        # from the mode (point 2: 2.3e-12 relative difference), hence 5e-12 there
        np.testing.assert_allclose(-lp, r["fval"][k], rtol=5e-12 if v == 2 else 1e-12)
    mYY, mYX, mXX = est._dvar_moments
    for a, b in ((mYY, r["mYY"]), (mYX, r["mYX"]), (mXX, r["mXX"])):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-12)
    # Dynare's sample size includes the dsge_varlag initial observations
    assert est.nobs + int(est.options["dsge_varlag"]) == r["T"]


def test_posterior_mode_matches_dynare(load):
    """csminwel (mode_compute=4 from the .mod file) reaches Dynare's mode: measured 1.9e-8 in the objective
    (convergence criterion 1e-7) and 3e-5 in the parameters."""
    r = ref(1)
    est = estimation.Estimation(load(1))
    mode = est.mode()
    assert abs(est.log_posterior(mode.x) - r["logpost_mode"]) < 1e-7
    assert np.max(np.abs(mode.x - r["xmode"])) < 1e-4


def test_refusals(load):
    m = load(0)
    with pytest.raises(ValueError, match="presample"):
        estimation.Estimation(m, presample=2).log_likelihood(estimation.Estimation(m).x0)
    est = estimation.Estimation(m, first_obs=2)
    with pytest.raises(estimation.EstimationError, match="first_obs"):
        est.log_likelihood(est.x0)
    with pytest.raises(NotImplementedError, match="per-observation"):
        estimation.Estimation(m).log_likelihood_contributions(est.x0)
