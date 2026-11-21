"""The loglinear option of stoch_simul, against Dynare 7.2 (reference/r_loglinear.m).

The model has an endogenous auxiliary variable (a lag of two on capital: logged by Dynare) and
auxiliary variables for leads and lags of exogenous variables (kept in levels by Dynare)."""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import moments

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))


def load(tmp_path, order=1, edit=None):
    text = (REF / "ll_rbc.mod").read_text()
    if edit:
        text = edit(text)
    (tmp_path / "ll_rbc.mod").write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(tmp_path / "ll_rbc.mod", extra_args=(f"-DORDER={order}",)))


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    m = load(tmp_path_factory.mktemp("ll"))
    return m, pydyn.first_order(m), loadmat(REF / "clean_loglinear.mat", squeeze_me=True)


def test_mask_follows_dynare(case):
    m, _, r = case
    assert m.endo_names == r["endo_names"].split(",")
    mask = pydyn.perturbation.log_mask(m)
    types = np.atleast_1d(r["aux_types"])
    expected = np.r_[np.ones(len(m.orig_endo_names), bool), ~np.isin(types, (2, 3))]
    np.testing.assert_array_equal(mask, expected)


def test_rules_and_steady_state_in_logs(case):
    m, sol, r = case
    assert sol.loglinear
    np.testing.assert_allclose(sol.ys, r["ys"], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(sol.order_var + 1, r["order_var"])
    np.testing.assert_allclose(sol.ghx, r["ghx"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sol.ghu, r["ghu"], rtol=1e-9, atol=1e-12)
    assert np.all(sol.steady_state[:5] > 0) and not np.allclose(sol.steady_state, sol.ys)


def test_irfs_and_simulation(case):
    m, sol, r = case
    chol = np.linalg.cholesky(m.Sigma_e)
    for label, series in zip(r["irf_names"].split(","), np.atleast_2d(r["irfs"])):
        shock = next(x for x in m.exo_names if label.endswith("_" + x))
        path = pydyn.irf(sol, chol[:, m.exo_names.index(shock)], periods=len(series))
        np.testing.assert_allclose(path[:, m.endo_names.index(label[: -len(shock) - 1])], series, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(pydyn.simulate(sol, r["ex"]), r["sim"][1:], rtol=1e-10, atol=1e-12)


def test_moments(case):
    m, sol, r = case
    mo = moments.stoch_simul_moments(m, m.orig_endo_names, sol.as_dr(), order=1, ar=3,
                                     conditional_variance_decomposition=[1, 4])
    np.testing.assert_allclose(mo.mean, r["mean"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(mo.var, r["var"], rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(np.hstack(mo.autocorr), r["autocorr"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(mo.variance_decomposition, r["vd"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(mo.conditional_variance_decomposition, r["cvd"], rtol=1e-9, atol=1e-10)


def test_levels_on_request_and_k_order(case, tmp_path):
    m, sol, r = case
    lev = pydyn.first_order(m, loglinear=False)
    assert not lev.loglinear
    np.testing.assert_allclose(lev.ys, sol.steady_state)
    try:
        pydyn.korder.load_library()
    except FileNotFoundError:
        pytest.skip("libpydyn_korder.so not built")
    dr = pydyn.k_order(m, order=1)
    np.testing.assert_allclose(dr["ghx"], r["ghx"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(dr["ys"], r["ys"], rtol=1e-12, atol=1e-14)
    m2 = load(tmp_path, order=2)
    with pytest.raises(pydyn.korder.KOrderError, match="only works at order 1"):    # as Dynare
        pydyn.k_order(m2)
    pydyn.k_order(m2, loglinear=False)


def test_non_positive_steady_state_is_refused(tmp_path):
    def add_negative_variable(t):
        t = t.replace("var c k y z inv;", "var c k y z inv nw;")
        t = t.replace("z = 1 - rho", "nw = -c;\nz = 1 - rho")
        return t.replace("c = y - inv;\nend;", "c = y - inv;\nnw = -c;\nend;")
    m = load(tmp_path, edit=add_negative_variable)
    with pytest.raises(pydyn.perturbation.LoglinearError, match="not strictly positive: nw"):
        pydyn.first_order(m)
    pydyn.first_order(m, loglinear=False)
