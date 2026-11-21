"""Model-local variables (#) and auxiliary variables for leads/lags beyond one period."""
import os
import warnings
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))


def load(name, tmp, **kw):
    shutil.copy(REF / f"{name}.mod", tmp)
    return pydyn.Model(pydyn.run_preprocessor(tmp / f"{name}.mod"), **kw)


@pytest.fixture(scope="module")
def a(tmp_path_factory):
    return load("aux_a", tmp_path_factory.mktemp("aux_a")), loadmat(REF / "clean_aux_a.mat", squeeze_me=True)


def test_auxiliary_variables_created_as_in_dynare(a):
    m, r = a
    assert m.endo_names == r["endo_names"].split(",")
    assert m.orig_endo_names == ["c", "k", "a", "r", "z"]
    kinds = {n.rsplit("_", 2)[0] if "LAG" in n else n.rsplit("_", 1)[0] for n in m.aux_names}
    assert kinds == {"AUX_ENDO_LEAD", "AUX_EXO_LEAD", "AUX_ENDO_LAG", "AUX_EXO_LAG"}


def test_steady_state_fills_auxiliary_variables(a):
    m, r = a
    ys, _ = pydyn.steady_state(m)
    np.testing.assert_allclose(ys, r["ys"], rtol=1e-10, atol=1e-12)


def test_first_order_rules_and_irfs(a):
    m, r = a
    sol = pydyn.first_order(m)
    np.testing.assert_array_equal(sol.order_var + 1, r["order_var"])
    np.testing.assert_allclose(sol.ghx, r["ghx"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(sol.ghu, r["ghu"], rtol=1e-9, atol=1e-11)
    chol = np.linalg.cholesky(m.Sigma_e)
    for i in range(m.n_exo):                       # first-order IRFs from Dynare's irf.m
        np.testing.assert_allclose(pydyn.irf(sol, chol[:, i], periods=12), r["irf1"][:, :, i],
                                   rtol=1e-8, atol=1e-12)


def test_second_order_rules_and_pruned_simulation(a):
    m, r = a
    try:
        dr = pydyn.k_order(m, order=2)
    except FileNotFoundError:
        pytest.skip("libpydyn_korder.so not built")
    for name in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2"):
        np.testing.assert_allclose(dr[name], np.asarray(r[name], float).reshape(dr[name].shape),
                                   rtol=1e-8, atol=1e-10, err_msg=name)
    np.testing.assert_allclose(pydyn.simulate(dr, r["ex"], order=1), r["y1"][1:], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(pydyn.simulate(dr, r["ex"], order=2, pruning=True), r["y2_pruned"][1:],
                               rtol=1e-9, atol=1e-11)


def test_local_variables_equal_hand_expansion(a, tmp_path):
    """The same model with the # definitions substituted by hand gives identical results."""
    m, _ = a
    b = load("aux_b", tmp_path)
    assert b.endo_names[:5] == m.endo_names[:5]
    sa, sb = pydyn.first_order(m), pydyn.first_order(b)
    np.testing.assert_allclose(sa.ys, sb.ys, rtol=1e-13)
    np.testing.assert_allclose(sa.G, sb.G, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sa.H, sb.H, rtol=1e-10, atol=1e-12)


def test_steady_state_function_need_not_set_auxiliary_variables(a, tmp_path):
    m, r = a

    def fn(model, params, exo):
        p = dict(zip(model.param_names, params))
        rr = 1 / p["beta"] - 1 + p["delta"]
        k = (rr / p["alpha"]) ** (1 / (p["alpha"] - 1))
        c = k ** p["alpha"] - p["delta"] * k
        return np.array([c, k, 0.0, rr, 0.1 * k])          # the five declared variables only

    b = load("aux_b", tmp_path, steady_state_fn=fn)
    ys, _ = pydyn.steady_state(b)
    np.testing.assert_allclose(ys, r["ys"], rtol=1e-10, atol=1e-12)


def test_numerical_steady_state_with_auxiliary_variables(a):
    m, r = a
    saved, m.steady_state_model = m.steady_state_model, None
    try:
        guess = np.where(np.arange(m.n_endo) < 5, np.r_[r["ys"][:5] * 1.05, np.zeros(m.n_endo - 5)], np.nan)
        ys, _ = pydyn.steady_state(m, guess=pydyn.steady.fill_auxiliary(m, guess, m.exo_steady(), m.params))
    finally:
        m.steady_state_model = saved
    np.testing.assert_allclose(ys, r["ys"], rtol=1e-8, atol=1e-10)


def test_perfect_foresight_example(tmp_path):
    """Dynare's perfect_foresight_rbc example: a one-period anticipated productivity shock."""
    examples = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
    shutil.copy(examples / "perfect_foresight" / "perfect_foresight_rbc.mod", tmp_path)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "perfect_foresight_rbc.mod"))
    assert m.aux_names                                  # z(+1) on an exogenous variable
    r = loadmat(REF / "clean_perfect_foresight_rbc.mat", squeeze_me=True)
    pf = pydyn.perfect_foresight(m)
    np.testing.assert_allclose(pf.exo[:, 0], r["exo_simul"], rtol=1e-12)
    np.testing.assert_allclose(pf.endo.T, r["endo_simul"], rtol=1e-6, atol=1e-7)   # Dynare stops at 1e-5
    assert pf.max_residual < 1e-9


# ------------------------------------------------------------------ var(log): LOG_x auxiliary variables
def test_var_log_auxiliary_variables_start_at_the_log(tmp_path):
    """var(log) x adds LOG_x with the equation x = exp(LOG_x), which is not linear in LOG_x: like
    Dynare's set_auxiliary_variables (LOG_x = log(x)) fill_auxiliary takes the log, where its two
    Newton steps gave LOG_k = 10.76 instead of log(12.76) = 2.55 on tests/log_transform/ramst.mod.
    That wrong start made the perfect-foresight 'steady' (pydyn's port of solve_algo=4) fail on ramst
    and the steady state of example1 fail. Dynare 7.2 references: reference/r_logtransform.m. Measured:
    ramst steady state 0 and paths 7e-15; example1 ys 0, ghx 6e-14; fs2000_nonstationary
    (var(log, deflator=A)) ys 5e-14, ghx 5e-14."""
    r = loadmat(REF / "clean_log_transform.mat", squeeze_me=True)
    m = load("log_ramst", tmp_path)
    assert m.endo_names == r["ramst_names"].split(",")
    y0 = np.array([m.initval.get(v, 0.0) for v in m.endo_names])
    y = pydyn.steady.fill_auxiliary(m, y0, m.exo_steady(), m.params)
    np.testing.assert_array_equal(y[m.endo_names.index("LOG_k")], np.log(y0[m.endo_names.index("k")]))
    np.testing.assert_array_equal(y[m.endo_names.index("LOG_c")], np.log(y0[m.endo_names.index("c")]))
    ys, _ = pydyn.steady_state(m)
    np.testing.assert_allclose(ys, r["ramst_ys"], rtol=1e-14, atol=1e-15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pydyn.perfect_foresight(m)
    assert res.status and int(r["ramst_status"]) == 1 and res.iterations == int(r["ramst_iterations"])
    np.testing.assert_allclose(res.endo.T, r["ramst_endo"], rtol=1e-12, atol=1e-13)
    for name, k in (("log_example1", "ex1"), ("log_fs2000ns", "fs")):
        d = tmp_path / name
        d.mkdir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = load(name, d)
            sol = pydyn.first_order(m)
        assert m.endo_names == r[k + "_names"].split(",")
        np.testing.assert_array_equal(sol.order_var + 1, r[k + "_order_var"])
        np.testing.assert_allclose(sol.ys, r[k + "_ys"], rtol=1e-12, atol=1e-14, err_msg=name)
        np.testing.assert_allclose(sol.ghx, r[k + "_ghx"], rtol=1e-10, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(sol.ghu, r[k + "_ghu"], rtol=1e-10, atol=1e-12, err_msg=name)
