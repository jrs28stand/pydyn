"""Compare pydyn first-order solutions with saved Dynare 7.2 (Octave) results."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))
EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
CASES = {
    "collard_2001_simulated_moments": EXAMPLES / "stochastic_simulations",
    "rbc_occbin": EXAMPLES / "occbin",
    # the estimation example with the estimation commands replaced by stoch_simul(order=1)
    "schorfheide_2000_sim": REF,
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, tmp_path_factory):
    name = request.param
    work = tmp_path_factory.mktemp(name)
    shutil.copy(CASES[name] / f"{name}.mod", work)
    jd = pydyn.run_preprocessor(work / f"{name}.mod")
    model = pydyn.Model(jd)
    ref = loadmat(REF / f"clean_{name}.mat", squeeze_me=True)
    return name, model, ref


ESTIMATED = set()


def test_symbols_and_calibration(case):
    name, m, ref = case
    assert m.endo_names == ref["endo_names"].split(",")
    assert m.exo_names == ref["exo_names"].split(",")
    if name not in ESTIMATED:
        np.testing.assert_allclose(m.params, ref["params"], rtol=1e-13)
        np.testing.assert_allclose(m.Sigma_e, np.atleast_2d(ref["Sigma_e"]), rtol=1e-12, atol=0)


def test_steady_state(case):
    """Dynare stops its nonlinear solver at a looser tolerance than we do, so compare
    loosely and require that our residual is no worse than Dynare's."""
    _, m, ref = case
    ys, params = pydyn.steady_state(m, params=ref["params"])
    np.testing.assert_allclose(ys, ref["ys"], rtol=1e-4, atol=1e-6)
    ours = np.max(np.abs(m.static_resid(ys, params=params)))
    theirs = np.max(np.abs(m.static_resid(ref["ys"], params=ref["params"])))
    assert ours <= max(theirs, 1e-12)


def test_decision_rules(case):
    """Decision rules at Dynare's own steady state and parameters must match tightly."""
    name, m, ref = case
    sol = pydyn.first_order(m, ys=np.asarray(ref["ys"], float), params=np.asarray(ref["params"], float))
    np.testing.assert_array_equal(sol.order_var + 1, ref["order_var"])
    ghx, ghu = sol.ghx, sol.ghu
    ref_ghx, ref_ghu = np.atleast_2d(ref["ghx"]), np.asarray(ref["ghu"]).reshape(ghu.shape)
    print(name, "max |d ghx| =", np.max(np.abs(ghx - ref_ghx)), " max |d ghu| =", np.max(np.abs(ghu - ref_ghu)))
    np.testing.assert_allclose(ghx, ref_ghx, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(ghu, ref_ghu, rtol=1e-9, atol=1e-11)


def test_irfs(case):
    """Dynare's IRFs: responses to a one-standard-deviation (Cholesky) shock."""
    name, m, ref = case
    if "irfs" not in ref:
        pytest.skip("no IRFs in reference")
    sol = pydyn.first_order(m, ys=np.asarray(ref["ys"], float), params=np.asarray(ref["params"], float))
    chol = np.linalg.cholesky(m.Sigma_e)
    names, irfs = ref["irf_names"].split(","), np.atleast_2d(ref["irfs"])
    worst = 0.0
    for label, series in zip(names, irfs):
        shock = next(x for x in sorted(m.exo_names, key=len, reverse=True) if label.endswith("_" + x))
        var = label[: -len(shock) - 1]
        path = sol.irf(chol[:, m.exo_names.index(shock)], periods=len(series))[:, m.endo_names.index(var)]
        worst = max(worst, np.max(np.abs(path - series)))
        np.testing.assert_allclose(path, series, rtol=1e-9, atol=1e-12)
    print(name, f"{len(names)} IRFs, max |diff| = {worst:.2e}")
