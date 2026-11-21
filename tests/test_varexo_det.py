"""varexo_det in stochastic models against Dynare 7.2 (reference/xdet: Dynare's tests/forecast/
example1_varexo_det.mod with a third deterministic period, orders 1 and 2; reference/xdet/xdet_o<k>.mat
from reference/xdet/r_varexo_det.m: decision rules of stochastic_solvers.m and a simult_varexo_det.m path)."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import korder, moments, varexo_det

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference"))) / "xdet"

try:
    pydyn.korder.load_library()
    HAVE_LIB = True
except FileNotFoundError:
    HAVE_LIB = False


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("xdet")
    out = {}
    for o in (1, 2):
        shutil.copy(REF / f"xdet_o{o}.mod", d)
        out[o] = (pydyn.Model(pydyn.run_preprocessor(d / f"xdet_o{o}.mod")), loadmat(REF / f"xdet_o{o}.mat", squeeze_me=True))
    return out


def close(a, b, what, tol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.max(np.abs(a - b)) <= tol * max(1.0, np.max(np.abs(b))), what


@pytest.mark.parametrize("order", [1, 2])
def test_rules_and_simulation_match_dynare(models, order):
    """At Dynare's steady state (its solver stops at a residual of 4e-6 from initval k=5, so pydyn's more
    accurate steady state differs by 6e-5 and the rules accordingly) every term agrees to rounding."""
    if order == 2 and not HAVE_LIB:
        pytest.skip("k-order library not available")
    m, r = models[order]
    dr = pydyn.first_order(m, ys=r["ys"]).as_dr() if order == 1 else korder.k_order(m, order=2, ys=r["ys"])
    assert varexo_det.exo_det_length(m) == r["exo_det_length"] == 3
    close(m.exo_det_steady(), np.atleast_1d(r["exo_det_steady_state"]), "exo_det_steady_state")
    d = varexo_det.decision_rules(m, dr)
    close(d["ghx"], r["ghx"], "ghx")
    for j in range(3):
        close(d["ghud"][j], r[f"ghud_{j + 1}"], f"ghud{{{j + 1}}}")
        if order == 2:
            close(d["ghxud"][j], r[f"ghxud_{j + 1}"], f"ghxud{{{j + 1}}}")
            close(d["ghuud"][j], r[f"ghuud_{j + 1}"], f"ghuud{{{j + 1}}}")
            for k in range(j + 1):
                close(d["ghudud"][(k, j)], r[f"ghudud_{k + 1}_{j + 1}"], f"ghudud{{{k + 1},{j + 1}}}")
    Y = varexo_det.simulate(m, d, r["sim_ex"], r["sim_ex_det"], y0=r["sim_y0"])
    close(Y.T, r["sim"], "simult_varexo_det path")


def test_own_steady_state_and_stochastic_view(models):
    """pydyn's own solution (steady state to 1e-14): the stochastic view of the model has no varexo_det
    columns, so ghu, moments, IRFs etc. work unchanged; the varexo_det terms match Dynare's to the accuracy
    of Dynare's steady state."""
    m, r = models[1]
    dr = moments.solve(m, 1)
    assert dr["ghu"].shape == (m.n_endo, m.n_exo) and m.jac_dims[1] == 3 * m.n_endo + m.n_exo
    d = varexo_det.decision_rules(m, dr)
    for j in range(3):
        close(d["ghud"][j], r[f"ghud_{j + 1}"], "ghud", tol=1e-4)
    mo = moments.theoretical_moments(m, dr, order=1)
    assert np.all(np.isfinite(mo.var))


def test_unsupported_cases_are_refused(models):
    m, r = models[1]
    with pytest.raises(NotImplementedError, match="varexo_det"):
        pydyn.perfect_foresight(m)
    if HAVE_LIB:
        m2, _ = models[2]
        with pytest.raises(NotImplementedError, match="order 3"):
            varexo_det.decision_rules(m2, dict(korder.k_order(m2, order=2), order=3))
