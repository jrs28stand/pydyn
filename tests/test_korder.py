"""Second- and third-order decision rules via Dynare's C++ k-order library vs Dynare 7.2."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.korder import k_order, load_library

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))
EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
CASES = [
    # order 2: reference from Dynare's default (MATLAB) second-order solver
    (EXAMPLES / "stochastic_simulations" / "collard_2001_simulated_moments.mod", "clean_collard_order2.mat", 2),
    # order 3: reference from Dynare's k_order_solver
    (REF / "collard_o3.mod", "clean_collard_order3.mat", 3),
]

try:
    load_library()
    HAVE_LIB = True
except FileNotFoundError:
    HAVE_LIB = False
pytestmark = pytest.mark.skipif(not HAVE_LIB, reason="libpydyn_korder.so not built")


@pytest.mark.parametrize("mod,refname,order", CASES, ids=["order2", "order3"])
def test_higher_order_rules(mod, refname, order, tmp_path):
    shutil.copy(mod, tmp_path)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / mod.name))
    ref = loadmat(REF / refname, squeeze_me=True)
    res = k_order(m, order=order, ys=np.asarray(ref["ys"], float), params=np.asarray(ref["params"], float))
    expected = {2: ["ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2"],
                3: ["ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2", "ghxxx", "ghxxu", "ghxuu", "ghuuu",
                    "ghxss", "ghuss", "g_0", "g_1", "g_2", "g_3"]}[order]
    for name in expected:
        want = np.asarray(ref[name], float).reshape(res[name].shape)
        np.testing.assert_allclose(res[name], want, rtol=1e-9, atol=1e-12, err_msg=name)


def test_first_order_agrees_with_python_solver(tmp_path):
    mod = CASES[0][0]
    shutil.copy(mod, tmp_path)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / mod.name))
    sol = pydyn.first_order(m)
    res = k_order(m, order=1, ys=sol.ys)
    np.testing.assert_allclose(res["ghx"], sol.ghx, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res["ghu"], sol.ghu, rtol=1e-9, atol=1e-12)


def test_order_beyond_available_derivatives(tmp_path):
    mod = CASES[0][0]                      # .mod file asks for order 2 only
    shutil.copy(mod, tmp_path)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / mod.name))
    with pytest.raises(pydyn.korder.KOrderError):
        k_order(m, order=3)
