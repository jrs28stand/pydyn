"""external_function in the model block, against Dynare 7.2 (reference/ext_fn: the models of Dynare's
tests/external_function without their final checks, function files copied unchanged; reference/ext_fn/
ext_<case>.mat produced by reference/ext_fn/r_external.m)."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import korder, moments

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference"))) / "ext_fn"
ANALYTIC = ["first_and_2nd_deriv_given", "first_and_2nd_deriv_given_namespace"]
NUMERICAL = ["no_deriv_given", "first_deriv_given", "extfun_in_mlv"]   # Hessian (and Jacobian) by finite differences
KEYS = ["ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghs2"]

try:
    pydyn.korder.load_library()
    HAVE_LIB = True
except FileNotFoundError:
    HAVE_LIB = False
pytestmark = pytest.mark.skipif(not HAVE_LIB, reason="libpydyn_korder.so not built")


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    d = tmp_path_factory.mktemp("ext")
    for p in REF.iterdir():
        if p.suffix in (".mod", ".m"):
            shutil.copy(p, d)
    shutil.copytree(REF / "+matlab", d / "+matlab")
    return d


def load(folder, case, **kw):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # addpath(pwd) in the namespace example
        return pydyn.Model(pydyn.run_preprocessor(folder / f"{case}.mod"), **kw)


def ref(case):
    return loadmat(REF / f"ext_{case}.mat", squeeze_me=True)


def maxdiff(a, b):
    return np.max(np.abs(np.ravel(a) - np.ravel(b)))


@pytest.mark.parametrize("case", ANALYTIC + NUMERICAL)
def test_derivatives_match_dynare_at_dynares_steady_state(folder, case):
    """Same steady state as Dynare: the decision rules agree to rounding, also where Dynare
    differentiates the external function numerically (jacob_element, hess_element reproduced)."""
    m, r = load(folder, case), ref(case)
    dr = korder.k_order(m, order=2, ys=r["ys"])
    for k in KEYS:
        assert maxdiff(dr[k], r[k]) < 1e-12 * max(1.0, np.max(np.abs(r[k]))), k


@pytest.mark.parametrize("case", ["benchmark"] + ANALYTIC + NUMERICAL)
def test_full_solution_matches_dynare(folder, case):
    """pydyn's own steady state, decision rules and theoretical moments.

    With analytic derivatives everything agrees to 1e-9. hess_element uses a step of 1e-6 for a
    second difference, so its result moves by about eps * |f| / h^2 ~ 1e-3 when the evaluation point
    moves by one unit in the last place; pydyn's steady state differs from Dynare's by ~4e-11, hence the
    second-order terms are compared with the tolerances Dynare's own tests use against the analytic
    benchmark (tests/external_function/no_deriv_given.mod)."""
    m, r = load(folder, case), ref(case)
    dr = moments.solve(m, 2)
    assert maxdiff(dr["ys"], r["ys"]) < 1e-9
    tol = dict(ghx=1e-9, ghu=1e-9, ghxx=1e-9, ghxu=1e-9, ghuu=1e-9, ghs2=1e-9)
    if case in NUMERICAL:
        tol.update(ghxx=1e-2, ghxu=1e-4, ghuu=1e-5)      # see docstring; Dynare: 1e-3, 1e-4, 1e-5 vs benchmark
    for k in KEYS:
        assert maxdiff(dr[k], r[k]) < tol[k], k
    mo = moments.theoretical_moments(m, dr, order=2)
    assert maxdiff(mo.var, r["var"]) < 1e-12
    # the order-2 mean correction depends on ghs2 (exact) and on the Hessian through ghxx (see above)
    assert maxdiff(mo.mean, r["mean"]) < (1e-10 if case not in NUMERICAL else 1e-5)


def test_python_callables(folder):
    """Model(functions=...) takes precedence over the .m files; a function that returns its own
    derivatives returns a tuple."""
    calls = []

    def f(a, b):
        calls.append(1)
        return a * b ** 2

    def f12(a, b):
        return a * b ** 2, np.array([b ** 2, 2 * a * b]), np.array([[0, 2 * b], [2 * b, 2 * a]])

    r = ref("no_deriv_given")
    m = load(folder, "no_deriv_given", functions={"extFunNoDerivs": f})
    dr = korder.k_order(m, order=2, ys=r["ys"])
    assert calls and all(maxdiff(dr[k], r[k]) < 1e-12 for k in KEYS)
    r = ref("first_and_2nd_deriv_given")
    m = load(folder, "first_and_2nd_deriv_given", functions={"extFunWithFirstAndSecondDerivs": f12})
    dr = korder.k_order(m, order=2, ys=r["ys"])
    assert all(maxdiff(dr[k], r[k]) < 1e-12 for k in KEYS)
    m = load(folder, "first_and_2nd_deriv_given", functions={"extFunWithFirstAndSecondDerivs": f})
    with pytest.raises(ValueError, match="must return 3 outputs"):
        m.dynamic_jac(np.tile(r["ys"], 3))


def test_missing_function_is_reported(folder, tmp_path):
    shutil.copy(folder / "no_deriv_given.mod", tmp_path)
    m = load(tmp_path, "no_deriv_given")
    with pytest.raises(NameError, match="extFunNoDerivs"):
        pydyn.steady_state(m)
