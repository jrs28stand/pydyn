"""One-constraint OccBin, higher-order simulation/IRFs, parameter loops and steady-state hooks,
all against Dynare 7.2 references (see reference/r_stage3.m)."""
import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.optimize import fsolve

import dynpy

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("DYNPY_REF", str(ROOT / "reference")))
EXAMPLES = Path(os.environ.get("DYNPY_EXAMPLES", "/home/claude/dl/src72/examples"))
SS = EXAMPLES / "stochastic_simulations"

try:
    dynpy.korder.load_library()
    HAVE_LIB = True
except FileNotFoundError:
    HAVE_LIB = False
needs_lib = pytest.mark.skipif(not HAVE_LIB, reason="libdynpy_korder.so not built")


def load(mod, tmp_path, **kw):
    shutil.copy(mod, tmp_path)
    return dynpy.Model(dynpy.run_preprocessor(tmp_path / Path(mod).name), **kw)


def ref(name):
    return loadmat(REF / name, squeeze_me=True)


# ------------------------------------------------------------------ OccBin, one constraint
@pytest.mark.parametrize("name,curb", [("borrcon_sim", False), ("borrcon_curb", True)])
def test_occbin_one_constraint(name, curb, tmp_path):
    m = load(REF / f"{name}.mod", tmp_path)
    r = ref(f"clean_{name}.mat")
    ob = dynpy.Occbin(m)
    res = ob.simulate(ob.surprise_shocks(), periods=m.options["occbin_setup"]["simul.periods"],
                      maxit=m.options["occbin_solver"].get("simul.maxit", 30), curb_retrench=curb)
    assert np.max(np.abs(r["piecewise"] - r["linear"])) > 1e-3
    np.testing.assert_allclose(res.linear, r["linear"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res.piecewise, r["piecewise"], rtol=1e-10, atol=1e-12)
    rh = r["regime"]                          # rows: regime, regimestart
    for t, per in enumerate(res.regime_history):
        assert list(np.atleast_1d(rh[0][t])) == per[0][0]
        assert list(np.atleast_1d(rh[1][t])) == per[0][1]


# ------------------------------------------------------------------ simulation and IRFs, orders 1-3
@pytest.fixture(scope="module")
def order2(tmp_path_factory):
    m = load(SS / "collard_2001_simulated_moments.mod", tmp_path_factory.mktemp("o2"))
    r = ref("clean_sim_order2.mat")
    dr = dynpy.k_order(m, order=2, ys=r["ys"].astype(float), params=r["params"].astype(float))
    return m, dr, r


@needs_lib
@pytest.mark.parametrize("key,order,pruning", [("y1", 1, False), ("y2_unpruned", 2, False), ("y2_pruned", 2, True)])
def test_simulation_order2(order2, key, order, pruning):
    _, dr, r = order2
    y = dynpy.simulate(dr, r["ex"], order=order, pruning=pruning)
    np.testing.assert_allclose(y, r[key][1:], rtol=1e-10, atol=1e-12)   # row 0 of Dynare's output is y0


@needs_lib
@pytest.mark.parametrize("key,pruning", [("irf2_unpruned", False), ("irf2_pruned", True)])
def test_irf_order2(order2, key, pruning):
    m, dr, r = order2
    chol = np.linalg.cholesky(m.Sigma_e)
    for i in range(m.n_exo):
        y = dynpy.irf(dr, chol[:, i], m.Sigma_e, periods=int(r["long"]), order=2, pruning=pruning,
                      drop=int(r["drop"]), draws=r["draws"])
        np.testing.assert_allclose(y, r[key][:, :, i].T, rtol=1e-9, atol=1e-13)


@needs_lib
@pytest.mark.parametrize("key,pruning", [("y3_pruned", True), ("y3_unpruned", False)])
def test_simulation_order3(key, pruning, tmp_path):
    m = load(REF / "collard_o3.mod", tmp_path)
    r = ref("clean_sim_order3.mat")
    dr = dynpy.k_order(m, order=3, ys=r["ys"].astype(float), params=r["params"].astype(float))
    y = dynpy.simulate(dr, r["ex"], order=3, pruning=pruning)
    np.testing.assert_allclose(y, r[key][1:], rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------ external parameter loop
def test_parameter_loop_without_reloading(tmp_path):
    """Load once, then change parameters and re-solve, as with set_param_value + resol in Dynare."""
    m = load(REF / "schorfheide_2000_sim.mod", tmp_path)
    r = ref("clean_param_loop.mat")
    for k in range(len(r["alp"])):
        m.set_params(alp=r["alp"][k], rho=r["rho"][k])
        sol = dynpy.first_order(m)                 # steady state is recomputed from the new parameters
        np.testing.assert_allclose(sol.ys, r["ys"][k], rtol=1e-10)
        np.testing.assert_allclose(sol.ghx, r["ghx"][k], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(sol.ghu, r["ghu"][k], rtol=1e-8, atol=1e-10)
        chol = np.linalg.cholesky(m.Sigma_e)
        np.testing.assert_allclose(dynpy.irf(sol, chol[:, 0], periods=20), r["irf"][k], rtol=1e-8, atol=1e-11)
    assert len({tuple(np.round(y, 8)) for y in r["ys"]}) > 1     # the loop really changes the steady state


# ------------------------------------------------------------------ steady-state hooks
def _nk_fn():
    spec = importlib.util.spec_from_file_location("nk_ss", ROOT / "examples" / "nk_baseline_steadystate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.steady_state


def test_steady_state_function_updates_parameters(tmp_path):
    """Python counterpart of a _steadystate.m file, including parameters it overwrites."""
    m = load(SS / "nk_baseline.mod", tmp_path, steady_state_fn=_nk_fn())
    r = ref("clean_nk_baseline.mat")
    assert m.param_names == r["param_names"].split(",")
    sol = dynpy.first_order(m)
    np.testing.assert_allclose(sol.ys, r["ys"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sol.params, r["params"], rtol=1e-10)
    np.testing.assert_allclose(sol.ghx, r["ghx"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sol.ghu, r["ghu"], rtol=1e-6, atol=1e-8)
    # change a deep parameter: the function must update the dependent parameter (Rbar)
    m.set_param("betta", 0.995)
    sol_b = dynpy.first_order(m)
    assert abs(sol_b.params[m._ip["Rbar"]] - sol.params[m._ip["Rbar"]]) > 1e-4
    np.testing.assert_allclose(sol_b.params, r["params_b"], rtol=1e-10)
    np.testing.assert_allclose(sol_b.ys, r["ys_b"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sol_b.ghx, r["ghx_b"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sol_b.ghu, r["ghu_b"], rtol=1e-6, atol=1e-8)


def test_helper_function_in_steady_state_model(tmp_path):
    """steady_state_model block that calls a user function (an .m helper in Dynare)."""
    def helper(alpha, beta, delta, psi, theta):
        kh = ((1 / beta - (1 - delta)) / alpha) ** (1 / (alpha - 1))
        f = lambda h: 1 - (((kh * h) ** (alpha - 1)) * h ** (1 - alpha)   # noqa: E731
                           - ((1 - alpha) * kh ** alpha / (theta * h ** psi)) / (kh * h) + (1 - delta))
        return float(fsolve(f, 0.2, xtol=1e-14)[0])

    m = load(SS / "collard_2001_analytical_steady_state.mod", tmp_path,
             functions={"collard_2001_analytical_steady_state_helper": helper})
    r = ref("clean_collard_2001_analytical_steady_state.mat")
    sol = dynpy.first_order(m)
    np.testing.assert_allclose(m.Sigma_e, r["Sigma_e"], rtol=1e-12)
    np.testing.assert_allclose(sol.ys, r["ys"], rtol=1e-7, atol=1e-9)
    at_ref = dynpy.first_order(m, ys=r["ys"].astype(float), params=r["params"].astype(float))
    np.testing.assert_allclose(at_ref.ghx, r["ghx"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(at_ref.ghu, r["ghu"], rtol=1e-9, atol=1e-11)
