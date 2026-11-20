"""Deterministic simulations against Dynare 7.2 (Octave 8.4).

References: ``reference/pf_*.mod`` run by ``reference/r_perfect_foresight.m`` (``pf_*.mat``).
Dynare was run with ``tolf=1e-8`` on the stacked problems (1e-6 for the homotopy model, whose
residuals have a round-off floor of 1.5e-8; 1e-12 for the period-by-period models; LMMCP TolFun
1e-10 and 1e-8) and pydyn reads the same tolerances from the .mod files, so both stop at the same
Newton iterate. Measured agreement of the paths: 5e-13 absolute or better on every case except
the homotopy family, which agrees to 2e-14 relative (levels of 1e7). The tests assert
rtol=1e-10, atol=1e-11.
"""
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.options import UnsupportedOptionWarning

REF = Path(__file__).resolve().parent.parent / "reference"
RTOL, ATOL = 1e-10, 1e-11


def build(tmp_path, name, text=None):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    if text is None:
        shutil.copy(REF / f"{name}.mod", d)
    else:
        (d / f"{name}.mod").write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # "statements not handled" from the loader
        return pydyn.Model(pydyn.run_preprocessor(d / f"{name}.mod"))


def ref(name):
    r = loadmat(REF / f"{name}.mat", squeeze_me=True)
    n = len(str(r["endo_names"]).split(","))
    r["endo"] = np.asarray(r["endo_simul"], float).reshape(n, -1).T           # one row per period
    r["exo"] = np.asarray(r["exo_simul"], float).reshape(r["endo"].shape[0], -1)
    return r


def solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnsupportedOptionWarning)
        return pydyn.perfect_foresight(m, **kw)


def check(res, r, rtol=RTOL, atol=ATOL):
    assert res.status and int(r["status"]) == 1
    assert res.endo.shape == r["endo"].shape
    np.testing.assert_allclose(res.exo, r["exo"], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(res.endo, r["endo"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(res.steady_state, r["steady_state"], rtol=rtol, atol=atol)
    assert res.homotopy_iterations == int(r["homotopy_iterations"])
    if "iterations" in r:
        assert res.iterations == int(r["iterations"])
    assert res.endo_names == str(r["endo_names"]).split(",")


# name -> (what it covers, solver pydyn must have used)
STACKED = {
    "pf_temp": "temporary shocks on two variables over ranges, from the steady state; long lags/leads, exo lag and lead",
    "pf_endval": "permanent shock: endval, steady after each block, temporary shocks on the way",
    "pf_endval_nosteady": "endval without steady: terminal condition = the values of the block",
    "pf_initval_nosteady": "initval without steady: start and end at the values of the block",
    "pf_histval": "histval with a lag beyond one and a lagged exogenous variable",
    "pf_mshocks": "mshocks relative to the terminal and to the initial steady state",
    "pf_mshocks0": "mshocks without endval; shocks(overwrite)",
    "pf_etp": "endogenous_terminal_period",
}


@pytest.mark.parametrize("name", sorted(STACKED))
def test_stacked_newton(tmp_path, name):
    m, r = build(tmp_path, name), ref(name)
    res = solve(m)
    check(res, r)
    assert res.solver == "stacked Newton" and res.homotopy_completion_share == 1
    assert res.endo.shape == (res.periods + 2, m.n_endo)            # oo_.endo_simul': initial row, periods, terminal row
    assert res.max_residual < float(r["tolf"]) and float(r["error"]) < float(r["tolf"])
    if "initial_steady_state" in r:
        np.testing.assert_allclose(res.initial_steady_state, r["initial_steady_state"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(res.endo[0], r["initial_steady_state"], rtol=RTOL, atol=ATOL)
    else:
        assert res.initial_steady_state is None


def test_initval_without_steady_is_not_a_steady_state(tmp_path):
    """Dynare's semantics: without ``steady`` the values of initval are y_0, the guess and the
    terminal condition; ``steady=True`` asks pydyn for the steady state instead."""
    m, r = build(tmp_path, "pf_initval_nosteady"), ref("pf_initval_nosteady")
    res = solve(m)
    k = m.endo_names.index("k")
    assert res.endo[0, k] == 6.0 and res.endo[-1, k] == 6.0 == r["endo"][-1, k]
    aux = m.endo_names.index("AUX_ENDO_LAG_1_1")                  # k(-1): set to k by the initval block
    assert res.endo[0, aux] == 6.0
    res2 = solve(m, steady=True)
    ys, _ = pydyn.steady_state(m)
    np.testing.assert_allclose(res2.endo[0], ys, rtol=1e-12)
    np.testing.assert_allclose(res2.endo[-1], ys, rtol=1e-12)


def test_endval_without_steady_keeps_the_values(tmp_path):
    m = build(tmp_path, "pf_endval_nosteady")
    res = solve(m)
    assert res["k"][-1] == 7.5 and res["z"][-1] == 1.05 and res["z"][0] == 1.0
    assert not res.info["recompute"]                # the endval values are not a steady state: kept as they are
    assert abs(res["k"][0] - 7.5) > 1               # starts from the initial steady state


def test_histval_conventions(tmp_path):
    m, r = build(tmp_path, "pf_histval"), ref("pf_histval")
    res = solve(m)
    assert res["k"][0] == 60 and res["x"][0] == 0.9
    assert res["c"][0] == 0 and res["z"][0] == 0            # not in the histval block: zero, not the steady state
    assert res["AUX_ENDO_LAG_2_1"][0] == 0.1                # z(-1)
    assert res["AUX_EXO_LAG_3_0"][0] == 0.9                 # x(0)
    np.testing.assert_allclose(res.endo, r["endo"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name, rows", [("pf_backward", "periods+1"), ("pf_forward", "periods+1"), ("pf_static", "periods")])
def test_period_by_period_models(tmp_path, name, rows):
    """Purely backward (with histval and a second lag), purely forward and static models."""
    m, r = build(tmp_path, name), ref(name)
    res = solve(m)
    check(res, r)
    assert res.solver == "period by period"
    assert res.endo.shape[0] == res.periods + (1 if rows == "periods+1" else 0)
    assert (res.maximum_lag, res.maximum_lead) == (int(r["maximum_lag"]), int(r["maximum_lead"]))


def test_linear_model_and_linear_approximation(tmp_path):
    for name in ("pf_linear", "pf_linapprox"):
        m, r = build(tmp_path, name), ref(name)
        res = solve(m)
        check(res, r)
        assert res.solver == "linear"
    # the nonlinear solution of the same model differs from its linear approximation
    exact = solve(m, linear_approximation=False)
    assert exact.solver == "stacked Newton" and np.max(np.abs(exact.endo - res.endo)) > 1e-5


def test_homotopy_is_needed(tmp_path):
    """Newton fails on the full permanent shock; Dynare 7's homotopy on the shock size solves it."""
    m, r0 = build(tmp_path, "pf_nohomotopy"), ref("pf_nohomotopy")
    assert int(r0["status"]) == 0
    with pytest.raises(pydyn.PerfectForesightError) as exc:
        solve(m)
    assert not exc.value.result.status and exc.value.result.homotopy_completion_share == 0
    assert exc.value.result.iterations == int(r0["iterations"])
    assert not solve(m, raise_on_failure=False).status
    r = ref("pf_homotopy")
    res = solve(m, no_homotopy=False)                       # keyword overrides the .mod file
    check(res, r)
    assert res.homotopy_iterations == 15 and res.homotopy_completion_share == 1
    for name, n_it in (("pf_homotopy", 15), ("pf_homotopy_opts", 12)):
        res = solve(build(tmp_path, name))
        check(res, ref(name))
        assert res.homotopy_iterations == n_it


def test_homotopy_linearization_fallbacks(tmp_path):
    for name, flag in (("pf_homotopy_lin", "homotopy_linearization"), ("pf_homotopy_marg", "homotopy_marginal_linearization")):
        m, r = build(tmp_path, name), ref(name)
        res = solve(m)
        check(res, r)
        assert getattr(res, flag) and int(r[flag]) == 1
        assert res.homotopy_completion_share == pytest.approx(float(r["homotopy_completion_share"])) == 0.8
        np.testing.assert_allclose(res.max_residual, float(r["error"]), rtol=1e-6)
    with pytest.raises(ValueError, match="homotopy_max_completion_share"):
        solve(m, homotopy_marginal_linearization_fallback=0.0)
    with pytest.raises(ValueError, match="no_homotopy"):
        solve(m, no_homotopy=True)


def test_zero_lower_bound_with_lmmcp(tmp_path):
    """[mcp = 'i > -0.01'] + perfect_foresight_solver(lmmcp): the bound binds for 11 periods."""
    m, r = build(tmp_path, "pf_zlb"), ref("pf_zlb")
    res = solve(m)
    check(res, r)
    assert res.solver == "lmmcp"
    i = res["i"][1:-1]
    assert np.sum(i <= -0.01 + 1e-12) == 11 and i.min() >= -0.01 - 1e-12 and i[20] > -0.01
    lb, ub, order = pydyn.perfect_foresight.complementarity_conditions(m)
    assert lb[m.endo_names.index("i")] == -0.01 and np.all(np.isinf(ub)) and list(order) == [0, 1, 3, 2]
    # without the lmmcp option Dynare ignores the condition; pydyn does the same and says so
    text = (REF / "pf_zlb.mod").read_text().replace("perfect_foresight_solver(lmmcp, no_homotopy)", "perfect_foresight_solver")
    m2 = build(tmp_path, "pf_zlb_off", text)
    with pytest.warns(UnsupportedOptionWarning, match="complementarity"):
        free = pydyn.perfect_foresight(m2)
    assert free["i"].min() < -0.02
    np.testing.assert_allclose(solve(m2, lmmcp=True, lmmcp_tolfun=1e-10).endo, r["endo"], rtol=RTOL, atol=ATOL)


def test_irreversible_investment_with_lmmcp(tmp_path):
    """Dynare's tests/lmmcp/rbcii.mod (the ``⟂`` syntax, lower and upper bound): investment is at
    its lower bound for 51 periods. Measured agreement 4e-14; LMMCP TolFun is 1e-8 in both."""
    m, r = build(tmp_path, "pf_rbcii"), ref("pf_rbcii")
    res = solve(m)
    check(res, r, rtol=1e-9, atol=1e-10)
    inv = res["i"][1:-1]
    assert np.sum(inv <= 1e-7) == 51 and inv.min() >= -1e-7        # the bound holds to the LMMCP tolerance
    np.testing.assert_allclose(res.max_residual, float(r["error"]), rtol=1e-3)


def test_expectation_errors(tmp_path):
    """perfect_foresight_with_expectation_errors_setup/_solver with shocks(learnt_in=...),
    mshocks(learnt_in=...), endval(learnt_in=...) (level, add, multiply, overwrite)."""
    m, r = build(tmp_path, "pf_pfwee"), ref("pf_pfwee")
    res = solve(m)
    check(res, r)
    np.testing.assert_array_equal(np.isnan(res.info["shocks_info"][0]), np.isnan(r["shocks_info"]))
    np.testing.assert_allclose(np.nan_to_num(res.info["shocks_info"][0]), np.nan_to_num(r["shocks_info"]), rtol=1e-14)
    np.testing.assert_allclose(res.info["terminal_info"][0], r["terminal_info"], rtol=1e-14)
    assert [p for p, _, _ in res.info["solves"]] == [1, 2, 3, 6]          # one solve per period with news


def test_unanticipated_shocks(tmp_path):
    """A sequence of surprises, from the .mod file and through the Python helper."""
    m, r = build(tmp_path, "pf_surprise"), ref("pf_surprise")
    check(solve(m), r)
    text = (REF / "pf_surprise.mod").read_text()
    text = re.sub(r"^shocks\(learnt_in[^;]*;.*?end;\s*", "", text, flags=re.S | re.M)
    text = re.sub(r"perfect_foresight_with_expectation_errors_setup\(periods = 60\);.*", "perfect_foresight_setup(periods=60);\n",
                  text, flags=re.S)
    m2 = build(tmp_path, "pf_surprise_py", text)
    res = pydyn.perfect_foresight.unanticipated_shocks(m2, [(1, {"x": 1.2}), (3, {"x": 0.9}), (4, {"x": 1.1})],
                                                       tolf=1e-8, constant_simulation_length=True)
    np.testing.assert_allclose(res.endo, r["endo"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(res.exo, r["exo"], rtol=1e-14)
    # the same shocks announced in period 1 give another path
    assert res.endo.shape[0] == 62 + 3          # constant_simulation_length: the horizon moves with each surprise
    anticipated = solve(m2, exo=res.exo[:62], tolf=1e-8)
    assert np.max(np.abs(anticipated.endo[:4] - res.endo[:4])) > 1e-4


def test_options_from_file_and_keywords(tmp_path):
    m, r = build(tmp_path, "pf_temp"), ref("pf_temp")
    base = solve(m)
    assert base.info["tolf"] == 1e-8 == float(r["tolf"])            # tolf read from the .mod file
    tight = solve(m, tolf=1e-13)
    assert tight.iterations == base.iterations + 1 and tight.max_residual < 1e-13
    assert solve(m, tol=1e-13).iterations == tight.iterations       # the older keyword
    with pytest.raises(pydyn.PerfectForesightError):
        solve(m, maxit=2, no_homotopy=True)
    short = solve(m, periods=40)
    assert short.endo.shape == (42, m.n_endo)
    np.testing.assert_allclose(pydyn.perfect_foresight.exo_path(m, 150), r["exo"], rtol=1e-14)
    with pytest.raises(pydyn.PerfectForesightError, match="after the terminal period"):
        solve(m, periods=8)
    with pytest.raises(TypeError):
        solve(m, not_an_option=1)
    with pytest.raises(ValueError, match="purely backward"):
        solve(build(tmp_path, "pf_backward"), endogenous_terminal_period=True)
    text = (REF / "pf_temp.mod").read_text().replace("tolf=1e-8, tolx=1e-8", "tolf=1e-8, stack_solve_algo=2, markowitz=0.3, robust_lin_solve")
    m2 = build(tmp_path, "pf_temp_opts", text)
    with pytest.warns(UnsupportedOptionWarning) as rec:
        res = pydyn.perfect_foresight(m2)
    msgs = " ".join(str(w.message) for w in rec)
    assert "stack_solve_algo=2" in msgs and "markowitz" in msgs and "robust_lin_solve" in msgs
    np.testing.assert_allclose(res.endo, r["endo"], rtol=RTOL, atol=ATOL)


def test_vectorised_and_per_period_assembly_agree(tmp_path):
    for name in ("pf_temp", "pf_rbcii"):
        m = build(tmp_path, name)
        a, b = solve(m, stacked="vector"), solve(m, stacked="loop")
        assert a.iterations == b.iterations
        np.testing.assert_allclose(a.endo, b.endo, rtol=1e-11, atol=1e-12)


def test_result_conveniences(tmp_path):
    m = build(tmp_path, "pf_endval")
    res = solve(m)
    p = res.paths()
    assert list(p) == m.endo_names and p["k"] is not None and p["k"].shape == (152,)
    np.testing.assert_array_equal(p["k"], res["k"])
    d0, dT = res.deviations("initial"), res.paths(deviation="terminal", exo=True, simulation_only=True)
    assert abs(d0["k"][0]) < 1e-12 and abs(dT["k"][-1]) < 0.05 and dT["k"].shape == (150,)
    assert abs(dT["z"][-1]) < 1e-15 and abs(res.deviations("initial", exo=True)["z"][-1] - 0.1) < 1e-15
    assert res.exo_names == ["z", "g"] and res.periods == 150
