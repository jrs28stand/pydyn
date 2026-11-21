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
    assert "stack_solve_algo=2" in msgs and "markowitz" in msgs and "robust_lin_solve" not in msgs   # robust_lin_solve is honoured
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


# ------------------------------------------------------------------ initval_file, histval_file, datafile
RAMST = (REF / "pf_ivf.mod").read_text().split("initval;")[0]


def _ramst_data(n_before=0, n_after=0):
    """ramst_initval_file_data.m of Dynare's tests (x = 1 except 1.2 in period 1, k = 13, c = 1.5),
    with extra observations before and after."""
    x = np.r_[np.ones(n_before), [1.0, 1.2], np.ones(200 + n_after)]
    return {"x": x, "k": np.full(x.size, 13.0), "c": np.full(x.size, 1.5)}


def check_paths(res, r):
    """The simulated paths and the solver's status (oo_.steady_state may differ: the files below run
    'steady' before the setup, pf_ivf does not, and the steady state does not enter this model)."""
    assert res.status and int(r["status"]) == 1
    np.testing.assert_allclose(res.exo, r["exo"], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(res.endo, r["endo"], rtol=RTOL, atol=ATOL)
    assert res.iterations == int(r["iterations"])


def test_initval_file_datafile_and_python_series(tmp_path):
    """initval_file / perfect_foresight_setup(datafile=...) / Python dicts with .mat (row and column
    vectors), .csv and .m files and first_obs/last_obs/nobs/first_simulation_period: all give Dynare's
    path of the same problem written with initval and shocks (pf_ivf; Dynare 7.2 here cannot read data
    files, its dseries class is missing)."""
    from scipy.io import savemat
    r = ref("pf_ivf")
    d = _ramst_data()
    head = RAMST + "initval; x = 1; k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1)); c = aa*k^alph-delt*k; end;\nsteady;\n"
    cases = {
        "ivf_col": ("initval_file(datafile = data_col);\nperfect_foresight_setup(periods=200);\n",
                    lambda p: savemat(p / "data_col.mat", {k: v.reshape(-1, 1) for k, v in d.items()})),
        "ivf_row": ("initval_file(filename = data_row);\nperfect_foresight_setup(periods=200);\n",
                    lambda p: savemat(p / "data_row.mat", {k: v.reshape(1, -1) for k, v in d.items()})),
        "ivf_setup": ("perfect_foresight_setup(periods=200, datafile = data_col);\n",
                      lambda p: savemat(p / "data_col.mat", {k: v.reshape(-1, 1) for k, v in d.items()})),
        "ivf_m": ("initval_file(datafile = 'data.m');\nperfect_foresight_setup;\n",            # periods from the file
                  lambda p: (p / "data.m").write_text("x = [1; 1.2; ones(200, 1)];\nk = 13*ones(202, 1);\nc = repmat(1.5, 202, 1);\n")),
        "ivf_csv": ("initval_file(datafile = 'data.csv', first_obs = 4, last_obs = 205);\nperfect_foresight_setup(periods=200);\n",
                    lambda p: (p / "data.csv").write_text(
                        ",c,k,x\n" + "".join(f"{i}Y,{c},{k},{x}\n" for i, (c, k, x) in
                                             enumerate(zip(*(_ramst_data(3, 5)[v] for v in "ckx")))))),
        "ivf_nobs": ("initval_file(datafile = 'data.csv', nobs = 202, last_obs = 205);\nperfect_foresight_setup(periods=200);\n",
                     lambda p: (p / "data.csv").write_text(
                         "c,k,x\n" + "".join(f"{c},{k},{x}\n" for c, k, x in zip(*(_ramst_data(3, 5)[v] for v in "ckx"))))),
    }
    for name, (tail, write) in cases.items():
        m = build(tmp_path, name, head + tail + "perfect_foresight_solver(tolf=1e-8, tolx=1e-8);\n")
        write(tmp_path / name)
        res = solve(m)
        check_paths(res, r)
        assert res.periods == 200
    # from Python: the series (and the same options) as keywords
    m = build(tmp_path, "ivf_py", head + "perfect_foresight_setup(periods=200);\nperfect_foresight_solver(tolf=1e-8, tolx=1e-8);\n")
    check_paths(solve(m, initval_series=_ramst_data(2, 0), initval_options={"first_obs": 3}), r)
    with pytest.raises(ValueError, match="missing"):
        solve(m, initval_series={"x": d["x"], "k": d["k"]})
    with pytest.raises(NotImplementedError, match="date"):
        solve(m, initval_series=d, initval_options={"first_obs": "dates('3Y')"})
    with pytest.raises(ValueError, match="inconsistent"):
        solve(m, initval_series=_ramst_data(3, 5), initval_options={"first_obs": 4, "last_obs": 205, "nobs": 10})
    # Dynare's driver sets options_.periods from the data when the setup command does not give it
    m = build(tmp_path, "ivf_m2", head + "initval_file(datafile = 'data.m');\nperfect_foresight_setup;\nperfect_foresight_solver;\n")
    (tmp_path / "ivf_m2" / "data.m").write_text("x = [1; 1.2; ones(210, 1)];\nk = 13*ones(212, 1);\nc = repmat(1.5, 212, 1);\n")
    assert solve(m).periods == 210


def test_histval_file(tmp_path):
    """histval_file with the values of pf_histval's histval block (lags beyond one, lagged exogenous
    variable): Dynare's pf_histval path."""
    r = ref("pf_histval")
    text = (REF / "pf_histval.mod").read_text()
    text = re.sub(r"histval;.*?end;", "histval_file(datafile = 'hist.csv');", text, flags=re.S)
    m = build(tmp_path, "pf_histval_file", text)
    # periods -1 and 0; variables the histval block does not mention are 0
    (tmp_path / "pf_histval_file" / "hist.csv").write_text("c,k,z,x\n0,0,0.1,0\n0,60,0,0.9\n")
    res = solve(m)
    np.testing.assert_allclose(res.endo, r["endo"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(res.exo, r["exo"], rtol=1e-14)
    # the same from Python: first_obs (or last_obs) selects the two observations of a longer series
    hist = {"c": [9, 0, 0, 9], "k": [9, 0, 60, 9], "z": [9, 0.1, 0, 9], "x": [9, 0, 0.9, 9]}
    for opts in ({"first_obs": 2}, {"last_obs": 3}):
        res2 = solve(m, histval_series=hist, histval_options=opts)
        np.testing.assert_allclose(res2.endo, r["endo"], rtol=RTOL, atol=ATOL)
    with pytest.raises(ValueError, match="nobs"):
        solve(m, histval_series=hist, histval_options={"nobs": 2})
    with pytest.raises(ValueError, match="maximum number of lags"):
        solve(m, histval_series=hist, histval_options={"first_obs": 1, "last_obs": 3})


@pytest.mark.parametrize("name", ["pf_ivfx", "pf_hvfx"])
def test_initval_and_histval_files_against_dynare(tmp_path, name):
    """Dynare 7.2 reading the same CSV files (its tests sim_exo_lead_lag_initvalf/_histvalf with the
    data written to a file): initval_file(first_obs, last_obs, nobs) and histval_file(first_obs,
    last_obs) with auxiliary variables for leads and lags up to 4. Measured agreement 1.6e-14 and
    1.2e-14, same iterations; the auxiliary leads of period 0 are NaN in both (histval_file)."""
    (tmp_path / name).mkdir()
    shutil.copy(REF / f"{name}_data.csv", tmp_path / name)
    m = build(tmp_path, name)
    r = ref(name)
    res = solve(m)
    assert res.status and int(r["status"]) == 1 and res.iterations == int(r["iterations"])
    np.testing.assert_array_equal(np.isnan(res.endo), np.isnan(r["endo"]))
    np.testing.assert_allclose(res.endo, r["endo"], rtol=RTOL, atol=ATOL, equal_nan=True)
    np.testing.assert_allclose(res.exo, r["exo"], rtol=1e-14, atol=1e-15)


# ------------------------------------------------------------------ Dynare's tests that load since 2026-11-21
# The reference files are Dynare 7.2 tests/deterministic_simulations/*.mod with the changes listed in
# their headers (tolerances; a steady(tolf=1e-12) where Dynare's steady-state search restarts from
# random points, which differ from pydyn's, so that both end at the root rather than at different
# points within the default tolerance).

def _solve_ref(tmp_path, name, extra=(), **kw):
    (tmp_path / name).mkdir(exist_ok=True)
    for f in extra:
        shutil.copy(REF / f, tmp_path / name)
    m = build(tmp_path, name)
    return m, solve(m, **kw), ref(name)


def test_steady_state_restart_and_homotopy_fallbacks(tmp_path):
    """homotopy_marginal_linearization / homotopy_endval_steady_linearization: `steady` from a zero
    guess (1/Consumption is not finite) restarts from random points, as Dynare's dynare_solve; then
    homotopy with steady_solve_algo=13 (pydyn's port of Dynare's block trust region) and the
    linearization fallbacks. Measured: 1.2e-13 and 2.4e-14 (scaled). In homotopy_endval_steady_linearization
    the endval steady state at share 0.375 fails in Dynare (trust region, slow progress) and succeeds
    in pydyn from a guess that differs in the last bits, so the homotopy takes 25 steps instead of 26."""
    m = build(tmp_path, "pf_hml")
    with pytest.warns(UserWarning, match="restarted from random guesses"):
        res = pydyn.perfect_foresight(m)
    r = ref("pf_hml")
    check(res, r)
    assert res.homotopy_marginal_linearization and res.homotopy_completion_share == 0.7
    m, res, r = _solve_ref(tmp_path, "pf_hesl")
    assert res.status and res.homotopy_linearization and int(r["homotopy_linearization"]) == 1
    np.testing.assert_allclose(res.endo, r["endo"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(res.steady_state, r["steady_state"], rtol=RTOL)
    assert (res.homotopy_iterations, int(r["homotopy_iterations"])) == (25, 26)


def test_expectation_errors_with_marginal_linearization(tmp_path):
    """pfwee_marginal_linearization: each information period reuses the two raw simulations of the
    previous one (Dynare's marginal_linearization_previous_raw_sims). Measured 8.7e-13."""
    m, res, r = _solve_ref(tmp_path, "pf_pfweeml")
    check(res, r, rtol=1e-9, atol=1e-10)
    assert res.homotopy_marginal_linearization and res.homotopy_completion_share == 0.8


def test_matlab_code_reading_and_writing_the_steady_state(tmp_path):
    """rbc_det: 'CapitalSS = oo_.steady_state(ik)' after a numerical steady state (varlist_indices,
    read of pydyn's port of Dynare's solver: the same iterate), then histval with CapitalSS/2.
    pfwee: verbatim code after the setup assigns oo_.steady_state and oo_.exo_steady_state."""
    d = tmp_path / "pf_rbcdet"
    d.mkdir()
    shutil.copy(REF / "pf_rbcdet.mod", d)
    with pytest.warns(UserWarning, match="reads oo_.steady_state after a steady state computed numerically"):
        m = pydyn.Model(pydyn.run_preprocessor(d / "pf_rbcdet.mod"))
    assert m.calibration_tolerance == pytest.approx(6.06e-6, rel=1e-2)
    check(solve(m), ref("pf_rbcdet"))
    assert m.workspace["CapitalSS"] == pytest.approx(ref("pf_rbcdet")["steady_state"][0], rel=1e-14)
    m, res, r = _solve_ref(tmp_path, "pf_pfweecsv", extra=["pf_pfweecsv.csv"])
    check(res, r)


def test_vector_values_in_shocks_and_mshocks(tmp_path):
    """ramst_vec, ramst_mshocks_vec: 'values (a);' with a MATLAB vector, one value per period."""
    m, res, r = _solve_ref(tmp_path, "pf_vec")
    check(res, r)
    assert list(res["x"][1:7]) == [1.2, 1.1, 1.0, 1.2, 0.8, 0.8]


def test_robust_lin_solve(tmp_path):
    """Irreversible_investment: the stacked Jacobian becomes exactly singular at the third Newton
    iteration; with robust_lin_solve Dynare's lin_solve_robust (A\\b, here Octave's solution of the
    consistent singular system, then BiCGSTAB which returns at once) solves it, without the option
    the simulation fails. Measured 2.4e-10 (scaled), same iterations: the solutions of the singular
    systems agree to about 1e-13, not exactly."""
    m, res, r = _solve_ref(tmp_path, "pf_irr")
    check(res, r, rtol=1e-9, atol=1e-9)
    assert not solve(m, robust_lin_solve=False, raise_on_failure=False, no_homotopy=True).status


@pytest.mark.parametrize("name", ["pf_solow", "pf_zlbfwd"])
def test_period_by_period_models_use_dynare_trust_region(tmp_path, name):
    """Solow_no_varexo (purely backward), simul_ZLB_purely_forward: each period is solved with pydyn's
    port of dynare_solve's solve_algo=4 (trust region on Dulmage-Mendelsohn blocks), so the iterates
    and the stopping points are Dynare's: measured 0 and 7e-16."""
    m, res, r = _solve_ref(tmp_path, name)
    assert res.solver == "period by period" and res.status and int(r["status"]) == 1
    np.testing.assert_allclose(res.endo, r["endo"], rtol=1e-13, atol=1e-14)


def test_block_bytecode_model_agrees_at_a_tight_tolerance(tmp_path):
    """rbc_det2 (model(block, bytecode)): Dynare's bytecode solver works block by block, so its Newton
    iterates differ from pydyn's on the whole stacked system; at tolf=1e-8 the paths differ by 1e-10,
    at tolf=1e-11 (this reference) by 5e-15."""
    m, res, r = _solve_ref(tmp_path, "pf_rbcdet2")
    np.testing.assert_allclose(res.endo, r["endo"], rtol=1e-13, atol=1e-13)


def test_controlled_paths(tmp_path):
    """perfect_foresight_controlled_paths (exogenize/endogenize) in the stacked Newton solver, the
    linear solver and with homotopy plus linearization (Dynare's ramst_pf_controlled_paths with
    periods instead of dates, linear_state_space_arma_condpaths, homotopy_linearization_condpaths).
    Measured 7e-15, 1.4e-14 and 3e-14; in the homotopy case the number of homotopy steps differs
    (43 against 45: endval steady states that succeed in one and fail in the other)."""
    m, res, r = _solve_ref(tmp_path, "pf_cp")
    check(res, r)
    assert list(res["c"][[2, 4, 5]]) == [1.6, 1.7, 1.7] and list(res["k"][7:10]) == [13, 13, 13]
    assert res["x"][1] == 1.2 and res["x"][2] != 1.0                 # x is solved for in the controlled periods
    m, res, r = _solve_ref(tmp_path, "pf_cplin")
    assert res.solver == "linear"
    np.testing.assert_allclose(res.endo, r["endo"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(res.exo, r["exo"], rtol=1e-12, atol=1e-13)
    m, res, r = _solve_ref(tmp_path, "pf_cphom")
    assert res.homotopy_linearization and res.status
    np.testing.assert_allclose(res.endo, r["endo"], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(res.exo, r["exo"], rtol=1e-12, atol=1e-14)
