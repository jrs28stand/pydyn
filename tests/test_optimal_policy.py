"""Optimal policy against Dynare 7.2 (reference/r_policy.sh writes reference/clean_pol_*.mat with
reference/pol_export.m): Ramsey steady state, rules and welfare (evaluate_planner_objective),
discretionary policy and optimal simple rules.

The .mod files are Dynare's tests/optimal_policy, tests/discretionary_policy and
examples/optimal_policy models (reference/pol_*.mod), with the MATLAB code that loads other runs'
results removed and evaluate_planner_objective added where it was missing."""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import optimal_policy as op
from pydyn.occbin_filter import csolve

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))


# find_c.m of Dynare's tests/optimal_policy/Ramsey (it calls csolve, which pydyn's MATLAB interpreter
# does not have): the same computation with pydyn's port of csolve
def _nk_ss(c, pai, beta, epsilon, phi, gamma, omega):
    return (pai * (pai - 1) / c - beta * pai * (pai - 1) / c - epsilon * phi * (c + (omega / 2) * (pai - 1) ** 2) ** (gamma + 1) / omega
            + (c + (omega / 2) * (pai - 1) ** 2) * (epsilon - 1) / (omega * c))


def find_c(c0, pai, beta, epsilon, phi, gamma, omega):
    f = lambda c: np.atleast_1d(_nk_ss(c[0], pai, beta, epsilon, phi, gamma, omega))  # noqa: E731

    def grad(c):
        f0 = f(c)
        return f0, ((f(c + 1e-6) - f0) / 1e-6).reshape(1, 1)
    return float(csolve(f, np.array([c0]), grad, 1e-8, 100)[0][0])


def load(tmp_path, mod, defs=(), files=()):
    shutil.copy(REF / f"{mod}.mod", tmp_path)
    for f in files:
        shutil.copy(REF / f, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(tmp_path / f"{mod}.mod", extra_args=tuple(defs)),
                           functions={"find_c": find_c})


def ref(case):
    return loadmat(REF / f"clean_{case}.mat", squeeze_me=True)


def rules(m, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.first_order(m).as_dr() if order == 1 else pydyn.k_order(m, order=order)


def assert_rules(dr, r, order, rtol=1e-9, atol=1e-11):
    np.testing.assert_array_equal(dr["order_var"] + 1, r["order_var"])
    # instruments found by csolve stop at its tolerance on a norm of residuals (not smooth at the
    # solution): steady states agree to about 1e-12 (1e-14 when Dynare's steady refines them first)
    np.testing.assert_allclose(dr["ys"], r["ys"], rtol=1e-12, atol=1e-11)
    for k in ("ghx", "ghu") + (("ghxx", "ghxu", "ghuu", "ghs2") if order == 2 else ()):
        np.testing.assert_allclose(dr[k], np.reshape(r[k], np.shape(dr[k])), rtol=rtol, atol=atol, err_msg=k)


def assert_welfare(v, r, ramsey=True, rtol=1e-11, atol=1e-11):
    np.testing.assert_allclose(v.unconditional, r["W_unc"], rtol=rtol, atol=atol)
    if ramsey:
        np.testing.assert_allclose(v.zero_initial_multiplier, r["W_cond_L0"], rtol=rtol, atol=atol)
        np.testing.assert_allclose(v.steady_initial_multiplier, r["W_cond_LSS"], rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(v.conditional, r["W_cond"], rtol=rtol, atol=atol)


# (case, .mod file, macro definitions, extra files): tests/optimal_policy models
RAMSEY = [
    ("pol_nk_ramsey", "pol_nk_ramsey", (), ()),                         # no steady-state file: full solve
    ("pol_nk_ramsey_o2", "pol_nk_ramsey", ("-DORDER=2",), ()),
    ("pol_neo_ramsey", "pol_neo_ramsey", (), ()),                       # order 2; 2 instruments for 1 missing equation
    ("pol_neo_ramsey_o1", "pol_neo_ramsey", ("-DORDER=1",), ()),
    ("pol_ramsey_ex", "pol_ramsey_ex", (), ("find_c.m",)),              # steady_state_model given the instrument
    ("pol_ramsey_ex_o2", "pol_ramsey_ex", ("-DORDER=2",), ("find_c.m",)),
    ("pol_ramsey_ex2", "pol_ramsey_ex2", (), ("find_c.m",)),            # csolve moves the instrument
    ("pol_ramsey_aux", "pol_ramsey_aux", (), ("find_c.m",)),            # leads, lags, expectation: auxiliary variables
    ("pol_nk_ramsey_exp", "pol_nk_ramsey_exp", (), ()),                 # expectation auxiliary before the multipliers
    ("pol_ramsey_lin", "pol_ramsey_lin", (), ()),                       # linear model
    ("pol_nk_ramsey_sf", "pol_nk_ramsey_sf", (), ("pol_nk_ramsey_sf_steadystate.m",)),   # _steadystate.m reading options_.instruments
    ("pol_gali_commit", "pol_gali_commit", (), ()),
    ("pol_mult_elim", "pol_mult_elim", (), ()),
    ("pol_ramsey_histval", "pol_ramsey_histval", (), ("find_c.m",)),    # histval and shocks in period 1: initial state
    ("pol_ramsey_histval_o2", "pol_ramsey_histval", ("-DORDER=2",), ("find_c.m",)),
    ("pol_ramsey_2inst", "pol_ramsey_2inst", (), ()),                   # two instruments: QR residual of the multipliers
    ("pol_ramsey_policy", "pol_ramsey_policy", (), ()),                 # deprecated ramsey_policy statement
    ("pol_ramsey_policy_o2", "pol_ramsey_policy", ("-DORDER=2",), ()),
]


@pytest.mark.parametrize("case,mod,defs,files", RAMSEY, ids=[c[0] for c in RAMSEY])
def test_ramsey(case, mod, defs, files, tmp_path):
    r = ref(case)
    m = load(tmp_path, mod, defs, files)
    assert op.is_ramsey(m) and not set(m.unhandled_statements) - {"resid"}
    assert m.endo_names == r["endo_names"].split(",")
    ri = op.ramsey_info(m)
    assert (ri.orig_endo_nbr, ri.orig_eq_nbr) == (r["ramsey_orig_endo_nbr"], r["ramsey_orig_eq_nbr"])
    order = int(r["order"])
    dr = rules(m, order)
    assert_rules(dr, r, order)
    u = np.atleast_1d(r["exo_simul1"]) if not m.det_shocks else None
    assert_welfare(op.evaluate_planner_objective(m, dr, u=u), r)


@pytest.mark.parametrize("case,defs", [("pol_ramsey_policy", ()), ("pol_ramsey_policy_o2", ("-DORDER=2",))])
def test_ramsey_policy_defaults(case, defs, tmp_path):
    """evaluate_planner_objective without rules solves the model at the order of the .mod file
    (ramsey_policy(order=...) acts as stoch_simul)."""
    r = ref(case)
    m = load(tmp_path, "pol_ramsey_policy", defs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_welfare(op.evaluate_planner_objective(m), r)


def test_nk_ramsey_osr_example(tmp_path):
    """examples/optimal_policy/nk_ramsey_osr.mod (Ramsey branch): stoch_simul(periods=500) leaves the
    first simulated shock in oo_.exo_simul, which Dynare's conditional welfare uses."""
    r = ref("pol_nk_ramsey_osr")
    m = load(tmp_path, "pol_nk_ramsey_osr")
    dr = rules(m, 1)
    assert_rules(dr, r, 1)
    assert_welfare(op.evaluate_planner_objective(m, dr, u=np.atleast_1d(r["exo_simul1"])), r)
    v0 = op.evaluate_planner_objective(m, dr)               # u = exogenous steady state
    assert v0.unconditional == pytest.approx(r["W_unc"], rel=1e-13)
    assert abs(v0.zero_initial_multiplier - r["W_cond_L0"]) > 1e-4


def test_ramsey_steady_state_errors(tmp_path):
    """Dynare's error 85: the steady-state file does not solve the original equations given the
    initial value of the instrument."""
    text = (REF / "pol_nk_ramsey_sf.mod").read_text().replace("R=1/beta-1;", "R=0.03;")
    (tmp_path / "pol_nk_ramsey_sf.mod").write_text(text)
    shutil.copy(REF / "pol_nk_ramsey_sf_steadystate.m", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "pol_nk_ramsey_sf.mod"))
    with pytest.raises(op.OptimalPolicyError) as e:
        pydyn.steady_state(m)
    assert e.value.info == 85


def test_ramsey_perfect_foresight_welfare(tmp_path):
    """tests/optimal_policy/neo_growth_ramsey_foresight: welfare summed along the simulated path.
    Dynare's solver stops at |residual| < 1e-5 (options_.dynatol.f): with the same tolerance the
    paths agree to 1e-13."""
    r, path = ref("pol_neo_ramsey_pf"), ref("pol_neo_ramsey_pf_path")["endo_simul"]
    m = load(tmp_path, "pol_neo_ramsey_pf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = pydyn.perfect_foresight(m, tol=1e-5)
    np.testing.assert_allclose(pf.endo.T, path, rtol=1e-11, atol=1e-11)
    v = op.evaluate_planner_objective_perfect_foresight(m, pf)
    assert_welfare(v, r, ramsey=False, rtol=1e-12)


def test_ramsey_constraints_perfect_foresight(tmp_path):
    """tests/optimal_policy/RamseyConstraints/test1: ramsey_constraints become complementarity
    conditions on the first-order conditions; Dynare runs them with options_.solve_algo=10 and
    stack_solve_algo=7 (set by MATLAB lines, which pydyn does not act on: lmmcp=True here). The steady
    state of this linear model is found to Dynare's tolerance, so the multipliers (about 1e-10 instead
    of 0) and the path differ from Dynare's at that level."""
    r = ref("pol_rcons")
    m = load(tmp_path, "pol_rcons")
    assert not set(m.unhandled_statements) - {"rplot"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ys, _ = pydyn.steady_state(m)
        np.testing.assert_allclose(ys, r["steady_state"], atol=1e-9)
        pf = pydyn.perfect_foresight(m, lmmcp=True)
    np.testing.assert_allclose(pf.endo.T, r["endo_simul"], atol=1e-9)
    assert np.min(pf.endo[:, m.endo_names.index("i")]) >= -1e-10        # the bound i >= 0 holds (to the solver tolerance)


# tests/discretionary_policy models: (case, statement's variable list given to stoch_simul's moments)
DISCRETION = ["pol_dennis",          # periods=200: Dynare's conditional welfare uses the first simulated shock
              "pol_gali_disc",       # discretionary_tol=1e-12
              "pol_nk_disc_fwd",     # purely forward-looking: no state variables
              "pol_gali15",          # STEADY_STATE(y) in the model
              "pol_gali15_nl"]       # nonlinear model with a steady_state_model block, STEADY_STATE(log(Y))


@pytest.mark.parametrize("case", DISCRETION)
def test_discretionary_policy(case, tmp_path):
    from pydyn import moments
    r = ref(case)
    m = load(tmp_path, case)
    assert not set(m.unhandled_statements) - {"resid"}
    assert op.is_discretionary(m) and not op.is_ramsey(m)
    u = np.atleast_1d(r["exo_simul1"]) if "exo_simul1" in r else None
    res = op.discretionary_policy(m, u=u)
    dr = res.dr
    np.testing.assert_array_equal(dr["order_var"] + 1, r["order_var"])
    np.testing.assert_allclose(dr["ys"], r["ys"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(dr["ghx"], np.reshape(r["ghx"], dr["ghx"].shape), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(dr["ghu"], np.reshape(r["ghu"], dr["ghu"].shape), rtol=1e-11, atol=1e-12)
    assert_welfare(res.welfare, r, ramsey=False, rtol=1e-12, atol=1e-13)
    # stoch_simul's theoretical moments of the variables of the statement, from the same rules
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = next(s for s in m.statements if s["statementName"] == "discretionary_policy")
        mo = moments.theoretical_moments(m, dr, st.get("symbol_list") or m.orig_endo_names, order=1)
    if not st.get("options", {}).get("periods"):        # with periods, Dynare reports simulated moments
        np.testing.assert_allclose(mo.var, np.reshape(r["var"], mo.var.shape), rtol=1e-10, atol=1e-13)
    # the iterations start from the previous solution (Dynare's persistent variable): same rules
    again = op.discretionary_policy(m, u=u)
    np.testing.assert_allclose(again.dr["ghx"], dr["ghx"], atol=1e-8)
    assert again.iterations <= res.iterations


def test_discretionary_policy_errors(tmp_path):
    m = load(tmp_path, "pol_dennis")
    m.options["discretionary_policy"] = {**m.options["discretionary_policy"], "instruments": {"symbol_list": ["i", "y"]}}
    m.statements = [dict(st, options={**st.get("options", {}), "instruments": {"symbol_list": ["i", "y"]}})
                    if st["statementName"] == "discretionary_policy" else st for st in m.statements]
    with pytest.raises(op.DiscretionaryPolicyError, match="more declared instruments"):
        op.discretionary_policy(m)


# ------------------------------------------------------------------------------------------------ OSR
# tests/optimal_policy/OSR models: osr_example (weights with a covariance term), objective_correctness
# (TolFun=1e-20; covariance weights), obj_corr_non_stat_vars (15 parameters, non-stationary
# variables in the weights), OSR_planner_objective (welfare at order 2; csminwel instead of CMA-ES
# and no bounds, which only constrained optimisers accept)
OSR_CASES = ["pol_osr", "pol_osr_corr", "pol_osr_cov", "pol_osr_planner", "pol_osr_ns"]


@pytest.mark.parametrize("case", OSR_CASES)
def test_osr_objective_at_dynare_optimum(case, tmp_path):
    """The loss (weighted moments, or minus welfare) at Dynare's optimal parameters is Dynare's."""
    r = ref(case)
    m = load(tmp_path, case)
    assert not m.unhandled_statements
    names = r["osr_names"].split(",")
    assert names == m.options["osr_params"]["symbol_list"]
    f = op.osr_objective(m, np.atleast_1d(r["osr_x"]), names)
    assert f == pytest.approx(r["osr_f"], rel=1e-12, abs=1e-13)


def test_osr_rules_at_optimum(tmp_path):
    """osr ends with stoch_simul at the optimum: the rules there."""
    r = ref("pol_osr")
    m = load(tmp_path, "pol_osr")
    for name, v in zip(r["osr_names"].split(","), np.atleast_1d(r["osr_x"])):
        m.set_param(name, v)
    dr = rules(m, 1)
    assert_rules(dr, r, 1, rtol=1e-12, atol=1e-13)


def test_osr_planner_objective_same_path(tmp_path):
    """Welfare-maximising OSR at order 2 (planner_objective): csminwel takes the same 20 iterations and
    76 evaluations as in Dynare and stops at the same point (zero gradient)."""
    r = ref("pol_osr_planner")
    m = load(tmp_path, "pol_osr_planner")
    res = op.osr(m)
    info = r["osr_info"]
    assert (res.iterations, res.func_count, res.exitflag) == (int(info["iterations"]), int(info["funcCount"]), int(info["exitflag"]))
    assert res.optim_params["alpha"] == pytest.approx(float(r["osr_x"]), rel=1e-10)
    assert res.objective_function == pytest.approx(r["osr_f"], rel=1e-13)
    assert m.get_param("alpha") == res.optim_params["alpha"]
    assert_rules(res.dr, r, 2, rtol=1e-8, atol=1e-10)


def test_osr_csminwel_minimum(tmp_path):
    """osr(optim=('TolFun',1e-20)): the minimum agrees with Dynare's to 1e-7 in the parameters. The
    numerical gradients near the minimum are dominated by rounding (the losses of pydyn and Dynare differ
    by about 1e-15), so the iterations are not the same ones."""
    r = ref("pol_osr_corr")
    m = load(tmp_path, "pol_osr_corr")
    res = op.osr(m)
    x = np.array([res.optim_params[n] for n in r["osr_names"].split(",")])
    np.testing.assert_allclose(x, r["osr_x"], rtol=1e-6)
    assert res.objective_function == pytest.approx(r["osr_f"], rel=1e-13)


@pytest.mark.parametrize("case,rel", [("pol_osr", 2e-6), ("pol_osr_cov", 1e-10), ("pol_osr_ns", 0.02)])
def test_osr_csminwel_stops_near_dynare(case, rel, tmp_path):
    """With Dynare's default TolFun=1e-7 csminwel stops in a flat region, as soon as an iteration improves
    the loss by less than that; with paths separated by rounding differences, the stopping points differ.
    On the 15-parameter problem pydyn stops 1% above Dynare's loss (Dynare iterates 164 times, pydyn 44:
    the first iterations agree to 1e-8)."""
    r = ref(case)
    m = load(tmp_path, case)
    res = op.osr(m)
    assert res.objective_function <= res.initial_objective
    assert res.objective_function == pytest.approx(r["osr_f"], rel=rel)


def test_osr_refusals(tmp_path):
    m = load(tmp_path, "pol_osr")
    with pytest.raises(NotImplementedError, match="opt_algo=9"):
        op.osr(m, opt_algo=9)
    with pytest.raises(op.OSRError, match="constrained optimizer"):
        op.osr(m, bounds=np.array([[0, 1]] * 4))
    with pytest.raises(op.OSRError, match="only supports order=1"):
        op.osr(m, order=2)


def test_evaluate_planner_objective_options_json(tmp_path):
    """Dynare 7.2's preprocessor writes evaluate_planner_objective(options) as invalid JSON (the
    options before the statement): Model repairs it. Welfare beyond order 2 is refused."""
    text = (REF / "pol_neo_ramsey.mod").read_text().replace(
        "evaluate_planner_objective;", "evaluate_planner_objective(periods=10000,drop=1000);")
    (tmp_path / "pol_neo_ramsey.mod").write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "pol_neo_ramsey.mod", extra_args=("-DORDER=3",)))
        assert m.options["evaluate_planner_objective"] == {"ramsey.drop": 1000, "ramsey.periods": 10000}
        dr = pydyn.k_order(m, order=3)
    with pytest.raises(NotImplementedError, match="k_order_welfare"):
        op.evaluate_planner_objective(m, dr, order=3)
