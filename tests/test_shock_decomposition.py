"""Smoother at estimated parameter sets, observation trends, and the shock-decomposition workflow
(groups, plotting numbers, realtime, squeeze, init2shocks) against Dynare 7.2.

References: reference/sd_*.mat, made by reference/r_shock_decomposition.m from reference/sd_nk.mod
(New Keynesian model, 3 observables, one with measurement error, correlated shocks) and
reference/sd_ur.mod (unit root, observation trends), with the simulated data est_nk_data.m,
est_ur_data.m and est_ur_miss.m. Dynare's own parameter vectors (posterior mode, mean, median,
ML mode) are fed through ``theta=``, so the comparisons do not depend on an optimiser or a sampler.
"""
import functools
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import smoother
from pydyn.estimation import Estimation
from pydyn.options import UnsupportedOptionWarning

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))

pytestmark = [pytest.mark.filterwarnings("ignore:statements not handled"),
              pytest.mark.filterwarnings("ignore::pydyn.options.UnsupportedOptionWarning")]

RTOL, ATOL = 1e-9, 1e-12
VL = ["y", "pi", "robs"]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sd")
    for f in ("sd_nk.mod", "sd_ur.mod", "est_nk_data.m", "est_ur_data.m", "est_ur_miss.m"):
        shutil.copy(REF / f, d)
    return d


@pytest.fixture(scope="module")
def get(workdir):
    @functools.lru_cache(maxsize=None)
    def model(name, case):
        return pydyn.Model(pydyn.run_preprocessor(workdir / f"{name}.mod", extra_args=(f"-DCASE={case}",)))

    @functools.lru_cache(maxsize=None)
    def est(name, case):
        return Estimation(model(name, case))

    return model, est


@functools.lru_cache(maxsize=None)
def ref(name):
    return loadmat(REF / f"{name}.mat", squeeze_me=True, struct_as_record=False)


def close(a, b, rtol=RTOL, atol=ATOL):
    a = np.asarray(a, float)
    np.testing.assert_allclose(a, np.reshape(b, a.shape), rtol=rtol, atol=atol)


def table(d, names):
    return np.column_stack([d[v] for v in names])


def field(r, key):
    return r[key] if isinstance(r, dict) else getattr(r, key, None)


def has(r, key):
    return key in r if isinstance(r, dict) else hasattr(r, key)


def compare(res, r, m):
    """Every smoother field Dynare stored (a dict from loadmat or a struct inside one)."""
    assert field(r, "endo_names").split(",") == m.endo_names and field(r, "varobs").split(",") == res.varobs
    close(table(res.SmoothedVariables, m.endo_names), field(r, "smoothed"))
    close(table(res.SmoothedShocks, m.exo_names), field(r, "shocks"))
    close(np.array([res.Constant[v][0] for v in m.endo_names]), field(r, "constant"))
    close(np.array([res.Init_State[v] for v in m.endo_names]), field(r, "init_state"))
    close(res.Init_State_Vector, field(r, "init_state_vector"))
    close(res.TrendCoeffs, field(r, "trend_coeffs"))
    close(res.SteadyState, field(r, "ys"))
    n = 0
    for key, mine in (("updated", res.UpdatedVariables), ("filtered", res.FilteredVariables), ("trend", res.Trend)):
        if has(r, key):
            close(table(mine, m.endo_names), field(r, key))
            n += 1
    if has(r, "meas_err"):
        close(table(res.SmoothedMeasurementErrors, res.varobs), field(r, "meas_err"))
    if has(r, "kstep"):
        assert list(np.atleast_1d(field(r, "filter_step_ahead"))) == res.filter_step_ahead
        close(res.FilteredVariablesKStepAhead, field(r, "kstep"))
    return n


# ------------------------------------------------------------------ parameter sets: Dynare's vectors
@pytest.mark.parametrize("pset,vec", [("posterior_mode", "theta_mode"), ("posterior_mean", "theta_mean"),
                                      ("posterior_median", "theta_median"), ("prior_mean", "theta_prior_mean")])
def test_smoother_at_dynare_parameter_vectors(get, pset, vec):
    """Bayesian estimation in Dynare (mode_compute=4, 1200 draws): its vectors through theta=.
    Covers shock and measurement-error standard deviations, a shock correlation, structural
    parameters and a parameter recomputed in the steady state (rbar from beta)."""
    model, est = get
    m, e, r = model("sd_nk", 1), est("sd_nk", 1), ref("sd_nk_case1")
    assert e.names == r["names"].split(";")
    before = (m.params.copy(), np.array(m.Sigma_e), dict(m.measurement_errors))
    res = smoother.smooth(m, estimation=e, theta=r[vec], filtered_vars=True)
    d = r[pset]
    compare(res, d, m)
    assert res.parameter_set == "theta" and np.array_equal(res.theta, r[vec])
    assert res.SmoothedMeasurementErrors.keys() == set(m.varobs)
    if hasattr(d, "shock_decomposition"):
        dec = smoother.shock_decomposition(m, res)
        close(dec.z, d.shock_decomposition)
        assert smoother.check(res, dec)["decomposition"] < 1e-12
    if hasattr(d, "initval_decomposition"):
        icd = smoother.initial_condition_decomposition(m, res)
        close(icd.z[..., d.icd_idx.astype(int) - 1], d.initval_decomposition)
    # the caller's model is untouched
    assert np.array_equal(m.params, before[0]) and np.array_equal(m.Sigma_e, before[1])
    assert m.measurement_errors == before[2]


def test_shock_decomposition_statement_at_posterior_mode(get):
    """shock_decomposition(parameter_set=posterior_mode) of the .mod file: Dynare's stored
    oo_.shock_decomposition; the statement's parameter_set is read from the file."""
    model, est = get
    m, e, r = model("sd_nk", 1), est("sd_nk", 1), ref("sd_nk_case1")
    dec = smoother.shock_decomposition(m, estimation=e, mode=r["theta_mode"])
    assert dec.parameter_set == "posterior_mode"
    close(dec.z, r["sd_in_mod"])
    with pytest.raises(smoother.SmootherError, match="estimation="):
        smoother.shock_decomposition(m)                       # posterior_mode without an estimation
    cal = smoother.shock_decomposition(m, parameter_set="calibration")
    assert cal.parameter_set == "calibration" and np.max(np.abs(cal.z - dec.z)) > 1e-3


def test_parameter_set_names_go_through_the_same_path(get):
    """parameter_set=... with a Python Estimation: mode / mcmc objects, plain vectors and vectors
    stored on the Estimation all give the result of theta=."""
    model, est = get
    m, e, r = model("sd_nk", 1), est("sd_nk", 1), ref("sd_nk_case1")
    base = smoother.smooth(m, estimation=e, theta=r["theta_mode"])

    class Mode:                                               # what est.mode() returns: has .x
        x = r["theta_mode"]

    for kw in (dict(mode=Mode()), dict(mode=r["theta_mode"])):
        res = smoother.smooth(m, estimation=e, parameter_set="posterior_mode", **kw)
        assert res.parameter_set == "posterior_mode"
        np.testing.assert_array_equal(res.smoothed, base.smoothed)
        np.testing.assert_array_equal(res.shocks, base.shocks)
    with pytest.raises(smoother.SmootherError, match="mode="):
        smoother.smooth(m, estimation=e, parameter_set="posterior_mode")
    with pytest.raises(smoother.SmootherError, match="does not fit"):
        smoother.smooth(m, estimation=e, parameter_set="mle_mode", mode=r["theta_mode"])
    with pytest.raises(smoother.SmootherError):
        smoother.smooth(m, theta=r["theta_mode"])             # no Estimation to map the vector
    with pytest.raises(smoother.SmootherError, match="entries"):
        smoother.smooth(m, estimation=e, theta=r["theta_mode"][:-1])
    with pytest.raises(ValueError):
        smoother.smooth(m, estimation=e, parameter_set="posterior")
    try:
        e.posterior_mode = r["theta_mode"]                    # stored on the object by the caller
        res = smoother.smooth(m, estimation=e, parameter_set="posterior_mode")
        np.testing.assert_array_equal(res.smoothed, base.smoothed)
    finally:
        del e.posterior_mode
    # posterior mean / median from draws: the statistics of the kept draws
    rng = np.random.default_rng(3)
    draws = r["theta_mode"] + 1e-3 * rng.standard_normal((2, 40, len(e.names)))
    from pydyn.estimation import MCMCResult
    mc = MCMCResult(draws=draws, log_posterior=np.zeros((2, 40)), acceptance_rate=np.ones(2), names=e.names,
                    x0=draws[:, 0], drop=0.5)
    kept = draws[:, 20:].reshape(-1, len(e.names))
    for pset, vec in (("posterior_mean", kept.mean(axis=0)), ("posterior_median", np.median(kept, axis=0))):
        close(smoother.parameter_vector(e, pset, mcmc=mc), vec, atol=0)
        a = smoother.smooth(m, estimation=e, parameter_set=pset, mcmc=mc)
        b = smoother.smooth(m, estimation=e, theta=vec)
        np.testing.assert_array_equal(a.smoothed, b.smoothed)
    close(smoother.parameter_vector(e, "prior_mean"), r["theta_prior_mean"])


def test_calibration_hook_and_fallback(get):
    """est.calibration(theta) is used when the Estimation has it; otherwise model_params /
    covariances. Both give the same numbers."""
    model, est = get
    m, e, r = model("sd_nk", 1), est("sd_nk", 1), ref("sd_nk_case1")
    theta = r["theta_mean"]
    Q, H = e.covariances(theta)
    params = e.model_params(theta)
    calls = []

    class WithHook:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "model_params" or name == "covariances":
                raise AssertionError("the hook must be preferred")
            return getattr(self._inner, name)

        def calibration(self, th):
            calls.append(np.array(th))
            return {"params": params, "Sigma_e": Q, "H": H}

    class WithoutHook:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "calibration":
                raise AttributeError(name)
            return getattr(self._inner, name)

    a = smoother.smooth(m, estimation=WithHook(e), theta=theta)
    b = smoother.smooth(m, estimation=WithoutHook(e), theta=theta)
    assert len(calls) == 1 and not hasattr(WithoutHook(e), "calibration")
    np.testing.assert_array_equal(a.smoothed, b.smoothed)
    np.testing.assert_array_equal(a.shocks, b.shocks)
    close(a.Sigma_e, Q, atol=0)
    close(a.me_cov, H, atol=0)
    close(table(a.SmoothedVariables, m.endo_names), r["posterior_mean"].smoothed)


def test_maximum_likelihood_mode(get):
    """ML estimation: Dynare's mle_mode vector through theta=, and the Python optimiser's mode
    through parameter_set="mle_mode" (optimisers differ: 1e-4 on that second comparison)."""
    model, est = get
    m, e, r = model("sd_nk", 2), est("sd_nk", 2), ref("sd_nk_case2")
    assert e.names == r["names"].split(";")
    res = smoother.smooth(m, estimation=e, theta=r["theta_mode"], filtered_vars=True)
    assert compare(res, r["mle_mode"], m) == 2
    dec = smoother.shock_decomposition(m, res)
    close(dec.z, r["mle_mode"].shock_decomposition)
    close(dec.z, r["sd_in_mod"])
    mode = e.mode(compute_hessian=False)
    close(mode.x, r["theta_mode"], rtol=1e-4, atol=1e-5)
    own = smoother.shock_decomposition(m, estimation=e, mode=mode)      # parameter_set=mle_mode from the file
    assert own.parameter_set == "mle_mode"
    close(own.z, r["sd_in_mod"], rtol=1e-4, atol=1e-4)
    with pytest.raises(smoother.SmootherError, match="priors"):
        smoother.smooth(m, estimation=e, parameter_set="prior_mean")


def test_prior_mode(get):
    """bayestopt_.p5 for inverse gamma (both types), gamma, beta (also generalised), normal,
    Weibull and uniform priors; Dynare's smoother fails at the prior mode when a prior is uniform."""
    model, est = get
    e, r = est("sd_nk", 1), ref("sd_nk_case1")
    pm = smoother.prior_mode(e.prior)
    close(pm, r["theta_prior_mode"])                          # NaN for the two uniform priors
    assert np.isnan(pm).sum() == 2 and "NaN" in r["error_prior_mode"]
    with pytest.raises(smoother.SmootherError, match="uniform"):
        smoother.smooth(model("sd_nk", 1), estimation=e, parameter_set="prior_mode")
    e4, r4 = est("sd_ur", 4), ref("sd_ur_case4")
    close(smoother.parameter_vector(e4, "prior_mode"), r4["theta_prior_mode"])


# ------------------------------------------------------------------ observation trends, lik_init=2
@pytest.mark.parametrize("case,algo", [(1, (4, 1)), (2, (3, 2))])
def test_observation_trends_calibrated(get, case, algo):
    """Linear trends in the measurement equations with the diffuse filter: Trend, TrendCoeffs,
    smoothed / updated / filtered / k-step variables. Case 2: prefilter (demeaned trend) and
    missing observations."""
    model, _ = get
    m, r = model("sd_ur", case), ref(f"sd_ur_case{case}")
    res = smoother.smooth(m, replicate_dynare_bugs=True)
    assert (res.kalman_algo, res.diffuse_periods) == algo
    assert compare(res, r, m) == 3
    close(smoother.shock_decomposition(m, res).z, r["shock_decomposition"])
    i = m.endo_names.index("yobs")
    assert res.TrendCoeffs[i] == 0.05 and np.all(res.Trend["g"] == 0)
    first = res.first_obs
    want = 0.05 * np.arange(first, first + res.nobs)
    close(res.Trend["yobs"], want - want.mean() if case == 2 else want, atol=1e-15)
    # the trend is part of the observed variables only, and not of the decomposition
    seen = ~np.isnan(res.data[:, 0])
    close(res.SmoothedVariables["yobs"][seen] - res.Trend["yobs"][seen] - res.Constant["yobs"][seen],
          res.data[seen, 0], atol=1e-10)
    close(res.SmoothedVariables["y"], res.smoothed[:, m.endo_names.index("y")])


@pytest.mark.parametrize("case,pset,algo", [(3, "prior_mean", 1), (4, "prior_mode", 4)])
def test_trends_with_estimated_slope(get, case, pset, algo):
    """estimation(mode_compute=0, smoother): the smoother at the initial values, and
    shock_decomposition(parameter_set=prior_mean / prior_mode); the trend slope is an estimated
    parameter. Case 3 uses lik_init=2, case 4 the diffuse filter."""
    model, est = get
    m, e, r = model("sd_ur", case), est("sd_ur", case), ref(f"sd_ur_case{case}")
    close(e.x0, r["theta0"], atol=0)
    res = smoother.smooth(m, estimation=e, theta=r["theta0"], replicate_dynare_bugs=True, filtered_vars=(case == 3))
    assert res.kalman_algo == algo
    compare(res, r["at_theta0"], m)
    assert res.TrendCoeffs[m.endo_names.index("yobs")] == 0.08
    at = smoother.smooth(m, estimation=e, parameter_set=pset, replicate_dynare_bugs=True, filtered_vars=True)
    assert compare(at, getattr(r, pset) if not isinstance(r, dict) else r[pset], m) == 3
    close(smoother.shock_decomposition(m, at).z, r[pset].shock_decomposition)
    # the statement in the file names the parameter set
    close(smoother.shock_decomposition(m, estimation=e, replicate_dynare_bugs=True).z, r["sd_in_mod"])


def test_lik_init_2_conventions(get):
    model, est = get
    m, e = model("sd_ur", 3), est("sd_ur", 3)
    res = smoother.smooth(m, estimation=e, theta=e.x0)
    assert res.options["lik_init"] == 2 and res.diffuse_periods == 0
    chk = smoother.check(res)
    # period 1 starts from the large variance directly (Dynare), so the identity only holds from period 2
    prev, G, H = res.smoothed[:-1], res.G, res.H
    assert np.max(np.abs(prev @ G.T + res.shocks[1:] @ H.T - res.smoothed[1:])) < 1e-10
    assert chk["measurement"] < 1e-10
    with pytest.raises(smoother.SmootherError, match="contradictory"):
        smoother.smooth(m, estimation=e, theta=e.x0, diffuse_filter=True)
    res5 = smoother.smooth(m, estimation=e, theta=e.x0, Harvey_scale_factor=50.0)
    assert np.max(np.abs(res5.smoothed - res.smoothed)) > 1e-6


# ------------------------------------------------------------------ filter_initial_state
@pytest.mark.parametrize("case,algo", [(4, 1), (5, 2)])
def test_filter_initial_state(get, case, algo):
    model, _ = get
    m, r = model("sd_nk", case), ref(f"sd_nk_case{case}")
    res = smoother.smooth(m, replicate_dynare_bugs=True)
    assert res.kalman_algo == algo
    assert compare(res, r, m) == 2
    if case == 4:
        close(smoother.shock_decomposition(m, res).z, r["shock_decomposition"])
    # Dynare centres the period-0 state on T*a: the transition identity fails in period 1 only
    assert smoother.check(res)["transition"] > 0.1
    good = smoother.smooth(m)
    np.testing.assert_array_equal(good.smoothed, res.smoothed)
    np.testing.assert_array_equal(good.shocks, res.shocks)
    assert max(smoother.check(good).values()) < 1e-12
    assert np.max(np.abs(good.init_state - res.init_state)) > 0.1
    # the block matters: without it the smoothed path differs
    stripped = [st for st in m.statements if st.get("statementName") != "filter_initial_state"]
    saved, m.statements = m.statements, stripped
    try:
        plain = smoother.smooth(m)
    finally:
        m.statements = saved
    assert np.max(np.abs(plain.smoothed - good.smoothed)) > 1e-3


# ------------------------------------------------------------------ groups and the numbers plotted
@pytest.fixture(scope="module")
def mode_run(get):
    model, est = get
    m, r = model("sd_nk", 1), ref("sd_nk_case1")
    res = smoother.smooth(m, estimation=est("sd_nk", 1), theta=r["theta_mode"], parameter_set="posterior_mode")
    return m, res, smoother.shock_decomposition(m, res), smoother.initial_condition_decomposition(m, res)


def test_group_shocks_matches_dynare(mode_run):
    """make_the_groups of plot_shock_decomposition.m with a full and a partial shock_groups block."""
    m, res, dec, _ = mode_run
    p = ref("sd_nk_case1_plot")
    iv = [m.endo_names.index(v) for v in VL]
    assert smoother.shock_groups(m, "grp") == {"supply": ["eu"], "demand and policy": ["ea", "er"]}
    g = smoother.group_shocks(dec, model=m, name="grp")
    assert g.columns == ["supply", "demand and policy", "Initial values", "smoothed"]
    close(g.z[iv], p["p_grp"])
    g = smoother.group_shocks(dec, model=m, name="partial")
    assert g.columns == ["policy", "Others + Initial Values", "smoothed"]
    close(g.z[iv], p["p_partial"])
    close(g.z[:, :-1].sum(axis=1), g.z[:, -1], atol=1e-12)
    assert smoother.shock_groups(m) == {"policy": ["er"]}     # the last block when no name is given
    with pytest.raises(smoother.SmootherError):
        smoother.shock_groups(m, "nope")
    with pytest.raises(smoother.SmootherError):
        smoother.group_shocks(g, model=m, name="grp")         # grouped already


def test_plot_numbers_match_dynare(mode_run):
    """plot_shock_decomposition(..., get_decomp_only): groups, type=yoy / aoa, diff, flip,
    init2shocks, and the steady state Dynare attaches to the axis."""
    m, res, dec, icd = mode_run
    r, p = ref("sd_nk_case1"), ref("sd_nk_case1_plot")
    f = functools.partial(smoother.decomposition_plot_data, dec, VL, m)
    last = r["ys_last_solved"]              # Dynare's "steady state" here: oo_.dr.ys, the posterior-mean one
    close(last, r["posterior_mean"].ys)
    T = res.nobs

    d = f(use_shock_groups="grp")
    close(d.z, p["p_grp"])
    assert d.labels == ["supply", "demand and policy", "Initial values"] and list(d.periods) == list(range(1, T + 1))
    # our default steady state is that of the parameter set (posterior mode); Dynare's is stale
    close(d.steady_state, res.SteadyState[[m.endo_names.index(v) for v in VL]])
    assert abs(d.steady_state[2] - p["p_grp_ss"][2]) > 0.05
    close(f(use_shock_groups="grp", steady_state=last).steady_state, p["p_grp_ss"])

    d = f(type="yoy", steadystate=True, steady_state=last)
    close(d.z, p["p_yoy"])
    close(d.steady_state, p["p_yoy_ss"])
    assert d.z.shape[2] == T - 3 and d.periods[0] == 4 and d.steadystate

    d = f(type="qoq", diff=True, flip=True, use_shock_groups="grp", steady_state=last)
    close(d.z, p["p_diff_flip"])                             # first period NaN in both
    assert np.isnan(d.z[:, :, 0]).all() and not np.any(d.steady_state)
    close(f(type="yoy", diff=True, flip=True).z, p["p_yoy_diff_flip"])

    d = f(type="aoa", steady_state=last)                     # every fourth period, sample starting in Q1
    close(d.z, p["p_aoa"])
    close(d.steady_state, p["p_aoa_ss"])
    assert list(d.periods) == list(range(1, T // 4 + 1))
    close(f(type="aoa", first_quarter=3).z, dec.z[[m.endo_names.index(v) for v in VL]][:, :, 1::4], atol=0)   # Q3, Q4, ...

    d = f(init2shocks="i2s", icd=icd)
    close(d.z, p["p_i2s"])
    assert d.labels == ["ea + a", "eu + u", "er", "Initial values"]
    assert smoother.init2shocks(m, "i2s") == [("a", "ea"), ("u", "eu")]
    d = f(init2shocks="i2s", icd=icd, use_shock_groups="grp")
    close(d.z, p["p_i2s_grp"])
    assert d.labels == ["supply", "demand and policy", "Initial values"]
    with pytest.raises(smoother.SmootherError, match="icd"):
        f(init2shocks="i2s")
    with pytest.raises(ValueError):
        f(type="mom")


def test_cropping_and_panels(mode_run):
    m, res, dec, _ = mode_run
    d = smoother.decomposition_plot_data(dec, VL, m, plot_init_date=11, plot_end_date=40)
    assert [p["name"] for p in d.panels] == VL
    pan = d.panels[1]
    assert list(pan["periods"]) == list(range(11, 41)) and pan["contributions"].shape == (4, 30)
    close(pan["contributions"], dec.z[m.endo_names.index("pi"), :4, 10:40], atol=0)
    close(pan["contributions"].sum(axis=0), pan["total"], atol=1e-12)
    assert d.z.shape[2] == res.nobs                          # z itself is not cropped (as get_decomp_only)
    with pytest.raises(smoother.SmootherError):
        smoother.decomposition_plot_data(dec, VL, m, plot_init_date=500)
    with pytest.warns(UserWarning, match="after the last period"):
        d = smoother.decomposition_plot_data(dec, VL, m, plot_end_date=500)
    assert d.panels[0]["periods"][-1] == res.nobs
    # a variable that never moves gives no figure, as in Dynare
    flat = smoother.ShockDecomposition(z=np.zeros((1, 3, 5)), endo_names=["v"], columns=["e", "initial_conditions", "smoothed"])
    assert smoother.decomposition_plot_data(flat).panels == []
    # default variable list: the model's original variables
    assert smoother.decomposition_plot_data(dec, model=m).var_list == m.orig_endo_names
    bottom, top = smoother.stack_bars(np.array([[1.0, -1.0, np.nan], [2.0, 3.0, 1.0], [-1.0, -2.0, 0.0]]))
    np.testing.assert_array_equal(bottom, [[0, 0, 0], [1, 0, 0], [0, -1, 0]])
    np.testing.assert_array_equal(top[:, :2], [[1, -1], [3, 3], [-1, -3]])


# ------------------------------------------------------------------ calibration: realtime, expand, squeeze
@pytest.fixture(scope="module")
def cal_run(get):
    model, _ = get
    m = model("sd_nk", 3)
    res = smoother.smooth(m)
    return m, res, smoother.shock_decomposition(m, res), smoother.realtime_shock_decomposition(m)


def test_calibrated_case_and_expand(cal_run):
    m, res, dec, _ = cal_run
    r, p = ref("sd_nk_case3"), ref("sd_nk_case3_plot")
    compare(res, r, m)
    close(dec.z, r["shock_decomposition"])
    d = smoother.decomposition_plot_data(dec, VL, m, use_shock_groups="grp", steadystate=True)
    close(d.z, p["p_grp"])
    close(d.steady_state, p["p_grp_ss"])                      # calibration: no stale steady state
    # Dynare numbers the sample from 1 whatever first_obs is (first_obs=5 here)
    assert res.first_obs == 5 and d.periods[0] == 1 == p["drawn_first_obs"].f1_p0_line_x[0]
    # "expand group": the shocks of one group, the rest with the initial values
    d = smoother.decomposition_plot_data(dec, ["pi"], m, use_shock_groups="grp", expand="demand and policy")
    close(d.z[0], p["p_expand"])
    assert d.labels == ["ea", "er", "Others + Initial Values"]
    with pytest.raises(smoother.SmootherError):
        smoother.group_shocks(dec, model=m, name="grp", expand="nope")


def test_realtime_shock_decomposition(cal_run):
    """realtime_shock_decomposition(forecast=4, presample=20, save_realtime=[30,45,60]) of the file:
    all four families of results Dynare stores."""
    m, res, dec, rt = cal_run
    r = ref("sd_nk_case3")
    assert (rt.nobs, rt.forecast) == (60, 4)
    stores = {"rt_": rt.realtime, "cond_": rt.conditional, "fcst_": rt.realtime_forecast,
              "rtcond_": rt.realtime_conditional}
    n = 0
    for prefix, store in stores.items():
        keys = {k[len(prefix):] for k in r if k.startswith(prefix)}
        assert keys == set(store), (prefix, keys, set(store))
        for k in keys:
            assert store[k].shape == r[prefix + k].shape
            close(store[k], r[prefix + k])
            n += 1
    assert n == 13
    assert rt.realtime["pool"].shape == (m.n_endo, m.n_exo + 2, 64) and rt.realtime["time_30"].shape[2] == 34
    # Dynare's presample rule: max(1, estimation presample - 1), the option can only lower it
    assert rt.presample == 1
    # Dynare's realtime recursion gives the shocks of period 1 to the initial conditions
    last = rt.realtime["time_60"][:, :, :60]
    assert np.all(last[:, :m.n_exo, 0] == 0)
    close(last[:, -1], dec.z[:, -1], atol=0)
    alt = smoother.shock_decomposition(m, res, init_state=1)
    close(last, alt.z, atol=1e-12)


def test_realtime_variants(cal_run):
    m, _, _, rt = cal_run
    r = ref("sd_nk_case3")
    rt0 = smoother.realtime_shock_decomposition(m, forecast=0, presample=0, save_realtime=[10, 60])
    close(rt0.realtime["pool"], r["rt0_pool"])
    close(rt0.realtime["time_10"], r["rt0_time_10"])
    close(rt0.conditional["pool"], r["rt0_cond_pool"])
    assert rt0.realtime_forecast == {} and rt0.realtime_conditional == {} and set(rt0.conditional) == {"pool"}
    fast = smoother.realtime_shock_decomposition(m, fast_realtime=[40, 50], save_realtime=[45])
    close(fast.realtime["pool"], r["fast_pool"])
    close(fast.conditional["pool"], r["fast_cond_pool"])
    close(fast.realtime_forecast["time_45"], r["fast_fcst_time_45"])
    assert np.max(np.abs(fast.realtime["pool"] - rt.realtime["pool"])) > 1e-4


@pytest.mark.parametrize("name,realtime,vintage,first", [("rt_pool", 1, 0, 1), ("rt_vintage", 1, 45, 1),
                                                         ("cond_pool", 2, 0, 1), ("cond_vintage", 2, 45, 45),
                                                         ("fcst_pool", 3, 0, 1), ("fcst_vintage", 3, 45, 45)])
def test_realtime_selection_for_plots(cal_run, name, realtime, vintage, first):
    """What plot_shock_decomposition(realtime=, vintage=, use_shock_groups=grp) takes from oo_."""
    m, _, _, rt = cal_run
    p = ref("sd_nk_case3_plot")
    sel = rt.select(realtime, vintage)
    d = smoother.decomposition_plot_data(sel, VL, m, use_shock_groups="grp")
    close(d.z, p["p_" + name])
    assert d.periods[0] == first
    with pytest.raises(smoother.SmootherError):
        rt.select(1, 31)
    with pytest.raises(ValueError):
        rt.select(4)


def test_squeeze(cal_run):
    m, _, dec, rt = cal_run
    p = ref("sd_nk_case3_plot")
    assert list(p["squeezed_i_var"]) == [m.endo_names.index(v) + 1 for v in ("pi", "y")]
    sq = smoother.squeeze_shock_decomposition(dec, ["pi", "y"])
    assert sq.endo_names == ["pi", "y"]
    close(sq.z, p["squeezed"])
    sq_rt = smoother.squeeze_shock_decomposition(rt, ["pi", "y"])
    close(sq_rt.realtime["pool"], p["squeezed_rt_pool"])
    assert sq_rt.realtime_forecast["time_45"].shape[0] == 2 and len(sq_rt.steady_state) == 2
    d = smoother.decomposition_plot_data(sq, model=m, use_shock_groups="grp")
    assert d.var_list == ["y", "pi"] or d.var_list == ["pi", "y"]
    with pytest.raises(smoother.SmootherError):
        smoother.squeeze_shock_decomposition(dec, ["nope"])


# ------------------------------------------------------------------ screening of shocks
def screened():
    p = ref("sd_nk_case3_plot")
    z = p["screen_z"]
    cols = [f"s{j}" for j in range(1, z.shape[1] - 1)] + ["initial_conditions", "smoothed"]
    return p, smoother.ShockDecomposition(z=z, endo_names=["v1", "v2"], columns=cols)


def test_screen_shocks_rule():
    """More than 18 components: the 16 largest shocks (mean absolute contribution) for each
    variable, the others summed; nothing happens with groups or 18 components or fewer."""
    p, dec = screened()
    d = smoother.decomposition_plot_data(dec, screen_shocks=True)
    want = p["drawn_screen"].f1_p0_labels.split("|")[1:][::-1]             # Dynare lists them bottom-up
    assert d.panels[0]["labels"] == want and want[-2:] == ["Others", "Initial values"]
    assert d.panels[1]["labels"] != d.panels[0]["labels"]                  # chosen variable by variable
    assert d.panels[0]["contributions"].shape == (18, 12) and "SCREEN" in d.title
    close(d.panels[0]["contributions"].sum(axis=0), dec.z[0, -1], atol=1e-12)
    plain = smoother.decomposition_plot_data(dec)
    assert plain.panels[0]["contributions"].shape == (22, 12)
    few = smoother.ShockDecomposition(z=dec.z[:, 4:], endo_names=dec.endo_names, columns=dec.columns[4:])
    assert smoother.decomposition_plot_data(few, screen_shocks=True).panels[0]["contributions"].shape == (18, 12)
    grouped = smoother.decomposition_plot_data(dec, screen_shocks=True, groups={"a": ["s1", "s2"]})
    assert grouped.panels[0]["labels"] == ["a", "Others + Initial Values"]


# ------------------------------------------------------------------ options that are not acted on
def test_unhonoured_options_are_reported(workdir):
    m = pydyn.Model(pydyn.run_preprocessor(workdir / "sd_nk.mod", extra_args=("-DCASE=3",)))
    extra = [
        {"statementName": "shock_decomposition", "options": {"shock_decomp.with_epilogue": True, "parameter_set": "calibration"}},
        {"statementName": "realtime_shock_decomposition", "options": {"shock_decomp.forecast_params": [1], "shock_decomp.forecast": 2}},
        {"statementName": "plot_shock_decomposition", "options": {"plot_shock_decomp.write_xls": True,
                                                                  "plot_shock_decomp.realtime": 1,
                                                                  "plot_shock_decomp.use_shock_groups": "grp"}},
        {"statementName": "initial_condition_decomposition", "options": {"initial_condition_decomp.write_xls": True}},
        {"statementName": "calib_smoother", "options": {"filter_decomposition": True, "datafile": "est_nk_data",
                                                        "first_obs": 5, "nobs": 60}},
    ]
    m.statements = list(m.statements) + extra

    def caught(fn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = fn()
        return out, " | ".join(str(x.message) for x in w if issubclass(x.category, UnsupportedOptionWarning))

    res, msg = caught(lambda: smoother.smooth(m))
    assert "filter_decomposition" in msg
    dec, msg = caught(lambda: smoother.shock_decomposition(m, res))
    assert "with_epilogue" in msg and "shock_decomposition options" in msg and "init_state" not in msg
    icd, msg = caught(lambda: smoother.initial_condition_decomposition(m, res))
    assert "write_xls" in msg and "initial_condition_decomposition options" in msg
    rt, msg = caught(lambda: smoother.realtime_shock_decomposition(m))
    assert "forecast_params" in msg and rt.forecast == 2 and "save_realtime" not in msg
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from pydyn import plotting
    figs, msg = caught(lambda: plotting.plot_shock_decomposition(dec, m, ["pi"]))
    # realtime=1 in the file cannot be honoured with a full-sample decomposition; groups are
    assert "write_xls" in msg and "realtime" in msg and "use_shock_groups" not in msg
    assert figs[0].axes[0].containers[0].get_label() == "supply"
    figs2, msg = caught(lambda: plotting.plot_shock_decomposition(rt, m, ["pi"]))
    assert "realtime" not in msg.replace("realtime_shock", "")
    import matplotlib.pyplot as plt
    plt.close("all")
