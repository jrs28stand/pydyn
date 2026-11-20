"""Kalman smoother and shock decomposition against Dynare 7.2 (calib_smoother, shock_decomposition,
initial_condition_decomposition). References: reference/smo_*_case*.mat, made by
reference/r_smoother.m from reference/smo_*.mod (one case per value of the macro variable CASE)
and the simulated data reference/smo_*_data.m (reference/smo_make_data.py).

Models: smo_fs (cash-in-advance model, 2 observables, loglinear), smo_me (linear NK model,
3 observables, measurement error on two, missing observations, correlated shocks), smo_ur (two
unit roots, missing observations, diffuse filter).
"""
import functools
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import smoother

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))

pytestmark = pytest.mark.filterwarnings("ignore:statements not handled")

RTOL, ATOL = 1e-9, 1e-12
CASES = [("smo_fs", 1), ("smo_fs", 2), ("smo_fs", 3), ("smo_fs", 4),
         ("smo_me", 1), ("smo_me", 2), ("smo_me", 3),
         ("smo_ur", 1), ("smo_ur", 2), ("smo_ur", 3), ("smo_ur", 4), ("smo_ur", 5), ("smo_ur", 6), ("smo_ur", 7)]
# filter Dynare ends up using (1/2 stationary multi/univariate, 3/4 diffuse) and diffuse periods
EXPECTED_ALGO = {("smo_fs", 1): (1, 0), ("smo_fs", 2): (2, 0), ("smo_fs", 3): (1, 0), ("smo_fs", 4): (1, 0),
                 ("smo_me", 1): (1, 0), ("smo_me", 2): (2, 0), ("smo_me", 3): (1, 0),
                 ("smo_ur", 1): (3, 2), ("smo_ur", 2): (4, 2), ("smo_ur", 3): (4, 2), ("smo_ur", 4): (3, 1),
                 ("smo_ur", 5): (4, 2), ("smo_ur", 6): (3, 2), ("smo_ur", 7): (4, 2)}
# cases whose Dynare output carries one of the defects described in smoother.smooth
DEFECT = {("smo_ur", 1), ("smo_ur", 2), ("smo_ur", 6), ("smo_ur", 7)}


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("smo")
    for f in list(REF.glob("smo_*.mod")) + list(REF.glob("smo_*data.m")):
        shutil.copy(f, d)
    return d


@pytest.fixture(scope="module")
def get(workdir):
    @functools.lru_cache(maxsize=None)
    def model(name, case):
        return pydyn.Model(pydyn.run_preprocessor(workdir / f"{name}.mod", extra_args=(f"-DCASE={case}",)))

    @functools.lru_cache(maxsize=None)
    def run(name, case, compat):
        return smoother.smooth(model(name, case), replicate_dynare_bugs=compat)

    return model, run


def ref(name, case):
    return loadmat(REF / f"{name}_case{case}.mat", squeeze_me=True)


def close(a, b, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(a, np.reshape(b, np.shape(a)), rtol=rtol, atol=atol)


def table(d, names):
    return np.column_stack([d[v] for v in names])


def idx(r, key="idx"):
    return np.atleast_1d(r[key]).astype(int) - 1


def compare(res, r, m, skip=()):
    """Every field Dynare stored for this case."""
    T = res.nobs
    assert r["endo_names"].split(",") == m.endo_names and r["varobs"].split(",") == res.varobs
    close(table(res.SmoothedVariables, m.endo_names), r["smoothed"])
    close(table(res.UpdatedVariables, m.endo_names), r["updated"])
    close(table(res.SmoothedShocks, m.exo_names), r["shocks"])
    close(np.array([res.Constant[v][0] for v in m.endo_names]), r["constant"])
    close(np.array([res.Init_State[v] for v in m.endo_names]), r["init_state"])
    close(res.Init_State_Vector, r["init_state_vector"])
    assert bool(r["loglinear"]) == res.loglinear
    if "filtered" in r:
        close(table(res.FilteredVariables, m.endo_names), r["filtered"])
    if "meas_err" in r:
        close(table(res.SmoothedMeasurementErrors, res.varobs), r["meas_err"])     # NaN where data are missing
    else:
        assert res.SmoothedMeasurementErrors == {}
    if "kstep" in r:
        assert list(np.atleast_1d(r["filter_step_ahead"])) == res.filter_step_ahead
        close(res.FilteredVariablesKStepAhead, r["kstep"])
    if "kstep_var" in r:
        close(res.FilteredVariablesKStepAheadVariances[..., idx(r, "kidx")], r["kstep_var"])
    if "variance" in r:
        close(res.Variance[..., list(idx(r)) + [T]], r["variance"])
    if "state_uncertainty" in r and "V" not in skip:
        close(res.State_uncertainty[..., idx(r)], r["state_uncertainty"])
        close(res.Init_State_uncertainty, r["init_state_uncertainty"])


# ------------------------------------------------------------------ smoother against Dynare
@pytest.mark.parametrize("name,case", CASES)
def test_smoother_matches_dynare(get, name, case):
    """Smoothed, updated and filtered variables, shocks, measurement errors, k-step forecasts and
    their variances, filter covariance, state uncertainty and the period-0 state, with the
    options read from the calib_smoother statement of the .mod file."""
    model, run = get
    res = run(name, case, True)
    assert (res.kalman_algo, res.diffuse_periods) == EXPECTED_ALGO[name, case]
    compare(res, ref(name, case), model(name, case))


@pytest.mark.parametrize("name,case", [c for c in CASES if c not in DEFECT])
def test_default_is_dynare_when_no_defect(get, name, case):
    model, run = get
    compare(run(name, case, False), ref(name, case), model(name, case))


def test_steady_state_and_rules_match(get):
    """The references were produced at the same steady state and decision rules."""
    model, run = get
    for name, case in (("smo_fs", 2), ("smo_me", 3), ("smo_ur", 1)):
        r, m = ref(name, case), model(name, case)
        sol = pydyn.first_order(m)
        close(sol.ys, r["ys"])
        res = run(name, case, True)
        ov = r["order_var"].astype(int) - 1
        close(res.H[ov], r["ghu"], atol=1e-11)                  # log-linear rules for smo_fs
        close(res.G[np.ix_(ov, sol.state_var)], r["ghx"], atol=1e-11)


# ------------------------------------------------------------------ Dynare's diffuse-smoother defects
def test_multivariate_diffuse_corrected_equals_dynare_univariate(get):
    """Period 1 has no observation (smo_ur case 1) or F_inf = 0 (case 6): Dynare's kalman_algo=3
    output differs from its kalman_algo=4 output (cases 5 and 7). The default here follows the
    univariate (correct) numbers while still running the multivariate recursions."""
    model, run = get
    for mv, uv, skip in ((1, 5, ()), (6, 7, ("V",))):
        res, r_uv, r_mv = run("smo_ur", mv, False), ref("smo_ur", uv), ref("smo_ur", mv)
        assert res.kalman_algo == 3
        compare(res, r_uv, model("smo_ur", mv), skip=skip)
        # size of the defect in Dynare's own output: multivariate against univariate
        assert np.max(np.abs(r_mv["smoothed"] - r_uv["smoothed"])) > 0.1
        assert np.max(np.abs(r_mv["shocks"] - r_uv["shocks"])) > 0.05
    # case 6/7: Dynare's univariate state uncertainty is itself off in the first period (next test)
    res, r = run("smo_ur", 6, False), ref("smo_ur", 7)
    close(res.State_uncertainty[..., idx(r)[1:]], r["state_uncertainty"][..., 1:])
    assert np.max(np.abs(res.State_uncertainty[..., 0] - r["state_uncertainty"][..., 0])) > 0.1


def brute_force(res, model, kappa=1e6):
    """Smoothed mean and covariance from the joint normal distribution of the states and the
    data, the diffuse part of the initial variance being replaced by a large number."""
    G, H, Q, mf = res.G, res.H, model.Sigma_e, res.obs_index
    n, N = G.shape[0], res.nobs
    Pstar, Pinf = smoother.compute_Pinf_Pstar(mf, G, H, Q)
    cov = [Pstar + kappa * Pinf]
    for _ in range(N):
        cov.append(G @ cov[-1] @ G.T + H @ Q @ H.T)
    C = np.zeros(((N + 1) * n, (N + 1) * n))
    for i in range(N + 1):
        blk = cov[i]
        for j in range(i, N + 1):
            C[i * n:(i + 1) * n, j * n:(j + 1) * n] = blk
            C[j * n:(j + 1) * n, i * n:(i + 1) * n] = blk.T
            blk = blk @ G.T
    seen = [(t, i) for t in range(N) for i in range(len(mf)) if not np.isnan(res.data[t, i])]
    sel = [(t + 1) * n + mf[i] for t, i in seen]
    y = np.array([res.data[t, i] for t, i in seen])
    Cyy = C[np.ix_(sel, sel)] + np.diag([res.me_variance[i] for _, i in seen])
    gain = np.linalg.solve(Cyy, C[sel]).T
    V = C - gain @ C[sel]
    return (gain @ y).reshape(N + 1, n), np.stack([V[k * n:(k + 1) * n, k * n:(k + 1) * n] for k in range(N + 1)], -1)


@pytest.mark.parametrize("case,algo", [(1, 3), (2, 4), (6, 3), (7, 4), (4, 3), (3, 4)])
def test_diffuse_smoother_against_brute_force(get, case, algo):
    """Independent check of the diffuse smoother, including the state uncertainty where the
    default departs from Dynare (cases 1, 2, 6, 7)."""
    model, _ = get
    m = model("smo_ur", case)
    kw = dict(nobs=6, kalman_algo=algo, smoothed_state_uncertainty=True)
    res = smoother.smooth(m, **kw)
    assert res.kalman_algo == algo
    mean, V = brute_force(res, m)
    tol = dict(rtol=0, atol=2e-5)                         # error of the large-kappa approximation
    np.testing.assert_allclose(res.smoothed, mean[1:], **tol)
    np.testing.assert_allclose(res.init_state, mean[0], **tol)
    np.testing.assert_allclose(res.State_uncertainty, V[..., 1:], **tol)
    np.testing.assert_allclose(res.Init_State_uncertainty, V[..., 0], **tol)
    if ("smo_ur", case) in DEFECT:
        bug = smoother.smooth(m, replicate_dynare_bugs=True, **kw)
        err = max(np.max(np.abs(bug.State_uncertainty - V[..., 1:])), np.max(np.abs(bug.Init_State_uncertainty - V[..., 0])))
        assert err > 0.1


# ------------------------------------------------------------------ consistency checks
@pytest.mark.parametrize("name,case", CASES)
def test_check_identities(get, name, case):
    """Smoothed shocks through G, H from the smoothed period-0 state reproduce the smoothed
    variables; the decomposition sums to the smoothed deviation."""
    model, run = get
    res = run(name, case, False)
    dec = smoother.shock_decomposition(model(name, case), res, init_state=0)
    chk = smoother.check(res, dec)
    assert set(chk) == {"transition", "simulation", "measurement", "decomposition", "decomposition_initial"}
    assert max(chk.values()) < 1e-10, chk
    if not res.loglinear:                                 # sol.G, sol.H are the rules used
        chk = smoother.check(res, dec, sol=pydyn.first_order(model(name, case)))
        assert max(chk.values()) < 1e-10, chk


@pytest.mark.parametrize("case", [1, 6])
def test_check_detects_dynare_defect(get, case):
    """Dynare's kalman_algo=3 output for these samples violates the transition equation."""
    _, run = get
    chk = smoother.check(run("smo_ur", case, True))
    assert chk["transition"] > 0.1 and chk["measurement"] < 1e-10


# ------------------------------------------------------------------ decompositions
@pytest.mark.parametrize("name,case,init_state", [("smo_fs", 1, 0), ("smo_fs", 2, 1), ("smo_fs", 4, 0),
                                                  ("smo_me", 1, 0), ("smo_me", 3, 0),
                                                  ("smo_ur", 1, 0), ("smo_ur", 3, 0)])
def test_shock_decomposition_matches_dynare(get, name, case, init_state):
    model, run = get
    m, r = model(name, case), ref(name, case)
    res = run(name, case, True)
    dec = smoother.shock_decomposition(m, res)            # init_state from the .mod file
    assert dec.init_state == init_state
    assert dec.z.shape == (m.n_endo, m.n_exo + 2, res.nobs) == r["shock_decomposition"].shape
    assert dec.columns == m.exo_names + ["initial_conditions", "smoothed"]
    close(dec.z, r["shock_decomposition"])
    close(dec.z[:, :-1].sum(axis=1), dec.z[:, -1], atol=1e-13)
    v = m.endo_names[3]
    close(dec[v]["smoothed"], res.smoothed[:, 3])


def test_shock_decomposition_runs_smoother_itself(get):
    model, run = get
    m = model("smo_me", 1)
    dec = smoother.shock_decomposition(m)                 # data and options from the .mod file
    close(dec.z, ref("smo_me", 1)["shock_decomposition"])
    with pytest.raises(TypeError):
        smoother.shock_decomposition(m, run("smo_me", 1, False), nobs=10)


@pytest.mark.parametrize("name,case", [("smo_fs", 1), ("smo_me", 1), ("smo_ur", 1)])
def test_initial_condition_decomposition_matches_dynare(get, name, case):
    model, run = get
    m, r = model(name, case), ref(name, case)
    res = run(name, case, True)
    dec = smoother.initial_condition_decomposition(m, res)
    assert dec.z.shape == (m.n_endo, m.n_endo + 2, res.nobs)
    close(dec.z[..., idx(r, "icd_idx")], r["initval_decomposition"])


def test_group_shocks(get):
    model, run = get
    m = model("smo_me", 1)
    dec = smoother.shock_decomposition(m, run("smo_me", 1, False))
    g = smoother.group_shocks(dec, {"supply": ["e_a"], "demand": ["e_g", "e_v"]})
    assert g.columns == ["supply", "demand", "Others + Initial Values", "smoothed"]
    close(g.z[:, 0], dec.z[:, 0])
    close(g.z[:, 1], dec.z[:, 1] + dec.z[:, 2])
    close(g.z[:, 2], dec.z[:, 3])
    g = smoother.group_shocks(dec, {"policy": ["e_v"]})   # shocks in no group join the initial values
    close(g.z[:, 1], dec.z[:, 0] + dec.z[:, 2] + dec.z[:, 3])
    close(g.z[:, :-1].sum(axis=1), g.z[:, -1], atol=1e-13)
    with pytest.raises(smoother.SmootherError):
        smoother.group_shocks(dec, {"x": ["nope"]})


# ------------------------------------------------------------------ inputs from Python
def test_data_from_python_and_csv(get, workdir, tmp_path):
    model, run = get
    m, base = model("smo_me", 1), run("smo_me", 1, False)
    series = smoother._parse_m_file(workdir / "smo_me_data.m")
    arr = np.column_stack([series[v] for v in m.varobs])
    assert np.isnan(arr).sum() == 14 and np.isnan(arr[17]).all()
    csv = tmp_path / "d.csv"
    names = list(series)[::-1]                             # column order must not matter
    rows = [",".join(names)] + [",".join("" if np.isnan(series[v][t]) else repr(float(series[v][t])) for v in names)
                                for t in range(len(arr))]
    csv.write_text("\n".join(rows) + "\n")
    for data in (arr, series, csv, str(workdir / "smo_me_data")):
        res = smoother.smooth(m, data)
        np.testing.assert_array_equal(res.smoothed, base.smoothed)
        np.testing.assert_array_equal(res.shocks, base.shocks)
    with pytest.raises(smoother.SmootherError):
        smoother.smooth(m, arr[:, :2])
    with pytest.raises(smoother.SmootherError):
        smoother.smooth(m, {"yobs": arr[:, 0]})


def test_options_from_python(get):
    """Same numbers when the options come from Python instead of the .mod file, and when the
    parameters are set with set_params (Dynare: set_param_value, smo_fs case 2)."""
    model, run = get
    m = model("smo_fs", 4)                                 # its own statement: levels, nobs=100
    old = m.params.copy()
    try:
        m.set_params(rho=0.5, alp=0.36)
        res = smoother.smooth(m, datafile="smo_fs_data", loglinear=True, first_obs=11, nobs=80, kalman_algo=2,
                              filtered_vars=True, smoothed_state_uncertainty=True)
        compare(res, ref("smo_fs", 2), m)
        close(smoother.shock_decomposition(m, res, init_state=1).z, ref("smo_fs", 2)["shock_decomposition"])
    finally:
        m.params[:] = old
    # logdata: case 3 sets options_.logdata in the .mod file; here the keyword does it
    res = smoother.smooth(m, datafile="smo_fs_logdata", loglinear=True, logdata=True, nobs=60, presample=4,
                          filtered_vars=False)
    r = ref("smo_fs", 3)
    close(res.smoothed + np.log(res.SteadyState), r["smoothed"])
    close(res.shocks, r["shocks"])
    assert res.FilteredVariablesKStepAhead is None and res.Variance is None and res.State_uncertainty is None


def test_dynare_conventions(get):
    model, run = get
    # loglinear: variables are returned in logs; Init_State_Vector adds the level steady state
    res, m = run("smo_fs", 1, False), model("smo_fs", 1)
    ys = res.SteadyState
    close(res.SmoothedVariables["c"], res.smoothed[:, m.endo_names.index("c")] + np.log(ys[m.endo_names.index("c")]))
    close(res.Init_State_Vector, res.init_state + ys)
    # FilteredVariables[t] is the forecast for t+1 made at t
    close(table(res.FilteredVariables, m.endo_names)[:-1] - np.log(ys), (res.updated @ res.G.T)[:-1])
    assert res.FilteredVariablesKStepAhead.shape == (3, m.n_endo, res.nobs + 4)
    assert res.filter_step_ahead == [1, 2, 4]
    # prefilter: observed variables are returned around the sample mean of the data
    res, m = run("smo_me", 3, False), model("smo_me", 3)
    raw = smoother.load_data("smo_me_data", m.varobs, base_dir=m.json_dir.parents[2])[:90]
    close(res.obs_constant, np.nanmean(raw, axis=0))
    i = m.endo_names.index("iobs")
    seen = ~np.isnan(raw[:, 2])
    close(res.SmoothedVariables["iobs"][seen], raw[seen, 2], atol=1e-10)      # no measurement error on iobs
    assert np.all(res.Constant["iobs"] == res.obs_constant[2]) and res.SteadyState[i] == 0
    # missing observations: the measurement error is NaN, the smoothed variable is not
    assert np.isnan(res.SmoothedMeasurementErrors["yobs"][2]) and np.isfinite(res.SmoothedVariables["yobs"][2])
    assert set(res.SmoothedMeasurementErrors) == set(m.varobs)                # all observables, as in Dynare


def test_errors(get):
    model, _ = get
    m = model("smo_ur", 1)
    with pytest.raises(smoother.SmootherError, match="unit root"):
        smoother.smooth(m, diffuse_filter=False)
    with pytest.raises(TypeError):
        smoother.smooth(m, no_such_option=1)
    with pytest.raises(NotImplementedError):
        smoother.smooth(m, lik_init=2, diffuse_filter=False)
    with pytest.raises(smoother.SmootherError):
        smoother.smooth(m, first_obs=80, nobs=20)
    with pytest.raises(smoother.SmootherError, match="prefilter"):
        smoother.smooth(model("smo_fs", 4), prefilter=1)
