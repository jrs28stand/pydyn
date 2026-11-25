"""OccBin likelihoods, smoother, shock decomposition and estimation against Dynare 7.2.

References: reference/occest_*.mat, written by reference/r_occbin_estimation.m (Dynare 7.2 under
Octave 8.4) for

* ``occest_nkm``: Dynare's tests/occbin/filter NKM model (ZLB, 24 variables, 3 shocks, 3 observables,
  9 estimated parameters) with its data (``dataobsfile.mat``); likelihood of the piecewise Kalman
  filter (``dsge_likelihood``) and of the inversion filter (``occbin.IVF_posterior``) at four
  parameter vectors around the mode stored with the test (``NKM_mh_mode_saved.mat``), the PKF's
  per-period likelihood, filtered states and regimes (instrumented copy of
  ``missing_observations_kalman_filter.m``), and ``evaluate_smoother`` (calib_smoother) at that mode;
* ``occest_rbc``: tests/occbin/model_irrcap_twoconstraints (two constraints, one shock observed
  through consumption), data simulated with OccBin; the smoother needs two iterations and the
  conditional-smoother attempt (Dynare: "converged to a different regime wrt realtime regime");
* ``occest_nkms``: the NKM model with ``small_model = 1`` and three estimated parameters; posterior
  mode by Dynare's ``mode_compute = 4`` from the prior mean.

Set ``PYDYN_SLOW_TESTS=1`` for the posterior-mode search from the prior mean (about 80 s) and a
longer Metropolis-Hastings run.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import occbin_filter as of
from pydyn.occbin_estimation import OccbinEstimation

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
SLOW = os.environ.get("PYDYN_SLOW_TESTS", "0") == "1"


def ref(name):
    return loadmat(REF / f"{name}.mat", squeeze_me=True)


def regimes(cell):
    """Dynare's regime history (cell: fields x periods) as pydyn's hashable regimes."""
    a = np.asarray(cell, dtype=object)
    k = a.shape[0] // 2
    as_t = lambda x: tuple(int(v) for v in np.atleast_1d(x).ravel())  # noqa: E731
    return [tuple((as_t(a[2 * j, t]), as_t(a[2 * j + 1, t])) for j in range(k)) for t in range(a.shape[1])]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("occest")
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("occest_nkm", "occest_rbc", "occest_nkms"):
            shutil.copy(REF / f"{name}.mod", d)
            out[name] = pydyn.Model(pydyn.run_preprocessor(d / f"{name}.mod"))
    return out


def estimation(model, r, **kw):
    data = np.atleast_2d(r["data"].T).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OccbinEstimation(model, data={v: data[:, j] for j, v in enumerate(model.varobs)},
                                nobs=data.shape[0], first_obs=1, **kw)


def points(r, n):
    X = np.atleast_2d(r["X"])
    return X if X.shape[0] == n else X.T


# ------------------------------------------------------------------ inversion filter
@pytest.mark.parametrize("case,ks", [("nkm", [0, 1, 2, 3]), ("rbc", [0]), ("nkms", [0, 1, 2])])
def test_inversion_filter_likelihood(models, case, ks):
    r = ref(f"occest_{case}_lik")
    est = estimation(models[f"occest_{case}"], r, occbin={"likelihood_inversion_filter": True})
    assert est.last_filter == "inversion filter"
    X = points(r, est.n)
    for k in ks:
        lp = est.log_posterior(X[:, k])
        assert lp == pytest.approx(-r["ivf_fval"][k], rel=1e-12, abs=1e-8), (k, est.last_info)
        res = est.last_result
        if f"ivf_atT_{k + 1}" in r:                # states (decision-rule order) and shocks
            order = np.atleast_1d(r["order_var"]).astype(int) - 1
            np.testing.assert_allclose(res.stateval[:, order].T, r[f"ivf_atT_{k + 1}"], atol=1e-10)
            innov = np.atleast_2d(r[f"ivf_innov_{k + 1}"])
            np.testing.assert_allclose(res.filtered_errs, innov.T.reshape(res.filtered_errs.shape), atol=1e-9)
        assert res.regime_history == regimes(r[f"ivf_regimes_{k + 1}"])[:len(res.regime_history)]


def test_inversion_filter_binds(models):
    """The NKM data reach the zero lower bound: the inversion filter goes through binding periods."""
    r = ref("occest_nkm_lik")
    rh = regimes(r["ivf_regimes_1"])
    assert sum(reg[0][0][0] for reg in rh) > 20


# ------------------------------------------------------------------ piecewise Kalman filter
@pytest.mark.parametrize("case,ks", [("nkm", [0, 1, 2, 3]), ("rbc", [0]), ("nkms", [0, 1, 2])])
def test_piecewise_kalman_filter_likelihood(models, case, ks):
    r = ref(f"occest_{case}_lik")
    est = estimation(models[f"occest_{case}"], r)
    assert est.last_filter == "piecewise Kalman filter"
    X = points(r, est.n)
    for k in ks:
        lp = est.log_posterior(X[:, k])
        assert lp == pytest.approx(-r["pkf_fval"][k], rel=1e-12, abs=1e-8), (k, est.last_info)
        res = est.last_result
        # Dynare's lik before the final halving (instrumented missing_observations_kalman_filter.m)
        np.testing.assert_allclose(2 * res.lik, np.atleast_1d(r[f"pkf_lik_{k + 1}"]), atol=1e-8)
        np.testing.assert_allclose(res.a0, np.atleast_2d(r[f"pkf_a0_{k + 1}"]), atol=1e-10)
        dyn = regimes(r[f"pkf_regimes_{k + 1}"])
        T = len(res.regimes)
        if case == "rbc" and k == 1:
            continue
        assert res.regimes == dyn[:T]


def test_pkf_regime_difference_rbc_point2(models):
    """At the second rbc vector one regime-history entry differs from Dynare's (period 53): the update
    finds two self-consistent regimes (second constraint expected to bind until period 91 or 95,
    paths within 4e-7 of the constraint's -1e-6 threshold) whose likelihoods differ by 1e-15. Both
    programs use the sum of the two densities (kalman_update_engine.m), hence the same likelihood;
    which regime is reported is decided by the strict ``<`` between the two values, i.e. by rounding."""
    r = ref("occest_rbc_lik")
    est = estimation(models["occest_rbc"], r)
    X = points(r, est.n)
    lp = est.log_posterior(X[:, 1])
    assert lp == pytest.approx(-r["pkf_fval"][1], rel=1e-12)
    res = est.last_result
    np.testing.assert_allclose(2 * res.lik, r["pkf_lik_2"], atol=1e-8)
    dyn = regimes(r["pkf_regimes_2"])
    diff = [t for t in range(len(res.regimes)) if res.regimes[t] != dyn[t]]
    assert diff == [52]
    assert res.regimes[52][0] == dyn[52][0] and res.regimes[52][1][0] == dyn[52][1][0]


# ------------------------------------------------------------------ smoother
def _check_smoother(sm, s, prefix, atol):
    ys = sm.ys
    np.testing.assert_allclose(sm.smoothed + ys, np.atleast_2d(s[f"{prefix}_SmoothedVariables"].T).T, atol=atol)
    np.testing.assert_allclose(sm.shocks, np.atleast_2d(s[f"{prefix}_SmoothedShocks"].T).T, atol=atol)


def test_occbin_smoother_nkm(models):
    r, s = ref("occest_nkm_lik"), ref("occest_nkm_smoother")
    est = estimation(models["occest_nkm"], r)
    sm = est.occbin_smoother(s["x0"])
    assert sm.error_flag == s["pkf_error_flag"] == 0 and sm.converged
    _check_smoother(sm, s, "pkf", 1e-10)
    np.testing.assert_allclose(sm.updated + sm.ys, s["pkf_UpdatedVariables"], atol=1e-10)
    np.testing.assert_allclose(sm.filtered + sm.ys, s["pkf_FilteredVariables"], atol=1e-10)
    assert sm.regime_history == regimes(s["pkf_regimes"])
    assert sm.realtime_regime_history == regimes(s["pkf_rt_regimes"])
    from pydyn.perturbation import dr_ordering
    ov = dr_ordering(models["occest_nkm"].lead_lag_incidence)[0]
    np.testing.assert_allclose(sm.linear_smoothed[:, ov].T, s["pkf_linear_alphahat"], atol=1e-10)
    np.testing.assert_allclose(sm.linear_shocks.T, s["pkf_linear_etahat"], atol=1e-10)
    # inversion filter as smoother (smoother_inversion_filter)
    iv = est.occbin_smoother(s["x0"], inversion_filter=True)
    _check_smoother(iv, s, "ivf", 1e-10)
    # shock decomposition (occbin.shock_decomposition, first period as TINIT)
    d = est.occbin_shock_decomposition(s["x0"], sm)
    rows = np.atleast_1d(s["sd_rows"]).astype(int) - 1
    n = len(models["occest_nkm"].endo_names)
    for key, dk in [("sd_decomp", "decomp"), ("sd_wdecomp", "wdecomp"), ("sd_linear", "linear"),
                    ("sd_init", "init_decomp"), ("sd_linear_init", "linear_init")]:
        z = d[dk].z[rows]
        if "init" in key:                     # Dynare keeps only the n variable columns
            z = z[:, :n]
        np.testing.assert_allclose(z, np.asarray(s[key]).reshape(z.shape), atol=1e-12, err_msg=key)
    # the decomposition adds up to the smoothed variables
    z = d["decomp"].z
    np.testing.assert_allclose(z[:, :-1].sum(axis=1), z[:, -1], atol=1e-12)
    np.testing.assert_allclose(z[:, -1].T, sm.smoothed, atol=1e-10)
    # squeeze
    names = [models["occest_nkm"].endo_names[i] for i in rows]
    sq = of.squeeze_occbin_shock_decomposition(d, names)
    np.testing.assert_allclose(sq["decomp"].z, np.asarray(s["sd_decomp"]).reshape(sq["decomp"].z.shape), atol=1e-12)
    assert sq["wdecomp"].endo_names == names


def test_occbin_smoother_two_constraints(models):
    r, s = ref("occest_rbc_lik"), ref("occest_rbc_smoother")
    est = estimation(models["occest_rbc"], r)
    sm = est.occbin_smoother(s["x0"])
    # Dynare: converged after iterating, to a regime different from the real-time one (warning 320)
    assert sm.error_flag == 0 and sm.warning_flag == 320 and sm.iterations == 2
    _check_smoother(sm, s, "pkf", 1e-12)
    np.testing.assert_allclose(sm.updated + sm.ys, np.atleast_2d(s["pkf_UpdatedVariables"].T).T, atol=1e-12)
    np.testing.assert_allclose(sm.filtered + sm.ys, np.atleast_2d(s["pkf_FilteredVariables"].T).T, atol=1e-12)
    assert sm.regime_history == regimes(s["pkf_regimes"])
    assert sm.realtime_regime_history == regimes(s["pkf_rt_regimes"])
    assert sm.regime_history != sm.realtime_regime_history
    iv = est.occbin_smoother(s["x0"], inversion_filter=True)
    _check_smoother(iv, s, "ivf", 1e-12)
    if "sd_decomp" in s:
        d = est.occbin_shock_decomposition(s["x0"], sm)
        np.testing.assert_allclose(d["decomp"].z, np.asarray(s["sd_decomp"]).reshape(d["decomp"].z.shape), atol=1e-12)
        np.testing.assert_allclose(d["wdecomp"].z, np.asarray(s["sd_wdecomp"]).reshape(d["wdecomp"].z.shape), atol=1e-12)
    # regime table (write_regimes_to_xls) and plotting data
    tab = of.regime_table(sm.regime_history, periods=[1, 2, 3])
    assert set(tab[0]) == {"time", "regime_sequence_1", "starting_period_of_regime_1", "regime_sequence_2",
                           "starting_period_of_regime_2"}
    reg = sm.regime_history[0]
    assert tab[0]["regime_sequence_1"] == "  ".join(str(x) for x in reg[0][0])
    pdat = of.plot_data(sm, models["occest_rbc"].endo_names, ["c"])
    np.testing.assert_allclose(pdat["c"]["smoothed"], np.atleast_2d(s["pkf_SmoothedVariables"].T).T[:, 1], atol=1e-12)
    assert pdat["c"]["binding"].shape == (len(sm.smoothed), 2)


# ------------------------------------------------------------------ estimation
def test_hessian_and_laplace_at_dynare_mode(models):
    """At Dynare's posterior mode (mode_compute = 4 from the prior mean, small NKM model, PKF): the
    objective, Dynare's Hessian (rtol 1e-3: finite differences on a piecewise-linear filter) and the
    Laplace approximation (Dynare printed 1515.166990)."""
    r, md = ref("occest_nkms_lik"), ref("occest_nkms_mode")
    est = estimation(models["occest_nkms"], r)
    x = np.asarray(md["xparam1"], float)
    assert est.log_posterior(x) == pytest.approx(-float(md["fval"]), rel=1e-12)
    H = est.hessian(x)
    np.testing.assert_allclose(H, md["hh"], rtol=2e-3)
    lap = est.laplace(x, est.inverse_hessian(H))
    assert lap == pytest.approx(1515.166990, abs=2e-3)


def test_short_metropolis_hastings(models):
    """A short random-walk Metropolis-Hastings chain on the OccBin posterior runs end to end."""
    r, md = ref("occest_nkms_lik"), ref("occest_nkms_mode")
    est = estimation(models["occest_nkms"], r)
    x = np.asarray(md["xparam1"], float)
    replic = 40 if SLOW else 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = est.sample(replic=replic, nblocks=1, cov=np.linalg.inv(md["hh"]), jscale=0.5, x0=x[None, :], seed=1)
    draws = np.asarray(mc.draws)
    assert draws.shape[-1] == 3 and np.all(np.isfinite(draws))
    assert np.all(np.isfinite(np.asarray(mc.log_posterior)))


@pytest.mark.skipif(not SLOW, reason="posterior mode from the prior mean takes about 80 s (PYDYN_SLOW_TESTS=1)")
def test_posterior_mode_matches_dynare(models):
    """csminwel (mode_compute = 4) from the prior mean, as Dynare's run. Both stop after 17
    iterations; the modes agree within 1e-3 posterior standard deviations (measured: 3e-4) and the
    log posterior within 1e-6 (measured 4.5e-8, below csminwel's stopping criterion 1e-7)."""
    r, md = ref("occest_nkms_lik"), ref("occest_nkms_mode")
    est = estimation(models["occest_nkms"], r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = est.mode(mode_compute=4)
    sd = np.sqrt(np.diag(np.linalg.inv(md["hh"])))
    assert np.all(np.abs(res.x - md["xparam1"]) < 1e-3 * sd)
    assert res.log_posterior == pytest.approx(-float(md["fval"]), abs=1e-6)


# ------------------------------------------------------------------ options
def test_options_from_occbin_setup(models, tmp_path):
    m = models["occest_rbc"]
    o = of.occbin_options(m)
    # occest_rbc.mod: occbin_setup(smoother_periods=200, smoother_maxit=200, smoother_curb_retrench, ...)
    assert o["smoother.periods"] == 200 and o["smoother.maxit"] == 200 and o["smoother.curb_retrench"]
    assert o["likelihood.periods"] == 3 and o["likelihood.maxit"] == 30          # Dynare's defaults
    assert not o["likelihood.inversion_filter"]
    o2 = of.occbin_options(m, {"likelihood_maxit": 7, "likelihood_inversion_filter": True,
                               "likelihood_max_kalman_iterations": 4})
    assert o2["likelihood.maxit"] == 7 and o2["likelihood.inversion_filter"]
    assert o2["likelihood.max_number_of_iterations"] == 4
    # options that Dynare's occbin.setup drops are reported once
    src = (REF / "occest_rbc.mod").read_text()
    src += "\noccbin_setup(smoother_max_number_of_iterations=3, filter_particle_number_of_particles=10, " \
           "likelihood_inversion_filter);\n"
    (tmp_path / "occest_rbc2.mod").write_text(src)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2 = pydyn.Model(pydyn.run_preprocessor(tmp_path / "occest_rbc2.mod"))
    with pytest.warns(pydyn.options.UnsupportedOptionWarning, match="smoother.max_number_of_iterations"):
        o3 = of.occbin_options(m2)
    assert o3["likelihood.inversion_filter"] and o3["smoother.max_number_of_iterations"] == 10
    assert o3["smoother.periods"] == 200                       # earlier occbin_setup kept (cumulative)


def test_linear_classes_still_refuse(models):
    with pytest.raises(NotImplementedError):
        pydyn.estimation.Estimation(models["occest_nkm"])


def test_ivf_requires_as_many_shocks_as_observables(models):
    r = ref("occest_nkm_lik")
    m = models["occest_nkm"]
    data = np.atleast_2d(r["data"].T).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(pydyn.estimation.EstimationError):
            OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)},
                             occbin={"likelihood_inversion_filter": True,
                                     "likelihood.IVF_shock_observable_mapping": ["epsg", "epsi"]})


def test_filter_options_against_dynare(models):
    """filter_use_relaxation, likelihood_max_kalman_iterations=3, likelihood_periodic_solution and
    likelihood_check_ahead_periods=100 (set in Dynare through options_.occbin): both filters agree
    with Dynare; the relaxation step runs (regime guesses) at these vectors."""
    r = ref("occest_nkmopt_lik")
    opts = {"filter_use_relaxation": True, "likelihood_max_kalman_iterations": 3,
            "likelihood_periodic_solution": True, "likelihood_check_ahead_periods": 100}
    for ivf in (False, True):
        est = estimation(models["occest_nkm"], r, occbin=dict(opts, likelihood_inversion_filter=ivf))
        assert est.occbin_options["filter.use_relaxation"] and est.occbin_options["likelihood.check_ahead_periods"] == 100
        X = points(r, est.n)
        for k in range(X.shape[1]):
            lp = est.log_posterior(X[:, k])
            assert lp == pytest.approx(-r["ivf_fval" if ivf else "pkf_fval"][k], rel=1e-12, abs=1e-8)


def test_write_regime_table(tmp_path):
    rh = [(((0,), (1,)),), (((1, 0), (1, 3)),), (((0, 1, 0), (1, 2, 4)),)]
    rows = of.write_regime_table(tmp_path / "regimes.csv", rh)
    assert rows[1] == {"time": 2, "regime_sequence": "1  0", "starting_period_of_regime": "1  3"}
    text = (tmp_path / "regimes.csv").read_text().splitlines()
    assert text[0] == "time,regime_sequence,starting_period_of_regime" and len(text) == 4
    # Dynare labels rows with ``periods`` but fills them from the first regimes
    assert of.regime_table(rh, periods=[3])[0]["regime_sequence"] == "0  1  0"
    assert of.regime_table(rh, periods=[3], dynare_indexing=True)[0]["regime_sequence"] == "0"


def test_occbin_estimation_uses_estimation_qz_criterium(models, monkeypatch):
    """occbin.solver keeps options_.qz_criterium when estimation has set it (1 - 1e-6 for a stationary
    initialisation, select_qz_criterium_value.m); it falls back to 1 + 1e-6 only when it is empty."""
    import pydyn.occbin as occ
    r = ref("occest_nkm_lik")
    est = estimation(models["occest_nkm"], r, occbin={"likelihood_inversion_filter": True})
    assert est.options["qz_criterium"] == 1 - 1e-6
    seen = []
    orig = occ.first_order
    monkeypatch.setattr(occ, "first_order", lambda *a, **k: (seen.append(k.get("qz_criterium")), orig(*a, **k))[1])
    est.occbin_at(points(r, est.n)[:, 0])
    assert seen == [1 - 1e-6]
