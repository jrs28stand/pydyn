"""OccBin: the loss-function regime guess (occbin.findmin), smoother_first_period_occbin_update > 1, OccBin
forecasts (occbin.forecast, qmc_sequence) against Dynare 7.2 under Octave 8.4.

References (reference/r_occ_*.m): occ_findmin_*.mat (NKM likelihood, Dynare's own findmin.m: Octave cannot run it
unchanged because the nested function ``cost_function`` shadows ``occbin.cost_function``, so the reference copy renames
the nested one), occ_smoother_fpou_*.mat, occ_forecast_nkm.mat (qmc_sequence draws too), occ_forecast_fc1.mat/.mod
(pydyn.dynare() against Dynare's replic = 0 and Gaussian-draw forecasts, shared random stream).
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
from pydyn.occbin_forecast import occbin_forecast
from pydyn.octave_rng import OctaveRandom
from pydyn.sobol import qmc_sequence

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))


def ref(name):
    return loadmat(REF / f"{name}.mat", squeeze_me=True)


def regimes(cell):
    a = np.asarray(cell, dtype=object)
    k = a.shape[0] // 2
    as_t = lambda x: tuple(int(v) for v in np.atleast_1d(x).ravel())  # noqa: E731
    return [tuple((as_t(a[2 * j, t]), as_t(a[2 * j + 1, t])) for j in range(k)) for t in range(a.shape[1])]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("occgap")
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("occest_nkm", "occest_rbc"):
            shutil.copy(REF / f"{name}.mod", d)
            out[name] = pydyn.Model(pydyn.run_preprocessor(d / f"{name}.mod"))
    return out


def estimation(model, r, occbin=None):
    data = np.atleast_2d(r["data"].T).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OccbinEstimation(model, data={v: data[:, j] for j, v in enumerate(model.varobs)},
                                nobs=data.shape[0], first_obs=1, occbin=occbin)


# ------------------------------------------------------------------ findmin
def test_fminsearch_octave_rosenbrock():
    x = of.fminsearch_octave(lambda z: (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2, [-1.2, 1.0])
    assert np.allclose(x, [1, 1], atol=5e-3)


@pytest.mark.parametrize("variant", ["early", "fail"])
def test_loss_function_regime_guess(models, variant):
    """findmin used (in the first 20 periods / when the regime iteration fails): likelihood, filtered states and call
    counts of Dynare."""
    fm = ref(f"occ_findmin_{variant}")
    ov = {"likelihood.loss_function_regime_guess": True, "likelihood.brute_force_regime_guess": False,
          "likelihood.brute_force_extra_regime_guess": False}
    ov.update({"early": {"likelihood.number_of_initial_periods_with_extra_regime_guess": 20},
               "fail": {"likelihood.max_number_of_iterations": 1}}[variant])
    est = estimation(models["occest_nkm"], fm, ov)
    calls = []
    orig = of._findmin
    of._findmin = lambda ctx, step: (calls.append(1), orig(ctx, step))[1]
    try:
        X = np.atleast_2d(fm["X"])
        X = X if X.shape[0] == est.n else X.T
        lp = est.log_posterior(X[:, 0])
    finally:
        of._findmin = orig
    assert lp == pytest.approx(-fm["fval"][0], rel=1e-12, abs=1e-8)
    assert len(calls) == fm["nfind"][0]
    res = est.last_result
    np.testing.assert_allclose(2 * res.lik, np.atleast_1d(fm["lik_1"]), atol=1e-8)
    np.testing.assert_allclose(res.a0, np.atleast_2d(fm["a0_1"]), atol=1e-10)


def test_regime_iteration_failure_without_guess(models):
    """Same limit (one regime iteration), no loss function guess: Dynare's likelihood is Inf (info 320), as here."""
    fm = ref("occ_findmin_failbase")
    ov = {"likelihood.max_number_of_iterations": 1, "likelihood.brute_force_regime_guess": False,
          "likelihood.brute_force_extra_regime_guess": False}
    est = estimation(models["occest_nkm"], fm, ov)
    X = np.atleast_2d(fm["X"])
    X = X if X.shape[0] == est.n else X.T
    assert est.log_posterior(X[:, 0]) == -np.inf and fm["info"][0] == 320


# ------------------------------------------------------------------ smoother_first_period_occbin_update
@pytest.mark.parametrize("name,case,fpous", [("occ_smoother_fpou_nkm", "nkm", (3, 40)),
                                             ("occ_smoother_fpou_nkm2", "nkm", (55, 100)),
                                             ("occ_smoother_fpou_rbc", "rbc", (60,)),
                                             ("occ_smoother_fpou_rbc2", "rbc", (10, 120))])
def test_smoother_first_period_occbin_update(models, name, case, fpous):
    s = ref(name)
    r = ref(f"occest_{case}_lik")
    for f in fpous:
        est = estimation(models[f"occest_{case}"], r, {"smoother.first_period_occbin_update": f})
        sm = est.occbin_smoother(s["x0"])
        pre = f"f{f}"
        assert sm.error_flag == s[pre + "_error_flag"] == 0
        ys = sm.ys
        tol = 1e-11
        np.testing.assert_allclose(sm.smoothed + ys, np.atleast_2d(s[pre + "_SmoothedVariables"].T).T, atol=tol)
        np.testing.assert_allclose(sm.shocks, np.atleast_2d(s[pre + "_SmoothedShocks"].T).T, atol=tol)
        np.testing.assert_allclose(sm.updated + ys, np.atleast_2d(s[pre + "_UpdatedVariables"].T).T, atol=tol)
        np.testing.assert_allclose(sm.filtered + ys, np.atleast_2d(s[pre + "_FilteredVariables"].T).T, atol=tol)
        assert sm.regime_history == regimes(s[pre + "_regimes"])
        assert sm.realtime_regime_history == regimes(s[pre + "_rt_regimes"])


def test_smoother_first_period_occbin_update_two_constraints_tie(models):
    """rbc with first_period_occbin_update = 3: the update at period 52 finds two self-consistent regimes whose
    likelihoods differ by 2e-11 (the second constraint expected to bind until period 109 or 113); Dynare and pydyn
    report different ones (rounding decides the strict ``<``), everything else agrees to 3e-12."""
    s, r = ref("occ_smoother_fpou_rbc"), ref("occest_rbc_lik")
    est = estimation(models["occest_rbc"], r, {"smoother.first_period_occbin_update": 3})
    sm = est.occbin_smoother(s["x0"])
    assert sm.error_flag == 0
    ys = sm.ys
    np.testing.assert_allclose(sm.smoothed + ys, np.atleast_2d(s["f3_SmoothedVariables"].T).T, atol=3e-12)
    np.testing.assert_allclose(sm.shocks, np.atleast_2d(s["f3_SmoothedShocks"].T).T, atol=1e-12)
    dyn = regimes(s["f3_regimes"])
    diff = [t for t in range(len(dyn)) if sm.regime_history[t] != dyn[t]]
    assert diff == [51]
    assert sm.regime_history[51][0] == dyn[51][0] and sm.regime_history[51][1][0] == dyn[51][1][0]


# ------------------------------------------------------------------ qmc_sequence
def test_qmc_sequence_bit_exact():
    f = ref("occ_forecast_nkm")
    for nm, args in (("q1", (24, 1, 1, 127)), ("q2", (7, 1, 0, 20)), ("q3", (1111, 1, 0, 3)), ("q4", (300, 5, 1, 6))):
        a, _ = qmc_sequence(*args)
        assert np.array_equal(a, f[nm]), nm
    assert qmc_sequence(300, 5, 1, 6)[1] == int(f["q4_seed"])
    with np.errstate(all="ignore"):
        a, _ = qmc_sequence(5, 1, 2, 4, 2.5)
    np.testing.assert_allclose(a, f["q5"], atol=2e-16)          # (nan in the first point: norm 0, as Dynare)
    np.testing.assert_allclose(qmc_sequence(3, 1, 1, 5, f["q6_L"])[0], f["q6"], atol=1e-15)


# ------------------------------------------------------------------ forecasts
STATS = ("Mean", "Median", "Var", "HPDinf", "HPDsup", "Deciles", "Min", "Max")


def _compare(f, pre, fc, names, stats, atol, skip=()):
    for fld in ("linear", "piecewise"):
        for st in stats:
            r = np.asarray(f[f"{pre}_{fld}_{st}"])
            r = r.reshape(r.shape[0], -1, len(names))
            for i, nm in enumerate(names):
                if (st, nm) in skip:
                    continue
                np.testing.assert_allclose(np.asarray(getattr(fc, fld)[st][nm]).reshape(r.shape[0], -1), r[:, :, i],
                                           atol=atol, err_msg=f"{pre} {fld} {st} {nm}")


@pytest.fixture(scope="module")
def nkm_forecast(models):
    f = ref("occ_forecast_nkm")
    r = ref("occest_nkm_lik")
    est = estimation(models["occest_nkm"], r)
    ob = est.occbin_at(f["x0"])
    Q = est.covariances(f["x0"])[0]
    np.testing.assert_allclose(Q, f["Sigma_e"], atol=0)
    return f, ob, Q, list(models["occest_nkm"].endo_names)


def test_forecast_zero_shocks(nkm_forecast):
    f, ob, Q, names = nkm_forecast
    fc = occbin_forecast(ob, 8, f["histval"], sigma_e=Q)
    _compare(f, "A", fc, names, ["Mean"], 1e-12)
    np.testing.assert_allclose(fc.yf, f["A_yf"], atol=1e-12)
    assert fc.error_flag == 0


def test_forecast_shocks0(nkm_forecast):
    f, ob, Q, names = nkm_forecast
    fc = occbin_forecast(ob, 8, f["histval"], shocks=f["D_shocks"], sigma_e=Q)
    _compare(f, "D", fc, names, ["Mean"], 1e-12)
    fc = occbin_forecast(ob, 8, f["histval"], shocks=[("epsi", [-1.5, -1, -0.5, 0, 0, 0, 0, 0])], sigma_e=Q)
    _compare(f, "E", fc, names, ["Mean"], 1e-12)


def test_forecast_gaussian_bands(nkm_forecast):
    f, ob, Q, names = nkm_forecast
    rng = OctaveRandom(7)
    fc = occbin_forecast(ob, 8, f["histval"], replic=60, rng=rng, sigma_e=Q)
    _compare(f, "C", fc, names, STATS, 1e-11)
    np.testing.assert_allclose(fc.yf, f["C_yf"], atol=1e-11)
    np.testing.assert_allclose(rng.randn(3), f["C_state_after"], atol=1e-14)      # same number of draws consumed
    rng = OctaveRandom(3)
    fc = occbin_forecast(ob, 8, f["histval"], shocks=f["D_shocks"], replic=40, rng=rng, sigma_e=Q)
    _compare(f, "D2", fc, names, STATS, 1e-11)


def test_forecast_qmc_bands(nkm_forecast):
    """tests/occbin/filter/NKM.mod: qmc, 127 replications. Variable mp (= sigi * epsi) has exact ties between the
    candidate HPD intervals (symmetric Sobol-Gaussian draws, relative gap 5e-16): the interval reported there is
    decided by rounding and is not compared."""
    f, ob, Q, names = nkm_forecast
    fc = occbin_forecast(ob, 8, f["histval"], replic=127, qmc=True, sigma_e=Q)
    assert fc.replic == 127
    _compare(f, "B", fc, names, STATS, 1e-11, skip={("HPDinf", "mp"), ("HPDsup", "mp")})
    np.testing.assert_allclose(fc.yf, f["B_yf"], atol=1e-11)


def test_dynare_forecast_with_shared_stream():
    d = ref("occ_forecast_fc1")
    r = pydyn.dynare(REF / "occ_forecast_fc1.mod")
    v = r.vars
    names = r.model.endo_names
    o = loadmat(REF / "occ_forecast_fc1.mat", squeeze_me=True, struct_as_record=False)
    for key in ("forecast", "forecast1"):
        for fld in ("linear", "piecewise"):
            blk = getattr(o[key], fld)
            for st in blk._fieldnames:
                for nm in names:
                    a, b = np.asarray(getattr(getattr(blk, st), nm)), np.asarray(v[key][fld][st][nm])
                    np.testing.assert_allclose(b, a.reshape(b.shape), atol=1e-13)
    for k in ("yf", "yf1"):
        np.testing.assert_allclose(v[k], d[k] if k in d else getattr(o[k], "__array__", o[k]), atol=1e-13)


def test_forecast_refuses_stale_exo_pos(tmp_path):
    """Dynare's forecast.m (replic = 0) keeps options_.occbin.simul.exo_pos from occbin_solver and fails on the size
    mismatch when the surprise shocks cover fewer shocks than the model has: refused here."""
    src = (REF / "occ_forecast_fc1.mod").read_text().replace("values 0.01;", "values 0;")
    (tmp_path / "e.mod").write_text(src)
    with pytest.raises(Exception, match="exo_pos"):
        pydyn.dynare(tmp_path / "e.mod")


def test_loglinear_refused_with_evidence(models):
    r = ref("occest_nkm_lik")
    with pytest.raises(NotImplementedError, match="LOGGED steady state"):
        estimation(models["occest_nkm"], r) if False else OccbinEstimation(
            models["occest_nkm"], data={v: np.atleast_2d(r["data"].T).T[:, j] for j, v in enumerate(models["occest_nkm"].varobs)},
            nobs=np.atleast_2d(r["data"].T).T.shape[0], first_obs=1, loglinear=True)
