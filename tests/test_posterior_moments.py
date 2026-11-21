"""Posterior distributions of theoretical moments and of smoothed / filtered variables (pydyn.posterior_moments)
against Dynare 7.2 under Octave 8.4.

References: reference/r_postmom.m runs Dynare on reference/postmom_nk.mod (3 chains x 2400 draws) and
reference/postmom_schorfheide.mod (2 x 3000 draws) with ``moments_varendo`` or ``smoother`` / ``filtered_vars`` /
``filter_step_ahead`` and stores the parameter sub-draws Dynare itself used, the statistics it computed
(oo_.PosteriorTheoreticalMoments.dsge, oo_.SmoothedVariables, ...; densities every 8th or 16th grid point) and the per-draw
moments / stocks it kept in its metropolis folder. Here the same sub-draws go through pydyn, so every comparison is
deterministic and entry by entry.
"""
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation, posterior_moments as pm

ROOT = Path(__file__).resolve().parent.parent
import os

REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
pytestmark = pytest.mark.filterwarnings("ignore:statements not handled")
_cache = {}


def ref(name):
    if name not in _cache:
        _cache[name] = loadmat(REF / name, squeeze_me=True)
    return _cache[name]


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    """load(model, case) -> Estimation for reference/postmom_<model>.mod preprocessed with -DPM=<case>."""
    work = tmp_path_factory.mktemp("postmom")
    made = {}

    def _load(mod, case):
        if (mod, case) not in made:
            sub = work / f"{mod}_{case}"
            sub.mkdir()
            for f in list(REF.glob("est_*_data.m")) + [REF / f"postmom_{mod}.mod", REF / f"post_{mod}_mode0.mat"]:
                shutil.copy(f, sub)
            m = pydyn.Model(pydyn.run_preprocessor(sub / f"postmom_{mod}.mod", extra_args=(f"-DPM={case}",)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                made[(mod, case)] = estimation.Estimation(m)
        return made[(mod, case)]
    return _load


def close(a, b, rtol=1e-9, what="", atol=0.0):
    """Entry-wise agreement relative to the entry plus the scale of the array; NaN must sit in the same places."""
    a, b = np.asarray(a, float), np.asarray(b, float).reshape(np.shape(a))
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=what)
    scale = np.nanmax(np.abs(b)) if np.isfinite(b).any() else 1.0
    err = np.abs(a - b) - rtol * (np.abs(b) + scale) - atol
    assert np.all(err[~np.isnan(err)] <= 0), f"{what}: largest excess {np.nanmax(err):.3e}"


MOMENT_CASES = [("nk", 1), ("nk", 2), ("nk", 4), ("nk", 5), ("schorfheide", 1)]


def _compare_family(fam, r, prefix, take, dt):
    """fam: MomentFamily; r: reference dict; take: function mapping an array of fam's layout to the reference's."""
    close(take(fam.mean), r[f"{prefix}_mean"], what=f"{prefix} mean")
    close(take(fam.median), r[f"{prefix}_median"], what=f"{prefix} median")
    close(take(fam.variance), r[f"{prefix}_var"], 1e-8, what=f"{prefix} variance")
    close(take(fam.hpd_inf), r[f"{prefix}_hpdinf"], what=f"{prefix} hpd inf")
    close(take(fam.hpd_sup), r[f"{prefix}_hpdsup"], what=f"{prefix} hpd sup")
    close(take(fam.deciles), r[f"{prefix}_deciles"], what=f"{prefix} deciles")
    close(take(fam.density)[..., ::dt, :], r[f"{prefix}_density"], 1e-8, what=f"{prefix} density")


@pytest.mark.parametrize("mod,case", MOMENT_CASES)
def test_posterior_moments_match_dynare(load, mod, case):
    """Dynare's sub-draws -> oo_.PosteriorTheoreticalMoments.dsge: every statistic of every moment."""
    est = load(mod, case)
    r = ref(f"postmom_{mod}_{case}.mat")
    assert est.names == r["param_names"].split(";")
    assert r["pdraws"].shape[0] == int(r["nd"])
    res = pm.posterior_theoretical_moments(est, params=r["pdraws"], keep_draws=True)
    v = r["vartan"].split(";")
    n, nexo = len(v), len(est.model.exo_names)
    assert res.var_names == v and res.exo_names == r["exo_names"].split(";")
    assert res.mh_conf_sig == r["mh_conf_sig"] and res.ar == int(r["ar"]) and res.n_draws == int(r["nd"])
    DT = int(r["dt"])          # density grid thinning of the reference (covariances: every 8th point)
    iu = np.triu_indices(n)
    _compare_family(res.covariance, r, "cov", lambda a: a[iu], 8)
    if r["contemporaneous_correlation"]:
        _compare_family(res.contemporaneous_correlation, r, "ccorr", lambda a: a[iu], 8)
    else:
        assert res.contemporaneous_correlation is None
    _compare_family(res.correlation, r, "corr", lambda a: a, DT)
    if "vd_mean" in r:
        _compare_family(res.variance_decomposition, r, "vd", lambda a: a, DT)
    else:
        assert res.variance_decomposition is None and r["nodecomposition"]
    if "vdme_mean" in r:
        assert res.me_var_names == r["me_vars"].split(";")
        _compare_family(res.variance_decomposition_me, r, "vdme", lambda a: a, DT)
    else:
        assert res.variance_decomposition_me is None
    if "cvd_mean" in r:
        assert res.steps == list(np.atleast_1d(r["steps"]).astype(int))
        # Dynare's layout (variable, shock, step); deciles of a step are 9 values
        _compare_family(res.conditional_variance_decomposition, r, "cvd", lambda a: a, DT)
        assert list(np.atleast_1d(r["cvd_steps"])) == res.steps
    else:
        assert res.conditional_variance_decomposition is None
    if "cvdme_mean" in r:
        _compare_family(res.conditional_variance_decomposition_me, r, "cvdme", lambda a: a, DT)
    else:
        assert res.conditional_variance_decomposition_me is None
    # the per-draw moments Dynare kept on disk: first draws, full precision
    d = res.draws
    k = r["draw_cov"].shape[0]
    # Schorfheide's second draw has a root of 0.99 in the state transition: the first-order solution and Lyapunov equation of
    # the two programs then differ by up to 3e-10 relative (1e-14 at the draws with roots below 0.97)
    tol = 1e-12 if mod == "nk" else 1e-9
    close(d["variance"][:k][:, iu[0], iu[1]], r["draw_cov"], tol, "draw covariances")
    close(d["autocorr"][:k], r["draw_corr"].reshape(d["autocorr"][:k].shape), tol, "draw autocorrelations")
    # (a case's metropolis folder may hold per-draw files of an earlier case for families it did not compute: skipped)
    if "draw_vd" in r and d["decomposition"] is not None:
        close(d["decomposition"][:k].reshape(k, -1), r["draw_vd"], tol, "draw variance decompositions")
    if "draw_vdme" in r and d["decomposition_me"] is not None:
        close(d["decomposition_me"][:k].reshape(k, -1), r["draw_vdme"], tol, "draw variance decompositions with ME")
    if "draw_cvd" in r and d["conditional"] is not None:
        ref_cvd = np.moveaxis(r["draw_cvd"], -1, 0)                    # (draws, var, step, shock)
        close(np.moveaxis(d["conditional"][:k], 3, 2), ref_cvd, tol, "draw conditional decompositions")
    if "draw_cvdme" in r and d["conditional_me"] is not None:
        ref_me = np.moveaxis(r["draw_cvdme"], -1, 0)
        close(np.moveaxis(d["conditional_me"][:k], 3, 2), ref_me, tol, "draw conditional decompositions with ME")


def test_moments_structure_and_options(load):
    """oo(): Dynare's nested structure; option handling of the .mod file."""
    est = load("nk", 1)
    assert est.options["moments_varendo"] and est.options["ar"] == 4 and est.options["contemporaneous_correlation"]
    assert est.options["conditional_variance_decomposition"] == [1, 4, 10] and est.var_list == ["y", "pi", "r", "robs", "yobs"]
    assert not any(k in est.unsupported_options for k in ("moments_varendo", "ar", "smoother", "filtered_vars",
                                                          "conditional_variance_decomposition"))
    r = ref("postmom_nk_1.mat")
    res = pm.posterior_theoretical_moments(est, params=r["pdraws"][:30], density=False)
    oo = res.oo()
    assert set(oo) == {"covariance", "contemporeaneous_correlation", "correlation", "VarianceDecomposition",
                       "VarianceDecompositionME", "ConditionalVarianceDecomposition", "ConditionalVarianceDecompositionME"}
    assert set(oo["covariance"]["Mean"]["y"]) == {"y", "pi", "r", "robs", "yobs"}            # var1 listed first
    assert set(oo["covariance"]["Mean"]["yobs"]) == {"yobs"}
    assert oo["covariance"]["Mean"]["y"]["pi"] == res.covariance.mean[0, 1] == res.covariance.mean[1, 0]
    assert len(oo["correlation"]["deciles"]["y"]["pi"]) == 4 and oo["ConditionalVarianceDecomposition"]["Steps"] == [1, 4, 10]
    assert set(oo["VarianceDecompositionME"]["Mean"]) == {"robs", "yobs"} and "ME" in oo["VarianceDecompositionME"]["Mean"]["yobs"]
    assert "density" not in oo["covariance"]
    # the variance decomposition sums to one, with and without the measurement error
    np.testing.assert_allclose(res.variance_decomposition.mean.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(res.variance_decomposition_me.mean.sum(axis=1), 1.0, rtol=1e-12)
    # no variable list and no consider_ option: all original endogenous variables (nointeractive); consider_ options
    with pytest.warns(UserWarning, match="specified twice"):
        assert pm.moment_variable_list(est, ["y", "y", "pi"]) == ["y", "pi"]
    with pytest.raises(ValueError, match="consider_all_endogenous"):
        pm.moment_variable_list(est, consider="all_endogenous_variables")
    est5 = load("nk", 5)
    assert pm.moment_variable_list(est5) == est5.varobs
    est4 = load("nk", 4)
    assert pm.moment_variable_list(est4) == est4.model.orig_endo_names
    with pytest.raises(ValueError, match="not endogenous"):
        pm.moment_variable_list(est, ["nonexistent"])
    with pytest.raises(ValueError, match="consider_all_endogenous"):
        pm.moment_variable_list(est4, ["y"])


def test_moments_from_chains_and_theta_function(load):
    """The sub-draws come from the chains (posterior_subsample) and a single draw's moments are the theoretical moments
    of stoch_simul at that calibration."""
    from pydyn import moments
    est = load("nk", 1)
    r = ref("postmom_nk_1.mat")
    th = r["pdraws"][0]
    one = pm.theoretical_moments_at(est, th)
    cal = est.calibration(th)
    ss = est.state_space(th)
    mod = est.model
    from pydyn.perturbation import first_order
    sol = first_order(mod, params=cal["params"])
    ref_m = moments.theoretical_moments(mod, sol.as_dr(), est.var_list, Sigma_e=cal["Sigma_e"], H=np.diag(cal["H"]), ar=4,
                                        conditional_variance_decomposition=[1, 4, 10])
    np.testing.assert_allclose(one["variance"], ref_m.var, rtol=1e-12)
    np.testing.assert_allclose(one["decomposition"], ref_m.variance_decomposition / 100, rtol=1e-12)
    np.testing.assert_allclose(one["decomposition_me"], ref_m.variance_decomposition_me / 100, rtol=1e-12)
    np.testing.assert_allclose(one["conditional"], ref_m.conditional_variance_decomposition, rtol=1e-12)
    assert ss is not None
    # chains: the stored Dynare chains of the posterior test
    c = loadmat(REF / "post_nk_chains.mat", squeeze_me=True)
    idx = c["index"].astype(int) - 1
    draws = np.stack([c["points"][idx[:, b]] for b in range(int(c["nblck"]))])
    lp = np.stack([c["points_lp"][idx[:, b]] for b in range(int(c["nblck"]))])
    mc = estimation.MCMCResult(draws=draws, log_posterior=lp, acceptance_rate=c["acceptance"], names=est.names,
                               x0=c["x_init"], drop=est.options["mh_drop"])
    res = pm.posterior_theoretical_moments(est, mc, rng=np.random.default_rng(3), density=False)
    assert res.n_draws == 90 and res.params.shape == (90, 12)          # sub_draws=90 of the .mod file
    assert np.all(res.covariance.hpd_inf <= res.covariance.hpd_sup)
    r2 = pm.posterior_theoretical_moments(est, mc, rng=np.random.default_rng(3), density=False, sub_draws=20, ar=2,
                                          steps=[3], nodecomposition=True, var_list=["y", "pi"], mh_conf_sig=0.5)
    assert r2.n_draws == 20 and r2.correlation.mean.shape == (2, 2, 2) and r2.variance_decomposition is None
    assert r2.conditional_variance_decomposition.mean.shape == (2, 3, 1) and r2.mh_conf_sig == 0.5


# ------------------------------------------------------------------ smoothed and filtered variables
SMOOTHER_CASES = [("nk", 3), ("schorfheide", 2), ("nk", 6)]


@pytest.mark.parametrize("mod,case", SMOOTHER_CASES)
def test_smoother_statistics_match_dynare(load, mod, case):
    """Dynare's sub-draws (metropolis/<model>_param1.mat) -> oo_.SmoothedVariables, UpdatedVariables, SmoothedShocks,
    SmoothedMeasurementErrors, Constant, Smoother.Trend / TrendCoeff / Init_State, FilteredVariables and
    Filtered_Variables_<k>_step_ahead."""
    est = load(mod, case)
    r = ref(f"postmom_{mod}_{case}.mat")
    assert est.names == r["param_names"].split(";")
    res = pm.posterior_smoother_statistics(est, params=r["sm_params"].reshape(-1, len(est.names)), keep_draws=True)
    names = r["sm_varlist"].split(";")
    assert res.var_names == names and res.n_draws == int(r["sm_nd"])
    pairs = [("smv", res.smoothed_variables), ("upd", res.updated_variables), ("inn", res.smoothed_shocks),
             ("sme", res.smoothed_measurement_errors), ("cst", res.constant), ("tco", res.trend_coeff),
             ("ini", res.init_state), ("trd", res.trend), ("flt", res.filtered_variables)]
    pairs += [(f"fk{k}", sd) for k, sd in res.filtered_k_step_ahead.items()]
    checked = 0
    pidx = np.atleast_1d(r["pidx"]).astype(int) - 1                      # periods stored (every 10th and the last)
    for pre, sd in pairs:
        if f"{pre}_mean" not in r:
            assert sd is None, pre
            continue
        checked += 1
        ix = np.array([0]) if sd.mean.shape[0] == 1 else pidx
        shp = sd.mean[ix].shape
        close(sd.mean[ix], r[f"{pre}_mean"].reshape(shp), 1e-9, f"{pre} mean")
        close(sd.median[ix], r[f"{pre}_median"].reshape(shp), 1e-9, f"{pre} median")
        # the variance over draws of a series that is the same at every draw (the constant of an unobserved variable) is
        # rounding noise of order 1e-31 in both programs
        close(sd.var[ix], r[f"{pre}_var"].reshape(shp), 1e-8, f"{pre} var", atol=1e-25)
        close(sd.hpd_inf[ix], r[f"{pre}_hpdinf"].reshape(shp), 1e-9, f"{pre} hpd inf")
        close(sd.hpd_sup[ix], r[f"{pre}_hpdsup"].reshape(shp), 1e-9, f"{pre} hpd sup")
        dec = sd.deciles[:, ix]
        close(dec, r[f"{pre}_deciles"].reshape(dec.shape), 1e-9, f"{pre} deciles")
        den = sd.density[::(int(r["dts"]) if "dts" in r else 32)][:, :, ix]
        close(den, r[f"{pre}_density"].reshape(den.shape), 1e-8, f"{pre} density")
    assert checked >= 8
    fsa = list(np.atleast_1d(r["filter_step_ahead"]).astype(int))
    assert res.filter_step_ahead == fsa and list(res.filtered_k_step_ahead) == fsa
    close(res.filtered_k_step_ahead_mean, r["kstep_mean"].reshape(res.filtered_k_step_ahead_mean.shape), 1e-9, "kstep")
    close(res.filtered_k_step_ahead_variances, r["kstep_var"].reshape(res.filtered_k_step_ahead_mean.shape), 1e-9, "kstep var")
    # the first three draws' stocks Dynare kept
    d = res.draws
    ids = [list(est.model.endo_names).index(v) for v in est.model.endo_names]          # all variables, declaration order
    nd = r["stock_smooth"].shape[2] if r["stock_smooth"].ndim == 3 else 1     # draws stored in the reference
    close(np.moveaxis(d["smooth"][:nd], 0, 2).transpose(1, 0, 2), r["stock_smooth"], 1e-12, "stock smooth")
    close(np.moveaxis(d["update"][:nd], 0, 2).transpose(1, 0, 2), r["stock_update"], 1e-12, "stock update")
    close(np.moveaxis(d["innov"][:nd], 0, 2).transpose(1, 0, 2), r["stock_inno"], 1e-12, "stock innovations")
    if "stock_error" in r:
        close(np.moveaxis(d["error"][:nd], 0, 2).transpose(1, 0, 2), r["stock_error"], 1e-10, "stock measurement errors")
    close(d["init_state"].T, r["stock_init"], 1e-12, "stock init state")
    close(d["trend_coeff"].T, r["stock_tcoef"], 1e-12, "stock trend coeff")
    close(np.moveaxis(d["constant"][:nd], 0, 2).transpose(1, 0, 2), r["stock_const"], 1e-12, "stock constant")
    close(np.moveaxis(d["trend"][:nd], 0, 2).transpose(1, 0, 2), r["stock_trend"], 1e-12, "stock trend")
    close(np.moveaxis(d["filter_step_ahead"][:nd], 0, 3), r["stock_fsa"], 1e-12, "stock filter step ahead")
    assert ids == list(range(len(ids)))


def test_smoother_statistics_conventions(load):
    """Dynare 7.2 quirks reproduced on purpose; structure of oo()."""
    est = load("nk", 6)
    r = ref("postmom_nk_6.mat")
    assert list(np.atleast_1d(r["filter_step_ahead"])) == [2, 1]                    # user order kept
    with pytest.warns(UserWarning, match="shifted by one period"):
        res = pm.posterior_smoother_statistics(est, params=r["sm_params"][:12])
    # FilteredVariables is the *first listed* step (2), shifted by one period: it is Filtered_Variables_2_step_ahead moved
    # by one period, not the one-step-ahead forecast
    np.testing.assert_array_equal(res.filtered_k_step_ahead_mean, res.filtered_k_step_ahead_variances)
    assert res.filter_step_ahead == [2, 1]
    oo = res.oo()
    assert {"SmoothedVariables", "UpdatedVariables", "SmoothedShocks", "Constant", "FilteredVariables",
            "Filtered_Variables_1_step_ahead", "Filtered_Variables_2_step_ahead", "Smoother",
            "FilteredVariablesKStepAhead"} <= set(oo)
    assert set(oo["Smoother"]) == {"Trend", "TrendCoeff", "Init_State"}
    assert oo["SmoothedVariables"]["Mean"]["y"].shape == (res.smoothed_variables.mean.shape[0],)
    assert oo["SmoothedVariables"]["deciles"]["y"].shape[0] == 9
    # trends of observed variables are in the smoothed values; of the others NaN in Trend, zero in TrendCoeff
    k = res.var_names.index("y")
    assert np.all(np.isnan(res.trend.mean[:, k])) and res.trend_coeff.mean[0, k] == 0.0
    j = res.var_names.index("yobs")
    assert res.trend_coeff.mean[0, j] == pytest.approx(0.05)
    with pytest.raises(NotImplementedError, match="smoothed_state_uncertainty"):
        pm.posterior_smoother_statistics(est, params=r["sm_params"][:5], smoothed_state_uncertainty=True)
    with pytest.raises(ValueError, match="neither smoother"):
        pm.posterior_smoother_statistics(est, params=r["sm_params"][:5], smoother=False, filtered_vars=False,
                                         filter_step_ahead=[])


# ------------------------------------------------------------------ prior
def test_prior_moments_distribution_matches_dynare(load):
    """``prior moments(distribution)``: Dynare's prior draws (two of them do not solve the model) -> its per-draw first and
    second order moments, and the prior expectation / variance printed from them."""
    est = load("nk", 5)
    r = ref("postmom_nk_7.mat")
    assert est.names == r["param_names"].split(";")
    assert (r["prior_info"] != 0).sum() == 2
    out = pm.prior_moments_distribution(est, params=r["prior_params"])
    assert out["variables"] == r["orig_endo_names"].split(";")
    close(out["FirstOrderMoments"], r["first_order"], 1e-12, "first order moments")
    close(out["SecondOrderMoments"], r["second_order"], 1e-11, "second order moments")
    bad = np.flatnonzero(r["prior_info"] != 0)
    assert np.all(np.isnan(out["FirstOrderMoments"][:, bad]))
    # Dynare's summary statistics, from its own arrays
    F = r["first_order"][:, ~np.isnan(r["first_order"]).any(axis=0)]
    S = r["second_order"][:, :, ~np.isnan(r["second_order"]).any(axis=(0, 1))]
    close(out["PriorExpectationOfFirstOrderMoments"], F.mean(axis=1), 1e-12)
    close(out["PriorVarianceOfFirstOrderMoments"], ((F - F.mean(axis=1)[:, None]) ** 2).mean(axis=1), 1e-12)
    close(out["PriorExpectationOfSecondOrderMoments"], S.mean(axis=2), 1e-11)
    close(out["PriorVarianceOfSecondOrderMoments"], ((S - S.mean(axis=2)[:, :, None]) ** 2).mean(axis=2), 1e-11)


def test_prior_draws_and_prior_theoretical_moments(load):
    est = load("nk", 1)
    rng = np.random.default_rng(5)
    x, share = pm.prior_draws(est, 400, rng)
    assert x.shape == (400, 12) and 0 <= share < 0.5
    lb, ub = est.prior.bounds()
    assert np.all(x >= lb) and np.all(x <= ub)
    big = np.array([pm._prior_draw(est.prior, rng) for _ in range(4000)])
    # the draws follow the priors: means and standard deviations of the marginals (corr ea,eu: normal(0, 0.3); beta: beta(0.99, .002))
    np.testing.assert_allclose(big[:, 4].mean(), 0.0, atol=0.03)
    np.testing.assert_allclose(big[:, 4].std(), 0.3, rtol=0.05)
    np.testing.assert_allclose(big[:, 5].mean(), 0.99, atol=1e-3)
    np.testing.assert_allclose(big[:, 6].mean(), 0.1, rtol=0.05)
    res = pm.prior_theoretical_moments(est, params=x[:25], density=False, var_list=["y", "pi", "r"], ar=2)
    assert res.kind == "prior" and res.n_draws == 25 and res.correlation.mean.shape == (3, 3, 2)
    post = pm.posterior_theoretical_moments(est, params=x[:25], density=False, var_list=["y", "pi", "r"], ar=2)
    np.testing.assert_array_equal(res.covariance.mean, post.covariance.mean)


def test_plot_posterior_series(load):
    import matplotlib
    matplotlib.use("Agg")
    from pydyn import posterior_plots
    est = load("nk", 6)
    r = ref("postmom_nk_6.mat")
    res = pm.posterior_smoother_statistics(est, params=r["sm_params"][:12], density=False)
    d = posterior_plots.posterior_series_data(res.smoothed_variables)
    assert set(d) == set(res.var_names) and d["y"]["deciles"].shape == (9, res.smoothed_variables.mean.shape[0])
    assert "y" not in posterior_plots.posterior_series_data(res.trend)             # all NaN: no panel, as pm3_core.m
    figs = posterior_plots.plot_posterior_series(res.smoothed_variables, "Smoothed variables")
    assert len(figs) == 1
    figs = posterior_plots.plot_posterior_series(res.smoothed_variables, style="dynare")
    assert len(figs[0].axes) >= 6
