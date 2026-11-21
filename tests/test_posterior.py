"""Posterior analysis (pydyn.posterior, pydyn.posterior_plots) against Dynare 7.2 under Octave 8.4.

References are produced by reference/r_posterior.m from two short Metropolis-Hastings runs of Dynare
(post_nk: 3 chains x 2400 draws with non-default mh_drop, mh_conf_sig, Geweke options and relative IRFs;
post_schorfheide: 2 x 3000 draws with Raftery-Lewis diagnostics). Dynare's own chains are stored
(post_*_chains.mat) and every statistic is recomputed from them here, so the comparisons are deterministic.
post_*_replay.mat holds the random numbers Dynare used, regenerated from the seeds it stored: with them
pydyn's sampler must return Dynare's chain draw for draw.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation, posterior

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
MODELS = ["post_nk", "post_schorfheide"]

pytestmark = pytest.mark.filterwarnings("ignore:statements not handled")
_cache = {}


def ref(name):
    if name not in _cache:
        _cache[name] = loadmat(REF / name, squeeze_me=True)
    return _cache[name]


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    """load(mod, post) -> Estimation for reference/<mod>.mod preprocessed with -DPOST=<post>."""
    work = tmp_path_factory.mktemp("post")
    made = {}

    def _load(mod, post=2):
        if (mod, post) not in made:
            sub = work / f"{mod}_{post}"
            sub.mkdir()
            for f in list(REF.glob("est_*_data.m")) + [REF / f"{mod}.mod", REF / f"{mod}_mode0.mat"]:
                shutil.copy(f, sub)
            m = pydyn.Model(pydyn.run_preprocessor(sub / f"{mod}.mod", extra_args=(f"-DPOST={post}",)))
            made[(mod, post)] = estimation.Estimation(m)
        return made[(mod, post)]
    return _load


def chains(mod):
    """Dynare's chains: (draws (chains, T, n), log posterior (chains, T), reference dict)."""
    c = ref(f"{mod}_chains.mat")
    idx = c["index"].astype(int) - 1
    nb = int(c["nblck"])
    draws = np.stack([c["points"][idx[:, b]] for b in range(nb)])
    lp = np.stack([c["points_lp"][idx[:, b]] for b in range(nb)])
    return draws, lp, c


def close(a, b, rtol=1e-10, scale=None):
    """Relative agreement; entries that are tiny compared with ``scale`` (default: the largest entry) are
    compared in absolute terms."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.nanmax(np.abs(b)) if scale is None else scale
    err = np.abs(a - b) - rtol * (np.abs(b) + scale)
    assert np.all(err <= 0), f"largest excess {np.nanmax(err):.3e}"


# ------------------------------------------------------------------ the sampler, draw for draw
@pytest.mark.parametrize("mod", MODELS)
def test_sampler_reproduces_dynare_chain_draw_for_draw(load, mod):
    """Same mode, proposal covariance and random numbers as Dynare's posterior_sampler: same chain."""
    est = load(mod)
    draws, lp, c = chains(mod)
    p = ref(f"{mod}_replay.mat")
    assert est.names == c["names"].split(";")
    assert est.options["mh_jscale"] == c["mh_jscale"] and est.options["mh_nblck"] == c["nblck"]
    lb, ub = est.prior.bounds(est.options["prior_trunc"])
    np.testing.assert_array_equal(lb, c["mh_lb"])
    np.testing.assert_array_equal(ub, c["mh_ub"])
    mode = est.mode()                                        # mode_compute=0 with Dynare's mode file
    np.testing.assert_array_equal(mode.x, c["xparam1"])
    np.testing.assert_array_equal(mode.hessian, c["hh"])
    np.testing.assert_allclose(np.linalg.cholesky(mode.inv_hessian).T, c["proposal_chol"], rtol=1e-11, atol=1e-15)
    nseg = p["seg_normal"].shape[0]
    mc = est.sample(replic=nseg, mode=mode, innovations=np.moveaxis(p["seg_normal"], 2, 0), uniforms=p["seg_uniform"].T,
                    uniform_stream=True, init_innovations=p["init_normal"])
    np.testing.assert_allclose(mc.x0, c["x_init"], rtol=0, atol=1e-12)        # starting points (mh_init_scale_factor)
    np.testing.assert_allclose(mc.draws, draws[:, :nseg], rtol=0, atol=1e-10)
    np.testing.assert_allclose(mc.log_posterior, lp[:, :nseg], rtol=1e-11, atol=0)   # measured: 1e-12 (Schorfheide), 3e-16
    moved = np.any(np.diff(draws[:, :nseg], axis=1) != 0, axis=2)
    np.testing.assert_array_equal(mc.accepted[:, 1:], moved)
    assert mc.accepted.sum() > 0.1 * mc.accepted.size
    # the whole chain is on the posterior kernel pydyn computes (every distinct point of Dynare's run)
    k = np.linspace(0, len(c["points"]) - 1, 60).astype(int)
    mine = [est.log_posterior(c["points"][i], bounds=(lb, ub)) for i in k]
    np.testing.assert_allclose(mine, c["points_lp"][k], rtol=1e-11, atol=0)
    np.testing.assert_allclose([np.mean(np.any(np.diff(d, axis=0) != 0, axis=1)) for d in draws],
                               c["acceptance"], atol=1.5 / draws.shape[1])


@pytest.mark.parametrize("mod", MODELS)
def test_tune_jscale_matches_dynare(load, mod):
    est = load(mod)
    c, p = ref(f"{mod}_chains.mat"), ref(f"{mod}_replay.mat")
    o = p["tune_opt"]
    scale, info = posterior.tune_jscale(est, c["xparam1"], p["tune_cov"], target=o[2], maxiter=int(o[0]), stepsize=int(o[1]),
                                        rho=o[3], c1=o[4], c2=o[5], c3=o[6], innovations=p["tune_normal"],
                                        uniforms=p["tune_uniform"], return_info=True)
    np.testing.assert_allclose(scale, p["tune_scale"], rtol=1e-12)
    assert info["uniforms_used"] == int(p["tune_used_uniform"])
    o = p["tune_opt2"]
    scale = posterior.tune_jscale(est, c["xparam1"], p["tune_cov"], target=o[2], maxiter=int(o[0]), stepsize=int(o[1]),
                                  rho=o[3], c1=o[4], c2=o[5], c3=o[6], guess=o[7], innovations=p["tune_normal"],
                                  uniforms=p["tune_uniform"])
    np.testing.assert_allclose(scale, p["tune_scale2"], rtol=1e-12)


# ------------------------------------------------------------------ statistics from Dynare's draws
@pytest.mark.parametrize("mod", MODELS)
def test_posterior_statistics_match_dynare(mod):
    draws, lp, c = chains(mod)
    r = ref(f"{mod}_stats.mat")
    st = posterior.posterior_statistics(draws, mh_conf_sig=float(c["mh_conf_sig"]), drop=float(c["mh_drop"]), log_posterior=lp)
    for mine, name in [(st.mean, "posterior_mean"), (st.median, "posterior_median"), (st.std, "posterior_std"),
                       (st.variance, "posterior_variance"), (st.hpd_inf, "posterior_hpdinf"), (st.hpd_sup, "posterior_hpdsup"),
                       (st.deciles, "posterior_deciles")]:
        np.testing.assert_allclose(mine, r[name], rtol=1e-10, atol=0, err_msg=name)
    np.testing.assert_array_equal(st.mode, r["mh_mode"])          # Dynare searches all draws, burn-in included
    assert st.mode_log_posterior == r["mh_mode_lp"]
    # kernel density estimates (every fourth of the 512 grid points is stored)
    np.testing.assert_allclose(st.density[:, ::4, 0], r["density_x"], rtol=1e-10)
    np.testing.assert_allclose(st.density[:, -1, 0], r["density_last"], rtol=1e-12)
    for j in range(draws.shape[2]):
        close(st.density[j, ::4, 1], r["density_f"][j])
    assert "HPD interval" in st.table() and len(st.table().splitlines()) == draws.shape[2] + 1


def test_posterior_moments_details():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1000)
    m = posterior.posterior_moments(x, 0.9)
    s = np.sort(x)
    widths = s[899:999] - s[:100]                   # xx(jj) - xx(ii), jj = n - hpd_draws + ii - 1
    k = np.argmin(widths)
    np.testing.assert_allclose(m["hpd"], [s[k], s[k] + widths[k]])
    np.testing.assert_array_equal(m["deciles"], s[np.arange(100, 1000, 100) - 1])
    assert np.all(np.isnan(posterior.posterior_moments(x[:15], 0.9)["hpd"]))        # round(0.1*15) = 2 draws: too few
    assert np.all(np.isnan(posterior.posterior_moments(x[:5], 0.9)["deciles"]))
    assert np.all(np.isnan(posterior.posterior_moments(np.ones(50), 0.9, density=True)["density"]))
    assert posterior._mround(2.5) == 3 and posterior._mround(-2.5) == -3 and posterior._mround(0.4999) == 0
    # repeated draws widen the bandwidth (Skold and Roberts): correction = mean(2 m_i - 1)
    d = np.sort(np.repeat(rng.standard_normal(50), 3))
    assert posterior._repeated_draws_correction(d) == pytest.approx(np.mean(np.tile([5, 3, 1], 50)))
    h0 = posterior.optimal_bandwidth(np.sort(rng.standard_normal(150)))
    assert posterior.optimal_bandwidth(d) > 1.2 * h0 * (np.std(d, ddof=1))
    xg, f = posterior.kernel_density_estimate(x, posterior.optimal_bandwidth(s), 2 ** 8)
    assert abs(np.sum(f) * (xg[1] - xg[0]) - 1) < 1e-6 and f.min() > -1e-12
    with pytest.raises(ValueError, match="power of 2"):
        posterior.kernel_density_estimate(x, 0.1, 100)
    with pytest.raises(NotImplementedError):
        posterior.optimal_bandwidth(s, bandwidth=-2)
    assert posterior.optimal_bandwidth(s[::4], bandwidth=-1) > 0


@pytest.mark.parametrize("mod", MODELS)
def test_modified_harmonic_mean_matches_dynare(mod):
    draws, lp, c = chains(mod)
    r = ref(f"{mod}_stats.mat")
    m = posterior.modified_harmonic_mean(draws, lp, drop=float(c["mh_drop"]))
    np.testing.assert_allclose(m.value, r["mhm"], rtol=1e-10)
    np.testing.assert_allclose(m.table, r["mhm_table"], rtol=1e-10)
    np.testing.assert_allclose(m.mean, r["mh_mean"], rtol=1e-10)
    close(m.covariance, r["mh_cov"], scale=np.sqrt(np.outer(np.diag(r["mh_cov"]), np.diag(r["mh_cov"]))))
    assert m.converged and float(m) == m.value
    # a tolerance that cannot be met: the covariance of the weighting density is scaled by 1.2 until the factor
    # exceeds 20, and the last table is kept
    t = posterior.modified_harmonic_mean(draws, lp, drop=float(c["mh_drop"]), tolerance=1e-5)
    np.testing.assert_allclose(t.table, r["mhm_table_tight"], rtol=1e-10)
    np.testing.assert_allclose(t.value, r["mhm_tight"], rtol=1e-10)
    assert not t.converged and t.increase == pytest.approx(1.2 ** 16)


def test_model_comparison():
    out = posterior.model_comparison([-310.0, -312.5, -309.0], names=["a", "b", "c"])
    w = np.exp(np.array([-310.0, -312.5, -309.0]) + 309)
    np.testing.assert_allclose(out["posterior_model_probability"], w / w.sum())
    np.testing.assert_allclose(out["bayes_ratio"], np.exp([0, -2.5, 1.0]))
    np.testing.assert_allclose(out["posterior_odds_ratio"], out["bayes_ratio"])
    np.testing.assert_allclose(out["prior"], 1 / 3)
    out = posterior.model_comparison([-310.0, -312.5], priors=[1, 3])              # improper: rescaled
    np.testing.assert_allclose(out["prior"], [0.25, 0.75])
    np.testing.assert_allclose(out["posterior_odds_ratio"], [1, 3 * np.exp(-2.5)])
    np.testing.assert_allclose(out["posterior_model_probability"].sum(), 1)
    with pytest.raises(ValueError):
        posterior.model_comparison([1.0, 2.0], priors=[1.0])


def test_model_comparison_and_jumping_covariance_match_dynare():
    """model_comparison.m (equal priors, user priors, improper priors rescaled) and set_mcmc_jumping_covariance.m
    run by Dynare on the same inputs (reference/r_posterior.m, stage 5)."""
    import types
    q = ref("post_misc.mat")
    keys = {"Prior": "prior", "Log_Marginal_Density": "log_marginal_density", "Bayes_Ratio": "bayes_ratio",
            "Posterior_Odds_Ratio": "posterior_odds_ratio", "Posterior_Model_Probability": "posterior_model_probability"}
    for c, (pri, kind) in enumerate([(None, "laplace"), (q["mc_priors2"], "laplace"), (q["mc_priors3"], "mhm")], start=1):
        out = posterior.model_comparison(q[kind], priors=pri)
        fields = q[f"mc{c}_fields"].split(";")
        for k, f in enumerate(fields):
            np.testing.assert_allclose(out[keys[f]], q[f"mc{c}"][k], rtol=1e-13, err_msg=f)
    est = types.SimpleNamespace(n=3, prior=types.SimpleNamespace(p2=q["jc_p2"]))
    np.testing.assert_array_equal(posterior.jumping_covariance(est, "hessian", q["jc_V0"]), q["jc_hessian"])
    np.testing.assert_allclose(posterior.jumping_covariance(est, "prior_variance"), q["jc_prior"], rtol=1e-14, atol=1e-16)
    np.testing.assert_array_equal(posterior.jumping_covariance(est, "identity_matrix"), q["jc_identity"])
    # Dynare inverts a matrix read from a file; pydyn uses it as the covariance unless as_dynare=True
    np.testing.assert_allclose(posterior.jumping_covariance(est, q["jc_file_matrix"], as_dynare=True), q["jc_file"], rtol=1e-13)
    np.testing.assert_allclose(q["jc_file"], np.linalg.inv(q["jc_file_matrix"]), rtol=1e-13)
    np.testing.assert_array_equal(posterior.jumping_covariance(est, q["jc_file_matrix"]), q["jc_file_matrix"])


@pytest.mark.parametrize("mod", MODELS)
def test_convergence_diagnostics_match_dynare(mod):
    draws, lp, c = chains(mod)
    r = ref(f"{mod}_stats.mat")
    drop = float(c["mh_drop"])
    np.testing.assert_allclose(posterior.inefficiency_factors(draws, drop=drop), r["ifac"], rtol=1e-9)
    g = posterior.geweke(draws, r["geweke_interval"], r["taper_steps"], drop=drop)
    fields = r["geweke_fields"].split(";")
    np.testing.assert_allclose(np.stack([g["moments"][k] for k in fields], axis=2), r["geweke"], rtol=1e-10)
    np.testing.assert_allclose(g["pooled_mean"], r["geweke_pooled_mean"], rtol=1e-10)
    np.testing.assert_allclose(g["pooled_nse"], r["geweke_pooled_nse"], rtol=1e-10)
    np.testing.assert_allclose(g["prob_chi2_test"], r["geweke_prob"], rtol=1e-8, atol=1e-12)
    bg = posterior.brooks_gelman(draws, lp, drop=drop)
    np.testing.assert_array_equal(bg["x"], r["bg_x"])
    n = draws.shape[2]
    # Dynare's figures: per parameter interval / m2 / m3, pooled then within; then the posterior kernel
    mine = np.concatenate([bg["univariate"][:, :, j] for j in range(n)] + [bg["multivariate"]], axis=1)
    np.testing.assert_allclose(mine, r["bg"], rtol=1e-10)
    if "raftery" in r:
        rl = posterior.raftery_lewis_diagnostics(draws, r["raftery_qrs"])
        mine = np.stack([np.stack([x[k] for k in r["raftery_fields"].split(";")], axis=1) for x in rl])
        np.testing.assert_allclose(mine, r["raftery"], rtol=1e-12)
        assert rl[0]["N_min"] == r["raftery_N_min"]


def test_convergence_diagnostics_options_and_limits(load):
    est = load("post_nk")
    draws, lp, c = chains("post_nk")
    r = ref("post_nk_stats.mat")
    assert list(est.options["taper_steps"]) == [4, 10, 20] and list(est.options["geweke_interval"]) == [0.1, 0.6]
    mc = estimation.MCMCResult(draws=draws, log_posterior=lp, acceptance_rate=c["acceptance"], names=est.names,
                               x0=c["x_init"], drop=est.options["mh_drop"])
    cd = posterior.convergence_diagnostics(mc, est)            # options of the estimation command
    np.testing.assert_allclose(cd["geweke"]["prob_chi2_test"], r["geweke_prob"], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(cd["inefficiency_factors"], r["ifac"], rtol=1e-9)
    assert "raftery_lewis" not in cd and cd["brooks_gelman"]["univariate"].shape == (101, 6, 12)
    assert "raftery_lewis" in posterior.convergence_diagnostics(mc, est, raftery_lewis_qrs=(0.025, 0.01, 0.9))
    short = posterior.convergence_diagnostics(draws[:, :2000], drop=0.5)       # Dynare: nothing but the factors
    assert set(short) == {"inefficiency_factors"}
    assert posterior.brooks_gelman(draws[:1], lp[:1])["univariate"] is None     # one chain
    bad = posterior.geweke(draws, (0.7, 0.2), (4,), drop=0.5)                   # invalid interval: Dynare's default
    np.testing.assert_array_equal(bad["prob_chi2_test"], posterior.geweke(draws, (0.2, 0.5), (4,), drop=0.5)["prob_chi2_test"])
    # white noise: Dynare's inefficiency factor is near three (its sum includes lag 0), Raftery-Lewis thinning is one
    z = np.random.default_rng(0).standard_normal((1, 20000, 1))
    assert abs(posterior.inefficiency_factors(z, drop=0)[0, 0] - 3) < 0.15
    assert posterior.raftery_lewis(z[0])["k_thin"][0] == 1


# ------------------------------------------------------------------ posterior IRFs
@pytest.mark.parametrize("mod", MODELS)
def test_posterior_irf_matches_dynare(load, mod):
    """Dynare's own sub-draws (metropolis/<model>_param_irf1.mat) -> oo_.PosteriorIRF.dsge."""
    est = load(mod)
    r = ref(f"{mod}_stats.mat")
    ir = posterior.posterior_irf(est, params=r["irf_params"], keep_draws=True)
    assert ir.names == r["irf_names"].split(";") and ir.variables == r["irf_varlist"].split(";")
    scale = np.abs(r["irf_mean"]).max()
    for mine, name in [(ir.mean, "irf_mean"), (ir.median, "irf_median"), (ir.hpd_inf, "irf_hpdinf"),
                       (ir.hpd_sup, "irf_hpdsup"), (ir.deciles, "irf_deciles")]:
        np.testing.assert_allclose(mine, r[name], rtol=1e-9, atol=1e-10 * scale, err_msg=name)
    np.testing.assert_allclose(ir.var, r["irf_var"], rtol=1e-8, atol=1e-10 * np.abs(r["irf_var"]).max())
    np.testing.assert_allclose(np.moveaxis(ir.draws[:3], 0, 3), r["irf_draws_first"], rtol=1e-9, atol=1e-10 * scale)
    np.testing.assert_array_equal(ir[ir.names[0]]["Mean"], ir.mean[:, 0])


def test_posterior_irf_conventions(load):
    est = load("post_nk")
    draws, lp, c = chains("post_nk")
    assert est.options["relative_irf"] and est.options["irf"] == 12 and est.options["irf_shocks"] == ["ea", "er"]
    assert est.var_list == ["y", "pi", "r", "robs"] and posterior.irf_shocks(est) == [0, 2]
    theta = c["xparam1"]
    rel = posterior.irf_at(est, theta, 12, relative_irf=True)
    lev = posterior.irf_at(est, theta, 12)
    Q = est.calibration(theta)["Sigma_e"]
    S = np.linalg.cholesky(Q)
    np.testing.assert_allclose(rel, 100 * lev / np.diag(S), rtol=1e-12, atol=1e-14)
    # the impulse is a column of the Cholesky factor: with corr(ea, eu) != 0 the ea impulse moves eu too
    ss = est.state_space(theta)
    np.testing.assert_allclose(lev[0, :, 0], ss["Hfull"] @ S[:, 0], rtol=1e-12, atol=1e-15)
    assert S[1, 0] != 0
    mc = estimation.MCMCResult(draws=draws, log_posterior=lp, acceptance_rate=c["acceptance"], names=est.names,
                               x0=c["x_init"], drop=est.options["mh_drop"])
    assert posterior.number_of_subdraws(mc, est) == 90                         # sub_draws from the .mod file
    assert posterior.number_of_subdraws(mc, None) == 1200                      # posterior_max_subsample_draws
    sub, sub_lp = posterior.posterior_subsample(mc, 50, rng=np.random.default_rng(1))
    kept = posterior.pooled_draws(mc)
    assert kept.shape[0] == 3 * 1440 and all((kept == s).all(axis=1).any() for s in sub)
    ir = posterior.posterior_irf(est, mc, sub_draws=25, rng=np.random.default_rng(2))
    assert ir.mean.shape == (12, 8) and ir.shocks == ["ea", "er"] and np.all(ir.hpd_inf <= ir.hpd_sup)
    assert ir.mh_conf_sig == 0.8
    # convenience methods of Estimation use the options of the estimation command
    st = est.posterior_statistics(mc, density=False)
    assert st.mh_conf_sig == 0.8 and st.prior_shape is not None and "norm" in st.table()
    np.testing.assert_allclose(st.mean, ref("post_nk_stats.mat")["posterior_mean"], rtol=1e-10)
    assert float(est.modified_harmonic_mean(mc)) == pytest.approx(float(ref("post_nk_stats.mat")["mhm"]), rel=1e-10)
    assert est.posterior_irf(params=sub[:20]).mean.shape == (12, 8)


# ------------------------------------------------------------------ mode check, prior densities
@pytest.mark.parametrize("mod", MODELS)
def test_mode_check_and_prior_densities_match_dynare(load, mod):
    est = load(mod, 1)
    m = ref(f"{mod}_mode.mat")
    assert est.options["mode_check"] is True
    for key, kw in [("check", {}), ("check2", dict(neighbourhood_size=0.2, symmetric_plots=False, number_of_points=8)),
                    ("check3", dict(neighbourhood_size=np.inf, symmetric_plots=False, number_of_points=6))]:
        mine = posterior.mode_check(est, m["xparam1"], **kw)
        for j, name in enumerate(est.names):
            want = m[key][j]
            want = want[~np.isnan(want[:, 0])]
            assert mine[name].shape == want.shape, (key, name)
            np.testing.assert_allclose(mine[name][:, 0], want[:, 0], rtol=1e-12, atol=1e-14, err_msg=f"{key} {name}")
            np.testing.assert_array_equal(np.isnan(mine[name]), np.isnan(want))
            # Dynare's grid for a parameter with infinite bounds reaches -10 (see mode_check_grid). For logmst = -10
            # and -9 (money growth of exp(-10)) the posterior kernel is about -2e7 and the two likelihoods agree to
            # 2% and 7e-6 only; everywhere else, values of that size included, they agree to 1e-9
            with np.errstate(invalid="ignore"):
                off = np.abs(mine[name] - want) > 1e-9 * np.abs(want) + 1e-12
            assert off.sum() <= (4 if (mod, name) == ("post_schorfheide", "logmst") else 0), (key, name)
            np.testing.assert_allclose(mine[name], want, rtol=0.05, err_msg=f"{key} {name}")
    X, F = posterior.prior_density(est)
    # rho in the Schorfheide model has a beta prior with an asymptote at zero; its 0.1% quantile is 3.0e-19
    # (scipy) but Octave's betainv returns 1.3e-9, which shifts Dynare's grid by that much
    odd = np.array([mod == "post_schorfheide" and n == "rho" for n in est.names])
    np.testing.assert_allclose(X[~odd], m["prior_x"][~odd], rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(F[~odd], m["prior_f"][~odd], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(X[odd], m["prior_x"][odd], rtol=0, atol=2e-9)
    np.testing.assert_array_equal(np.isnan(F), np.isnan(m["prior_f"]))
    np.testing.assert_allclose(F[odd][:, 1:], m["prior_f"][odd][:, 1:], rtol=1e-6)
    assert np.isnan(F).sum() == (2 if mod == "post_schorfheide" else 0)      # densities above 10 at an end are cut


def test_mode_check_grid_rules():
    z, lo, hi = posterior.mode_check_grid(2.0, 0.0, 10.0, 0.5)
    np.testing.assert_allclose(z, np.linspace(1, 3, 21))
    z, lo, hi = posterior.mode_check_grid(2.0, 1.8, 10.0, 0.5)                 # cut below: made symmetric
    np.testing.assert_allclose([z[0], z[-1]], [1.8, 2.2])
    z, lo, hi = posterior.mode_check_grid(2.0, 1.8, 10.0, 0.5, symmetric_plots=False)
    np.testing.assert_allclose([z[0], z[-1]], [1.8, 3.0])
    z, lo, hi = posterior.mode_check_grid(1.0, 1.0, 10.0, 0.5)                 # at the lower bound: one-sided
    assert lo and not hi and z[0] == 1.0 and z[-1] == 1.5 and z.size == 11
    # Dynare's rule for an infinite bound: from -p2 up to the mode and from the mode down to -1e-6*huge_number
    z, _, _ = posterior.mode_check_grid(0.3, -np.inf, np.inf, 0.2, symmetric_plots=False)
    assert z[0] == -10 and z[-1] == 0.3
    z, _, _ = posterior.mode_check_grid(0.3, -np.inf, np.inf, 0.2, symmetric_plots=False, as_dynare=False)
    np.testing.assert_allclose([z[0], z[-1]], [0.1, 0.5])


# ------------------------------------------------------------------ sampler options, chain files
def test_jumping_covariance_tuning_and_chain_files(load, tmp_path):
    est = load("post_nk")
    mode = est.mode()
    V = mode.inv_hessian
    np.testing.assert_array_equal(posterior.jumping_covariance(est, "hessian", V), V)
    np.testing.assert_array_equal(posterior.jumping_covariance(est, "identity_matrix"), np.eye(12))
    with pytest.raises(ValueError, match="Infinite prior variances"):       # the uniform priors have finite ones
        posterior.jumping_covariance(load("post_schorfheide"), "prior_variance")
    np.testing.assert_array_equal(posterior.jumping_covariance(est, "prior_variance"), np.diag(est.prior.p2 ** 2))
    np.savez(tmp_path / "jc.npz", jumping_covariance=2 * V)
    np.testing.assert_array_equal(posterior.jumping_covariance(est, tmp_path / "jc.npz"), 2 * V)
    # Dynare 7.2 inverts a matrix read from a file (set_mcmc_jumping_covariance.m)
    np.testing.assert_allclose(posterior.jumping_covariance(est, 2 * V, as_dynare=True), np.linalg.inv(2 * V), rtol=1e-8)
    with pytest.raises(ValueError, match="positive definite"):
        posterior.jumping_covariance(est, -np.eye(12))
    with pytest.raises(ValueError, match="12 rows"):
        posterior.jumping_covariance(est, np.eye(3))
    # sample(): covariance choice, starting points, saving and continuing
    a = est.sample(replic=60, nblocks=2, mode=mode, seed=5, save=tmp_path / "run.npz")
    assert a.jscale == 0.9 and a.draws.shape == (2, 60, 12) and a.drop == 0.4
    np.testing.assert_allclose(a.proposal_chol.T @ a.proposal_chol, V, rtol=1e-10)
    b = est.sample(replic=60, nblocks=2, mode=mode, seed=5, jumping_covariance="prior_variance")
    np.testing.assert_allclose(b.proposal_chol, np.diag(est.prior.p2), atol=1e-15)
    back = posterior.load_chains(tmp_path / "run.npz")
    for f in ("draws", "log_posterior", "accepted", "x0", "proposal_chol", "jscale_vector"):
        np.testing.assert_array_equal(getattr(back, f), getattr(a, f))
    assert back.names == a.names and back.jscale == a.jscale and back.drop == a.drop
    more = est.sample(replic=40, load=tmp_path / "run.npz", seed=5)
    assert more.draws.shape == (2, 100, 12)
    np.testing.assert_array_equal(more.draws[:, :60], a.draws)
    lb, ub = est.prior.bounds(0.0)
    # the continuation starts from the last draw: every move is a valid step of the same kernel
    step = more.draws[:, 60] - a.draws[:, 59]
    assert np.all(np.isfinite(more.log_posterior)) and not np.array_equal(more.draws[:, 60:], a.draws[:, :40])
    assert np.all((step == 0).all(axis=1) | (np.abs(step).max(axis=1) > 0))
    assert est.sample(replic=0, load=back).draws.shape == (2, 60, 12)
    one = est.sample(replic=5, nblocks=1, mode=mode)                           # a single chain starts at the mode
    np.testing.assert_array_equal(one.x0[0], mode.x)
    # tuning: the calibrated scale replaces mh_jscale for all parameters
    est2 = estimation.Estimation(est.model, mh_jscale=None, mh_tune_jscale=0.25)
    assert est2.options["mh_tune_jscale"] == 0.25
    t = est2.sample(replic=30, nblocks=1, mode=mode, tune_options=dict(maxiter=240, stepsize=40, c1=0.2, c2=0.2, c3=1))
    assert t.jscale != pytest.approx(2.38 / np.sqrt(12)) and np.all(t.jscale_vector == t.jscale)
    with pytest.raises(estimation.EstimationError, match="incompatible"):
        estimation.Estimation(est.model, mh_tune_jscale=0.3)                   # the .mod file sets mh_jscale
    with pytest.raises(NotImplementedError, match="DIME"):                     # samplers: tests/test_samplers.py
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimation.Estimation(est.model, posterior_sampling_method="dime_mcmc").sample(replic=5, mode=mode)


def test_tuning_reaches_target_acceptance(load):
    est = load("post_nk")
    c = ref("post_nk_chains.mat")
    V = est.inverse_hessian(c["hh"])
    scale, info = posterior.tune_jscale(est, c["xparam1"], V, target=0.3, maxiter=4000, stepsize=200, c1=0.06, c2=0.06,
                                        c3=1, rng=np.random.default_rng(4), return_info=True)
    mc = est.sample(replic=1500, nblocks=1, mode=c["xparam1"], cov=V, jscale=scale, seed=9)
    assert 0.2 < mc.acceptance_rate[0] < 0.4 and 0.3 < scale < 0.9


# ------------------------------------------------------------------ figures: the arrays that are drawn
def test_plot_data_and_figures(load, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from pydyn import posterior_plots as pp
    est = load("post_nk")
    draws, lp, c = chains("post_nk")
    r = ref("post_nk_stats.mat")
    mc = estimation.MCMCResult(draws=draws, log_posterior=lp, acceptance_rate=c["acceptance"], names=est.names,
                               x0=c["x_init"], drop=est.options["mh_drop"])
    data = pp.brooks_gelman_data(mc, est)
    np.testing.assert_array_equal(data["x"], r["bg_x"])
    np.testing.assert_allclose(data["stderr ea (Interval)"][0], r["bg"][:, 0], rtol=1e-10)
    np.testing.assert_allclose(data["stderr ea (m3)"][1], r["bg"][:, 5], rtol=1e-10)
    np.testing.assert_allclose(data["m2"][0], r["bg"][:, -4], rtol=1e-10)
    dens = pp.posterior_density_data(est, mc, mode=c["xparam1"])
    np.testing.assert_allclose(dens["beta"]["posterior"][1][::4], r["density_f"][5], rtol=1e-8, atol=1e-9 * r["density_f"][5].max())
    assert dens["beta"]["mode"] == c["xparam1"][5] and dens["beta"]["prior"][0].shape == (200,)
    assert pp.trace_data(mc)["sig"].shape == (3, 2400)
    figs = pp.plot_brooks_gelman(mc, est)
    assert len(figs) == 5 and len(figs[0].axes) == 9 and len(figs[-1].axes) == 3
    np.testing.assert_allclose(figs[0].axes[0].lines[0].get_ydata(), r["bg"][:, 0], rtol=1e-10)
    figs += pp.plot_posteriors(est, mc, mode=c["xparam1"], style="dynare") + pp.plot_trace(mc) + pp.plot_priors(est)
    small = pp.plot_mode_check(est, c["xparam1"], number_of_points=4)
    assert len(small) == 2 and len(small[0].axes[0].lines) == 3
    assert pp.plot_brooks_gelman(draws[:1]) == []
    import matplotlib.pyplot as plt
    figs[0].savefig(tmp_path / "bg.png", dpi=40)
    assert (tmp_path / "bg.png").stat().st_size > 1000
    plt.close("all")


# ------------------------------------------------------------------ long runs: pydyn's chains against Dynare's
def _compare_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("post_sampler_compare", REF / "post_sampler_compare.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    return cmp


@pytest.mark.parametrize("mod", MODELS)
def test_sampler_statistics_agree_with_dynare_long_run(mod):
    """Recorded runs (reference/r_posterior.m stages 3 and 4, reference/post_sampler_compare.py): Dynare 2 x 20000 draws
    with its default seed (0) and with seeds 1000, 2000 and 3000, pydyn 32 (post_schorfheide) and 16 (post_nk) chains of
    20000 draws with the same mode, proposal covariance and scale; batch statistics of the second halves of all chains
    are stored.

    Chains are the replicates: each chain's mean, standard deviation and 5%/95% quantiles of every parameter are
    compared between the samplers with pooled-variance t statistics, and a permutation test (chains relabelled at
    random) gives exact p-values for their root mean square and their largest absolute value.

    History: the first comparison pooled 500-draw batches as if independent and compared pydyn with Dynare's
    default-seed run only; it gave z = -3.7 for the posterior mean of stderr e_m (post_schorfheide). Dynare's own runs
    with other seeds differ from that run by as much (batch-means z = 3.45 for the same statistic, 3.20 for its 5%
    quantile), and the sampler reproduces Dynare's chains draw for draw from Dynare's random numbers (the replay test
    above; all 2 x 20000 draws of both default-seed long runs in a recorded run): Monte Carlo error, understated by
    batch means. Measured here: permutation p-values 0.87 / 0.34 (post_schorfheide) and 0.95 / 0.83 (post_nk) for
    the root mean square / largest |t|; t = -0.15 for the posterior mean of stderr e_m."""
    cmp = _compare_module()
    dyn = cmp.dynare_runs(mod)
    with np.load(REF / f"{mod}_long_pydyn.npz") as z:
        pb, acc = z["batch"], z["acceptance"]
    assert len(dyn) >= 2 and pb.shape[0] == 20 * acc.size and acc.size >= 16
    dyn_batch = np.concatenate([b for _, b, _ in dyn], axis=0)
    assert dyn_batch.shape[1:] == pb.shape[1:]
    pt = cmp.permutation_test(dyn_batch, pb, n_perm=2000)
    assert pt["rms"][1] > 0.01 and pt["max"][1] > 0.01, (pt["rms"], pt["max"])
    t, d, p, dof = cmp.compare_chains(dyn_batch, pb)
    assert abs(t["acceptance"][0]) < 3
    assert abs(np.mean(acc) - np.mean(np.concatenate([a for _, _, a in dyn]))) < 0.01
    if mod == "post_schorfheide":
        # the statistic that looked wrong: Dynare's default run against Dynare's other runs, and pydyn against all of them
        others = np.concatenate([b for _, b, _ in dyn[1:]], axis=0)
        z_dd, *_ = cmp.compare(others, dyn[0][1])
        assert z_dd["mean"][1] > 2.5                       # Dynare against Dynare: as far apart as pydyn was
        assert abs(t["mean"][1]) < 2 and abs(t["q05"][1]) < 2
        lo, hi = p["mean"][:, 1].min(), p["mean"][:, 1].max()
        assert np.all((d["mean"][:, 1] > lo) & (d["mean"][:, 1] < hi))    # every Dynare chain inside pydyn's range
    # the batch statistics are those of the Octave script (checked on Dynare's own chains in a recorded run too)
    x = np.random.default_rng(0).standard_normal((4000, 2)).cumsum(axis=0)
    b = cmp.batch_stats(x)
    np.testing.assert_allclose(b[3, :, 0], x[2300:2400].mean(axis=0))
    assert b.shape == (20, 2, 5) and np.all(b[:, :, 4] == 1)
    np.testing.assert_allclose(cmp.chain_stats(b)["mean"][0], x[2000:].mean(axis=0))
