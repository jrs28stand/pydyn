"""Estimation (data handling, priors, Kalman-filter likelihood, posterior mode, Laplace approximation,
Metropolis-Hastings, smoother) against Dynare 7.2 under Octave 8.4; see reference/r_estimation.m.

Models: est_schorfheide (Dynare's examples/estimation/schorfheide_2000.mod: loglinear, logdata,
steady_state_model, an auxiliary variable), est_nk (linear New Keynesian model: measurement
errors, shock correlation, all prior families, a parameter overwritten by the steady state) and
est_ur (unit root: diffuse filter, lik_init=2, maximum likelihood, observation trends).
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation, kalman
from pydyn.priors import Prior, PriorError

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))

# pydyn.Model warns about statements it does not interpret itself; estimation.py reads them
pytestmark = pytest.mark.filterwarnings("ignore:statements not handled")

_models = {}


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("est")
    for f in REF.glob("est_*"):
        if f.suffix in (".mod", ".m", ".csv"):
            shutil.copy(f, d)
    return d


@pytest.fixture(scope="module")
def load(workdir):
    """load(mod, variant) -> Model, preprocessed once per (mod, variant)."""
    def _load(mod, variant=0):
        key = (mod, variant)
        if key not in _models:
            sub = workdir / f"{mod}_{variant}"
            sub.mkdir()
            for f in workdir.iterdir():
                if f.is_file():
                    shutil.copy(f, sub)
            args = (f"-D{variant}",) if isinstance(variant, str) else (f"-DVARIANT={variant}",)
            _models[key] = pydyn.Model(pydyn.run_preprocessor(sub / f"{mod}.mod", extra_args=args))
        return _models[key]
    return _load


def ref(name):
    return loadmat(REF / name, squeeze_me=True)


# ------------------------------------------------------------------ data files
def test_m_file_reader():
    text = """
    % a comment
    a = [1 2 3
         4 5 6];            # another comment
    b = a(:,2)';  c = a(2:end, 1) + a(1,3)*2;
    d = diff(log(a(:,3)));
    e = [1 -2, 3 - 4; a(1,:)] ...
        * 2;
    f = 1e2*[0.5; NaN; -.25]./a(1:3)';
    g = a(end, end) - 2^3 + 4/8;
    """
    v = estimation._MFile(text).run()
    np.testing.assert_array_equal(v["a"], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(v["b"], [[2, 5]])
    np.testing.assert_array_equal(v["c"], [[10]])
    np.testing.assert_allclose(v["d"], [[np.log(6) - np.log(3)]])
    np.testing.assert_array_equal(v["e"], [[2, -4, -2], [2, 4, 6]])
    np.testing.assert_array_equal(v["f"], [[50.0], [np.nan], [-12.5]])      # a(1:3) is [1 4 2] (column-major)
    assert v["g"].item() == 6 - 8 + 0.5
    with pytest.raises(estimation.EstimationError):
        estimation._MFile("x = load('file.txt');").run()
    with pytest.raises(estimation.EstimationError):
        estimation._MFile("for i=1:3, x = i; end").run()


def test_data_sources_agree(load, workdir):
    """.m file, CSV file, dict and array give the same data and the same likelihood."""
    m = load("est_nk")
    est = estimation.Estimation(m)
    r = ref("est_nk_v0.mat")
    np.testing.assert_array_equal(est.data, r["data"])
    table = estimation.read_m_data(workdir / "est_nk_data.m")
    alt = [estimation.Estimation(m, data=str(workdir / "est_nk_csv.csv")),
           estimation.Estimation(m, data=table),
           estimation.Estimation(m, data=np.column_stack([table[v] for v in m.varobs]))]
    ll = est.log_likelihood(est.x0)
    for e in alt:
        np.testing.assert_allclose(e.data, est.data, rtol=0, atol=1e-12)
        assert abs(e.log_likelihood(e.x0) - ll) < 1e-9
    with pytest.raises(estimation.EstimationError):
        estimation.Estimation(m, data={"yobs": table["yobs"]})
    with pytest.raises(estimation.EstimationError):
        estimation.Estimation(m, first_obs=100, nobs=80)


def test_schorfheide_data_transformations(load):
    """The data file computes growth rates with diff(log(.)); logdata stops a second log."""
    r = ref("est_schorfheide_v0.mat")
    est = estimation.Estimation(load("est_schorfheide"))
    assert est.data.shape == (192, 2)
    np.testing.assert_array_equal(est.data, r["data"])
    unlogged = estimation.Estimation(load("est_schorfheide"), data=np.exp(est.rawdata), logdata=False)
    np.testing.assert_allclose(unlogged.data, r["data"], rtol=0, atol=1e-15)


# ------------------------------------------------------------------ likelihood and prior
CASES = [("est_schorfheide", 0), ("est_nk", 0), ("est_nk", 1), ("est_nk", 2), ("est_nk", 3), ("est_nk", 4),
         ("est_nk", 8), ("est_nk", 9), ("est_ur", 0), ("est_ur", 1), ("est_ur", 2), ("est_ur", 5), ("est_ur", 6)]
# what each case exercises (for the reader; the options come from the .mod files)
CASE_NOTES = {
    ("est_schorfheide", 0): "loglinear, logdata, stationary initialisation",
    ("est_nk", 0): "measurement error, estimated shock correlation, steady state updates a parameter",
    ("est_nk", 1): "first_obs=11, nobs=80, presample=4",
    ("est_nk", 2): "missing observations",
    ("est_nk", 3): "prefilter=1",
    ("est_nk", 4): "lik_init=2",
    ("est_nk", 8): "kalman_algo=2 (univariate filter), missing observations",
    ("est_nk", 9): "two measurement errors with an estimated correlation, missing observations",
    ("est_ur", 0): "diffuse_filter; Dynare falls back to the univariate diffuse stage",
    ("est_ur", 1): "lik_init=2 with a unit root",
    ("est_ur", 2): "diffuse_filter, missing observations, presample=3",
    ("est_ur", 5): "diffuse_filter, observation_trends with an estimated slope, first_obs=5",
    ("est_ur", 6): "kalman_algo=4 (univariate diffuse filter), missing observations, presample=2",
}


@pytest.mark.parametrize("mod,variant", CASES)
def test_likelihood_and_prior_match_dynare(load, mod, variant):
    r = ref(f"est_{mod[4:]}_v{variant}.mat")
    m = load(mod, variant)
    params_before, sigma_before = m.params.copy(), m.Sigma_e.copy()
    est = estimation.Estimation(m)
    # setup: ordering, initial values, bounds, hyper-parameters, data
    assert est.names == r["names"].split(";")
    np.testing.assert_allclose(est.x0, r["x0"], rtol=1e-14)
    np.testing.assert_array_equal(est.lb, r["lb"])
    np.testing.assert_array_equal(est.ub, r["ub"])
    np.testing.assert_array_equal(est.prior.shape, r["pshape"])
    for key in ("p6", "p7"):
        np.testing.assert_allclose(getattr(est.prior, key), r[key], rtol=1e-12)
    np.testing.assert_array_equal(est.data, np.atleast_2d(r["data"]))
    assert est.options["lik_init"] == int(r["lik_init"])
    assert est.options["qz_criterium"] == float(r["qz_criterium"])
    # values
    X = r["X"]
    n_finite = 0
    for k in range(X.shape[1]):
        x = X[:, k]
        post = est.log_posterior(x)
        if np.isfinite(r["fval"][k]):
            n_finite += 1
            np.testing.assert_allclose(est.log_likelihood(x), r["loglik"][k], rtol=1e-10)
            np.testing.assert_allclose(est.log_prior(x), r["lnprior"][k], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(post, -r["fval"][k], rtol=1e-10)
            contrib = est.log_likelihood_contributions(x)
            assert contrib.shape == (est.nobs - est.options["presample"],)
            np.testing.assert_allclose(contrib.sum(), r["loglik"][k], rtol=1e-10)
        else:
            assert post == -np.inf, (k, int(r["info"][k]))
            info = int(r["info"][k])
            if info == 41:
                assert "lower bound" in est.last_info
            elif info == 42:
                assert "upper bound" in est.last_info
    assert n_finite >= 4
    # the model object is left untouched
    np.testing.assert_array_equal(m.params, params_before)
    np.testing.assert_array_equal(m.Sigma_e, sigma_before)


def test_rejections_match_dynare(load):
    """Blanchard-Kahn failure (Dynare info 3), a steady state that cannot be computed (info 22)
    and bound violations give a log posterior of -inf; the log prior is still Dynare's."""
    r = ref("est_schorfheide_v0.mat")
    est = estimation.Estimation(load("est_schorfheide"))
    infos = [int(i) for i in r["info"]]
    k = infos.index(3)                                   # rho = 0.999999: root above 1 - 1e-6
    assert est.log_likelihood(r["X"][:, k]) == -np.inf
    assert "Blanchard-Kahn" in est.last_info
    np.testing.assert_allclose(est.log_prior(r["X"][:, k]), r["lnprior"][k], rtol=1e-10)
    assert est.log_posterior(r["X"][:, k]) == -np.inf
    assert sorted(set(infos)) == [0, 3, 41, 42]

    r = ref("est_nk_v0.mat")
    est = estimation.Estimation(load("est_nk"))
    infos = [int(i) for i in r["info"]]
    k = infos.index(22)                                  # sig = 0: 1/sig in the model
    assert est.log_posterior(r["X"][:, k]) == -np.inf
    assert "steady state" in est.last_info
    assert all(est.log_posterior(r["X"][:, j]) == -np.inf for j, i in enumerate(infos) if i != 0)


def test_steady_state_parameter_update_is_used(load):
    """est_nk: the steady_state_model block overwrites the parameter rbar = 100*(1/beta - 1), and
    beta is estimated. The constant of the robs measurement equation depends on rbar."""
    m = load("est_nk")
    r = ref("est_nk_v0.mat")
    est = estimation.Estimation(m)
    ib, irbar = est.names.index("beta"), m.param_names.index("rbar")
    assert m.params[irbar] == 0.0                        # calibrated value in the .mod file
    finite = [k for k in range(r["X"].shape[1]) if np.isfinite(r["fval"][k])]
    betas = {round(float(r["X"][ib, k]), 12) for k in finite}
    assert len(betas) >= 3                               # the reference points differ in beta
    for k in finite:
        x = r["X"][:, k]
        ss = est.state_space(x)
        rbar = 100 * (1 / x[ib] - 1)
        np.testing.assert_allclose(ss["params"][irbar], rbar, rtol=1e-14)
        np.testing.assert_allclose(ss["constant"][m.varobs.index("robs")], rbar + x[est.names.index("pibar")], rtol=1e-13)
        np.testing.assert_allclose(est.log_likelihood(x), r["loglik"][k], rtol=1e-10)
    assert m.params[irbar] == 0.0                        # and the model is not modified
    # with the stale value of rbar the likelihood would be very different
    x = r["X"][:, finite[0]]
    ss = est.state_space(x)
    stale = est.data.T - (ss["constant"] - np.array([0, 0, 100 * (1 / x[ib] - 1)]))[:, None]
    QQ = ss["R"] @ ss["Q"] @ ss["R"].T
    P = kalman.lyapunov_symm(ss["T"], QQ)
    bad = -kalman.kalman_filter(stale, np.zeros(len(P)), P, ss["T"], QQ, ss["H"], ss["z"])
    assert abs(bad - r["loglik"][finite[0]]) > 10


def test_covariances_from_theta(load):
    """stderr of shocks and measurement errors, and corr, map into Q and H as set_all_parameters.m."""
    est = estimation.Estimation(load("est_nk"))
    x = est.x0.copy()
    d = dict(zip(est.names, x))
    Q, H = est.covariances(x)
    assert est.names[:5] == ["stderr ea", "stderr eu", "stderr er", "stderr yobs", "corr ea, eu"]
    np.testing.assert_allclose(np.sqrt(np.diag(Q)), [d["stderr ea"], d["stderr eu"], d["stderr er"]])
    np.testing.assert_allclose(Q[0, 1], d["corr ea, eu"] * d["stderr ea"] * d["stderr eu"])
    assert Q[0, 2] == 0 and Q[1, 2] == 0
    np.testing.assert_allclose(H, np.diag([d["stderr yobs"] ** 2, 0, 0]))


def test_kalman_algo_choices_agree(load):
    """Multivariate and univariate filters compute the same likelihood. With Dynare's default
    riccati_tol=1e-6 both switch to a steady-state gain at slightly different points, which moves
    the value by about 1e-8; with riccati_tol=0 they agree to rounding."""
    m = load("est_nk")
    a, b = estimation.Estimation(m, riccati_tol=0), estimation.Estimation(m, kalman_algo=2, riccati_tol=0)
    assert abs(a.log_likelihood(a.x0) - b.log_likelihood(b.x0)) < 1e-9
    assert (a.last_filter, b.last_filter) == ("multivariate", "univariate")
    default = estimation.Estimation(m)
    assert 0 < abs(default.log_likelihood(a.x0) - a.log_likelihood(a.x0)) < 1e-5
    m = load("est_ur", 2)
    a, b = estimation.Estimation(m, riccati_tol=0), estimation.Estimation(m, kalman_algo=4, riccati_tol=0)
    assert abs(a.log_likelihood(a.x0) - b.log_likelihood(b.x0)) < 1e-9
    assert (a.last_filter, b.last_filter) == ("multivariate (diffuse)", "univariate (diffuse)")


# ------------------------------------------------------------------ priors
def test_prior_families_match_dynare():
    """Hyper-parameters (set_prior.m and helpers) and log densities (priordens.m) on a grid, for
    beta, gamma, normal, inverse gamma 1 and 2, uniform and Weibull, with and without p3/p4."""
    q = ref("est_priors.mat")
    shapes_seen = set()
    for i, row in enumerate(q["spec"]):
        pr = Prior([int(row[0])], [row[1]], [row[2]], [row[3]], [row[4]])
        shapes_seen.add(int(row[0]))
        np.testing.assert_allclose([pr.p6[0], pr.p7[0]], [q["p6"][i], q["p7"][i]], rtol=1e-12)
        assert (pr.p3[0], pr.p4[0]) == (q["p3"][i], q["p4"][i])
        lp = np.array([pr.logpdf([x]) for x in q["grid"]])
        expected = q["lpdf"][i]
        finite = np.isfinite(expected)
        assert finite.sum() >= 20
        np.testing.assert_array_equal(lp[~finite], expected[~finite])       # -inf outside the support
        np.testing.assert_allclose(lp[finite], expected[finite], rtol=1e-11, atol=1e-11)
    assert shapes_seen == {1, 2, 3, 4, 5, 6, 8}


def test_prior_errors_and_bounds():
    with pytest.raises(PriorError):
        Prior([1], [0.5], [0.6])                         # beta: std too large for the mean
    with pytest.raises(PriorError):
        Prior([4], [-1.0], [1.0])                        # inverse gamma: mean below the lower bound
    with pytest.raises(PriorError):
        Prior([5], [0.5], [0.1], [0.0], [1.0])           # uniform: bounds and moments together
    pr = Prior([1, 2, 3, 4, 5, 6, 8], [0.5, 1, 0, 1, np.nan, 1, 1], [0.1, 0.5, 1, 1, np.nan, 1, 0.5],
               [np.nan] * 4 + [-1] + [np.nan] * 2, [np.nan] * 4 + [2] + [np.nan] * 2)
    lb, ub = pr.bounds()
    np.testing.assert_array_equal(lb, [0, 0, -np.inf, 0, -1, 0, 0])
    np.testing.assert_array_equal(ub, [1, np.inf, np.inf, np.inf, 2, np.inf, np.inf])
    lb, ub = pr.bounds(1e-3)                             # prior_trunc: quantiles
    assert np.all(lb > [0, 0, -4, 0, -1, 0, 0]) and np.all(np.isfinite(ub)) and np.all(lb < ub)
    np.testing.assert_allclose([lb[2], ub[2]], [-3.090232306167813, 3.090232306167813], rtol=1e-10)
    np.testing.assert_allclose([lb[4], ub[4]], [-1 + 3e-3, 2 - 3e-3], rtol=1e-12)


# ------------------------------------------------------------------ Hessian, Laplace, mode
def test_hessian_and_laplace_at_dynare_mode(load):
    """At the mode Dynare found for the Schorfheide example (shipped settings, mode_compute=4)."""
    r = ref("est_schorfheide_full.mat")
    est = estimation.Estimation(load("est_schorfheide"))
    assert est.names == r["names"].split(";")
    x = r["xparam1"]
    assert abs(r["logpost"] - 1343.6795) < 1e-4 and abs(r["laplace"] - 1299.02) < 1e-2
    np.testing.assert_allclose(est.log_posterior(x), r["logpost"], rtol=0, atol=1e-8)
    hh = est.hessian(x)
    # finite differences of the same function with the same steps: rounding noise only
    np.testing.assert_allclose(hh, r["hh"], rtol=1e-4)
    inv = est.inverse_hessian(hh)
    np.testing.assert_allclose(inv, r["Vmode"], rtol=1e-4)
    np.testing.assert_allclose(est.laplace(x, inv), r["laplace"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(est.laplace(x, r["Vmode"]), r["laplace"], rtol=0, atol=1e-8)


def test_log_posterior_at_dynare_mcmc_draws(load):
    """40 draws of Dynare's own Metropolis chains (shipped settings, 2 x 20000) with the log
    posterior Dynare stored for them: the target density is the same across the posterior."""
    d = ref("est_schorfheide_draws.mat")
    est = estimation.Estimation(load("est_schorfheide"))
    lp = np.array([est.log_posterior(x) for x in d["X"]])
    assert d["X"].shape == (40, 9) and np.ptp(d["LP"]) > 5
    np.testing.assert_allclose(lp, d["LP"], rtol=0, atol=1e-6)


def test_posterior_mode_schorfheide(load):
    """Mode search from the initial values (prior means), as Dynare's run starts."""
    r = ref("est_schorfheide_full.mat")
    est = estimation.Estimation(load("est_schorfheide"))
    np.testing.assert_allclose(est.x0, [0.035449, 0.008862, 0.356, 0.993, 0.0085, 0.0002, 0.129, 0.65, 0.01])
    assert (est.options["mh_nblck"], est.options["mh_jscale"], est.options["loglinear"]) == (2, 0.8, True)
    np.testing.assert_array_equal(est.jscale()[0], 0.8)
    mode = est.mode()
    # the optimum is flat: Dynare's csminwel stops 6.5e-6 below the value found here
    assert -1e-7 < mode.log_posterior - r["logpost"] < 1e-4
    dyn_std = np.sqrt(np.diag(r["Vmode"]))
    assert np.max(np.abs(mode.x - r["xparam1"]) / dyn_std) < 0.01      # within 1% of a posterior std
    np.testing.assert_allclose(mode.x, r["xparam1"], rtol=2e-3)
    np.testing.assert_allclose(mode.std, dyn_std, rtol=5e-3)
    assert abs(mode.laplace - r["laplace"]) < 0.01


@pytest.mark.parametrize("variant,bayesian", [(3, True), (4, False)])
def test_mode_unit_root_model(load, variant, bayesian):
    """Posterior mode (variant 3) and maximum likelihood (variant 4) with the diffuse filter."""
    r = ref(f"est_ur_mode_v{variant}.mat")
    est = estimation.Estimation(load("est_ur", variant))
    assert est.prior.is_bayesian == bayesian
    np.testing.assert_allclose(est.log_posterior(r["xparam1"]), r["logpost"], rtol=1e-10)
    mode = est.mode()
    assert abs(mode.log_posterior - r["logpost"]) < 1e-6
    dyn_std = np.sqrt(np.diag(r["Vmode"]))
    assert np.max(np.abs(mode.x - r["xparam1"]) / dyn_std) < 0.01
    np.testing.assert_allclose(mode.std, dyn_std, rtol=1e-3)
    if bayesian:
        assert abs(mode.laplace - r["laplace"]) < 1e-3
    else:
        assert np.isnan(mode.laplace) and est.log_prior(mode.x) == 0.0
        with pytest.raises(estimation.EstimationError):
            est.sample(replic=10, mode=mode)


def test_hessian_function():
    A = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, -0.2], [0.0, -0.2, 0.5]])
    x0 = np.array([0.3, -1.2, 2.0])
    hh = estimation.hessian(lambda x: 0.5 * x @ A @ x + np.sum(x), x0)
    np.testing.assert_allclose(hh, A, rtol=1e-6, atol=1e-8)


# ------------------------------------------------------------------ Metropolis-Hastings
def test_metropolis_accept_reject_path():
    """Fixed innovations and uniforms: the chain equals an independent implementation of Dynare's
    rule (posterior_sampler_iteration.m), including proposals outside the bounds."""
    rng = np.random.default_rng(5)
    n, n_draws = 3, 400
    S = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, -0.3], [0.0, -0.3, 0.5]])
    Sinv = np.linalg.inv(S)
    calls = []

    def log_post(x):
        calls.append(x.copy())
        if x[2] > 1.2:                                   # region where the model "cannot be solved"
            return -np.inf
        return -0.5 * x @ Sinv @ x

    lb, ub = np.array([-1.0, -np.inf, -np.inf]), np.array([np.inf, 2.5, np.inf])
    d = np.linalg.cholesky(S).T * np.array([0.9, 1.1, 1.0])      # d * diag(jscale)
    eps, u = rng.standard_normal((n_draws, n)), rng.random(n_draws)
    x0 = np.zeros(n)
    draws, logpo, acc = estimation.metropolis_hastings(log_post, x0, log_post(x0), d, lb, ub, eps, u)
    n_calls = len(calls) - 1

    # independent check, written out step by step
    def target(x):
        return -np.inf if x[2] > 1.2 else float(-0.5 * x @ Sinv @ x)

    last, last_lp, n_inside, n_out, n_unsolved = x0.copy(), target(x0), 0, 0, 0
    for i in range(n_draws):
        prop = last + d.T @ eps[i]                       # row vector times upper factor
        inside = bool(np.all(prop > lb) and np.all(prop < ub))
        lp = target(prop) if inside else -np.inf
        n_inside += inside
        n_out += not inside
        n_unsolved += inside and lp == -np.inf
        accept = lp > -np.inf and np.log(u[i]) < lp - last_lp
        if accept:
            last, last_lp = prop, lp
        assert bool(acc[i]) == bool(accept), i
        np.testing.assert_allclose(draws[i], last, rtol=0, atol=1e-13)
        np.testing.assert_allclose(logpo[i], last_lp, rtol=0, atol=1e-12)
    assert n_calls == n_inside                           # proposals outside the bounds are not evaluated
    assert n_out > 10 and n_unsolved > 5 and 0.1 < acc.mean() < 0.7
    assert np.all(draws > lb) and np.all(draws < ub) and np.all(draws[:, 2] <= 1.2)


def test_sampler_on_model_with_fixed_random_numbers(load):
    """est.sample with given innovations: proposal covariance jscale^2 * inverse Hessian and the
    accept/reject path, checked against a loop written directly on log_posterior."""
    r = ref("est_ur_mode_v3.mat")
    est = estimation.Estimation(load("est_ur", 3))
    V, xm = r["Vmode"], r["xparam1"]
    rng = np.random.default_rng(11)
    replic, js = 60, 0.7
    eps, u = rng.standard_normal((2, replic, est.n)), rng.random((2, replic))
    start = np.vstack([xm, xm * 1.02])
    mc = est.sample(replic=replic, nblocks=2, mode=xm, cov=V, jscale=js, x0=start, innovations=eps, uniforms=u)
    assert mc.draws.shape == (2, replic, est.n) and mc.names == est.names
    L = np.linalg.cholesky(V)
    for j in range(2):
        last, last_lp = start[j], est.log_posterior(start[j])
        for i in range(replic):
            prop = last + js * (L @ eps[j, i])
            lp = est.log_posterior(prop)
            if np.isfinite(lp) and np.log(u[j, i]) < lp - last_lp:
                last, last_lp = prop, lp
            np.testing.assert_allclose(mc.draws[j, i], last, rtol=0, atol=1e-12)
            np.testing.assert_allclose(mc.log_posterior[j, i], last_lp, rtol=0, atol=1e-9)
    np.testing.assert_allclose(mc.acceptance_rate, mc.accepted.mean(axis=1))
    # proposals really have covariance jscale^2 * V: implied by the identity above for every draw
    np.testing.assert_allclose(js ** 2 * (L @ L.T), js ** 2 * V, rtol=1e-12)


def test_sampler_rejects_proposals_outside_estimated_params_bounds(load):
    """Dynare's sampler evaluates the posterior with the prior bounds only (set_mcmc_prior_bounds.m), but
    posterior_sampler_iteration.m checks the proposal against sampler_options.bounds, the bounds of the mode search
    (prior support intersected with the estimated_params bounds): a proposal outside the estimated_params bounds is
    rejected without being evaluated (Dynare's own chain under tight bounds is replayed in
    tests/test_samplers.py::test_rwmh_estimated_params_bounds_match_dynare). est_nk: phipi has a normal prior and the
    bounds [1.01, 4]."""
    est = estimation.Estimation(load("est_nk"))
    i = est.names.index("phipi")
    assert (est.lb[i], est.ub[i]) == (1.01, 4.0)
    plb, pub = est.prior.bounds()
    assert (plb[i], pub[i]) == (-np.inf, np.inf)
    x = est.x0.copy()
    x[i] = 1.012
    below = x.copy()
    below[i] = 1.007
    assert est.log_posterior(below) == -np.inf and "lower bound" in est.last_info
    lp = est.log_posterior(below, bounds=(plb, pub))
    assert np.isfinite(lp) and lp == est.log_likelihood(below) + est.log_prior(below)
    # a proposal that moves phipi from 1.012 to 1.007 is rejected, one to 1.017 accepted
    V = np.diag((0.01 * np.abs(est.x0)) ** 2)
    for target, moved in ((1.007, False), (1.017, True)):
        y = x.copy()
        y[i] = target
        eps = np.linalg.solve(np.linalg.cholesky(V), y - x)[None, None, :]
        mc = est.sample(replic=1, nblocks=1, mode=x, cov=V, jscale=1.0, x0=x[None, :], innovations=eps,
                        uniforms=np.array([[1e-300]]))
        assert bool(mc.accepted[0, 0]) == moved
        assert abs(mc.draws[0, 0, i] - (target if moved else 1.012)) < 1e-12


def test_sampler_smoke(load):
    r = ref("est_ur_mode_v3.mat")
    est = estimation.Estimation(load("est_ur", 3))
    kw = dict(replic=150, nblocks=2, mode=r["xparam1"], cov=r["Vmode"])
    mc = est.sample(seed=3, **kw)
    again = est.sample(seed=3, **kw)
    other = est.sample(seed=4, **kw)
    np.testing.assert_array_equal(mc.draws, again.draws)             # reproducible
    assert not np.array_equal(mc.draws, other.draws)
    assert not np.array_equal(mc.draws[0], mc.draws[1])              # chains have their own streams
    assert not np.array_equal(mc.x0[0], mc.x0[1])                    # over-dispersed starting points
    assert np.all((mc.acceptance_rate > 0.1) & (mc.acceptance_rate < 0.6))   # default jscale 2.38/sqrt(n)
    lb, ub = est.prior.bounds()
    assert np.all(mc.draws > lb) and np.all(mc.draws < ub) and np.all(np.isfinite(mc.log_posterior))
    assert mc.kept().shape == (150, est.n) and mc.kept(0.0).shape == (300, est.n)
    std = np.sqrt(np.diag(r["Vmode"]))
    assert np.all(np.abs(mc.mean() - r["xparam1"]) < 2 * std)        # loose: 150 kept draws
    hpd = mc.hpd_interval(0.9)
    assert np.all(hpd[:, 0] < mc.mean()) and np.all(mc.mean() < hpd[:, 1])
    one = est.sample(replic=5, nblocks=1, mode=r["xparam1"], cov=r["Vmode"])
    np.testing.assert_array_equal(one.x0[0], r["xparam1"])           # a single chain starts at the mode


# ------------------------------------------------------------------ smoother
@pytest.mark.parametrize("mod,variant,name", [("est_nk", 5, "est_nk_smoother.mat"),
                                              ("est_nk", 7, "est_nk_smoother_miss.mat"),
                                              ("est_schorfheide", "SMOOTHER", "est_schorfheide_smoother.mat")])
def test_smoother_matches_dynare(load, mod, variant, name):
    """Smoothed and updated variables, smoothed shocks and measurement errors at the initial
    parameter values: complete data, missing observations, loglinear model."""
    s = ref(name)
    est = estimation.Estimation(load(mod, variant))
    res = est.smoother(est.x0)
    for i, v in enumerate(s["endo_names"].split(";")):
        np.testing.assert_allclose(res["variables"][v], s["smoothed"][:, i], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(res["updated"][v], s["updated"][:, i], rtol=1e-9, atol=1e-10)
    for i, v in enumerate(s["exo_names"].split(";")):
        np.testing.assert_allclose(res["shocks"][v], s["shocks"][:, i], rtol=1e-9, atol=1e-10)
        assert np.max(np.abs(s["shocks"][:, i])) > 1e-3
    if "me_yobs" in s:
        np.testing.assert_allclose(res["measurement_errors"]["yobs"], s["me_yobs"], rtol=1e-9, atol=1e-10)
        assert np.nanmax(np.abs(s["me_yobs"])) > 1e-2
        assert np.isnan(s["me_yobs"]).sum() == np.isnan(est.data[:, 0]).sum()   # NaN where yobs is missing


# ------------------------------------------------------------------ building blocks
def test_lyapunov_and_option_checks(load):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    A *= 0.9 / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.standard_normal((6, 6))
    B = B @ B.T
    X = kalman.lyapunov_symm(A, B)
    np.testing.assert_allclose(X, A @ X @ A.T + B, rtol=1e-10, atol=1e-10)
    m = load("est_ur")
    with pytest.raises(estimation.EstimationError):
        estimation.Estimation(m, diffuse_filter=True, lik_init=2)
    with pytest.raises(NotImplementedError):
        estimation.Estimation(m, order=2)
    with pytest.raises(estimation.EstimationError, match="prefilter"):
        estimation.Estimation(load("est_nk"), prefilter=1)      # non-zero steady state of piobs, robs
    assert estimation.Estimation(load("est_nk", 3)).check_prefilter()
    assert estimation.Estimation(m, noconstant=True).ignored_options == ["nodisplay", "noconstant"]
    # the unit root is rejected under the stationary initialisation, as in Dynare (qz_criterium = 1 - 1e-6)
    est = estimation.Estimation(m, diffuse_filter=False, lik_init=1)
    assert est.log_posterior(est.x0) == -np.inf and "Blanchard-Kahn" in est.last_info


def test_estimated_params_init_and_unsupported_blocks(workdir):
    """estimated_params_init sets starting values; blocks that cannot be honoured raise.
    (Python-side behaviour only: no Dynare reference for this test.)"""
    base = (workdir / "est_ur.mod").read_text()
    d = workdir / "init"
    d.mkdir()
    shutil.copy(workdir / "est_ur_data.m", d)

    def model(extra):
        (d / "est_init.mod").write_text(base.replace("varobs yobs xobs;", extra + "\nvarobs yobs xobs;", 1))
        return pydyn.Model(pydyn.run_preprocessor(d / "est_init.mod"))

    est = estimation.Estimation(model("estimated_params_init;\nrho, 0.55;\nstderr eg, 0.33;\nend;"))
    x0 = dict(zip(est.names, est.x0))
    assert (x0["rho"], x0["stderr eg"], x0["phi"], x0["lam"]) == (0.55, 0.33, 0.4, 0.2)
    with pytest.raises(NotImplementedError, match="several estimated_params"):
        estimation.Estimation(model("estimated_params;\nrho, beta_pdf, 0.5, 0.2;\nend;"))
    with pytest.raises(NotImplementedError, match="estimated_params_bounds"):
        estimation.Estimation(model("estimated_params_bounds;\nlam, 0, 1;\nend;"))


# ====================================================================== optimisers (mode_compute), see reference/r_posterior.m
@pytest.fixture(scope="module")
def post(tmp_path_factory):
    """post(mod, n) -> Estimation for reference/<mod>.mod preprocessed with -DPOST=n."""
    work = tmp_path_factory.mktemp("post_est")
    made = {}

    def _post(mod, n=0, **kw):
        if (mod, n) not in made:
            sub = work / f"{mod}_{n}"
            sub.mkdir()
            for f in list(REF.glob("est_*_data.m")) + [REF / f"{mod}.mod", REF / f"{mod}_mode0.mat"]:
                shutil.copy(f, sub)
            made[(mod, n)] = pydyn.Model(pydyn.run_preprocessor(sub / f"{mod}.mod", extra_args=(f"-DPOST={n}",)))
        return estimation.Estimation(made[(mod, n)], **kw)
    return _post


def _rosenbrock(x):
    """The objective of reference/r_posterior.m (post_testfun), with the same floating-point operations."""
    k = np.flatnonzero(x > 1.5)
    if k.size:
        s = 0.0
        for j in k:
            s = s + (x[j] - 1.5) * (x[j] - 1.5)
        return np.inf, 42, s
    f = 0.0
    for i in range(x.size - 1):
        a = x[i + 1] - x[i] * x[i]
        b = 1 - x[i]
        f = f + 100 * (a * a) + b * b
    return f, 0, 0.0


def test_csminwel_on_test_function_matches_dynare():
    """csminwel1.m and the port on a function both evaluate with identical arithmetic. The first evaluations are
    bit-identical; later ones drift apart because Octave and numpy round their matrix products differently and the
    search amplifies that (measured: 1e-13 after 27 evaluations, 1e-9 after 75, 1e-7 after 140). Iteration and
    evaluation counts stay equal where the run is short or the gradient accurate."""
    t = ref("post_csminwel_fun.mat")
    counts = {}
    for v in range(1, t["settings"].shape[0] + 1):
        method, eps, nit, crit = t["settings"][v - 1]
        p = f"s{v}_"
        X, F = [], []

        def fobj(x):
            out = _rosenbrock(x)
            X.append(np.array(x))
            F.append(out[0])
            return out
        res = estimation.csminwel(fobj, t["x0"], method=int(method), epsilon=eps, nit=int(nit), crit=crit)
        Fd = t[p + "F"]
        np.testing.assert_array_equal(F[:20], Fd[:20])                       # bit for bit
        np.testing.assert_allclose(F[:60], Fd[:60], rtol=1e-6)
        counts[v] = (res["iterations"], res["fcount"], res["retcode"])
        assert res["fcount"] == len(F)
        assert abs(res["f"] - t[p + "fh"]) < 1e-6 and res["retcode"] == t[p + "retcode"]
        np.testing.assert_allclose(res["x"], t[p + "xh"], atol=5e-4)
        if v in (2, 3, 6):                                                   # same path to the end
            assert counts[v][:2] == (t[p + "itct"], t[p + "fcount"])
            idx = t[p + "Xidx"].astype(int) - 1
            np.testing.assert_allclose(np.array(X)[idx].T, t[p + "X"], atol=2e-2 if v < 6 else 1e-5)
            np.testing.assert_allclose(res["x"], t[p + "xh"], atol=1e-6)
    assert counts[6] == (13, 102, 8)                                         # iteration count termination
    with pytest.raises(ValueError, match="gradient"):
        estimation.csminwel(_rosenbrock, t["x0"], method=4)
    bad = estimation.csminwel(_rosenbrock, np.full(5, 2.0))
    assert bad["retcode"] == -1 and bad["x"] is None
    flat = estimation.csminwel(lambda x: (1.0, 0, 0.0), np.zeros(3))         # zero gradient
    assert flat["retcode"] == 1 and flat["iterations"] == 1


@pytest.mark.parametrize("mod", ["post_nk", "post_schorfheide"])
def test_csminwel_follows_dynare_on_the_posterior(post, mod):
    """mode_compute=4 from the same starting point as Dynare, with every evaluation of Dynare's run recorded.
    Measured: the evaluation points agree to 1e-9 for the first 90 (nk) and to 1e-8 for the first 146 (Schorfheide)
    evaluations, then drift (rounding differences of 1e-15 to 1e-12 in the likelihood divided by the gradient step
    1e-6); the optima differ by 3e-8 in the objective (convergence criterion 1e-7) and 1e-4 posterior standard
    deviations in the parameters."""
    est = post(mod, 1)
    r = ref(f"{mod}_mode.mat")
    assert est.options["mode_compute"] == 4 and est.names == r["names"].split(";")
    np.testing.assert_array_equal(est.x0, r["x0"])
    # penalties outside the domain: Dynare's info(1) and info(4)
    for j in range(r["pen_X"].shape[1]):
        fval, code, pen = est.objective(r["pen_X"][:, j])
        assert code == r["pen_info"][0, j]
        np.testing.assert_allclose(pen, r["pen_info"][1, j], rtol=1e-12, atol=1e-18)
        assert np.isfinite(fval) == np.isfinite(r["pen_f"][j])
    X, F = [], []
    original = est.objective

    def logged(x, bounds=None):
        out = original(x, bounds)
        X.append(np.array(x))
        F.append(out[0])
        return out
    est.objective = logged
    cliff = iter(np.atleast_2d(r["c2_cliff_rand"].T))
    mode = est.mode(rand=lambda n: next(cliff))                 # mode_compute=4 from the .mod file
    del est.objective
    res = est.last_csminwel
    idx = r["c2_Xidx"].astype(int) - 1
    early = idx[idx < 120]
    np.testing.assert_allclose(np.array(X)[early].T, r["c2_X"][:, :early.size], rtol=0, atol=1e-7)
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(F[:120], r["c2_F"][:120], rtol=1e-6)      # steep: looser than the points
    assert abs(res["iterations"] - r["c2_itct"]) <= 3 and abs(res["fcount"] - r["c2_fcount"]) < 80
    assert abs(res["f"] - r["c2_fh"]) < 2e-7
    sd = np.sqrt(np.diag(r["Vmode"]))
    assert np.max(np.abs(mode.x - r["xparam1"]) / sd) < 2e-3
    np.testing.assert_array_equal(r["c2_xh"], r["xparam1"])     # Dynare's own run is the recorded one
    # Hessian, covariance and Laplace approximation at Dynare's mode
    at = estimation.ModeResult(x=r["xparam1"], names=est.names, log_posterior=est.log_posterior(r["xparam1"]))
    est.add_hessian(at)
    np.testing.assert_allclose(at.log_posterior, r["logpost"], rtol=1e-11)
    np.testing.assert_allclose(at.inv_hessian, r["Vmode"], rtol=2e-4, atol=2e-5 * np.abs(r["Vmode"]).max())
    np.testing.assert_allclose(at.laplace, r["laplace"], atol=2e-4)
    np.testing.assert_allclose(mode.laplace, r["laplace"], atol=5e-3)
    # the other gradient formulas and option values: a few iterations each, same path as Dynare
    for v in range(1, 6):
        method, eps, nit, crit = r["variants"][v - 1]
        X.clear()
        est.objective = logged
        m2 = est.mode(optim={"NumgradAlgorithm": int(method), "NumgradEpsilon": eps, "MaxIter": int(nit), "TolFun": crit},
                      compute_hessian=False)
        del est.objective
        res = est.last_csminwel
        assert (res["iterations"], res["fcount"], res["retcode"]) == (r[f"v{v}_itct"], r[f"v{v}_fcount"], r[f"v{v}_retcode"])
        k = r[f"v{v}_Xidx"].astype(int) - 1
        np.testing.assert_allclose(np.array(X)[k].T, r[f"v{v}_X"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(m2.x, r[f"v{v}_xh"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(res["H"], r[f"v{v}_H"], rtol=0, atol=1e-5 * np.abs(r[f"v{v}_H"]).max())
        assert m2.hessian is None and np.all(np.isnan(m2.std)) and np.isnan(m2.laplace)


def test_mode_compute_choices(post, tmp_path):
    r = ref("post_nk_mode.mat")
    est = post("post_nk", 0)                                    # the .mod file says mode_compute=0, no mode_file
    assert est.options["mode_compute"] == 0
    with pytest.warns(pydyn.options.UnsupportedOptionWarning, match="mode_compute=0 without mode_file"):
        m0 = est.mode()                                         # explicit request: the default optimiser (csminwel) runs
    assert "csminwel" in m0.message and abs(m0.log_posterior - r["logpost"]) < 1e-6
    # no mode_compute at all: Dynare's default is 5 (newrat), pydyn's stays csminwel and a warning says so
    e = post("post_nk", 0, mode_compute=None)
    with pytest.warns(pydyn.options.UnsupportedOptionWarning, match="newrat"):
        dflt = e.mode(compute_hessian=False)
    assert "csminwel" in dflt.message and abs(dflt.log_posterior - r["logpost"]) < 1e-6
    sp = e.mode(method="scipy", x0=r["xparam1"] + 0.05 * np.sqrt(np.diag(r["Vmode"])), compute_hessian=False)
    assert "SLSQP" in sp.message and abs(sp.log_posterior - r["logpost"]) < 1e-5   # pydyn's bounded routine
    # saving and reloading a mode; Dynare's own mode file
    est.save_mode(tmp_path / "mode.npz", m0)
    back = est.mode(mode_compute=0, mode_file=tmp_path / "mode.npz")
    np.testing.assert_array_equal(back.x, m0.x)
    np.testing.assert_array_equal(back.hessian, m0.hessian)
    assert back.laplace == m0.laplace and back.nfev < 5 and "mode read from" in back.message
    dyn = post("post_nk", 2).mode()                             # mode_compute=0, mode_file=post_nk_mode0 (a Dynare .mat file)
    np.testing.assert_array_equal(dyn.x, r["xparam1"])
    np.testing.assert_array_equal(dyn.hessian, r["hh"])
    np.testing.assert_allclose(dyn.inv_hessian, r["Vmode"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(dyn.laplace, r["laplace"], rtol=1e-10)
    np.savez(tmp_path / "perm.npz", xparam1=m0.x[::-1], parameter_names=np.array(est.names[::-1]), fval=1.0)
    perm = est.mode(mode_compute=0, mode_file=tmp_path / "perm.npz", compute_hessian=False)
    np.testing.assert_array_equal(perm.x, m0.x)                 # matched by name; no Hessian in the file
    assert perm.hessian is None
    with pytest.raises(estimation.EstimationError, match="no such"):
        est.mode(mode_compute=0, mode_file=tmp_path / "missing")
    # scipy stand-ins: one warning each, same optimum
    e = post("post_nk", 0, mode_compute=1)
    with pytest.warns(pydyn.options.UnsupportedOptionWarning, match="fmincon.*not a port"):
        sqp = e.mode(compute_hessian=False)
    assert abs(sqp.log_posterior - r["logpost"]) < 1e-3
    e = post("post_nk", 0, mode_compute=3)                      # unbounded, like fminunc: started near the mode
    with pytest.warns(pydyn.options.UnsupportedOptionWarning, match="fminunc.*not a port"):
        bfgs = e.mode(x0=r["xparam1"] + 0.05 * np.sqrt(np.diag(r["Vmode"])), compute_hessian=False)
    assert abs(bfgs.log_posterior - r["logpost"]) < 1e-3
    e = post("post_nk", 0, mode_compute=7, optim="'MaxIter',120,'TolFun',1e-6")
    simplex = e.mode(x0=r["xparam1"] + 0.05 * np.sqrt(np.diag(r["Vmode"])), compute_hessian=False)   # ported: no warning
    assert simplex.log_posterior > e.log_posterior(r["xparam1"] + 0.05 * np.sqrt(np.diag(r["Vmode"])))
    # additional_optimizer_steps: csminwel stopped early, then continued by a second run
    e = post("post_nk", 0, mode_compute=4, optim=("MaxIter", 3), additional_optimizer_steps=[4], cova_compute=0)
    two = e.mode()
    one = post("post_nk", 0, mode_compute=4, optim=("MaxIter", 3)).mode(compute_hessian=False)
    assert two.log_posterior > one.log_posterior and two.message.count("csminwel") == 2
    assert two.hessian is None and two.inv_hessian is None      # cova_compute=0
    # mode_compute=9 (CMA-ES in Dynare): differential evolution on a box, then L-BFGS-B
    e = post("post_nk", 0, mode_compute=9, optim="'MaxIter',3")
    with pytest.warns(pydyn.options.UnsupportedOptionWarning, match="CMA-ES.*not a port"):
        de = e.mode(x0=r["xparam1"], compute_hessian=False, rand=3)
    assert "differential_evolution" in de.message and abs(de.log_posterior - r["logpost"]) < 1e-4
    for code in (2, 6, 10):                                                     # 5: tests/test_newrat.py
        with pytest.raises(NotImplementedError, match=f"mode_compute={code}"):
            post("post_nk", 0, mode_compute=code).mode()
    assert estimation._parse_optim("''MaxIter'',20,''TolFun'',1e-5,''InitialInverseHessian'',''1e-3*eye(12)''") == \
        {"MaxIter": 20, "TolFun": 1e-5, "InitialInverseHessian": "1e-3*eye(12)"}
    e = post("post_nk", 0, mode_compute=4, optim="'MaxIter',2,'InitialInverseHessian','1e-3*eye(12)'")
    h0 = e.mode(compute_hessian=False)                          # a larger first step than with the default 1e-4*I
    assert e.last_csminwel["iterations"] == 3 and np.isfinite(h0.log_posterior) and not np.allclose(h0.x, one.x)
    # penalised objective for the Hessian: same result where the objective is defined around the mode
    a = estimation.ModeResult(x=r["xparam1"], names=est.names, log_posterior=est.log_posterior(r["xparam1"]))
    est.add_hessian(a, penalized=True)
    np.testing.assert_allclose(a.hessian, est.hessian(r["xparam1"]), rtol=1e-12)


def test_calibration_and_option_report(post):
    est = post("post_nk", 2)
    theta = est.x0 * 1.05
    before = (est.model.params.copy(), np.array(est.model.Sigma_e, float).copy())
    cal = est.calibration(theta)
    assert set(cal) == {"params", "Sigma_e", "H"}
    Q, H = est.covariances(theta)
    np.testing.assert_array_equal(cal["Sigma_e"], Q)
    np.testing.assert_array_equal(cal["H"], H)
    np.testing.assert_array_equal(cal["params"], est.model_params(theta))
    k = est.model.param_names.index("kappa")
    assert cal["params"][k] == theta[est.names.index("kappa")] and cal["H"][0, 0] == theta[3] ** 2
    np.testing.assert_array_equal(est.model.params, before[0])                 # the model is not modified
    np.testing.assert_array_equal(np.array(est.model.Sigma_e, float), before[1])
    with pytest.raises(estimation.EstimationError):
        est.calibration(theta[:3])
    # every option of this estimation command is acted on
    assert est.unsupported_options == [] and est.ignored_options == ["nodisplay"]
    o = est.options
    assert (o["mh_drop"], o["mh_conf_sig"], o["mh_nblck"], o["sub_draws"], o["mode_file"]) == (0.4, 0.8, 3, 90, "post_nk_mode0")
    # options that are not (moments_varendo is acted on since pydyn.posterior_moments): reported once, by name
    with pytest.warns(pydyn.options.UnsupportedOptionWarning) as rec:
        e2 = estimation.Estimation(est.model, forecast=8, analytic_derivation=True, mh_recover=True,
                                   load_mh_file=True, conditional_likelihood=True, nodiagnostic=True)
    msg = str(rec[0].message)
    for name in ("forecast", "analytic_derivation", "mh_recover", "load_mh_file",
                 "conditional_likelihood"):
        assert name in msg and name in e2.unsupported_options
    assert "nodiagnostic" not in msg and "mode_file" not in msg and "sub_draws" not in msg
    # aliases of the JSON names
    e3 = estimation.Estimation(est.model, **{"mh_tune_jscale.status": True, "mh_jscale": None,
                                             "MCMC_jumping_covariance": "identity_matrix",
                                             "convergence.rafterylewis.indicator": True, "mode_check.number_of_points": 10})
    assert e3.options["mh_tune_jscale"] == 0.33 and e3.options["mcmc_jumping_covariance"] == "identity_matrix"
    assert e3.options["raftery_lewis_diagnostics"] is True and e3.options["mode_check_number_of_points"] == 10


def test_subclass_hook_for_occbin(post):
    """A model with occbin_constraints is refused by Estimation, accepted by a subclass that sets handles_occbin."""
    m = post("post_nk", 2).model
    saved = getattr(m, "occbin_constraints", None)
    m.occbin_constraints = [{"name": "zlb"}]
    try:
        with pytest.raises(NotImplementedError, match="occbin_constraints"):
            estimation.Estimation(m)

        class Sub(estimation.Estimation):
            handles_occbin = True

            def _minus_loglik(self, theta, return_lik=False):        # documented extension point
                return super()._minus_loglik(theta, return_lik)
        e = Sub(m)
        assert m.occbin_constraints == [{"name": "zlb"}] and np.isfinite(e.log_posterior(e.x0))
    finally:
        m.occbin_constraints = saved


# ------------------------------------------------------------------ standard deviations of correlated shocks (R11)
@pytest.mark.parametrize("variant", [1, 2, 3])
def test_estimated_stderr_with_correlated_shocks(load, variant):
    """est_corr.mod (reference/r_shock_corr.m, Dynare 7.2): Sigma_e from set_all_parameters and the
    likelihood at three points. 1: 'corr e, u' calibrated, stderr e estimated: the correlation is
    kept. 2: 'var e, u = 0.06' (a covariance) calibrated: Dynare rebuilds Sigma_e from
    M_.Correlation_matrix, which only 'corr' sets, so the covariance is dropped (pydyn kept the
    correlation: likelihood off by 8.7) and pydyn warns. 3: corr u, w estimated as well: the
    calibrated covariance of e, u keeps its value (check_for_calibrated_covariances), so at the third
    point (stderr e = 0.2) the correlation is 1 and both reject the point. Measured: Sigma_e 2e-18,
    log-likelihood 1e-13 relative."""
    m = load("est_corr", variant)
    r = ref(f"clean_est_corr_v{variant}.mat")
    if variant == 2:
        with pytest.warns(UserWarning, match="calibrated covariances of e and u are not kept"):
            est = estimation.Estimation(m)
    else:
        est = estimation.Estimation(m)
    assert ";".join(est.names) == r["names"]
    for k in range(r["X"].shape[1]):
        x = r["X"][:, k]
        Q, _ = est.covariances(x)
        np.testing.assert_allclose(Q, r["Sigma_e"][:, :, k], rtol=1e-14, atol=1e-16)
        if np.isfinite(r["fval"][k]):
            np.testing.assert_allclose(est.log_likelihood(x), -r["fval"][k], rtol=1e-12)
        else:
            assert est.log_likelihood(x) == -np.inf


def test_set_shock_stderr_keeps_the_covariance(load):
    """R11. Dynare's set_shock_stderr_value sets M_.Sigma_e(i,i) only: the covariance with a
    correlated shock is kept, so their correlation changes (pydyn's Model.set_shock_stderr zeroed
    the row and column). Reference: est_corr.mod variant 1 after set_shock_stderr_value('e', 0.8)."""
    m = load("est_corr", 1)
    r = ref("clean_est_corr_v1.mat")
    saved = m.Sigma_e.copy()
    np.testing.assert_allclose(m.Sigma_e, r["Sigma_e0"], rtol=1e-15)
    try:
        m.set_shock_stderr("e", 0.8)
        np.testing.assert_allclose(m.Sigma_e, r["Sigma_e_set"], rtol=1e-15, atol=0)
    finally:
        m.Sigma_e[:] = saved                      # the model is shared with the other tests of the module


@pytest.mark.parametrize("variant", [1, 2])
def test_singular_forecast_error_variance_in_the_likelihood(load, variant):
    """R25 (likelihood side). est_sing.mod: one shock, two observed variables. Dynare 7.2's
    dsge_likelihood switches to the univariate filter when the multivariate one finds F singular
    (use_univariate_filters_if_singularity_is_detected, default 1) and returns info 50 when the
    option is 0 (variant 2); pydyn already did both. (Dynare's estimation command itself stops
    earlier, in initial_estimation_checks, which pydyn does not port; reference/r_est_sing.m calls
    dsge_likelihood directly.) Measured 3e-16 relative."""
    m = load("est_sing", variant)
    r = ref(f"clean_est_sing_v{variant}.mat")
    est = estimation.Estimation(m)
    assert ";".join(est.names) == r["names"]
    for k in range(r["X"].shape[1]):
        ll = est.log_likelihood(r["X"][:, k])
        if variant == 1:
            assert est.last_filter == "univariate" and int(r["info"][k]) == 0
            np.testing.assert_allclose(ll, -r["fval"][k], rtol=1e-12)
        else:
            assert ll == -np.inf and int(r["info"][k]) == 50 and "info 50" in est.last_info
