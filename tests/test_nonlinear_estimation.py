"""Second-order estimation with Dynare's nonlinear filters, against Dynare 7.2 under Octave 8.4.

References (reference/r_nonlinear_filters.m, reference/r_octave_rng.m), for the RBC model of Dynare's
tests/particle (``nl_rbc.mod``, 50 observations of y, l, i with estimated measurement errors):

* ``nl_rbc_lik.mat``: ``non_linear_dsge_likelihood`` (minus log posterior) for 35 filter
  configurations (sis, apf, cpf, gf, nlkf and their options) at three parameter vectors, the
  per-period log likelihoods, and Dynare's reduced form at the first vector;
* ``nl_rbc_seeds.mat``: sis and apf (1000 particles) with the filter seed set to 1..30;
* ``nl_rbc_smooth.mat``: inputs and outputs of ``multivariate_smooth_resampling`` in sis;
* ``octave_rng.mat``: Octave's ``rand``/``randn`` numbers for four seeds.

Random numbers: pydyn's default stream reproduces Octave's generators, so the particle filters are
compared exactly (same draws), not statistically. Accuracy: run on Dynare's own reduced form the
filters agree with Dynare to about 1e-10 per period. Through pydyn's own second-order solution, whose
coefficients differ from Dynare's by about 1e-12 (rounding in the model derivatives), the likelihoods
agree to about 1e-9 relative to the sum of the absolute per-period log likelihoods: the measurement
errors of this model are tiny (standard deviation 0.001), which magnifies every difference in the
predicted observables by about 1e6.

Not compared exactly, with the cause:
* ``resampling_method=smooth``: Dynare's ``eig(P'*(P/N))`` takes the symmetric or the general
  eigensolver according to whether the product is exactly symmetric, which depends on BLAS rounding;
  the function is checked step by step with Dynare's solver choice instead;
* ``gf`` with ``distribution_approximation=montecarlo`` and resampling: the particles collapse
  (all weight on one particle), and whether the Cholesky factorisation of their (numerically zero)
  covariance succeeds is decided by rounding; Dynare then either stops with a dimension error or
  continues with a meaningless factor. Compared only where neither stops.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.io import loadmat

import pydyn
from pydyn import nonlinear_filters as nlf
from pydyn.estimation import _Rejected
from pydyn.nonlinear_estimation import NonlinearEstimation, particle_options
from pydyn.random_streams import NumpyRandomStream, OctaveRandomStream, ReplayStream

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))

pytestmark = pytest.mark.skipif(not (ROOT / "korder" / "libpydyn_korder.so").exists()
                                and not os.environ.get("PYDYN_KORDER_LIB"),
                                reason="k-order library not built")

# name -> filter_algorithm, number_of_particles, keyword options (as in r_nonlinear_filters.m)
CASES = {
    "gf_uu": ("gf", 1000, {}),
    "gf_cu": ("gf", 1000, {"proposal_approximation": "cubature"}),
    "gf_uc": ("gf", 1000, {"distribution_approximation": "cubature"}),
    "gf_cc": ("gf", 1000, {"proposal_approximation": "cubature", "distribution_approximation": "cubature"}),
    "gf_uu_abk": ("gf", 1000, {"unscented_alpha": 0.8, "unscented_beta": 1, "unscented_kappa": 2}),
    "nlkf_u": ("nlkf", 1000, {}),
    "nlkf_c": ("nlkf", 1000, {"proposal_approximation": "cubature"}),
    "nlkf_u_abk": ("nlkf", 1000, {"unscented_alpha": 0.8, "unscented_beta": 1, "unscented_kappa": 2}),
    "gf_init3": ("gf", 1000, {"nonlinear_filter_initialization": 3, "initial_state_prior_std": 0.5}),
    "nlkf_m": ("nlkf", 500, {"proposal_approximation": "montecarlo"}),
    "gf_mu": ("gf", 500, {"proposal_approximation": "montecarlo"}),
    "gf_um": ("gf", 500, {"distribution_approximation": "montecarlo"}),
    "gf_mm": ("gf", 500, {"proposal_approximation": "montecarlo", "distribution_approximation": "montecarlo"}),
    "gf_um_generic": ("gf", 500, {"distribution_approximation": "montecarlo", "resampling": "generic"}),
    "gf_um_none": ("gf", 500, {"distribution_approximation": "montecarlo", "resampling": "none"}),
    "sis": ("sis", 1000, {}),
    "sis_strat": ("sis", 1000, {"resampling_method": "stratified"}),
    "sis_smooth": ("sis", 1000, {"resampling_method": "smooth"}),
    "sis_generic": ("sis", 1000, {"resampling": "generic"}),
    "sis_generic_t": ("sis", 1000, {"resampling": "generic", "resampling_threshold": 19}),
    "sis_generic_low": ("sis", 1000, {"resampling": "generic", "resampling_threshold": 0.001}),
    "sis_none": ("sis", 1000, {"resampling": "none"}),
    "sis_pruning": ("sis", 1000, {"particle_pruning": True}),
    "sis_init3": ("sis", 1000, {"nonlinear_filter_initialization": 3}),
    "apf": ("apf", 1000, {}),
    "apf_none": ("apf", 1000, {"resampling": "none"}),
    "apf_strat": ("apf", 1000, {"resampling_method": "stratified"}),
    "apf_generic": ("apf", 1000, {"resampling": "generic", "resampling_threshold": 19}),
    "apf_generic_low": ("apf", 1000, {"resampling": "generic", "resampling_threshold": 0.001}),
    "apf_pruning": ("apf", 1000, {"particle_pruning": True}),
    "cpf_u": ("cpf", 100, {}),
    "cpf_c": ("cpf", 100, {"proposal_approximation": "cubature"}),
    "cpf_m": ("cpf", 100, {"proposal_approximation": "montecarlo"}),
    "cpf_u_mjp": ("cpf", 100, {"cpf_weights": "murrayjonesparslow"}),
    "cpf_u_none": ("cpf", 100, {"resampling": "none"}),
}
SMOOTH = {"sis_smooth"}
GF_MC_RESAMPLED = {"gf_um", "gf_mm", "gf_um_generic"}


def ref(name):
    return loadmat(REF / f"{name}.mat", squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("nlrbc")
    for f in ("nl_rbc.mod", "nl_rbc_data.m"):
        shutil.copy(REF / f, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(d / "nl_rbc.mod"))


@pytest.fixture(scope="module")
def lik_ref():
    return ref("nl_rbc_lik")


def make(model, name, **extra):
    alg, n, kw = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NonlinearEstimation(model, filter_algorithm=alg, number_of_particles=n, **kw, **extra)


def dynare_reduced_form(R):
    a2 = lambda v: np.atleast_2d(np.asarray(v, float))   # noqa: E731
    col = lambda v: np.asarray(v, float).reshape(-1, 1)  # noqa: E731  (one shock: column vectors)
    vec = lambda v: np.asarray(v, float).ravel()          # noqa: E731
    return nlf.ReducedForm(ghx=a2(R.ghx), ghu=col(R.ghu), ghxx=a2(R.ghxx), ghuu=col(R.ghuu), ghxu=a2(R.ghxu),
                           ghs2=vec(R.ghs2), steadystate=vec(R.steadystate), constant=vec(R.constant),
                           state_variables_steady_state=vec(R.state_variables_steady_state),
                           mf0=np.asarray(R.mf0, int).ravel() - 1, mf1=np.asarray(R.mf1, int).ravel() - 1,
                           Q=a2(R.Q), H=a2(R.H), StateVectorMean=vec(R.StateVectorMean),
                           StateVectorVariance=a2(R.StateVectorVariance))


# ---------------------------------------------------------------------------- random numbers
def test_octave_random_streams():
    r = ref("octave_rng")
    idx = np.asarray(r["idx"], int) - 1
    for k, seed in enumerate(np.atleast_1d(r["seeds"]), start=1):
        s = OctaveRandomStream(int(seed))
        u = s.rand(200000, 1).ravel()
        x = s.randn(200000, 1).ravel()
        assert np.array_equal(u[:50], r[f"u_head_{k}"]) and np.array_equal(x[:50], r[f"x_head_{k}"])
        assert np.array_equal(u[idx], r[f"u_{k}"]) and np.array_equal(x[idx], r[f"x_{k}"])
        # sums of all 200000 numbers (sequential summation, as Octave's sum)
        assert np.cumsum(u)[-1] == pytest.approx(r[f"u_sum_{k}"], rel=1e-14)
        assert np.cumsum(x)[-1] == pytest.approx(r[f"x_sum_{k}"], rel=1e-12, abs=1e-9)
        assert np.max(np.abs(x)) == r[f"x_max_{k}"]          # a draw from the Ziggurat's tail
    s = OctaveRandomStream(0)
    a, b, c, d = s.rand(2, 3), s.randn(3, 2), s.rand(1, 4), s.randn(5, 1)
    for got, name in ((a, "a"), (b, "b"), (c, "c"), (d, "d")):
        assert np.array_equal(got, np.atleast_2d(r[name]).reshape(got.shape))


def test_replay_and_numpy_streams():
    rp = ReplayStream(normals=np.arange(6.0), uniforms=[0.5])
    assert np.array_equal(rp.randn(2, 3), np.arange(6.0).reshape(2, 3, order="F"))
    assert rp.rand(1, 1)[0, 0] == 0.5
    with pytest.raises(RuntimeError):
        rp.rand(1, 1)
    a, b = NumpyRandomStream(3).randn(4, 5), NumpyRandomStream(3).randn(4, 5)
    assert np.array_equal(a, b) and a.shape == (4, 5)


# ---------------------------------------------------------------------------- reduced form
def test_reduced_form(model, lik_ref):
    est = make(model, "gf_uu")
    assert est.names == lik_ref["names"].split(";")
    rf = est.reduced_form(lik_ref["X"][:, 0])
    R = dynare_reduced_form(lik_ref["rf"])
    assert np.array_equal(rf.mf0, R.mf0) and np.array_equal(rf.mf1, R.mf1)
    for f in ("ghx", "ghu", "ghxx", "ghuu", "ghxu", "ghs2", "constant", "steadystate", "StateVectorMean",
              "StateVectorVariance", "Q", "H"):
        a, b = np.asarray(getattr(rf, f)), np.asarray(getattr(R, f)).reshape(np.shape(getattr(rf, f)))
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9 * np.max(np.abs(b)), err_msg=f)
    np.testing.assert_array_equal(est.data, lik_ref["data"])


# ---------------------------------------------------------------------------- filters, same reduced form
@pytest.mark.parametrize("name", [c for c in CASES if "init3" not in c and c not in SMOOTH
                                  and c not in ("gf_mm", "gf_um_generic")])
def test_filter_on_dynare_reduced_form(model, lik_ref, name):
    """The filter alone: Dynare's reduced form and the same random numbers give Dynare's per-period
    log likelihoods to 1e-9 (the log densities are of order 10 to 1000)."""
    est = make(model, name)
    rf = dynare_reduced_form(lik_ref["rf"])
    L = np.asarray(lik_ref[f"{name}_lik"])[:, 0]
    kw = {"model_pruning": False} if CASES[name][0] == "cpf" else {}
    if name == "gf_um":                    # Dynare stops: the particles collapse (see module docstring)
        assert lik_ref[f"{name}_info"][0] == -1
        with pytest.raises(nlf.NonlinearFilterError):
            est.filter(rf, est.data.T, 1, est.particle, est.new_stream(), **kw)
        return
    with np.errstate(all="ignore"):
        LIK, lik = est.filter(rf, est.data.T, 1, est.particle, est.new_stream(), **kw)
    finite = np.isfinite(L)
    assert np.array_equal(finite, np.isfinite(lik))
    np.testing.assert_allclose(lik[finite], L[finite], rtol=0, atol=1e-9)
    if lik_ref[f"{name}_info"][0] == 0:
        assert LIK == pytest.approx(-np.sum(L), abs=1e-8)


# ---------------------------------------------------------------------------- full likelihood
@pytest.mark.parametrize("name", [c for c in CASES if c not in SMOOTH])
def test_likelihood_against_dynare(model, lik_ref, name):
    est = make(model, name)
    X = lik_ref["X"]
    fv, info = np.atleast_1d(lik_ref[f"{name}_fval"]), np.atleast_1d(lik_ref[f"{name}_info"])
    L = np.atleast_2d(lik_ref[f"{name}_lik"])
    compared = 0
    for k in range(X.shape[1]):
        x = X[:, k]
        try:
            LIK, lik, _ = est.run_filter(x)
            code = 0
        except _Rejected as exc:
            code = exc.code
        except nlf.NonlinearFilterError:
            assert name in GF_MC_RESAMPLED          # collapsed particles, see the module docstring
            continue
        if name in GF_MC_RESAMPLED and (code != 0 or info[k] != 0):
            continue
        assert code == info[k], f"{name}, point {k}: info {code} instead of {info[k]}"
        if code == 0:
            scale = np.nansum(np.abs(L[:, k]))
            assert LIK - est.log_prior(x) == pytest.approx(fv[k], abs=1e-9 * scale), f"{name}, point {k}"
            if name not in GF_MC_RESAMPLED:    # there even the last bits of the solution decide on a collapse
                assert -est.log_posterior(x) == pytest.approx(fv[k], abs=1e-9 * scale)
            compared += 1
    assert compared >= (1 if name in GF_MC_RESAMPLED else (0 if np.all(info != 0) else 3))


def test_seeds_reproduced(model):
    """Dynare with the filter seed set to 1..30 (sis and apf, 1000 particles): pydyn with the same seed."""
    s = ref("nl_rbc_seeds")
    x0 = s["x0"]
    for alg in ("sis", "apf"):
        for seed, fval in zip(s["seeds"], s[f"{alg}_fval"]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = NonlinearEstimation(model, filter_algorithm=alg, number_of_particles=1000, particle_seed=int(seed))
            got = -est.log_posterior(x0)
            assert got == pytest.approx(fval, abs=1e-8 * max(1.0, abs(fval)) + 2e-6), (alg, seed)


def test_seed_distribution_numpy_stream(model):
    """numpy's generator instead of Octave's: the likelihood across 30 seeds has the same distribution as
    Dynare's across its seeds 1..30."""
    s = ref("nl_rbc_seeds")
    x0 = s["x0"]
    for alg in ("sis", "apf"):
        vals = []
        for seed in range(30):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = NonlinearEstimation(model, filter_algorithm=alg, number_of_particles=1000, rng="numpy",
                                          particle_seed=seed)
            vals.append(-est.log_posterior(x0))
        vals, dyn = np.array(vals), np.asarray(s[f"{alg}_fval"])
        # heavy tails (tiny measurement errors): a two-sample Kolmogorov-Smirnov test, not moments
        assert stats.ks_2samp(vals, dyn).pvalue > 0.01, alg


def test_smooth_resampling_steps():
    """multivariate_smooth_resampling on the particles Dynare passed to it in sis (200 particles), with
    the eigensolver Dynare used (symmetric or general) and the same uniform numbers."""
    r = ref("nl_rbc_smooth")
    u = OctaveRandomStream(0).rand(12, 1).ravel()
    for k in range(1, 7):
        P, w, out = r[f"P_{k}"], r[f"w_{k}"], r[f"out_{k}"]
        stream = ReplayStream(uniforms=u[2 * (k - 1):2 * k])
        got = nlf.multivariate_smooth_resampling(P, w, stream, symmetric=bool(r[f"sym_{k}"]))
        np.testing.assert_allclose(got, out, rtol=0, atol=1e-9 * np.max(np.abs(out)))


# ---------------------------------------------------------------------------- building blocks
def test_local_state_space_iteration_2_kronecker():
    rng = np.random.default_rng(0)
    m, n, q, N = 4, 3, 2, 5
    ghx, ghu = rng.normal(size=(m, n)), rng.normal(size=(m, q))
    ghxx, ghuu, ghxu = rng.normal(size=(m, n * n)), rng.normal(size=(m, q * q)), rng.normal(size=(m, n * q))
    c, yhat, eps, yp, ss = rng.normal(size=m), rng.normal(size=(n, N)), rng.normal(size=(q, N)), \
        rng.normal(size=(n, N)), rng.normal(size=m)
    y, y_ = nlf.local_state_space_iteration_2(yhat, eps, ghx, ghu, c, ghxx, ghuu, ghxu, yp, ss)
    for j in range(N):        # the loop of Dynare's local_state_space_iteration_2.m
        e = c + ghx @ yhat[:, j] + ghu @ eps[:, j] + 0.5 * ghxx @ np.kron(yp[:, j], yp[:, j]) \
            + 0.5 * ghuu @ np.kron(eps[:, j], eps[:, j]) + ghxu @ np.kron(yp[:, j], eps[:, j])
        np.testing.assert_allclose(y[:, j], e, atol=1e-12)
        np.testing.assert_allclose(y_[:, j], ghx @ yp[:, j] + ghu @ eps[:, j] + ss, atol=1e-12)


def test_resampling_quirks():
    w = np.array([0.1, 0.2, 0.3, 0.4])
    # Kitagawa: (i - 1 + u)/N against the cumulated weights
    assert np.array_equal(nlf.traditional_resampling(None, w, [0.5]), [1, 2, 3, 3])     # 0-based
    # Dynare's "stratified": the sorted uniforms are cumulative products of the uniforms
    noise = np.array([0.9, 0.5, 0.9, 0.9])
    assert np.array_equal(nlf.traditional_resampling(None, w, noise), [3, 2, 2, 2])   # 0-based
    po = nlf.ParticleOptions(resampling="generic", resampling_threshold=0.5)
    assert not po.must_resample(np.full(100, 0.01), 50)          # neff 100 >= 0.5 * 50 periods
    assert po.must_resample(np.r_[0.99, np.full(99, 0.01 / 99)], 50)


# ---------------------------------------------------------------------------- options and refusals
def test_options_from_mod_file(tmp_path, lik_ref):
    text = (REF / "nl_rbc.mod").read_text().replace(
        "estimation(order=2, datafile=nl_rbc_data, nobs=50, number_of_particles=1000,",
        "estimation(order=2, datafile=nl_rbc_data, nobs=50, number_of_particles=1000, filter_algorithm=apf, "
        "resampling_method=stratified, particle_filter_options=('unscented_alpha', 0.5),")
    (tmp_path / "nl_rbc.mod").write_text(text)
    shutil.copy(REF / "nl_rbc_data.m", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "nl_rbc.mod"))
        est = NonlinearEstimation(m)
    po = est.particle
    assert (po.filter_algorithm, po.number_of_particles, po.resampling_method, po.unscented_alpha) == \
        ("apf", 1000, "stratified", 0.5)
    x = lik_ref["X"][:, 0]
    scale = np.nansum(np.abs(lik_ref["apf_strat_lik"][:, 0]))
    assert -est.log_posterior(x) == pytest.approx(lik_ref["apf_strat_fval"][0], abs=1e-9 * scale)
    with warnings.catch_warnings():                 # a keyword overrides the .mod file
        warnings.simplefilter("ignore")
        assert NonlinearEstimation(m, resampling_method="kitagawa").particle.resampling_method == "kitagawa"


def test_particle_options_parsing():
    po, rest = particle_options({"particle.filter_algorithm": "sis", "particle.resampling.status.generic": True,
                                 "particle.resampling.status.systematic": False,
                                 "particle.resampling.threshold": 0.3, "particle.initialization": 3,
                                 "particle.particle_filter_options": "''pruning'',true,''initial_state_prior_std'',0.2",
                                 "mh_replic": 0})
    assert (po.resampling, po.resampling_threshold, po.initialization, po.pruning, po.initial_state_prior_std) == \
        ("generic", 0.3, 3, True, 0.2)
    assert rest == {"mh_replic": 0}


@pytest.mark.parametrize("opts, exc", [
    ({"order": 3}, NotImplementedError),
    ({"loglinear": True}, NotImplementedError),
    ({"filter_algorithm": "gmf"}, NotImplementedError),
    ({"nonlinear_filter_initialization": 2}, NotImplementedError),
    ({"resampling_method": "residual"}, NotImplementedError),
    ({"filter_algorithm": "cpf", "pruning": True}, NotImplementedError),
    ({"filter_algorithm": "gf", "particle_pruning": True}, NotImplementedError),
    ({"filter_algorithm": "apf", "resampling_method": "smooth"}, NotImplementedError),
    ({"filter_algorithm": "cpf", "proposal_approximation": "cubature", "cpf_weights": "murrayjonesparslow"},
     NotImplementedError),
    ({"prefilter": 1}, NotImplementedError),
    ({"dsge_var": 0.5}, NotImplementedError),
    ({"heteroskedastic_filter": True}, NotImplementedError),
    ({"endogenous_prior": True}, NotImplementedError),
])
def test_refusals(model, lik_ref, opts, exc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if exc is not None:
            with pytest.raises(exc):
                NonlinearEstimation(model, **opts)
            return
        est = NonlinearEstimation(model, number_of_particles=100, **opts)
    with pytest.raises(NotImplementedError):
        est.run_filter(lik_ref["X"][:, 0])


def test_occbin_refused(model):
    model.occbin_constraints = [{"dummy": True}]
    try:
        with pytest.raises(NotImplementedError, match="OccBin"):
            NonlinearEstimation(model)
    finally:
        model.occbin_constraints = []


# ---------------------------------------------------------------------------- estimation hooks
def test_mode_and_sampler(model, lik_ref):
    """The particle likelihood is deterministic in the parameters (common random numbers, as Dynare),
    so the optimisers and the sampler of Estimation work unchanged."""
    est = make(model, "gf_uu")
    x0 = lik_ref["X"][:, 0]
    # deterministic up to the last bits of the k-order solution (libkorder varies by ~1e-15 between calls)
    assert est.log_posterior(x0) == pytest.approx(est.log_posterior(x0), rel=1e-12)
    sis = make(model, "sis")
    assert sis.log_posterior(x0) == pytest.approx(sis.log_posterior(x0), rel=1e-11)
    assert sis.log_posterior(x0) == pytest.approx(-lik_ref["sis_fval"][0], rel=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = est.mode(x0=x0, mode_compute=7, compute_hessian=False, optim={"MaxIter": 120})
    assert md.log_posterior > est.log_posterior(x0) + 1.0
    cov = np.diag((0.001 * np.abs(md.x) + 1e-6) ** 2)
    mc = est.sample(replic=30, nblocks=1, mode=md.x, cov=cov, seed=1)
    assert mc.draws.shape == (1, 30, est.n) and np.all(np.isfinite(mc.log_posterior))
    with pytest.raises(NotImplementedError):
        est.smoother(x0)


@pytest.mark.parametrize("case", ["gf_300", "sis_300"])
def test_simplex_mode_compute_8(model, case):
    """Dynare's simplex (mode_compute=8, dynare_minimize_objective) from the initial values, stopped
    after 300 function calls (optim MaxFunEvals): the same path, hence the same point, value, number of
    calls and of iterations (reference/nl_rbc_simplex.mat; the full run, 14969 calls, also agrees but
    takes ten minutes)."""
    r = ref("nl_rbc_simplex")
    alg, n = {"gf_300": ("gf", 1000), "sis_300": ("sis", 200)}[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = NonlinearEstimation(model, filter_algorithm=alg, number_of_particles=n)
        md = est.mode(mode_compute=8, optim={"MaxFunEvals": 300}, compute_hessian=False)
    assert md.nfev == r[f"{case}_funcCount"]
    assert -md.log_posterior == pytest.approx(r[f"{case}_fval"], abs=1e-8)
    np.testing.assert_allclose(md.x, r[f"{case}_x"], rtol=1e-8)
