"""pydyn.samplers and pydyn.octave_rng against Dynare 7.2 under Octave 8.4.

References: reference/r_samplers.m runs Dynare on reference/smp_nk.mod and reference/smp_schorfheide.mod (one case per
-DCASE=k) and writes smp_case<k>.mat (the chains, starting points, bounds, rotations, ...) and smp_rng.mat (Octave's
random numbers and a few small functions). No random number of Dynare's runs is stored: pydyn regenerates them from
Dynare's seeds (pydyn.octave_rng), so these tests replay Dynare's chains from the seeds alone.

Bounds: with prior_trunc > 0 (the slice sampler sets 1e-10) Dynare's bounds come from the inverse CDFs of Octave's
statistics package (at 1 - prior_trunc, which rounds to 1 - 1.0000000827e-10), which differ from scipy's by up to
2.5e-8 (relative; measured: the lower bound of the Weibull prior of sig, the others 6e-9 or less); the slice replays pass
Dynare's bounds (sampler_bounds / kernel_bounds) and the bounds themselves are compared with that tolerance.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation, octave_rng, samplers
from pydyn.octave_rng import OctaveRandom

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
MOD = {1: "smp_nk", 2: "smp_schorfheide", 3: "smp_nk", 4: "smp_schorfheide", 5: "smp_nk", 6: "smp_nk",
       7: "smp_schorfheide", 8: "smp_nk", 9: "smp_schorfheide", 10: "smp_schorfheide", 11: "smp_schorfheide",
       12: "smp_schorfheide", 13: "smp_nk", 14: "smp_schorfheide"}

pytestmark = pytest.mark.filterwarnings("ignore:statements not handled", "ignore:Estimation does not act")
_cache = {}


def ref(name):
    if name not in _cache:
        _cache[name] = loadmat(REF / name, squeeze_me=True)
    return _cache[name]


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    """load(case) -> Estimation for the .mod file of that case preprocessed with -DCASE=<case>."""
    work = tmp_path_factory.mktemp("smp")
    made = {}

    def _load(case, **options):
        key = (case, tuple(sorted(options.items())))
        if key not in made:
            mod = MOD[case]
            sub = work / f"case{case}_{len(made)}"
            sub.mkdir()
            for f in ["est_nk_data.m", "est_schorfheide_data.m", "post_nk_mode0.mat", "post_schorfheide_mode0.mat",
                      "smp_schorfheide_modes.mat", f"{mod}.mod"]:
                shutil.copy(REF / f, sub)
            m = pydyn.Model(pydyn.run_preprocessor(sub / f"{mod}.mod", extra_args=(f"-DCASE={case}",)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                made[key] = estimation.Estimation(m, **options)
        return made[key]
    return _load


def chains(r):
    x2 = r["x2"] if r["x2"].ndim == 3 else r["x2"][None]
    lp = np.atleast_2d(r["logpo2"])
    return x2, lp, np.atleast_2d(r["x_init"])


def dynare_bounds(r):
    return dict(sampler_bounds=(r["sampler_lb"], r["sampler_ub"]), kernel_bounds=(r["mh_lb"], r["mh_ub"]))


# ------------------------------------------------------------------ Octave's random numbers
def test_octave_rng_matches_octave():
    q = ref("smp_rng.mat")
    for sd in (1, 7):
        p = f"s{sd}_"
        r = OctaveRandom(sd)
        u = r.rand(2000000)
        z = r.randn(2000000)
        np.testing.assert_array_equal(u[:50], q[p + "u_first"])
        np.testing.assert_array_equal(u[::997], q[p + "u_sub"])
        np.testing.assert_array_equal(z[:50], q[p + "z_first"])
        np.testing.assert_array_equal(z[::997], q[p + "z_sub"])
        tail = np.flatnonzero(np.abs(z) > 3.6541528853610088) + 1               # the ziggurat's tail algorithm
        np.testing.assert_array_equal(tail, q[p + "tail_idx"])
        np.testing.assert_array_equal(z[tail - 1], q[p + "tail_z"])
        np.testing.assert_allclose([u.sum(), z.sum()], [q[p + "u_sum"], q[p + "z_sum"]], rtol=1e-12)
        m = octave_rng.MTStream(sd)
        np.testing.assert_array_equal(m.get_state(), q[p + "state0"])
        [m.u53() for _ in range(1001)]
        np.testing.assert_array_equal(m.get_state(), q[p + "state1001"])
        m2 = octave_rng.MTStream(state=q[p + "state1001"])                        # restarting from a stored state
        assert m2.u53() == m.u53()
    r = OctaveRandom(7)                                                            # randperm uses the uniform stream
    np.testing.assert_array_equal(r.randn(3), q["mix_a"])
    np.testing.assert_array_equal(r.randperm(6) + 1, q["mix_p1"])
    np.testing.assert_array_equal(r.rand(2), q["mix_b"])
    np.testing.assert_array_equal(r.randperm(6) + 1, q["mix_p2"])
    assert r.randn() == q["mix_c"]
    np.testing.assert_array_equal(r.randperm(5, 3) + 1, q["mix_p3"])
    assert r.rand() == q["mix_d"]
    with pytest.raises(ValueError, match="integers"):
        OctaveRandom(-3)
    assert octave_rng.dynare_chain_seeds(0, 4) == [1, 3, 6, 10] and octave_rng.dynare_chain_seeds(1000, 2) == [1001, 1003]
    a = octave_rng.ArrayRandom([0.25, 0.5], [1.0])
    assert a.rand() == 0.25 and a.randn() == 1.0 and a.rand() == 0.5
    with pytest.raises(IndexError):
        a.rand()


@pytest.mark.parametrize("mod", ["post_nk", "post_schorfheide"])
def test_octave_rng_reproduces_dynare_chain_streams(mod):
    """The random numbers r_posterior.m regenerated from the seeds Dynare stored are those of Dynare's seeding rule."""
    p = ref(f"{mod}_replay.mat")
    nb, n = p["seg_uniform"].shape[1], p["seg_normal"].shape[1]
    for b, s in enumerate(octave_rng.dynare_chain_seeds(0, nb)):
        r = OctaveRandom(s)
        np.testing.assert_array_equal(r.randn((n, p["seg_normal"].shape[0])).T, p["seg_normal"][:, :, b])
        np.testing.assert_array_equal(r.rand(p["seg_uniform"].shape[0]), p["seg_uniform"][:, b])
    np.testing.assert_array_equal(OctaveRandom(0).randn((n, 30)).T, p["init_normal"])


def test_small_functions_match_dynare():
    q = ref("smp_rng.mat")
    for i in (1, 2, 3):
        R, indef, E = samplers.chol_SE(q[f"cse_A{i}"])
        np.testing.assert_allclose(R, q[f"cse_R{i}"], rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(E, q[f"cse_E{i}"], rtol=1e-13, atol=1e-15)
        assert indef == bool(q[f"cse_indef{i}"])
        np.testing.assert_allclose(R.T @ R, q[f"cse_A{i}"] + np.diag(E), atol=1e-12)
    assert not q["cse_indef1"] and q["cse_indef2"]
    C, X, M = q["pdf_C"], q["pdf_X"], q["pdf_M"]
    np.testing.assert_allclose(samplers.multivariate_normal_pdf(X, M, C, 3), q["pdf_normal"], rtol=1e-14)
    np.testing.assert_allclose(samplers.multivariate_student_pdf(X, M, C, 5), q["pdf_student"], rtol=1e-14)
    r = OctaveRandom(11)
    np.testing.assert_allclose(samplers.rand_multivariate_student(r, M, C, 5), q["student_draw"], rtol=1e-15)
    assert r.randn() == q["student_next"]                     # dim + df normals consumed


# ------------------------------------------------------------------ random-walk Metropolis-Hastings
def test_rwmh_replays_whole_dynare_chains_from_the_seed(tmp_path):
    """post_nk (r_posterior.m stage 2): 3 chains x 2400 draws; pydyn with rng='octave' (Dynare's seeds, starting points
    drawn from the global seed) reproduces all of them; the default sampler does too when fed the same numbers."""
    from test_posterior import chains as post_chains
    c = ref("post_nk_chains.mat")
    draws, lp, _ = post_chains("post_nk")
    sub = tmp_path / "pn"
    sub.mkdir()
    for f in ("post_nk.mod", "est_nk_data.m", "post_nk_mode0.mat"):
        shutil.copy(REF / f, sub)
    m = pydyn.Model(pydyn.run_preprocessor(sub / "post_nk.mod", extra_args=("-DPOST=2",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimation.Estimation(m)
    mode = est.mode()
    T = draws.shape[1]
    mc = est.sample(replic=T, mode=mode, rng="octave")
    np.testing.assert_allclose(mc.x0, c["x_init"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(mc.draws, draws, rtol=0, atol=1e-10)
    np.testing.assert_allclose(mc.log_posterior, lp, rtol=1e-11, atol=0)
    np.testing.assert_allclose(mc.acceptance_rate, c["acceptance"], rtol=1e-12)


def test_rwmh_estimated_params_bounds_match_dynare(load):
    """CASE 8: rhoa has the estimated_params bounds [0.5, 0.86] and its posterior mode is 0.857. Dynare rejects every
    proposal above 0.86 (posterior_sampler_iteration.m checks proposals against sampler_options.bounds), so its chain
    never exceeds 0.86; pydyn reproduces the chain through both code paths."""
    r = ref("smp_case8.mat")
    est = load(8)
    x2, lp, xi = chains(r)
    i = est.names.index("rhoa")
    assert x2[0, :, i].max() < 0.86 and est.ub[i] == 0.86
    mode = est.mode()
    mc = est.sample(replic=x2.shape[1], nblocks=1, mode=mode, rng="octave")
    np.testing.assert_allclose(mc.draws, x2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mc.log_posterior, lp, rtol=1e-12)
    o = OctaveRandom(1)
    z = o.randn((est.n, x2.shape[1])).T
    u = o.rand(x2.shape[1])
    mc2 = est.sample(replic=x2.shape[1], nblocks=1, mode=mode, innovations=z[None], uniforms=u[None], uniform_stream=True)
    np.testing.assert_allclose(mc2.draws, x2, rtol=0, atol=1e-12)


def test_rwmh_student_proposal_matches_dynare(load):
    """CASE 7: proposal_distribution='rand_multivariate_student', 4 degrees of freedom, 2 chains x 400 draws."""
    r = ref("smp_case7.mat")
    est = load(7)
    x2, lp, xi = chains(r)
    mc = est.sample(replic=x2.shape[1], mode=est.mode(), rng="octave")
    assert isinstance(mc, estimation.MCMCResult) and mc.method == samplers.RWMH
    np.testing.assert_allclose(mc.x0, xi, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mc.draws, x2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mc.log_posterior, lp, rtol=1e-11)
    np.testing.assert_allclose(mc.acceptance_rate, r["acceptance"], rtol=1e-12)
    np.testing.assert_allclose(mc.proposal_chol, r["proposal_chol"], rtol=1e-10, atol=1e-16)


# ------------------------------------------------------------------ independent Metropolis-Hastings
@pytest.mark.parametrize("case", [6, 10])
def test_independent_mh_matches_dynare(load, case):
    """CASE 6: normal proposal, 2 chains x 400 draws (mh_jscale 1.2); CASE 10: Student proposal, 5 degrees of freedom."""
    r = ref(f"smp_case{case}.mat")
    est = load(case)
    x2, lp, xi = chains(r)
    mc = est.sample(replic=x2.shape[1], mode=est.mode(), rng="octave")
    assert mc.method == samplers.IMH
    np.testing.assert_allclose(mc.x0, xi, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mc.draws, x2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mc.log_posterior, lp, rtol=1e-11)
    np.testing.assert_allclose(mc.acceptance_rate, r["acceptance"], rtol=1e-12)


# ------------------------------------------------------------------ slice
def test_slice_from_prior_draws_matches_dynare(load):
    """CASE 1: two chains started from prior draws (Dynare's starting points are taken: Octave's gamrnd/betarnd are not
    seeded by Dynare), 40 iterations of the univariate slice sampler over 12 parameters within the estimation bounds.

    Chain 1 is reproduced draw for draw. Chain 2 is not compared: its first slice step moves stderr ea to 8.5e3 (inverse
    gamma prior with infinite variance; prior_trunc=1e-10 gives an upper bound of 2.3e4), where Dynare's and pydyn's
    Kalman filters agree to only 1e-3 to 20 in the log likelihood (Dynare also reports info 46 at some of those
    points), so the slice sampler, which compares kernel values with the slice level, goes elsewhere."""
    r = ref("smp_case1.mat")
    est = load(1)
    assert est.options["prior_trunc"] == 1e-10                 # dynare_estimation_init.m: slice needs bounded support
    np.testing.assert_allclose(est.lb, r["sampler_lb"], rtol=3e-8)
    np.testing.assert_allclose(est.ub, r["sampler_ub"], rtol=3e-8)
    lb, ub = est.prior.bounds(1e-10)
    np.testing.assert_allclose(lb, r["mh_lb"], rtol=3e-8)
    np.testing.assert_allclose(ub, r["mh_ub"], rtol=3e-8)
    x2, lp, xi = chains(r)
    mc = est.sample(replic=x2.shape[1], nblocks=1, x0=xi[:1], rng="octave", **dynare_bounds(r))
    assert mc.method == "slice" and np.all(mc.accepted)
    np.testing.assert_allclose(mc.draws[0], x2[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(mc.log_posterior[0], lp[0], rtol=1e-12)
    np.testing.assert_allclose(r["W1"], 0.8 * (r["sampler_ub"] - r["sampler_lb"]), rtol=1e-15)


def test_slice_from_prior_mode_matches_dynare(load):
    """CASE 2: one chain from the prior mode (bayestopt_.p5; uniform priors and modes outside the bounds take the prior
    mean), initial_step_size 0.5, 25 iterations; the number of function evaluations per iteration is Dynare's too."""
    r = ref("smp_case2.mat")
    est = load(2)
    x2, lp, xi = chains(r)
    p5 = samplers.prior_mode(est.prior)
    np.testing.assert_allclose(p5, r["p5"], rtol=1e-13)
    mc = est.sample(replic=x2.shape[1], rng="octave", **dynare_bounds(r))
    np.testing.assert_allclose(mc.x0, xi, rtol=1e-15)
    np.testing.assert_allclose(mc.draws, x2, rtol=0, atol=1e-13)
    np.testing.assert_allclose(mc.log_posterior, lp, rtol=1e-12)
    np.testing.assert_allclose(mc.neval.mean(axis=1), r["feval"], rtol=1e-15)
    np.testing.assert_allclose(r["W1"], 0.5 * (r["sampler_ub"] - r["sampler_lb"]), rtol=1e-15)


def test_rotated_slice_matches_dynare(load):
    """CASE 3: rotated slice initialised at the mode (mode file): directions and widths from the eigen decomposition
    of the inverse Hessian (Dynare's V1 to 1.3e-13), starting points drawn around the mode from the global seed, two
    chains x 30 iterations."""
    r = ref("smp_case3.mat")
    est = load(3)
    x2, lp, xi = chains(r)
    mode = est.mode()
    V1, WR = samplers.rotation_from_covariance(mode.inv_hessian)
    np.testing.assert_allclose(V1, r["V1"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(WR, r["WR"], rtol=1e-11)
    mc = est.sample(replic=x2.shape[1], mode=mode, rng="octave", **dynare_bounds(r))
    np.testing.assert_allclose(mc.x0, xi, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mc.draws, x2, rtol=0, atol=1e-12)            # measured 4.8e-15
    np.testing.assert_allclose(mc.log_posterior, lp, rtol=1e-12)
    np.testing.assert_allclose(mc.neval.mean(axis=1), r["feval"], rtol=1e-15)
    mc2 = est.sample(replic=8, mode=mode, x0=xi, rng="octave", rotation=(r["V1"], r["WR"]), **dynare_bounds(r))
    np.testing.assert_array_equal(mc2.draws, x2[:, :8])                     # with Dynare's rotation: bit for bit


def test_multimodal_slice_matches_dynare(load):
    """CASE 4: mode_files with two modes: a rotated step along the direction between the modes (randperm of the
    directions) then the univariate slice, 12 iterations."""
    r = ref("smp_case4.mat")
    est = load(4)
    x2, lp, xi = chains(r)
    mc = est.sample(replic=x2.shape[1], mode=est.mode(), rng="octave", **dynare_bounds(r))
    np.testing.assert_allclose(mc.x0, xi, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mc.draws, x2, rtol=0, atol=1e-13)
    np.testing.assert_allclose(mc.log_posterior, lp, rtol=1e-12)
    np.testing.assert_allclose(mc.neval.mean(axis=1), r["feval"], rtol=1e-15)


# ------------------------------------------------------------------ TaRB
@pytest.mark.parametrize("case", [13, 14])
def test_tarb_blocks_match_dynare(load, case):
    """Every TaRB block of Dynare's run (CASE 13: smp_nk, 4 iterations, 15 blocks; CASE 14: smp_schorfheide with a
    Student proposal and new_block_probability 0.5) replayed from Dynare's state: the current draw and the states of
    rand and randn before the block. Same blocks (randperm, rand), same acceptance decisions, same draw afterwards.

    Tolerances: the block's conditional mode is found by csminwel, whose path depends on the objective at trial points
    far in the tails (for instance the posterior kernel at 1115 against 283 at the optimum) where the two likelihood
    implementations agree to about 1e-9 relative; csminwel's stopping rule (improvement below 1e-7) then leaves optima
    that differ by up to 1.4e-5 (relative; measured), and the proposals by up to 2.6e-6. Where csminwel's paths
    differ it can also meet a "cliff" in one implementation and not in the other, and so use a uniform more or less:
    the random streams after such a block differ, so the uniform of the acceptance step, the decision and the streams are
    compared only where csminwel used as many evaluations as Dynare's."""
    r = ref(f"smp_case{case}.mat")
    est = load(case)
    opts = samplers.sampler_options(samplers.TARB, est.options["posterior_sampler_options"])
    js = est.jscale()[0]
    opt = samplers._tarb_optimizer(est, opts)
    nb = int(r["nblocks_logged"])
    agree = 0
    for k in range(1, nb + 1):
        g = lambda f: r[f"b{k}_{f}"]                                           # noqa: E731
        o = OctaveRandom()
        o.u.set_state(g("su"))
        o.n.set_state(g("sn"))
        idx = np.atleast_1d(g("idx")).astype(int) - 1
        cur, lpost, acc, rec = samplers.tarb_block(est, np.atleast_1d(g("current")), float(g("last_posterior")), idx,
                                                   r["sampler_lb"], r["sampler_ub"], r["mh_lb"], r["mh_ub"], js, opts, o,
                                                   opt)
        xo = np.atleast_1d(g("xopt"))
        np.testing.assert_allclose(rec["xopt"], xo, rtol=5e-5)
        np.testing.assert_allclose(rec["proposal"], np.atleast_1d(g("prop")), rtol=2e-5, atol=1e-9)
        if rec["fcount"] != int(g("fc")):
            continue                                       # csminwel's paths differ: uniforms may be used differently
        agree += 1
        if np.isfinite(g("accprob")):
            assert rec["u"] == g("u")
            np.testing.assert_allclose(rec["accprob"], g("accprob"), atol=2e-5)
        np.testing.assert_allclose(cur, np.atleast_1d(g("after")), rtol=2e-5, atol=1e-9)   # same decision
        if f"b{k + 1}_su" in r and sum(np.atleast_1d(r[f"b{j}_idx"]).size for j in range(1, k + 1)) % est.n != 0:
            np.testing.assert_array_equal(o.u.get_state(), r[f"b{k + 1}_su"])    # next block of the same iteration
            np.testing.assert_array_equal(o.n.get_state(), r[f"b{k + 1}_sn"])
    assert agree >= 0.25 * nb               # measured: 13 of 15 blocks (CASE 13), 4 of 16 (CASE 14)


def test_tarb_first_iteration_and_starting_points_match_dynare(load):
    """CASE 5 (12 draws) and CASE 9 (two chains): pydyn's TaRB chain from Dynare's seed follows Dynare's first
    iteration (to csminwel's accuracy, see above) and the starting points of several chains are Dynare's."""
    r = ref("smp_case5.mat")
    est = load(5)
    x2, lp, xi = chains(r)
    mc = est.sample(replic=1, mode=est.mode(), rng="octave")
    np.testing.assert_allclose(mc.x0, xi, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mc.draws[0, 0], x2[0, 0], rtol=1e-6)
    assert mc.method == samplers.TARB and 0 < mc.acceptance_rate[0] <= 1
    r9 = ref("smp_case9.mat")
    est9 = load(9)
    st = est9.sample(replic=0, mode=est9.mode(), rng="octave")
    np.testing.assert_allclose(st.x0, np.atleast_2d(r9["x_init"]), rtol=0, atol=1e-15)


# ------------------------------------------------------------------ sequential Monte Carlo
def test_hssmc_matches_dynare(load):
    """CASE 11 (300 particles, 8 stages). Dynare's initial particles are taken (prior draws: gamrnd/betarnd are not
    seeded by Dynare); after the initialisation Dynare's random numbers come from the seed of the last particle,
    which rng='octave' reproduces."""
    r = ref("smp_case11.mat")
    est = load(11)
    opts = samplers.sampler_options("hssmc", est.options["posterior_sampler_options"])
    res = samplers.hssmc(est, opts=opts, rng="octave", initial=r["p_first"])
    # the likelihoods agree to about 2e-11 (relative; Kalman filters with different operation orders) and the particles,
    # which depend on them only through accept/reject decisions and the weighted covariance, to 3e-10
    np.testing.assert_allclose(res.stages[0]["log_likelihood"], r["ll_first"], rtol=1e-10)
    mid = int(r["mid"]) - 1
    np.testing.assert_allclose(res.stages[mid]["particles"], r["p_mid"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.particles, r["p_last"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.log_posterior, r["lp_last"], rtol=1e-10)
    np.testing.assert_allclose(res.log_marginal_density, r["mdd"], rtol=1e-11)


def test_dsmh_matches_dynare(load):
    """CASE 12 (N=30, G=10, M=5, K=3, H=15, tau=2), from Dynare's initial particles."""
    r = ref("smp_case12.mat")
    est = load(12)
    opts = samplers.sampler_options("dsmh", est.options["posterior_sampler_options"])
    res = samplers.dsmh(est, opts=opts, rng="octave", initial=r["p_first"])
    np.testing.assert_allclose(res.particles, r["p_last"], rtol=1e-9, atol=1e-12)        # see test_hssmc_matches_dynare
    np.testing.assert_allclose(res.log_posterior, r["lp_last"], rtol=1e-10)
    np.testing.assert_allclose(res.log_marginal_density, r["mdd"], rtol=1e-11)


# ------------------------------------------------------------------ options, dispatch, refusals
def test_options_parsing_and_refusals(load):
    kv = samplers.read_key_value_string("''rotated'',1,''initial_step_size'',0.5,''proposal_distribution'',"
                                        "''rand_multivariate_student'',''mode'',[1 2; 3 4]")
    assert kv[:3] == [("rotated", 1), ("initial_step_size", 0.5), ("proposal_distribution", "rand_multivariate_student")]
    np.testing.assert_array_equal(kv[3][1], [[1, 2], [3, 4]])
    o = samplers.sampler_options("slice", "'rotated',1,'initial_step_size',0.3")
    assert o["rotated"] == 1 and o["initial_step_size"] == 0.3 and o["use_mh_covariance_matrix"] == 0
    o = samplers.sampler_options("slice", {"mode": np.array([[1.0, 2.0], [3.0, 4.0]])})
    assert o["rotated"] == 1 and o["WR"] is None and np.array_equal(o["mode"][1], [2.0, 4.0])
    o = samplers.sampler_options(samplers.TARB, "'new_block_probability',0.4,'mode_compute',4")
    assert o["new_block_probability"] == 0.4 and o["mode_compute"] == 4
    with pytest.warns(UserWarning, match=r"tarb_sampler: Unknown option \(rotated\)"):
        samplers.sampler_options(samplers.TARB, "'rotated',1")
    for method, spec, msg in [("slice", "'initial_step_size',1", "between 0 and 1"),
                              (samplers.TARB, "'new_block_probability',2", "between 0 and 1"),
                              (samplers.RWMH, "'proposal_distribution','rand_uniform'", "takes either"),
                              (samplers.IMH, "'student_degrees_of_freedom',0", "positive integer"),
                              ("dsmh", "'N',7,'G',3,'M',4", "M to divide"),
                              ("nonsense", None, "Unknown posterior_sampling_method")]:
        with pytest.raises(samplers.SamplerError, match=msg):
            samplers.sampler_options(method, spec)
    for method, msg in [("dime_mcmc", "DIME"), ("online", "online")]:
        with pytest.raises(NotImplementedError, match=msg):
            samplers.sampler_options(method)
    for spec in ("'draw_init_state_from_smoother',1", "'maximize',1"):
        with pytest.raises(NotImplementedError):
            samplers.sampler_options("slice", spec)
    with pytest.raises(samplers.SamplerError, match="needs priors|Rotated slice cannot start"):
        load(1).sample(replic=2, nblocks=1, sampler_options="'rotated',1", rng="octave")
    est = load(7)
    assert samplers.dispatch(est, {})                                        # Student proposal in the .mod file
    assert samplers.dispatch(load(8), {"rng": "octave"}) and not samplers.dispatch(load(8), {})
    with pytest.raises(TypeError, match="unexpected keyword"):
        load(8).sample(replic=1, nonsense=1)
    with pytest.raises(samplers.SamplerError, match="ArrayRandom"):
        est.sample(replic=1, innovations=np.zeros((2, 1, est.n)))
    with pytest.raises(samplers.SamplerError, match="cannot continue"):
        est.sample(replic=1, load=est.sample(replic=2, mode=est.mode(), seed=3), rng="octave")


def test_default_generators_save_and_continue(load, tmp_path):
    """numpy generators (rng=None): reproducible, chains continued from their last draws with load=..."""
    est = load(6)
    mode = est.mode()
    a = est.sample(replic=30, mode=mode, seed=4, save=tmp_path / "imh.npz")
    b = est.sample(replic=30, mode=mode, seed=4)
    np.testing.assert_array_equal(a.draws, b.draws)
    more = est.sample(replic=20, load=tmp_path / "imh.npz", mode=mode, seed=4)
    assert more.draws.shape == (2, 50, est.n)
    np.testing.assert_array_equal(more.draws[:, :30], a.draws)
    est1 = load(2)
    s = est1.sample(replic=3, seed=1)
    s2 = est1.sample(replic=2, load=s, seed=1)
    assert s2.draws.shape == (1, 5, est1.n) and np.all(np.isfinite(s2.log_posterior))
    lb, ub = est1.lb, est1.ub
    assert np.all((s2.draws >= lb) & (s2.draws <= ub))


# ------------------------------------------------------------------ independent runs against Dynare's posterior
def _dynare_posterior_chains():
    import importlib.util
    spec = importlib.util.spec_from_file_location("post_sampler_compare", REF / "post_sampler_compare.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    dyn = cmp.dynare_runs("post_schorfheide")
    return cmp.chain_stats(np.concatenate([b for _, b, _ in dyn], axis=0))


def _welch_min_p(a, b):
    """Smallest p-value of the Welch t statistics comparing the per-chain statistics a (chains x parameters) and b."""
    from scipy import stats
    va, vb = a.var(axis=0, ddof=1) / a.shape[0], b.var(axis=0, ddof=1) / b.shape[0]
    t = (a.mean(axis=0) - b.mean(axis=0)) / np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (a.shape[0] - 1) + vb ** 2 / (b.shape[0] - 1))
    return float(np.min(2 * stats.t.sf(np.abs(t), dof)))


@pytest.mark.parametrize("name", ["slice", "imh", "student"])
def test_independent_runs_agree_with_dynare_posterior(name):
    """Recorded runs (reference/smp_compare.py): pydyn's samplers with numpy random numbers on post_schorfheide
    (slice 6 chains x 150 iterations, independent MH and Student random walk 8 x 4000), against the posterior of
    Dynare's random-walk runs (r_posterior.m stages 3 and 4: 8 chains of 20000 draws; same posterior, another sampler).
    Chains are the replicates: the mean and standard deviation over the second half of every chain, compared with
    Welch t statistics (18 per sampler). Measured smallest p-values: slice 0.10, independent 0.14, Student 0.016; the
    threshold 0.001 is a Bonferroni-corrected 2 percent level."""
    d = _dynare_posterior_chains()
    with np.load(REF / f"smp_compare_{name}.npz") as z:
        p_min = min(_welch_min_p(z[key], d[key]) for key in ("mean", "std"))
        assert p_min > 0.001, p_min
        if name in ("imh", "student"):
            assert np.all((z["acceptance"] > 0.15) & (z["acceptance"] < 0.6))


def test_tarb_independent_runs_agree_with_dynare_tarb():
    """TaRB costs about 300 evaluations per iteration, so the recorded chains are short: pydyn 4 chains x 150
    iterations (reference/smp_compare.py), Dynare 2 x 2 chains x 150 iterations with seeds 1000 and 2000
    (smp_schorfheide.mod CASE 15, smp_tarb_seed*.mat); statistics over the second halves (75 iterations). Like is
    compared with like (pydyn's TaRB chains against Dynare's), and the means also against the long random-walk run.

    Measured: smallest p-values 0.023 (means) and 0.11 (standard deviations) against Dynare's TaRB, 0.18 for the means
    against the random-walk run. This is a weak test (4 + 4 chains): pydyn's within-chain standard deviations are on
    average 0.8 of the long run's (Dynare's TaRB chains: 0.94), and one of pydyn's chains (acceptance 0.27) did not move
    logmst, rho and del in its second half. The block-by-block replay (test_tarb_blocks_match_dynare) is the evidence
    that the algorithm is Dynare's; longer independent runs would settle the statistical question."""
    d = _dynare_posterior_chains()
    runs = [loadmat(REF / f"smp_tarb_seed{s}.mat", squeeze_me=True) for s in (1000, 2000)]
    dyn = {k: np.array([r[f"{k}{b}"] for r in runs for b in (1, 2)]) for k in ("mean", "std")}
    with np.load(REF / "smp_compare_tarb.npz") as z:
        for key in ("mean", "std"):
            assert _welch_min_p(z[key], dyn[key]) > 0.001
        assert _welch_min_p(z["mean"], d["mean"]) > 0.001


def test_hssmc_approaches_dynare_posterior_with_more_particles():
    """Herbst-Schorfheide SMC (exactly Dynare's algorithm, see test_hssmc_matches_dynare) is biased for small settings:
    4 runs of 1000 particles and 12 stages miss the posterior means by up to 1.5 posterior standard deviations and
    understate standard deviations by up to 45 percent; one run of 5000 particles and 25 stages is within 0.4 standard
    deviations and 15 percent (measured). Recorded runs of reference/smp_compare.py."""
    d = _dynare_posterior_chains()
    sd = d["std"].mean(axis=0)
    with np.load(REF / "smp_compare_hssmc.npz") as small, np.load(REF / "smp_compare_hssmc_big.npz") as big:
        err_small = np.abs(small["mean"].mean(axis=0) - d["mean"].mean(axis=0)) / sd
        err_big = np.abs(big["mean"].mean(axis=0) - d["mean"].mean(axis=0)) / sd
        ratio_big = big["std"].mean(axis=0) / sd
        assert err_big.max() < 0.5 and np.all((ratio_big > 0.8) & (ratio_big < 1.2))
        assert err_big.max() < err_small.max()
