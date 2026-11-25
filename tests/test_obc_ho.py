"""Order-2 and cubature prototype of the news-shock OBC method (pydyn.obc_ho): internal-consistency checks.

No DynareOBC reference exists for these (it needs MATLAB and Dynare 4.5). The checks are: order 1 reproduces
NewsOBC; the base simulation is pydyn's pruned simulation (validated against Dynare 7.2); the expected path
equals the Monte Carlo mean of pruned simulations; the cubature covariance equals the sample covariance; cubature
with vanishing shock variance equals no cubature. Model: Dynare's NKM OccBin estimation example.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import pydyn
from pydyn.obc import NewsOBC, OBCError, solve_active_set
from pydyn.obc_ho import HigherOrderNews as _HigherOrderNews


def HigherOrderNews(*a, **k):
    """These tests are about the LCP implementation (NKM is approx class, for which 'auto' is now the regime formulation)."""
    k.setdefault("obc_formulation", "lcp")
    return _HigherOrderNews(*a, **k)


from pydyn.simulate import simulate

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))


@pytest.fixture(scope="module")
def nkm(tmp_path_factory):
    d = tmp_path_factory.mktemp("obc_ho")
    shutil.copy(REF / "occest_nkm.mod", d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(d / "occest_nkm.mod"))
    return m, np.random.default_rng(4).normal(size=(300, m.n_exo))


def test_order_one_equals_news(nkm):
    m, S = nkm
    h = HigherOrderNews(m, order=1)
    a = h.simulate(S)
    b = NewsOBC(h.ob, horizon=40, check_ahead=40).simulate(S, periods=300)
    np.testing.assert_allclose(a["with_bound"], b.piecewise, atol=1e-11)
    assert (a["wedges"].max(axis=1) > 1e-10).sum() >= 10


def test_order_two_base_is_pruned_simulation(nkm):
    m, S = nkm
    h = HigherOrderNews(m, order=2)
    r = h.simulate(S)
    np.testing.assert_allclose(r["without_bound"], simulate(h.dr, S, order=2, pruning=True), atol=1e-11)
    assert (r["wedges"].max(axis=1) > 1e-10).sum() >= 10
    small = h.simulate(S * 0.05)                                  # the bound is never reached
    assert not small["wedges"].any()
    np.testing.assert_array_equal(small["with_bound"], small["without_bound"])


def test_expected_path_equals_monte_carlo_mean(nkm):
    m, S = nkm
    h = HigherOrderNews(m, order=2)
    n = m.n_endo
    y1, y2 = np.zeros(n), np.zeros(n)

    def step(y1, y2, u):
        x1, x2 = y1[h.ist], y2[h.ist]
        y2 = h.ghx @ x2 + 0.5 * (h.ghxx @ np.kron(x1, x1) + h.ghuu @ np.kron(u, u) + 2 * h.ghxu @ np.kron(x1, u) + h.ghs2)
        return h.ghx @ x1 + h.ghu @ u, y2

    for t in range(151):
        y1, y2 = step(y1, y2, S[t])
    path = h.expected_path(y1, y2, np.zeros(n))
    K, R = 8, 4000
    rg = np.random.default_rng(9)
    acc, acc2 = np.zeros((K, n)), np.zeros((K, n))
    for _ in range(R):
        z1, z2 = y1, y2
        for k in range(K):
            z1, z2 = step(z1, z2, rg.normal(size=m.n_exo))
            tot = np.empty(n)
            tot[h.ov] = z1 + z2
            acc[k] += tot
            acc2[k] += tot * tot
    mean = acc / R
    se = np.sqrt(np.maximum(acc2 / R - mean ** 2, 0) / R)
    z = np.abs(mean - path[1:K + 1]) / np.maximum(se, 1e-12)
    assert z.max() < 4.5                                           # 8 x 24 comparisons


def test_cubature_limits_and_covariance(nkm):
    m, S = nkm
    tiny = m.Sigma_e * 1e-12
    c = HigherOrderNews(m, order=2, cubature=True, Sigma_e=tiny, qmc_level=5).simulate(S[:100])
    b = HigherOrderNews(m, order=2, Sigma_e=tiny).simulate(S[:100])
    np.testing.assert_allclose(c["with_bound"], b["with_bound"], atol=1e-9)
    h = HigherOrderNews(m, order=1, cubature=True, taper=False, periods_of_uncertainty=16, qmc_level=5,
                        pruning_cutoff=None)
    Gq, N = h.news.Qm[:, m.n_endo:], h.news.N
    cov = np.zeros((N, N))                                         # exact covariance of the gap path
    for i in range(1, 17):
        B = np.zeros((N, m.n_exo))
        B[i:] = Gq[:N - i]
        cov += B @ m.Sigma_e @ B.T
    np.testing.assert_allclose(h.R @ h.R.T, cov, atol=1e-12 * max(1.0, np.abs(cov).max()))
    w = HigherOrderNews(m, order=1, cubature=True, periods_of_uncertainty=16, qmc_level=5)
    assert np.trace(w.R @ w.R.T) < np.trace(cov)                   # the cosine taper shrinks it


def test_cubature_makes_agents_react_before_the_bound(nkm):
    m, S = nkm
    a = HigherOrderNews(m, order=2).simulate(S)
    c = HigherOrderNews(m, order=2, cubature=True, qmc_level=7, regions=1).simulate(S)
    assert (c["wedges"].max(axis=1) > 1e-10).sum() > 3 * (a["wedges"].max(axis=1) > 1e-10).sum()


# ---------------------------------------------------------------------------------------- performance work
def test_matches_stored_baseline():
    """Paths equal those of the unperturbed prototype (commit f173780) stored in tests/data/obc_ho_baseline.npz
    (made by ``bench/obc_ho_bench.py baseline`` with the unmodified code): nk and nkm, orders 1 and 2, cubature at
    qmc_level 9 and the default."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import obc_ho_bench as b
    base = np.load(Path(__file__).resolve().parent / "data" / "obc_ho_baseline.npz")
    new = b.paths("after")
    assert set(new) == set(base.files)
    for k in base.files:
        np.testing.assert_allclose(new[k], base[k], rtol=0, atol=1e-13, err_msg=k)


def test_precomputed_gap_equals_expected_path(nkm):
    m, S = nkm
    for order in (1, 2):
        h = HigherOrderNews(m, order=order)
        rg = np.random.default_rng(1)
        n = m.n_endo
        for _ in range(5):
            y1, y2, o, prev = (0.01 * rg.normal(size=n) for _ in range(4))
            if order == 1:
                y2 = np.zeros(n)
            u = 0.01 * rg.normal(size=m.n_exo)
            q0 = h.gap_path(h.expected_path(y1, y2, o), prev, u)
            np.testing.assert_allclose(h.gap_from_state(y1, y2, o, prev, u), q0, rtol=0, atol=1e-13)


def test_regions_match_stored_baseline():
    """regions='all' and regions=k (levels 9; both models) against paths of the code before the batch solver
    (tests/data/obc_ho_regions_baseline.npz, ``bench/obc_ho_bench.py baseline --regions``)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import obc_ho_bench as b
    base = np.load(Path(__file__).resolve().parent / "data" / "obc_ho_regions_baseline.npz")
    new = b.rpaths("after", prefix="L9")
    assert new and all(k in base.files for k in new)
    for k in new:
        np.testing.assert_allclose(new[k], base[k], rtol=0, atol=1e-12, err_msg=k)
    assert any("rall" in k for k in new)


def test_batch_active_set_equals_scalar():
    """solve_active_set_batch returns, column by column, what solve_active_set returns (and fails where it fails)."""
    from pydyn.obc import solve_active_set, solve_active_set_batch
    rg = np.random.default_rng(3)
    n = 12
    n_ok = [0]
    for trial in range(30):
        A = rg.normal(size=(n, n)) * 0.3
        M = np.vstack([np.eye(n) + np.tril(A), rg.normal(size=(6, n)) * 0.2])        # n_s = n complementarity rows + 6 extra
        Q = rg.normal(size=(n + 6, 40)) * (0.5 if trial % 2 else 1.0) + (0.6 if trial % 3 else 0.1)
        cache = {}
        Y, ok = solve_active_set_batch(M, Q, n, cache=cache)
        Y2, ok2 = solve_active_set_batch(M, Q, n, cache=cache)                         # warm cache gives the same
        np.testing.assert_array_equal(ok, ok2)
        n_ok[0] += int(ok.sum())
        for j in range(Q.shape[1]):
            y = solve_active_set(M, Q[:, j], n)
            assert (y is not None) == bool(ok[j])
            if y is not None:
                np.testing.assert_allclose(Y[:, j], y, rtol=0, atol=1e-11)
                np.testing.assert_allclose(Y2[:, j], y, rtol=0, atol=1e-11)
    assert n_ok[0] > 200                                       # the comparison is not only about failures


# ------------------------------------------------------------------------------------ several constraints
@pytest.fixture(scope="module")
def zlb2(tmp_path_factory):
    d = tmp_path_factory.mktemp("obc_zlb2")
    shutil.copy(REF / "obc_zlb2.mod", d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(d / "obc_zlb2.mod"))
    return m, np.random.default_rng(1).normal(size=(150, m.n_exo)) * 0.02


def test_two_constraints_order_one_equals_news(zlb2):
    m, S = zlb2
    h = HigherOrderNews(m, order=1, horizon=20, check_ahead=20)
    assert h.n_c == 2
    a = h.simulate(S)
    b = NewsOBC(h.ob, horizon=20, check_ahead=20).simulate(S, periods=150)
    np.testing.assert_allclose(a["with_bound"], b.piecewise, atol=1e-11)
    w = a["wedges"]
    assert (w[:, :20] > 1e-10).any(axis=1).sum() > 10 and (w[:, 20:] > 1e-10).any(axis=1).sum() > 10


def test_two_constraints_order_two_and_gap_map(zlb2):
    m, S = zlb2
    h = HigherOrderNews(m, order=2, horizon=20, check_ahead=20)
    r = h.simulate(S)
    np.testing.assert_allclose(r["without_bound"], simulate(h.dr, S, order=2, pruning=True), atol=1e-11)
    rg = np.random.default_rng(2)
    n = m.n_endo
    for _ in range(4):
        y1, y2, o, prev = (0.01 * rg.normal(size=n) for _ in range(4))
        u = 0.01 * rg.normal(size=m.n_exo)
        np.testing.assert_allclose(h.gap_from_state(y1, y2, o, prev, u), h.gap_path(h.expected_path(y1, y2, o), prev, u),
                                   rtol=0, atol=1e-13)


def test_two_constraints_cubature_and_batch_equal_scalar(zlb2):
    m, S = zlb2
    tiny = m.Sigma_e * 1e-12
    c = HigherOrderNews(m, order=2, horizon=20, check_ahead=20, cubature=True, Sigma_e=tiny, qmc_level=5).simulate(S[:60])
    b = HigherOrderNews(m, order=2, horizon=20, check_ahead=20, Sigma_e=tiny).simulate(S[:60])
    np.testing.assert_allclose(c["with_bound"], b["with_bound"], atol=1e-9)
    h = HigherOrderNews(m, order=2, horizon=20, check_ahead=20, cubature=True, qmc_level=6, regions="all",
                         cubature_undecided_tolerance=0)
    assert h.R.shape[0] == 2 * h.N
    qs, h1 = [], HigherOrderNews(m, order=2, horizon=20, check_ahead=20)       # realistic gap paths from a simulation
    orig = h1.news_shocks
    h1.news_shocks = lambda q: (qs.append(q.copy()), orig(q))[1]
    h1.simulate(S)
    seen = checked = failed = 0
    for q in qs:
        idx = h._relevant_points(q)
        if idx.size == 0:
            continue
        seen += 1
        pr = q[:, None] + h.R @ h.xi[:, idx]
        try:
            ref = sum(h._solve(pr[:, j]) for j in range(idx.size)) / h.xi.shape[1]
        except OBCError:                                             # a perturbed path without solution: both routes raise
            with pytest.raises(OBCError):
                h.news_shocks(q)
            failed += 1
            continue
        np.testing.assert_allclose(h.news_shocks(q), ref, rtol=0, atol=1e-11)
        checked += 1
    assert checked >= 3 and failed >= 1                      # some perturbed paths have no solution: both routes raise


def test_batch_cache_is_bounded_and_harmless():
    from pydyn.obc import solve_active_set_batch
    rg = np.random.default_rng(7)
    n = 10
    M = np.vstack([np.eye(n) + np.tril(rg.normal(size=(n, n)) * 0.3), rg.normal(size=(4, n)) * 0.2])
    Q = rg.normal(size=(n + 4, 300)) * 0.5 + 0.2
    Y0, ok0 = solve_active_set_batch(M, Q, n, cache={})
    cache = {}
    Y1, ok1 = solve_active_set_batch(M, Q, n, cache=cache, max_cache_bytes=2000)           # forces clearing at every pass
    np.testing.assert_array_equal(ok0, ok1)
    np.testing.assert_allclose(Y0[:, ok0], Y1[:, ok0], rtol=0, atol=1e-12)
    assert cache.get("_bytes", 0) < 200_000


# ------------------------------------------------------------------ cubature defaults and failure handling
from pydyn.obc_ho import CubatureFailure  # noqa: E402


@pytest.fixture(scope="module")
def borrcon(tmp_path_factory):
    d = tmp_path_factory.mktemp("borrcon")
    shutil.copy(REF / "occ_borrcon.mod", d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(d / "occ_borrcon.mod"))
    sd = np.sqrt(np.diag(m.Sigma_e))
    return m, np.random.default_rng(4).normal(size=(30, m.n_exo)) * sd * 0.1


@pytest.fixture(scope="module")
def sw_cp():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import obc_ho_scale as sc
    m = sc.sw_cp_model()
    sd = np.sqrt(np.diag(m.Sigma_e))
    return m, np.random.default_rng(4).normal(size=(30, m.n_exo)) * sd * 0.25


def _qs(m, S, horizon):
    qs, h1 = [], HigherOrderNews(m, order=2, horizon=horizon, check_ahead=horizon)
    orig = h1.news_shocks
    h1.news_shocks = lambda q: (qs.append(q.copy()), orig(q))[1]
    h1.simulate(S)
    return qs


def test_cubature_defaults(nkm):
    m, _ = nkm
    h = HigherOrderNews(m, order=2, cubature=True)
    assert h.regions == "all" and h.qmc_level == 11 and h.xi.shape[1] == 2 ** 12 - 1
    assert h.failure_tolerance == 0.0 and h.milp_time_limit == 1.0
    assert HigherOrderNews(m, order=2).cubature is False


def test_m_diagnostic_warns_only_on_non_s_matrix(nkm, sw_cp):
    m, _ = nkm
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        h = HigherOrderNews(m, order=1, cubature=True, qmc_level=3)          # nkm: S-matrix, no warning
    assert h.m_diag["is_s"] is True and h.m_diag["level"] == "s" and h.m_diag["seconds"] >= 0
    assert HigherOrderNews(m, order=1, cubature=True, qmc_level=3).m_diag is h.m_diag      # cached per parameter set
    assert HigherOrderNews(m, order=1, cubature=True, qmc_level=3, m_diagnostic=False).m_diag is None
    assert HigherOrderNews(m, order=1, qmc_level=3).m_diag is None           # no cubature: not computed
    with pytest.warns(UserWarning, match="not an S-matrix"):
        w = HigherOrderNews(sw_cp[0], order=1, cubature=True, qmc_level=3)
    assert w.m_diag["is_s"] is False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert HigherOrderNews(m, order=1, cubature=True, qmc_level=3, m_diagnostic="full").m_diag["level"] == "full"


def test_raise_by_default_with_classified_counts(borrcon, sw_cp):
    m, S = borrcon                                    # failures are class (b): the check-ahead rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = HigherOrderNews(m, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=5, cubature_undecided_tolerance=0)
        with pytest.raises(CubatureFailure) as ei:
            h.simulate(S)
        e = ei.value
        assert isinstance(e, OBCError) and e.period >= 1
        assert e.n_points == 63 and 0 < e.n_relevant <= 63 and e.lower_bound and e.n_failed >= 1
        assert sum(e.counts.values()) == e.n_failed and e.counts["b"] >= 1 and e.status == "horizon_too_short"
        assert e.n_active_set_failed >= e.n_failed and any("longer `horizon`" in a for a in e.advice)
        # the advice works here: with a longer horizon no point fails
        HigherOrderNews(m, order=2, horizon=96, check_ahead=96, cubature=True, qmc_level=5).simulate(S[:10])
        m2, S2 = sw_cp                                 # class (a): proved no solution
        h2 = HigherOrderNews(m2, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=5)
        with pytest.raises(CubatureFailure) as e2:
            h2.simulate(S2)
        assert e2.value.counts["a"] >= 1 and e2.value.status == "no_solution" and e2.value.period >= 1


def test_tolerance_drops_and_renormalises(borrcon):
    m, S = borrcon
    kw = dict(order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=5, m_diagnostic=False, cubature_undecided_tolerance=0)
    h = HigherOrderNews(m, cubature_failure_tolerance=0.9, **kw)
    with pytest.warns(UserWarning, match="dropped") as rec:
        r = h.simulate(S[:12])
    assert sum("dropped" in str(w.message) for w in rec) == 1                  # one warning per simulation
    assert r["dropped"].sum() > 0 and r["dropped"].sum() == h.stats["dropped_points"]
    assert (r["dropped"] <= 0.9 * r["relevant"]).all() and r["relevant"].max() <= 63
    qs = _qs(m, S[:12], 32)
    checked = 0
    for q in qs:                                                              # weights: sum over solved / (points - failed)
        idx = h._relevant_points(q)
        if not idx.size:
            continue
        pr = q[:, None] + h.R @ h.xi[:, idx]
        out = [h._solve_unit(pr[:, j]) for j in range(idx.size)]
        nf = sum(c is not None for _, c in out)
        if nf == 0 or nf > 0.9 * idx.size:
            continue
        expect = sum(y for y, c in out if c is None) / (63 - nf)
        np.testing.assert_allclose(h.news_shocks(q), expect, rtol=0, atol=1e-11)
        assert h._dropped_last == nf
        checked += 1
    assert checked >= 2
    # the failed share of relevant points above the tolerance raises, with the exceeded count as a lower bound
    h2 = HigherOrderNews(m, cubature_failure_tolerance=0.02, **kw)
    with pytest.raises(CubatureFailure) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h2.simulate(S)
    assert ei.value.tolerance == 0.02 and ei.value.n_failed > 0.02 * ei.value.n_relevant
    with pytest.raises(ValueError):
        HigherOrderNews(m, cubature_failure_tolerance=1.0, **kw)


def test_milp_time_limit_gives_undecided_class(borrcon):
    m, S = borrcon
    h = HigherOrderNews(m, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=5, m_diagnostic=False, cubature_undecided_tolerance=0)
    q = _qs(m, S[:3], 32)[0]
    idx = h._relevant_points(q)
    pr = q[:, None] + h.R @ h.xi[:, idx]
    p = next(pr[:, j] for j in range(idx.size) if solve_active_set(h.news.Mo, pr[:, j], h.nws) is None)
    assert h._classify(p) == (None, "b")                                      # the LP proof
    h._lp_gap = lambda p, rows: 0.0                                           # pretend the LP is feasible: force the MILP
    h.milp_time_limit = 1e-6
    t = __import__("time").perf_counter()
    assert h._classify(p) == (None, "c")
    assert __import__("time").perf_counter() - t < 0.5
    h.milp_time_limit = 10.0
    assert h._classify(p) == (None, "b")                                      # MILP proof on the first n_s rows
    h.milp_time_limit = 1e-6                                                  # (c) counts as a failure
    h.failure_tolerance = 0.0
    with pytest.raises(CubatureFailure) as ei:
        h.news_shocks(q)
    assert ei.value.counts["c"] >= 1 and ei.value.status == "not_converged"
