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
from pydyn.obc import NewsOBC
from pydyn.obc_ho import HigherOrderNews
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
    c = HigherOrderNews(m, order=2, cubature=True, qmc_level=7).simulate(S)
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
    for trial in range(30):
        A = rg.normal(size=(n, n)) * 0.3
        M = np.vstack([np.eye(n) + np.tril(A), rg.normal(size=(6, n)) * 0.2])        # n_s = n complementarity rows + 6 extra
        Q = rg.normal(size=(n + 6, 40)) + 0.6
        cache = {}
        Y, ok = solve_active_set_batch(M, Q, n, cache=cache)
        Y2, ok2 = solve_active_set_batch(M, Q, n, cache=cache)                         # warm cache gives the same
        np.testing.assert_array_equal(ok, ok2)
        for j in range(Q.shape[1]):
            y = solve_active_set(M, Q[:, j], n)
            assert (y is not None) == bool(ok[j])
            if y is not None:
                np.testing.assert_allclose(Y[:, j], y, rtol=0, atol=1e-11)
                np.testing.assert_allclose(Y2[:, j], y, rtol=0, atol=1e-11)
