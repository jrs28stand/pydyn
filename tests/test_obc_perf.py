"""Checks on the fast paths of pydyn.obc (docs/OBC_PERF.md): the per-period step without a full path, q and the
next state as maps from the state variables only, the O(T) construction of M and the wedge responses, the
cached post-processor.  Each is compared with the slower construction it replaces (kept as the lazily built
``Yw``, ``base_path`` and ``stacked_M``)."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import pydyn
from pydyn.obc import NewsOBC, RegimeOBC, constraint_objects
from pydyn.occbin import Occbin

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))
EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
DYNARE_TESTS = Path(os.environ.get("PYDYN_DYNARE_TESTS", str(EXAMPLES.parent / "tests")))


def _load(tmp, name):
    shutil.copy(REF / name, tmp)
    return pydyn.Model(pydyn.run_preprocessor(tmp / name))


@pytest.fixture(scope="module")
def zlb2(tmp_path_factory):
    m = _load(tmp_path_factory.mktemp("perf_zlb2"), "obc_zlb2.mod")
    return m, Occbin(m)


@pytest.fixture(scope="module")
def nkm(tmp_path_factory):
    src = DYNARE_TESTS / "occbin" / "filter"
    if not src.exists():
        pytest.skip("Dynare's tests folder not found")
    work = tmp_path_factory.mktemp("perf_nkm") / "filter"
    shutil.copytree(src, work)
    m = pydyn.Model(pydyn.run_preprocessor(work / "NKM_0_std_shocks.mod"))
    return m, Occbin(m)


def _random_inputs(m, seed, k=8, scale=0.05):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, scale, m.n_endo), rng.normal(0, scale, m.n_exo)) for _ in range(k)]


def test_M_of_the_shift_recursion_matches_stacked_system(zlb2, nkm):
    """M from the diagonal recursion against one stacked linear system, with one and with two constraints."""
    for _, ob in (zlb2, nkm):
        nb = NewsOBC(ob, horizon=12, check_ahead=10)
        np.testing.assert_allclose(nb.Mo, nb.stacked_M(pad=60), atol=1e-10)


def test_wedge_responses_and_path_match_the_full_recursion(zlb2, nkm):
    """The path from the backward/forward recursion equals base path + Yw y, and the step gives its first period."""
    for m, ob in (zlb2, nkm):
        for cls in (NewsOBC, RegimeOBC):
            obj = cls(ob, horizon=15, check_ahead=10)
            rng = np.random.default_rng(1)
            for init, shock in _random_inputs(m, 2):
                w = rng.normal(0, 0.1, obj.n_c * obj.T if cls is NewsOBC else len(obj.cols) * obj.T)
                full = obj.base_path(init, shock) + obj.Yw @ w
                np.testing.assert_allclose(obj._path(init, shock, w), full, atol=1e-11)
                np.testing.assert_allclose(obj._path(init, shock, w, upto=3), full[:4], atol=1e-11)


def test_state_only_q_and_next_state(zlb2, nkm):
    """q as a map from the state variables and the shocks equals q from all variables; the dropped columns are
    numerically zero."""
    for m, ob in (zlb2, nkm):
        nb = NewsOBC(ob, horizon=20, check_ahead=20)
        assert 0 < len(nb._st) <= m.n_endo
        dropped = np.setdiff1d(np.arange(m.n_endo), nb._st)
        assert np.abs(nb.Qm[:, dropped]).max(initial=0.0) < 1e-12 and np.abs(ob.P[:, dropped]).max(initial=0.0) < 1e-12
        for init, shock in _random_inputs(m, 3):
            q = nb.q0 + nb._Qx @ np.concatenate([init[nb._st], shock])
            np.testing.assert_allclose(q, nb.q_of(init, shock)[0], atol=1e-13)
            y, nxt, _ = nb._step(init, shock)
            y2, path, _ = nb.solve_period(init, shock)
            np.testing.assert_allclose(y, y2, atol=1e-13)
            np.testing.assert_allclose(nxt, path[0], atol=1e-12)


def test_regime_step_returns_first_period_of_the_path(zlb2, nkm):
    for m, ob in (zlb2, nkm):
        rb = RegimeOBC(ob, horizon=20, check_ahead=20)
        for init, shock in _random_inputs(m, 4, scale=0.1):
            z, w, nxt, _ = rb._step(init, shock)
            z2, w2, path, _ = rb.solve_period(init, shock)
            assert (z == z2).all()
            np.testing.assert_allclose(nxt, path[0], atol=1e-12)


def test_constraint_objects_cached_per_occbin_instance(zlb2):
    _, ob = zlb2
    a, b = constraint_objects(ob), constraint_objects(ob)
    assert all(x is y for x, y in zip(a, b))
    assert all(x is not y for x, y in zip(a, constraint_objects(ob, tol=1e-5)))


def test_simulation_with_news_beyond_the_shocks_uses_the_last_period_path(zlb2):
    """periods > number of shocks: the continuation after the last shock comes from the single path computed once."""
    m, ob = zlb2
    rng = np.random.default_rng(5)
    shocks = rng.normal(0, 0.04, (30, 1))
    a = ob.simulate(shocks, periods=90, check_ahead_periods=80)
    for cls in (NewsOBC, RegimeOBC):
        b = cls(ob, horizon=40, check_ahead=40).simulate(shocks, periods=90)
        np.testing.assert_allclose(b.piecewise, a.piecewise, atol=1e-11)
        assert b.regime_history == a.regime_history
