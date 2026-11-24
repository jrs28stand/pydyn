"""Equilibrium selection for obc_method='news' (``obc_selection``): 'occbin' (default), 'shortest', 'longest', 'min_wedge'.

Benchmark: reference/obc_sw_zlb.mod (Smets-Wouters with a lower bound whose rule also switches its lag term; see
docs/OBC_PROGRESS.md): a -4 shock to ``eb`` from the steady state has exactly two single-spell solutions, binding for 9
or 13 periods.  Unique-solution model: reference/obc_zlb.mod.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import pydyn
from pydyn.obc import NewsOBC, RegimeOBC

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))
SELECTIONS = ("occbin", "shortest", "longest", "min_wedge")


def _load(tmp, name, extra=()):
    for f in (name, *extra):
        shutil.copy(REF / f, tmp)
    return pydyn.Model(pydyn.run_preprocessor(tmp / name))


@pytest.fixture(scope="module")
def sw(tmp_path_factory):
    m = _load(tmp_path_factory.mktemp("sel_sw"), "obc_sw_zlb.mod")
    return m, pydyn.Occbin(m)


@pytest.fixture(scope="module")
def zlb(tmp_path_factory):
    m = _load(tmp_path_factory.mktemp("sel_zlb"), "obc_zlb.mod")
    return m, pydyn.Occbin(m)


def _passes_occbin_test(ob, res, sp=0):
    """OccBin's own test on a returned path: no constraint binding (bind condition true) in the reference-regime periods,
    none released (relax condition true) in the alternative-regime periods, in the shock period's regime history."""
    (regime, start), = res.regime_history[sp]
    ind = np.zeros(res.piecewise.shape[0], bool)
    for r, s0, s1 in zip(regime, start, list(start[1:]) + [len(ind) + 1]):
        ind[s0 - 1:s1 - 1] = bool(r)
    binding, relax, _eb, _er = ob.check(res.piecewise)
    return not (binding[:, 0] & ~ind).any() and not (relax[:, 0] & ind).any()


def _spell_length(res, sp=0):
    (regime, start), = res.regime_history[sp]
    assert regime[0] == 1 and len(regime) == 2            # one spell, starting now
    return start[1] - 1


def test_sw_selection_on_the_two_equilibria(sw):
    m, ob = sw
    S = np.zeros((1, m.n_exo))
    S[0, m._ix["eb"]] = -4.0
    out = {sel: ob.simulate(S, periods=60, check_ahead_periods=60, obc_method="news", obc_selection=sel)
           for sel in SELECTIONS}
    occ = ob.simulate(S, periods=60, check_ahead_periods=100)
    assert {sel: _spell_length(r) for sel, r in out.items()} == {"occbin": 9, "shortest": 9, "longest": 13, "min_wedge": 9}
    for sel, r in out.items():
        assert _passes_occbin_test(ob, r), sel
        assert r.news["selection"] == sel and r.news["formulation"] == "regime"
        np.testing.assert_array_equal(r.news["selection_fallback"], [False])
    np.testing.assert_array_equal(out["occbin"].news["n_single_spell"], [-1])          # default: not looked for
    for sel in SELECTIONS[1:]:
        np.testing.assert_array_equal(out[sel].news["n_single_spell"], [2])
    # default and shortest are OccBin's path; longest is another equilibrium, far away
    np.testing.assert_allclose(out["occbin"].piecewise, occ.piecewise, atol=1e-9)
    np.testing.assert_allclose(out["shortest"].piecewise, occ.piecewise, atol=1e-9)
    assert np.abs(out["longest"].piecewise - occ.piecewise).max() > 1.0


def test_sw_default_is_unchanged(sw):
    """Without the keyword, or with obc_selection='occbin': identical to what obc_method='news' returned before."""
    m, ob = sw
    rng = np.random.default_rng(7)
    sd = np.sqrt(np.diag(m.Sigma_e))
    S = rng.normal(size=(120, m.n_exo)) * sd * 0.25
    a = ob.simulate(S, periods=120, obc_method="news", check_ahead_periods=60)
    b = ob.simulate(S, periods=120, obc_method="news", check_ahead_periods=60, obc_selection="occbin")
    np.testing.assert_array_equal(a.piecewise, b.piecewise)
    assert a.regime_history == b.regime_history
    assert (a.news["n_single_spell"] == -1).all() and not a.news["selection_fallback"].any()
    assert (a.news["wedges"].any(axis=1)).sum() >= 5            # the sample binds


def test_zlb_unique_solution_all_selections_equal_occbin(zlb):
    m, ob = zlb
    rng = np.random.default_rng(1)
    S = rng.normal(0, 0.02, (200, 1))
    occ = ob.simulate(S, periods=200)
    nb = None
    for sel in SELECTIONS:
        r = ob.simulate(S, periods=200, obc_method="news", obc_selection=sel)
        np.testing.assert_allclose(r.piecewise, occ.piecewise, atol=1e-11)
        assert r.regime_history == occ.regime_history
        if sel != "occbin":
            n = r.news["n_single_spell"]
            assert set(np.unique(n)) <= {-1, 1}                   # unique solution wherever something binds
            nb = int((n == 1).sum())
            assert not r.news["selection_fallback"].any()
    assert nb > 20                                                # the sample binds


def test_selection_fallback_is_flagged(zlb, monkeypatch):
    """No single-spell solution: the default solver's answer stands and the period is flagged."""
    m, ob = zlb
    S = np.full((5, 1), -0.05)
    base = ob.simulate(S, periods=20, obc_method="news")
    monkeypatch.setattr(NewsOBC, "single_spells", lambda self, q, l_max=None, tol=1e-8: [])
    r = ob.simulate(S, periods=20, obc_method="news", obc_selection="longest")
    np.testing.assert_array_equal(r.piecewise, base.piecewise)
    assert r.news["selection_fallback"][base.news["wedges"].any(axis=1)].all()
    assert (r.news["n_single_spell"][r.news["selection_fallback"]] == 0).all()


def test_errors(tmp_path, zlb):
    m, ob = zlb
    S = np.array([[-0.05]])
    for bad in ("fastest", "", None, 1):
        with pytest.raises(ValueError, match="unknown obc_selection"):
            ob.simulate(S, obc_method="news", obc_selection=bad)
    with pytest.raises(ValueError, match="needs obc_method='news'"):
        ob.simulate(S, obc_selection="longest")                     # OccBin has no choice to make
    with pytest.raises(ValueError, match="unknown obc_selection"):
        ob.simulate(S, obc_selection="fastest")
    with pytest.raises(ValueError, match="unknown obc_selection"):
        NewsOBC(ob).simulate(S, selection="fastest")
    m2 = _load(tmp_path, "obc_zlb2.mod")                            # two constraints
    ob2 = pydyn.Occbin(m2)
    with pytest.raises(ValueError, match="one constraint only"):
        ob2.simulate(S, periods=20, obc_method="news", obc_selection="shortest")
    ob2.simulate(S, periods=20, obc_method="news", obc_selection="occbin")      # the default is fine
    with pytest.raises(ValueError, match="one constraint only"):
        RegimeOBC(ob2).simulate(S, selection="min_wedge")


def test_selection_from_mod_file_and_irf(tmp_path, sw):
    """``options_.occbin.obc_selection`` (native line) reaches simulate through mod_options and run(); irf_occbin passes
    the keyword on."""
    src = (REF / "obc_sw_zlb.mod").read_text()
    (tmp_path / "obc_sw_sel.mod").write_text(
        src + "\noptions_.occbin.obc_method = 'news';\noptions_.occbin.obc_selection = 'longest';\n")
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "obc_sw_sel.mod"))
    ob = pydyn.Occbin(m)
    kw = ob.mod_options()
    assert kw["obc_method"] == "news" and kw["obc_selection"] == "longest"
    m0, ob0 = sw
    S = np.zeros((1, m0.n_exo))
    S[0, m0._ix["eb"]] = -4.0
    kw.update(periods=40, check_ahead_periods=60)
    a = ob.simulate(S, **kw)
    assert _spell_length(a) == 13
    ir = ob0.irf_occbin(periods=30, shocksize=4.0, exo_names=["eb"], shocksigns=("neg",), obc_method="news",
                        obc_selection="longest", check_ahead_periods=60)
    ir0 = ob0.irf_occbin(periods=30, shocksize=4.0, exo_names=["eb"], shocksigns=("neg",), obc_method="news",
                         check_ahead_periods=60)
    assert np.abs(ir[("eb", "neg")]["piecewise"] - ir0[("eb", "neg")]["piecewise"]).max() > 1.0


def test_minnorm_is_news_with_min_wedge(tmp_path, sw):
    """``obc_method='minnorm'`` (JDyn's name) is ``'news'`` with ``obc_selection='min_wedge'``: identical results,
    also from the native mod-file line and in irf_occbin; another explicit selection is refused."""
    m0, ob0 = sw
    S = np.zeros((1, m0.n_exo))
    S[0, m0._ix["eb"]] = -4.0
    a = ob0.simulate(S, periods=40, check_ahead_periods=60, obc_method="minnorm")
    b = ob0.simulate(S, periods=40, check_ahead_periods=60, obc_method="news", obc_selection="min_wedge")
    c = ob0.simulate(S, periods=40, check_ahead_periods=60, obc_method="news")
    assert np.array_equal(a.piecewise, b.piecewise) and a.regime_history == b.regime_history
    assert a.news["selection"] == "min_wedge" and _spell_length(a) == 9 == _spell_length(c)
    ob0.simulate(S, periods=40, check_ahead_periods=60, obc_method="minnorm", obc_selection="min_wedge")
    with pytest.raises(ValueError, match="minnorm"):
        ob0.simulate(S, periods=40, obc_method="minnorm", obc_selection="longest")
    src = (REF / "obc_sw_zlb.mod").read_text()
    (tmp_path / "obc_sw_mn.mod").write_text(src + "\noptions_.occbin.obc_method = 'minnorm';\n")
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "obc_sw_mn.mod"))
    kw = pydyn.Occbin(m).mod_options()
    kw.update(periods=40, check_ahead_periods=60)
    d = pydyn.Occbin(m).simulate(S, **kw)
    assert np.array_equal(d.piecewise, a.piecewise) and d.news["selection"] == "min_wedge"
    ir = ob0.irf_occbin(periods=30, shocksize=4.0, exo_names=["eb"], shocksigns=("neg",), obc_method="minnorm",
                        check_ahead_periods=60)
    ir1 = ob0.irf_occbin(periods=30, shocksize=4.0, exo_names=["eb"], shocksigns=("neg",), obc_method="news",
                         obc_selection="min_wedge", check_ahead_periods=60)
    assert np.array_equal(ir[("eb", "neg")]["piecewise"], ir1[("eb", "neg")]["piecewise"])
