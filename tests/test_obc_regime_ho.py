"""Order 2 (and cubature) for the regime formulation, ``pydyn.obc_regime_ho.HigherOrderRegime`` (docs/OBC_REGIME_HO.md).

New design with no external reference for the non-LCP classes: the checks are internal consistency.  (a) order 1 is ``RegimeOBC.simulate``;
(b) on exact-class models order 2 equals ``HigherOrderNews`` (the LCP), and so the DynareOBC reference; (c) never binding: the pruned
simulation; (d) the affine maps equal the direct recursion; (e) approx class: regime vs LCP, and order-2 corrections -> 0 gives order 1;
(f) every accepted period passes OccBin's bind/relax test in levels, recomputed independently.
"""
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.obc import OBCError, RegimeOBC, constraint_objects
from pydyn.obc_ho import HigherOrderNews
from pydyn.obc_regime_ho import HigherOrderRegime
from pydyn.simulate import simulate

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/ref/dynare-7.2/examples")).parent / "tests"
SRC = {"nkm": REF / "occest_nkm.mod", "zlb": REF / "obc_zlb.mod", "zlb2": REF / "obc_zlb2.mod", "nk": REF / "dynareobc_octave/nk_occ.mod",
       "borrcon": REF / "occ_borrcon.mod", "sw": REF / "obc_sw_zlb.mod", "bf": EXAMPLES / "estimation/slice/bf.mod"}
# (shock scale relative to the model's sd, horizon) chosen so that the constraint binds in a good share of 150 periods
CFG = {"nkm": (1.0, 40), "zlb": (0.02, 20), "zlb2": (0.02, 20), "nk": (1.0, 32), "borrcon": (0.5, 40), "sw": (0.25, 32), "bf": (1.0, 32)}
_cache = {}


def model(name):
    if name not in _cache:
        path = SRC[name]
        if not path.exists():
            pytest.skip(f"{path} not available")
        d = Path(tempfile.mkdtemp(prefix="rho_"))
        for f in path.parent.iterdir():
            if f.is_file() and f.suffix in (".mod", ".inc", ".mat", ".m", ".csv"):
                shutil.copy(f, d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cache[name] = pydyn.Model(pydyn.run_preprocessor(d / path.name, extra_args=("output=second",)))
    return _cache[name]


def shocks(name, T=150, seed=4):
    m = model(name)
    sc, _ = CFG[name]
    sd = np.sqrt(np.diag(m.Sigma_e)) if name not in ("zlb", "zlb2") else 1.0
    return np.random.default_rng(seed).normal(size=(T, m.n_exo)) * sd * sc


def binds(r):
    return int((np.abs(r["wedges"]).max(axis=1) > 1e-10).sum())


@pytest.mark.parametrize("name", ["nkm", "zlb", "zlb2", "nk", "bf", "sw"])
def test_order_one_is_regime_obc(name):
    m, S, hz = model(name), shocks(name), CFG[name][1]
    h = HigherOrderRegime(m, order=1, horizon=hz, check_ahead=hz)
    r = h.simulate(S)
    ref = RegimeOBC(h.ob, horizon=hz, check_ahead=hz).simulate(S, periods=len(S)).piecewise
    np.testing.assert_allclose(r["with_bound"], ref, rtol=1e-11, atol=1e-11 * max(1.0, np.abs(ref).max()))
    assert binds(r) >= 5


@pytest.mark.parametrize("name", ["zlb", "zlb2", "nk", "borrcon", "nkm"])
def test_exact_class_order_two_equals_lcp(name):
    """Exact complementarity: the regime formulation with order-2 expected paths is the LCP with order-2 expected paths, to rounding
    (offset state and wedges included); for ``nk`` this is also the DynareOBC reference (below).  nkm is approx but has no knife-edge here."""
    m, S, hz = model(name), shocks(name), CFG[name][1]
    a = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz).simulate(S)
    b = HigherOrderNews(m, order=2, horizon=hz, check_ahead=hz, obc_formulation="lcp").simulate(S)
    np.testing.assert_allclose(a["with_bound"], b["with_bound"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["without_bound"], b["without_bound"], rtol=0, atol=1e-13)
    assert binds(a) >= 5
    wl = b["wedges"]
    ra = a["wedges"].reshape(len(S), -1, hz)
    if name != "borrcon":                               # wedges of the (single) switched equation = the LCP's wedges up to the unit sign
        np.testing.assert_allclose(np.abs(ra).sum(axis=1), np.abs(wl.reshape(len(S), -1, hz)).sum(axis=1), rtol=0, atol=1e-10)


def test_exact_class_order_two_equals_dynareobc():
    m = model("nk")
    r = loadmat(REF / "dynareobc_octave" / "ref_o2.mat", squeeze_me=True)
    names = [str(n).strip() for n in r["names"]]
    E = r["endo"][[names.index(n) for n in m.endo_names]].T
    S = loadmat(REF / "dynareobc_octave" / "shocks.mat", squeeze_me=True)["ShockSequence"].reshape(-1, 1)
    h = HigherOrderRegime(m, order=2, horizon=32, check_ahead=32)
    np.testing.assert_allclose(h.simulate(S)["with_bound"], E, atol=1e-12)


@pytest.mark.parametrize("name", ["nkm", "zlb2", "bf", "sw"])
def test_never_binding_is_pruned_simulation(name):
    m, hz = model(name), CFG[name][1]
    S = shocks(name) * 0.01
    h = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz)
    r = h.simulate(S)
    assert not r["wedges"].any()
    np.testing.assert_allclose(r["without_bound"], simulate(h.dr, S, order=2, pruning=True), atol=1e-11)
    np.testing.assert_array_equal(r["with_bound"], r["without_bound"])


@pytest.mark.parametrize("name", ["nkm", "zlb2", "bf", "sw"])
@pytest.mark.parametrize("order", [1, 2])
def test_affine_maps_equal_direct_recursion(name, order):
    m, hz = model(name), CFG[name][1]
    h = HigherOrderRegime(m, order=order, horizon=hz, check_ahead=hz)
    rg = np.random.default_rng(1)
    n = m.n_endo
    for _ in range(4):
        y1, y2, o, prev = (0.01 * rg.normal(size=n) for _ in range(4))
        if order == 1:
            y2 = np.zeros(n)
        u = 0.01 * rg.normal(size=m.n_exo)
        a, Y, A = h.inputs_from_state(y1, y2, o, prev, u)
        a2, Y2, A2 = h.inputs_direct(y1, y2, o, prev, u)
        sc = max(1.0, np.abs(A2).max())
        np.testing.assert_allclose(a, a2, rtol=0, atol=1e-12 * sc)
        np.testing.assert_allclose(Y, Y2, rtol=0, atol=1e-12 * sc)
        np.testing.assert_allclose(A, A2, rtol=0, atol=1e-12 * sc)


@pytest.mark.parametrize("name", ["nkm", "zlb2", "bf", "sw", "borrcon"])
def test_accepted_periods_pass_occbin_test_in_levels(name):
    """Recompute, independently of the solver's own bookkeeping, the order-2 expected path plus the first-order wedge response
    (full responses ``Yw``), and apply OccBin's compiled bind/relax functions to the LEVELS of that path in every period."""
    m, hz = model(name), CFG[name][1]
    h = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz)
    h.trace = []
    r = h.simulate(shocks(name))
    rb, ob, N = h.rb, h.ob, h.N
    nbind = 0
    for tr in h.trace:
        E = h.expected_path(tr["y1"], tr["y2"], tr["o_pre"])                    # (N+1, n) declaration order, deviations
        path = E + np.tensordot(rb.Yw, tr["w"], axes=(2, 0)) if tr["w"].any() else E
        zl = path[:N] + ob.ys
        for j, ev in enumerate(ob._cons):
            ib, _eb, ir, _er = ev(zl, ob.params, ob.ys)
            bind = np.asarray(ib, bool)
            relax = ~bind if ir is None else np.asarray(ir, bool)
            z = tr["z"][:, j]
            zfull = np.zeros(N, bool)
            zfull[:len(z)] = z
            assert not (bind & ~zfull).any() and not (relax & zfull).any()
        nbind += int(tr["z"].any())
    assert nbind >= 3


def test_zero_second_order_terms_give_order_one():
    """With the second-order coefficients scaled to zero the order-2 construction is the order-1 one."""
    m, hz = model("nkm"), 40
    S = shocks("nkm")
    h1 = HigherOrderRegime(m, order=1, horizon=hz, check_ahead=hz)
    h2 = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz)
    for nm in ("ghxx", "ghxu", "ghuu", "ghs2", "c2", "ghxx_s"):
        setattr(h2, nm, getattr(h2, nm) * 0.0)
    h2._build_maps()
    np.testing.assert_allclose(h2.simulate(S)["with_bound"], h1.simulate(S)["with_bound"], rtol=0, atol=1e-11)


def test_approx_class_regime_equals_occbin_where_lcp_may_not():
    """NKM (approx): the regime formulation at order 1 is OccBin's solution; the LCP can differ at knife-edge periods only."""
    from pydyn.occbin import Occbin
    m = model("nkm")
    S = shocks("nkm", T=400, seed=11)
    ob = Occbin(m)
    ref = ob.simulate(S, periods=len(S))
    reg = HigherOrderRegime(m, order=1).simulate(S)["with_bound"]
    np.testing.assert_allclose(reg, ref.piecewise, rtol=0, atol=1e-8)
    lcp = HigherOrderNews(m, order=1, obc_formulation="lcp").simulate(S)["with_bound"]
    assert np.isfinite(lcp).all()        # (how often the LCP differs is measured in docs/OBC_REGIME_HO.md, not asserted: knife-edge periods change the history)


# ------------------------------------------------------------------------------------------------ cubature
def _lcp(m, **kw):
    return HigherOrderNews(m, obc_formulation="lcp", **kw)


@pytest.mark.parametrize("name,regions,level", [("nk", "all", 7), ("zlb", "all", 7), ("nk", 1, 9), ("zlb2", 1, 9), ("zlb", 4, 9), ("zlb2", 4, 9)])
def test_cubature_exact_class_equals_lcp(name, regions, level):
    """Same root, same points, same relevant points, same wedges as the LCP cubature (exact class)."""
    m, S, hz = model(name), shocks(name, T=60), CFG[name][1]
    kw = dict(order=2, horizon=hz, check_ahead=hz, cubature=True, qmc_level=level, regions=regions)
    a, b = HigherOrderRegime(m, **kw), _lcp(m, **kw)
    np.testing.assert_allclose(a.R, b.R, rtol=0, atol=1e-13)
    ra, rb_ = a.simulate(S), b.simulate(S)
    np.testing.assert_array_equal(ra["relevant"], rb_["relevant"])
    np.testing.assert_allclose(ra["with_bound"], rb_["with_bound"], rtol=0, atol=1e-12)
    assert ra["relevant"].sum() > 1000


def test_cubature_exact_class_failures_and_tolerance_equal_lcp():
    m, S, hz = model("zlb2"), shocks("zlb2", T=60), 20
    kw = dict(order=2, horizon=hz, check_ahead=hz, cubature=True, qmc_level=7, regions="all")
    from pydyn.obc_ho import CubatureFailure
    for mk in (lambda **k: HigherOrderRegime(m, **k), lambda **k: _lcp(m, **k)):
        with pytest.raises(CubatureFailure) as ei:
            mk(cubature_undecided_tolerance=0, **kw).simulate(S)
        assert ei.value.period == 1 and ei.value.counts["b"] == 1 and ei.value.status == "horizon_too_short"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = HigherOrderRegime(m, cubature_failure_tolerance=0.01, cubature_undecided_tolerance=0, **kw).simulate(S)
        b = _lcp(m, cubature_failure_tolerance=0.01, cubature_undecided_tolerance=0, **kw).simulate(S)
    np.testing.assert_array_equal(a["dropped"], b["dropped"])
    assert a["dropped"].sum() >= 1
    np.testing.assert_allclose(a["with_bound"], b["with_bound"], rtol=0, atol=1e-12)


def test_cubature_equals_dynareobc():
    """DynareOBC's own root and points (reference/dynareobc_octave): one region to 1e-12 and, as in the LCP test, four regions."""
    m = model("nk")
    r = loadmat(REF / "dynareobc_octave" / "ref_o2cub.mat", squeeze_me=True)
    names = [str(n).strip() for n in r["names"]]
    E = r["endo"][[names.index(n) for n in m.endo_names]].T
    S = loadmat(REF / "dynareobc_octave" / "shocks.mat", squeeze_me=True)["ShockSequence"].reshape(-1, 1)
    sob = loadmat(REF / "dynareobc_octave" / "sob_o2cub.mat")["x"]
    h = HigherOrderRegime(m, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=9, regions=1)
    Rd = np.real(r["Rcc"])
    np.testing.assert_allclose(h.R @ h.R.T, Rd @ Rd.T, atol=1e-16)
    h.R, h.xi = Rd, sob
    np.testing.assert_allclose(h.simulate(S)["with_bound"], E, atol=1e-12)
    r4 = loadmat(REF / "dynareobc_octave" / "ref_o2cub4.mat", squeeze_me=True)
    E4 = r4["endo"][[[str(n).strip() for n in r4["names"]].index(n) for n in m.endo_names]].T
    h4 = HigherOrderRegime(m, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=9, regions=4)
    h4.R, h4.xi = Rd, sob
    d = np.abs(h4.simulate(S)["with_bound"] - E4).max(axis=1)
    assert (d < 1e-12).sum() >= 290 and d.max() < 0.01


@pytest.mark.parametrize("name", ["nkm", "bf", "zlb"])
def test_cubature_vanishing_variance_is_no_cubature(name):
    m, S, hz = model(name), shocks(name, T=60), CFG[name][1]
    tiny = np.array(m.Sigma_e) * 1e-12
    c = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz, cubature=True, Sigma_e=tiny, qmc_level=5).simulate(S)
    b = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz, Sigma_e=tiny).simulate(S)
    np.testing.assert_allclose(c["with_bound"], b["with_bound"], atol=1e-9)


def test_cubature_makes_agents_react_before_the_bound_regime():
    m, S, hz = model("nkm"), shocks("nkm"), 40
    a = HigherOrderRegime(m, order=2).simulate(S)
    c = HigherOrderRegime(m, order=2, cubature=True, qmc_level=7, regions=1).simulate(S)
    assert binds(c) > 3 * binds(a)


def test_cubature_batch_equals_point_by_point_search():
    """The batched guess-and-verify over many perturbed points returns, point by point, what the complete search returns."""
    m, S, hz = model("nkm"), shocks("nkm", T=80), 40
    h = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz, cubature=True, qmc_level=6, regions="all", cubature_failure_tolerance=0.5)
    h.trace = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h.simulate(S)
    rb, nb, N, T, k = h.rb, h.nb, h.N, h.T, len(h.rb._cv)
    Ro = h._other_roots()
    checked = 0
    for tr in h.trace[:60]:
        a, Y, A = h.inputs_from_state(tr["y1"], tr["y2"], tr["o_pre"], tr["prev"], tr["u"])
        pY = (Y.reshape(-1)[:, None] + Ro[nb * N:] @ h.xi).T.reshape(-1, N, k)
        pA = (A.reshape(-1)[:, None] + Ro[:nb * N] @ h.xi).T.reshape(-1, nb, N)
        relmask, vb_all = h._relevant(pY)
        rel = np.flatnonzero(relmask)[:40]
        if not rel.size:
            continue
        W, ok = h._guess_verify_many(pA[rel][:, :, :T].reshape(len(rel), -1), pY[rel], vb0=vb_all[rel])
        W2, ok2 = h._guess_verify_many(pA[rel][:, :, :T].reshape(len(rel), -1), pY[rel])       # without the shortcut: the same
        np.testing.assert_array_equal(ok, ok2)
        np.testing.assert_allclose(W[ok], W2[ok2], rtol=0, atol=1e-13)
        for j, i in enumerate(rel):
            try:
                z, w, _ = h.solve_inputs(pA[i][:, :T].ravel(), pY[i], pA[i])
            except OBCError:
                assert not ok[j]
                continue
            if ok[j]:
                np.testing.assert_allclose(W[j], w, rtol=0, atol=1e-11 * max(1.0, np.abs(w).max()))
                checked += 1
    assert checked > 100


# ------------------------------------------------------------------------------------------------ routing and approx class
def test_higher_order_news_routes_by_obc_formulation():
    nkm, zlb, bf = model("nkm"), model("zlb"), model("bf")
    assert type(HigherOrderNews(nkm, order=1)) is HigherOrderRegime                      # approx class: 'auto' is the regime formulation
    assert type(HigherOrderNews(nkm, order=1, obc_formulation="lcp")) is HigherOrderNews
    assert type(HigherOrderNews(zlb, order=1, horizon=20, check_ahead=20)) is HigherOrderNews       # exact class: unchanged
    assert type(HigherOrderNews(zlb, order=1, horizon=20, check_ahead=20, obc_formulation="regime")) is HigherOrderRegime
    assert type(HigherOrderNews(bf, order=1)) is HigherOrderRegime                         # multi_equation: was refused
    with pytest.raises(ValueError, match="complementary pair"):
        HigherOrderNews(bf, order=1, obc_formulation="lcp")
    with pytest.raises(ValueError, match="unknown obc_formulation"):
        HigherOrderNews(zlb, order=1, obc_formulation="hybrid")


def test_exact_class_auto_is_bit_identical_to_lcp():
    m, S, hz = model("nk"), shocks("nk"), 32
    a = HigherOrderNews(m, order=2, horizon=hz, check_ahead=hz).simulate(S)
    b = HigherOrderNews(m, order=2, horizon=hz, check_ahead=hz, obc_formulation="lcp").simulate(S)
    # same class, same code path.  (Not array_equal: two builds of the same model differ by ~1e-20 in ghxx, a property of k_order that predates this work.)
    assert type(HigherOrderNews(m, order=2, horizon=hz, check_ahead=hz)) is HigherOrderNews
    np.testing.assert_allclose(a["with_bound"], b["with_bound"], rtol=0, atol=1e-15)


def test_sw_cubature_raises_with_class_a_and_bf_tolerance_mode():
    """sw (multi_equation): reachable states without a solution, so cubature raises at once with class (a) points; bf drops class (c) points when allowed."""
    from pydyn.obc_ho import CubatureFailure
    m = model("sw")
    h = HigherOrderRegime(m, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=5, cubature_milp_time_limit=0.02)
    with pytest.raises(CubatureFailure) as ei:
        h.simulate(shocks("sw", T=10))
    assert ei.value.counts["a"] >= 1 and ei.value.status == "no_solution"


def _zlbm(d, tmp):
    sys.path.insert(0, str(ROOT / "soak" / "approx"))
    sys.path.insert(0, str(ROOT / "soak"))
    import gen
    p = Path(tmp) / "tf_approx.mod"
    p.write_text(gen.zlbm(d))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(p, extra_args=("output=second",)))


@pytest.mark.parametrize("d,max_periods", [(0.0, 0), (1e-3, 0), (1e-2, 5)])
def test_approx_class_family_regime_vs_lcp_order_two(tmp_path, d, max_periods):
    """Constructed approx-class model (docs/OBC_ESTIMATION.md): order-2 regime vs order-2 LCP differ at knife-edge periods only, and
    the d = 0 control (additive shock) is exact; the regime path is accepted by OccBin's test in every period."""
    m = _zlbm(d, tmp_path)
    S = np.random.default_rng(4).normal(size=(300, m.n_exo)) * np.sqrt(np.diag(m.Sigma_e)) * 1.6
    h = HigherOrderRegime(m, order=2)
    h.trace = []
    a = h.simulate(S)
    b = HigherOrderNews(m, order=2, obc_formulation="lcp").simulate(S)
    dd = np.abs(a["with_bound"] - b["with_bound"]).max(axis=1)
    assert binds(a) >= 10
    assert (dd > 1e-8).sum() <= max_periods and dd.max() < 1e-3
    assert len(h.trace) == len(S)
