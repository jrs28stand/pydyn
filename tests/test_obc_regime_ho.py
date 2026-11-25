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
from pydyn.obc import RegimeOBC, constraint_objects
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
