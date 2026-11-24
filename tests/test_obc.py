"""News-shock method for occasionally binding constraints (pydyn.obc, obc_method='news').

References: reference/obc_zlb.mat (r_obc_zlb.m: M and q from Dynare 7.2's first-order solution of the model
with news shocks added by hand, the LCP solved with GLPK in Holden's formulation, and Dynare's own OccBin path),
and pydyn's OccBin solver, which is itself checked against Dynare 7.2 in test_occbin*.py.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.obc import (NewsOBC, OBCClassError, OBCError, RegimeOBC, constraint_objects, m_diagnostics, solve_active_set,
                       solve_news, verify_lcp)

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))
EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
DYNARE_TESTS = Path(os.environ.get("PYDYN_DYNARE_TESTS", str(EXAMPLES.parent / "tests")))


def _load(tmp, name, src=REF, extra=()):
    for f in (name, *extra):
        shutil.copy(Path(src) / f, tmp)
    return pydyn.Model(pydyn.run_preprocessor(tmp / name))


@pytest.fixture(scope="module")
def zlb(tmp_path_factory):
    m = _load(tmp_path_factory.mktemp("obc_zlb"), "obc_zlb.mod")
    return m, pydyn.Occbin(m)


@pytest.fixture(scope="module")
def zlb2(tmp_path_factory):
    m = _load(tmp_path_factory.mktemp("obc_zlb2"), "obc_zlb2.mod")
    return m, pydyn.Occbin(m)


# ------------------------------------------------------------------------------------------ post-processor
def test_constraint_object_lower_bound(zlb):
    m, ob = zlb
    (co,) = constraint_objects(ob)
    assert co.cls == "complementarity" and co.approx == 0.0
    assert len(co.equations) == 1
    assert co.s_g == 1.0 and co.s_y == 1.0           # g = r - rbar, y = r - rn
    assert co.g_ss == pytest.approx(0.01, abs=1e-12)
    ir, irn, n = m._ie["r"], m._ie["rn"], m.n_endo
    np.testing.assert_allclose(co.alt[n + ir], 1.0)
    np.testing.assert_allclose(co.ref[n + ir], 1.0)
    np.testing.assert_allclose(co.ref[n + irn], -1.0)


def test_two_constraints_classified(zlb2):
    _, ob = zlb2
    cos = constraint_objects(ob)
    assert [c.cls for c in cos] == ["complementarity", "complementarity"]
    assert [c.name for c in cos] == ["zlb", "floor"]


# -------------------------------------------------------------------------------- against Dynare 7.2 / GLPK
def test_M_and_q_match_dynare(zlb):
    m, ob = zlb
    ref = loadmat(REF / "obc_zlb.mat", squeeze_me=True)
    nb = NewsOBC(ob, horizon=int(ref["T"]), check_ahead=int(ref["N"]) - int(ref["T"]))
    np.testing.assert_allclose(nb.Mo, ref["M"], atol=1e-12)
    q, _ = nb.q_of(np.zeros(m.n_endo), np.array([float(ref["shock"])]))
    np.testing.assert_allclose(q, ref["q"], atol=1e-13)


def test_wedges_and_path_match_reference(zlb):
    m, ob = zlb
    ref = loadmat(REF / "obc_zlb.mat", squeeze_me=True)
    assert int(ref["lcp_ok"]) == 1
    T, N = int(ref["T"]), int(ref["N"])
    nb = NewsOBC(ob, horizon=T, check_ahead=N - T)
    for method in ("news", "milp"):
        y, path, info = nb.solve_period(np.zeros(m.n_endo), np.array([float(ref["shock"])]), method=method)
        np.testing.assert_allclose(y, ref["y"], atol=1e-9)        # GLPK's own tolerance
        r = path[:30, m._ie["r"]] + ob.ys[m._ie["r"]]
        np.testing.assert_allclose(r, ref["r_occbin"][:30], atol=1e-12)   # Dynare's OccBin path
    assert info["used"] == "milp"


def test_milp_on_reference_problem():
    ref = loadmat(REF / "obc_zlb.mat", squeeze_me=True)
    T = int(ref["T"])
    y, status = solve_news(ref["M"], ref["q"], T)
    assert status == "ok" and verify_lcp(ref["M"], ref["q"], y, T)
    np.testing.assert_allclose(y, ref["y"], atol=1e-9)
    np.testing.assert_allclose(solve_active_set(ref["M"], ref["q"], T), y, atol=1e-12)


# ------------------------------------------------------------------------------------- against pydyn OccBin
def _compare(ob, shocks, periods, horizon=40, atol=1e-11, **kw):
    a = ob.simulate(shocks, periods=periods, **kw)
    b = ob.simulate(shocks, periods=periods, obc_method="news", obc_horizon=horizon)
    np.testing.assert_allclose(b.piecewise, a.piecewise, atol=atol)
    np.testing.assert_allclose(b.linear, a.linear, atol=1e-13)
    assert b.regime_history == a.regime_history
    return a, b


def test_M_matches_stacked_system(zlb, zlb2):
    for _, ob in (zlb, zlb2):
        nb = NewsOBC(ob, horizon=30, check_ahead=30)
        np.testing.assert_allclose(nb.Mo, nb.stacked_M(), atol=1e-10)


def test_q_affine_map(zlb2):
    m, ob = zlb2
    nb = NewsOBC(ob, horizon=20, check_ahead=20)
    rng = np.random.default_rng(0)
    init, shock = rng.normal(0, 0.02, m.n_endo), rng.normal(0, 0.02, m.n_exo)
    np.testing.assert_allclose(nb.q_of(init, shock)[0], nb.q_direct(init, shock), atol=1e-13)


def test_zlb_equals_occbin(zlb):
    m, ob = zlb
    _compare(ob, ob.surprise_shocks(), 30)
    rng = np.random.default_rng(1)
    _, b = _compare(ob, rng.normal(0, 0.02, (200, 1)), 200)
    assert b.news["solver"].count("active_set") > 20          # the constraint really binds in the sample


def test_two_constraints_equal_occbin(zlb2):
    m, ob = zlb2
    _compare(ob, ob.surprise_shocks(), 40)
    rng = np.random.default_rng(3)
    _, b = _compare(ob, rng.normal(0, 0.02, (200, 1)), 200)
    both = sum(1 for h in b.regime_history if 1 in h[0][0] and 1 in h[1][0])
    assert both > 5                                           # periods in which both constraints bind


def test_method_from_mod_file(tmp_path):
    m = _load(tmp_path, "obc_zlb_opt.mod", extra=("obc_zlb.mod",))
    ob = pydyn.Occbin(m)
    kw = ob.mod_options()
    assert kw["obc_method"] == "news" and kw["obc_horizon"] == 30
    res = ob.run()
    assert res.news["method"] == "news" and res.news["horizon"] == 30
    np.testing.assert_allclose(res.piecewise, ob.run(obc_method="occbin").piecewise, atol=1e-12)


def test_irf_occbin_with_news(zlb):
    _, ob = zlb
    a = ob.irf_occbin(periods=20, shocksize=0.05)
    b = ob.irf_occbin(periods=20, shocksize=0.05, obc_method="news")
    for key in a:
        np.testing.assert_allclose(b[key]["piecewise"], a[key]["piecewise"], atol=1e-11)


# ------------------------------------------------------------------------------------------- Dynare's files
needs_dynare = pytest.mark.skipif(not (DYNARE_TESTS / "occbin").exists(),
                                  reason="Dynare's tests folder not found (PYDYN_DYNARE_TESTS)")


def _dynare_model(tmp, rel, base=DYNARE_TESTS):
    src = Path(base) / rel
    work = tmp / src.parent.name
    shutil.copytree(src.parent, work)
    return pydyn.Model(pydyn.run_preprocessor(work / src.name))


@needs_dynare
def test_borrcon_equals_occbin(tmp_path):
    """Dynare's borrowing-constraint test: the reference regime is the one in which the constraint binds."""
    m = _dynare_model(tmp_path, "occbin/model_borrcon/borrcon.mod")
    ob = pydyn.Occbin(m)
    (co,) = constraint_objects(ob)
    assert co.cls == "complementarity" and co.approx == 0.0 and co.s_y == -1.0
    _compare(ob, ob.surprise_shocks(), 100, horizon=80)
    rng = np.random.default_rng(7)
    _compare(ob, rng.normal(0, 0.02, (200, m.n_exo)), 200, horizon=80)
    d = NewsOBC(ob, horizon=40).diagnostics()
    assert d["is_p"] is True and d["is_s"] is True


@needs_dynare
def test_nkm_lower_bound_approximate_class(tmp_path):
    """Dynare's NKM filter test: 'inom = inomlb*exp(mp)' against 'bind inomnot <= inomlb'. The linearised
    equations carry a term in mp that the condition lacks, so the pair is flagged as approximate."""
    m = _dynare_model(tmp_path, "occbin/filter/NKM_0_std_shocks.mod")
    ob = pydyn.Occbin(m)
    (co,) = constraint_objects(ob)
    assert co.cls == "complementarity" and 0 < co.approx < 0.05
    rng = np.random.default_rng(7)
    sd = np.sqrt(np.diag(m.Sigma_e))
    sd[sd < 1e-9] = 0.01
    _compare(ob, rng.normal(size=(200, m.n_exo)) * sd * 2, 200, horizon=80, atol=1e-10)


@needs_dynare
@pytest.mark.parametrize("rel,base,expected", [
    ("occbin/rbc_occbin.mod", EXAMPLES, {"IRR": "complementarity", "INEG": "inconsistent"}),
    ("occbin/ppf/bf_ppf.mod", DYNARE_TESTS, {"zlb": "multi_equation"}),
    ("estimation/slice/bf.mod", DYNARE_TESTS, {"zlb": "multi_equation"}),
])
def test_classification_of_dynare_files(tmp_path, rel, base, expected):
    m = _dynare_model(tmp_path, rel, base)
    ob = pydyn.Occbin(m)
    assert {c.name: c.cls for c in constraint_objects(ob)} == expected
    with pytest.raises(OBCClassError, match="obc_method='occbin'"):
        NewsOBC(ob)                                     # the LCP class refuses; the dispatcher uses RegimeOBC


# --------------------------------------------------------------------------- general regime formulation
def test_regime_formulation_equals_lcp_and_occbin(zlb, zlb2):
    rng = np.random.default_rng(5)
    for m, ob in (zlb, zlb2):
        S = rng.normal(0, 0.02, (150, 1))
        a = ob.simulate(S, periods=150)
        b = RegimeOBC(ob, horizon=40, check_ahead=40).simulate(S, periods=150)
        c = NewsOBC(ob, horizon=40, check_ahead=40).simulate(S, periods=150)
        np.testing.assert_allclose(b.piecewise, a.piecewise, atol=1e-11)
        np.testing.assert_allclose(b.piecewise, c.piecewise, atol=1e-11)
        assert b.regime_history == a.regime_history


def test_regime_integer_programme_one_constraint(zlb):
    m, ob = zlb
    rb = RegimeOBC(ob, horizon=20, check_ahead=40)
    z, w, path, info = rb.solve_period(np.zeros(m.n_endo), np.array([-0.05]), method="milp")
    z2, w2, path2, info2 = rb.solve_period(np.zeros(m.n_endo), np.array([-0.05]))
    assert info["used"] == "milp" and info2["used"] == "guess_verify"
    np.testing.assert_array_equal(z, z2)
    np.testing.assert_allclose(path, path2, atol=1e-12)


@needs_dynare
@pytest.mark.parametrize("rel", ["occbin/ppf/bf_ppf.mod", "estimation/slice/bf.mod"])
def test_multi_equation_lower_bound_equals_occbin(tmp_path, rel):
    """Lower bound in which the rule's lag term also switches (two equations per constraint)."""
    m = _dynare_model(tmp_path, rel)
    ob = pydyn.Occbin(m)
    rng = np.random.default_rng(7)
    sd = np.sqrt(np.diag(m.Sigma_e))
    sd[sd < 1e-9] = 0.01
    a, b = _compare(ob, rng.normal(size=(200, m.n_exo)) * sd * 2, 200, atol=1e-10)
    assert b.news["formulation"] == "regime" and b.news["solver"].count("guess_verify") > 20


@needs_dynare
def test_rbc_occbin_two_constraints_equals_occbin(tmp_path):
    """Dynare's example: IRR is a complementary pair, INEG switches adjustment costs on (not a bound), and
    INEG stays in its alternative regime for more than 100 periods."""
    m = _dynare_model(tmp_path, "occbin/rbc_occbin.mod", EXAMPLES)
    ob = pydyn.Occbin(m)
    S = ob.surprise_shocks()
    a = ob.simulate(S, periods=200, check_ahead_periods=200)
    b = ob.simulate(S, periods=200, check_ahead_periods=70, obc_method="news", obc_horizon=130)
    np.testing.assert_allclose(b.piecewise, a.piecewise, atol=1e-11)
    assert b.regime_history == a.regime_history
    with pytest.raises(OBCError) as err:
        ob.simulate(S, periods=200, obc_method="news", obc_horizon=40)
    assert err.value.status == "horizon_too_short"


# ---------------------------------------------------------------------------------------------- the solvers
def test_horizon_too_short(zlb):
    _, ob = zlb
    with pytest.raises(OBCError) as err:
        ob.simulate(np.array([[-0.05]]), periods=30, obc_method="news", obc_horizon=4)
    assert err.value.status == "horizon_too_short" and err.value.period == 1


def test_unknown_method_and_occbin_only_options(zlb):
    _, ob = zlb
    with pytest.raises(ValueError, match="unknown obc_method"):
        ob.simulate(np.array([[-0.05]]), obc_method="holden")
    with pytest.raises(NotImplementedError):
        ob.simulate(np.array([[-0.05]]), obc_method="minnorm")
    with pytest.raises(ValueError, match="OccBin options"):
        ob.simulate(np.array([[-0.05]]), obc_method="news", periodic_solution=True)


def test_no_solution_and_multiple_solutions():
    # M = -1: q < 0 has no solution (y >= 0 makes q - y more negative)
    y, status = solve_news(np.array([[-1.0]]), np.array([-1.0]), 1)
    assert y is None and status == "no_solution"
    # M = [[1, 2], [2, 1]] is not a P-matrix; q = (-1, -1) has three solutions, and any returned one must verify
    M, q = np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([-1.0, -1.0])
    y, status = solve_news(M, q, 2)
    assert status == "ok" and verify_lcp(M, q, y, 2)
    d = m_diagnostics(M)
    assert d["is_p"] is False and d["is_s"] is True


def test_diagnostics_small_matrices():
    assert m_diagnostics(np.eye(3))["is_p"] is True
    d = m_diagnostics(np.array([[-1.0]]))
    assert d["is_p"] is False and d["is_s"] is False
    # P but not positive definite in its symmetric part
    M = np.array([[1.0, -3.0], [0.0, 1.0]])
    d = m_diagnostics(M)
    assert d["is_p"] is True and d["symmetric_part_positive_definite"] is False


def test_p_matrix_recursion_matches_minors():
    from pydyn.obc import _is_p_exact, _is_p_minors
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(300):
        n = int(rng.integers(1, 7))
        A = np.eye(n) * rng.uniform(0.2, 2.0) + rng.normal(0, 0.45, (n, n))
        r = _is_p_exact(A)
        assert r == _is_p_minors(A)
        seen.add(r)
    assert seen == {True, False}


# ------------------------------------------------------------------------------- single-spell fast path
def test_single_spell_fast_path_equals_general_solver(zlb):
    from pydyn.obc_lk import LKNews
    m, ob = zlb
    S = np.random.default_rng(1).normal(0, 0.02, (500, 1))
    a = ob.simulate(S, periods=520)
    b = NewsOBC(ob, horizon=40, check_ahead=40).simulate(S, periods=520)
    lk = LKNews(ob, horizon=40, check_ahead=40)
    c = lk.simulate(S, periods=520)
    np.testing.assert_allclose(c.piecewise, b.piecewise, atol=1e-12)
    np.testing.assert_allclose(c.piecewise, a.piecewise, atol=1e-11)
    assert c.regime_history == a.regime_history
    st = c.news["stats"]
    assert st["general"] == 0 and st["built"] < 40 and st["shifted"] + st["cached"] > 100


def test_single_spell_fast_path_one_constraint_only(zlb2):
    from pydyn.obc_lk import LKNews
    with pytest.raises(ValueError, match="one constraint"):
        LKNews(zlb2[1])


# ------------------------------------------- several equations per constraint written as a complementarity problem
@pytest.fixture(scope="module")
def sw(tmp_path_factory):
    m = _load(tmp_path_factory.mktemp("obc_sw"), "obc_sw_zlb.mod")
    return m, pydyn.Occbin(m)


def test_sw_tie_and_diagnostics(sw):
    """Smets-Wouters with a lower bound whose rule also switches its lag term: the second wedge is tied to the
    first, and the M of the equivalent LCP is neither an S- nor a P-matrix (no solution for some states,
    several for others)."""
    m, ob = sw
    rb = RegimeOBC(ob, horizon=40, check_ahead=60)
    tie = rb._tie
    assert tie is not None and tie["mirror"] and len(tie["secondary"]) == 1
    np.testing.assert_allclose(tie["d"], [1.0, -1.0], atol=1e-10)
    d = rb.diagnostics(max_exact=12)
    assert d["is_s"] is False and d["is_p"] is False and d["largest_p"] == 10


def test_sw_multiple_equilibria_and_occbin_choice(sw):
    m, ob = sw
    rb = RegimeOBC(ob, horizon=40, check_ahead=60)
    S = np.zeros(m.n_exo)
    S[m._ix["eb"]] = -4.0
    sols = rb.single_spell_solutions(np.zeros(m.n_endo), S)
    assert sols == [(0, 9), (0, 13)]
    # each is a solution by OccBin's own test, and they are far apart
    paths = []
    x = np.r_[np.zeros(m.n_endo), S]
    Y, a = np.tensordot(rb.Yb, x, axes=(2, 0)), rb.a0 + rb.Am @ x
    for l, k in sols:
        z = np.zeros((rb.N, 1), bool)
        z[l:l + k] = True
        path = Y + rb.Yw @ rb._wedges(z, a)
        vb, vr = rb._violations(z, path)
        assert not vb.any() and not vr.any()
        paths.append(path)
    assert np.abs(paths[0] - paths[1]).max() > 1.0
    # OccBin and the news method both return the shorter spell
    a_occ = ob.simulate(S[None, :], periods=60, check_ahead_periods=100)
    b = ob.simulate(S[None, :], periods=60, check_ahead_periods=60, obc_method="news")
    assert a_occ.regime_history[0] == [([1, 0], [1, 10])] == b.regime_history[0]
    np.testing.assert_allclose(b.piecewise, a_occ.piecewise, atol=1e-9)


def test_sw_no_solution_is_proved(sw):
    """A larger shock: OccBin cycles (error 313); the news method reports that no solution exists, by an
    infeasible linear programme, in a fraction of a second."""
    m, ob = sw
    S = np.zeros((1, m.n_exo))
    S[0, m._ix["eb"]] = -6.0
    with pytest.raises(pydyn.OccbinError) as e_occ:
        ob.simulate(S, periods=60, check_ahead_periods=100)
    assert e_occ.value.error_flag == 313
    with pytest.raises(OBCError) as err:
        ob.simulate(S, periods=60, check_ahead_periods=60, obc_method="news")
    assert err.value.status == "no_solution" and "linear programme infeasible" in str(err.value)
    assert RegimeOBC(ob, horizon=40, check_ahead=60).single_spell_solutions(np.zeros(m.n_endo), S[0]) == []


def test_sw_offsets_when_the_policy_shock_is_active(sw):
    """With the monetary shock non-zero the tied wedge needs a state-dependent offset (G != 0); solutions found
    through the complementarity problem pass OccBin's test."""
    m, ob = sw
    rb = RegimeOBC(ob, horizon=40, check_ahead=60)
    S = np.zeros(m.n_exo)
    S[m._ix["eb"]], S[m._ix["em"]] = -3.0, 0.5
    M, q, G, info = rb.lcp(np.zeros(m.n_endo), S)
    assert G is not None
    sols = rb.single_spell_solutions(np.zeros(m.n_endo), S)
    assert len(sols) >= 1
    z, w, path = rb._solve_lcp(np.zeros(m.n_endo), S)
    vb, vr = rb._violations(z, path)
    assert not vb.any() and not vr.any()
