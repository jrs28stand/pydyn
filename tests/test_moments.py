"""Theoretical, pruned and simulated moments against Dynare 7.2 (reference/mom_*.mat, produced by
reference/r_moments.m from reference/mom_*.mod), plus independent checks of the building blocks.

Agreement is checked elementwise with rtol=1e-9 and an absolute tolerance of 1e-11 times the largest
entry of the array (tiny entries such as a 1e-17 mean are numerical zeros). Simulated moments of
HP-filtered paths are limited by the conditioning of the HP system and use 1e-8 / 1e-9.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg
from scipy.io import loadmat

import pydyn
from pydyn import moments

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
MODELS = ["mom_collard", "mom_unitroot", "mom_aux", "mom_schorfheide"]
SUBSET = {"mom_collard": ["c", "y", "a"], "mom_unitroot": ["z", "d", "q", "w"],
          "mom_aux": ["z", "c", "AUX_ENDO_LAG_2_1", "AUX_EXO_LAG_6_1"],
          "mom_schorfheide": ["y", "R", "gy_obs", "AUX_ENDO_LEAD_65"]}
CVD = [1, 2, 4, 10, 40]

try:
    pydyn.korder.load_library()
    HAVE_LIB = True
except FileNotFoundError:
    HAVE_LIB = False
needs_lib = pytest.mark.skipif(not HAVE_LIB, reason="libpydyn_korder.so not built")

# the same option sets as reference/r_moments.m: name -> (options, uses the variable subset)
TH_CASES = {
    "th1": (dict(order=1, steps=CVD), False),
    "th1_hp": (dict(order=1, hp_filter=1600, ar=4, steps=[1, 5]), False),
    "th1_bp": (dict(order=1, bandpass_filter=(6, 32), ar=4), False),
    "th1_hp128": (dict(order=1, hp_filter=100, ar=2, filtered_theoretical_moments_grid=128), False),
    "th1_sub": (dict(order=1, ar=2, steps=[1, 3, 8]), True),
    "th1_nodec": (dict(order=1, ar=2, nodecomposition=True, steps=[2, 6]), False),
    "th2": (dict(order=2, pruning=False, ar=3, steps=[1, 4]), False),
    "th2p": (dict(order=2, pruning=True, ar=3, steps=[1, 4, 10]), False),
    "th2p_sub": (dict(order=2, pruning=True, ar=2, steps=[2, 5]), True),
    "th2p_nodec": (dict(order=2, pruning=True, ar=2, nodecomposition=True, steps=[2, 5]), False),
    "th2_sub": (dict(order=2, pruning=False, ar=2), True),
}
SIM_CASES = {
    "s1": (dict(order=1, ar=4), False),
    "s1_hp": (dict(order=1, ar=4, hp_filter=1600), False),
    "s1_os": (dict(order=1, ar=3, one_sided_hp_filter=1600), False),
    "s1_bp": (dict(order=1, ar=3, bandpass_filter=(6, 32)), False),
    "s1_sub": (dict(order=1, ar=2), True),
    "s2p": (dict(order=2, pruning=True, ar=4), False),
    "s2p_hp": (dict(order=2, pruning=True, ar=3, hp_filter=1600), False),
}


# ------------------------------------------------------------------------------------ fixtures
class Case:
    def __init__(self, name, tmp):
        shutil.copy(REF / f"{name}.mod", tmp)
        self.name = name
        self.m = pydyn.Model(pydyn.run_preprocessor(Path(tmp) / f"{name}.mod"))
        self.r = loadmat(REF / f"{name}.mat", squeeze_me=True)
        self._dr = {}

    def has(self, order):
        return self.m.max_deriv_order >= order and (order == 1 or HAVE_LIB)

    def dr(self, order):                      # pydyn's own decision rules
        if order not in self._dr:
            self._dr[order] = moments.solve(self.m, order)
        return self._dr[order]

    def dynare_dr(self, order):               # Dynare's decision rules, to test the moment code in isolation
        r, base = self.r, self.dr(1)
        d = dict(ys=r["dr_ys"].astype(float), ghx=np.atleast_2d(r["dr_ghx"]), ghu=np.atleast_2d(r["dr_ghu"]),
                 order_var=r["dr_order_var"].astype(int) - 1, state_var=base["state_var"], order=order)
        if order == 2:
            d.update(ghxx=r["dr_ghxx"], ghxu=r["dr_ghxu"], ghuu=r["dr_ghuu"], ghs2=np.atleast_1d(r["dr_ghs2"]))
        return d


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    return {n: Case(n, tmp_path_factory.mktemp(n)) for n in MODELS}


def close(a, b, rtol=1e-9, atol=1e-11, what=""):
    a, b = np.asarray(a, float), np.asarray(b, float)
    b = b.reshape(a.shape) if (a.shape != b.shape and a.size == b.size and b.ndim < a.ndim) else b
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{what}: NaN pattern differs"
    ok = ~np.isnan(b)
    scale = np.max(np.abs(b[ok])) if ok.any() else 1.0
    np.testing.assert_allclose(a[ok], b[ok], rtol=rtol, atol=atol * scale, err_msg=what)


def th_params(models=MODELS):
    out = []
    for n in models:
        for c, (kw, sub) in TH_CASES.items():
            if n == "mom_unitroot" and kw["order"] == 2:
                continue
            out.append(pytest.param(n, c, id=f"{n}-{c}"))
    return out


def compare_theoretical(mo, r, c, tol=dict(rtol=1e-9, atol=1e-11)):
    kw = TH_CASES[c][0]
    assert list(mo.var_index) == list(r[f"{c}_ivar"] - 1)
    close(mo.mean, r[f"{c}_mean"], what="mean", **tol)
    close(mo.var, r[f"{c}_var"], what="var", **tol)
    close(mo.variance, np.diag(r[f"{c}_var"]), what="variance", **tol)
    if f"{c}_contemporaneous_correlation" in r:
        close(mo.corr, r[f"{c}_contemporaneous_correlation"], what="corr", **tol)
    close(np.stack(mo.autocorr, axis=2), r[f"{c}_autocorr"], what="autocorr", **tol)
    ar = mo.ar
    if f"{c}_variance_decomposition" in r:
        close(mo.variance_decomposition, r[f"{c}_variance_decomposition"], what="variance decomposition", **tol)
        close(mo.gamma_y[ar + 1] * 100, r[f"{c}_variance_decomposition"], what="gamma_y vd", **tol)
    else:
        assert mo.variance_decomposition is None
    if f"{c}_variance_decomposition_ME" in r:
        close(mo.variance_decomposition_me, r[f"{c}_variance_decomposition_ME"], what="vd with ME", **tol)
    else:
        assert mo.variance_decomposition_me is None
    for f, attr in (("conditional_variance_decomposition", "conditional_variance_decomposition"),
                    ("conditional_variance_decomposition_ME", "conditional_variance_decomposition_me")):
        if f"{c}_{f}" in r:
            close(getattr(mo, attr), r[f"{c}_{f}"], what=f, **tol)
        else:
            assert getattr(mo, attr) is None
    if kw["order"] == 2 and not kw["pruning"]:
        close(mo.mean_correction, r[f"{c}_gy{ar + 3}"], what="mean correction", **tol)


# ------------------------------------------------------------------------------------ theoretical moments
@pytest.mark.parametrize("name,case", th_params())
def test_theoretical_moments_against_dynare(cases, name, case):
    """Everything stoch_simul reports at periods=0, from pydyn's own solution of the model."""
    cs = cases[name]
    kw, sub = TH_CASES[case]
    if kw["order"] == 2 and not cs.has(2):
        pytest.skip("k-order library not available")
    mo = moments.theoretical_moments(cs.m, cs.dr(kw["order"]), SUBSET[name] if sub else None, **kw)
    assert f"{case}_error" not in cs.r
    compare_theoretical(mo, cs.r, case)


@needs_lib
@pytest.mark.parametrize("name,case", [("mom_collard", "th1"), ("mom_collard", "th2"), ("mom_collard", "th2p"),
                                       ("mom_aux", "th1"), ("mom_aux", "th2p"), ("mom_schorfheide", "th1"),
                                       ("mom_schorfheide", "th1_hp"), ("mom_schorfheide", "th2p"),
                                       ("mom_unitroot", "th1"), ("mom_unitroot", "th1_bp")])
def test_theoretical_moments_with_dynare_decision_rules(cases, name, case):
    """Same, feeding Dynare's own ghx, ghu, ghxx, ... so that only the moment code is tested."""
    cs = cases[name]
    kw, sub = TH_CASES[case]
    mo = moments.theoretical_moments(cs.m, cs.dynare_dr(kw["order"]), SUBSET[name] if sub else None, **kw)
    compare_theoretical(mo, cs.r, case, tol=dict(rtol=1e-11, atol=1e-12))


def test_default_variable_list_options_and_aux_variables(cases):
    """The .mod file's stoch_simul options are the defaults; auxiliary variables are not reported by default."""
    cs = cases["mom_aux"]
    assert cs.m.options["stoch_simul"]["order"] == 2
    mo = moments.theoretical_moments(cs.m, cs.dr(1), order=1, ar=2)
    assert mo.var_names == cs.m.orig_endo_names and len(mo.var_names) == 5 < cs.m.n_endo
    assert len(mo.autocorr) == 2
    cc = cases["mom_collard"]
    assert cc.m.options["stoch_simul"]["ar"] == 3
    mo = moments.stoch_simul_moments(cc.m, dr=cc.dr(2))                 # order 2, ar=3 from the .mod file
    assert (mo.order, mo.ar, mo.kind) == (2, 3, "theoretical")
    close(mo.mean, cc.r["th2_mean"])
    close(mo.var, cc.r["th2_var"])
    with pytest.raises(ValueError):
        moments.theoretical_moments(cc.m, cc.dr(1), ["nonexistent"], order=1)
    with pytest.warns(UserWarning):
        mo = moments.theoretical_moments(cc.m, cc.dr(1), ["c", "y", "c"], order=1)
    assert mo.var_names == ["c", "y"]


# ------------------------------------------------------------------------------------ unit roots
def test_unit_root_gives_nan_for_nonstationary_variables(cases):
    cs = cases["mom_unitroot"]
    names = cs.m.endo_names
    assert names == ["x", "w", "q", "d", "z", "p", "s"]
    mo = moments.theoretical_moments(cs.m, cs.dr(1), ar=3, steps=[1, 8])
    nonstat = ["x", "q", "s"]          # x random walk; q = x + w; s loads on x(-1)
    stat = [v for v in names if v not in nonstat]
    ist, ins = [names.index(v) for v in stat], [names.index(v) for v in nonstat]
    assert list(mo.stationary) == [v in stat for v in names]
    for arr in (mo.mean, mo.std, mo.variance):
        assert np.all(np.isnan(arr[ins])) and np.all(np.isfinite(arr[ist]))
    assert np.all(np.isnan(mo.var[ins, :])) and np.all(np.isnan(mo.var[:, ins]))
    assert np.all(np.isfinite(mo.var[np.ix_(ist, ist)]))
    assert np.all(np.isnan(mo.variance_decomposition[ins])) and np.all(np.isfinite(mo.variance_decomposition[ist]))
    for a in mo.autocorr:
        assert np.all(np.isnan(a[ins, :])) and np.all(np.isnan(a[:, ins]))
    # correct values for the stationary ones, derived by hand
    se2, su2, rho = 0.01 ** 2, 0.02 ** 2, 0.8
    w, d, z, p = (names.index(v) for v in "wdzp")
    var_w = su2 / (1 - rho ** 2)
    phi1, phi2 = 0.5 + rho, -0.5 * rho                    # z is AR(2) with innovation u
    var_z = su2 * (1 - phi2) / ((1 + phi2) * ((1 - phi2) ** 2 - phi1 ** 2))
    np.testing.assert_allclose(mo.variance[[w, d, z, p]], [var_w, se2, var_z, var_w / 0.6 ** 2], rtol=1e-12)
    np.testing.assert_allclose(mo.mean[ist], 0, atol=1e-15)
    np.testing.assert_allclose([mo.autocorr[k][w, w] for k in range(3)], rho ** np.arange(1, 4), rtol=1e-12)
    r1 = phi1 / (1 - phi2)                                # Yule-Walker for the AR(2) z
    r2 = phi1 * r1 + phi2
    r3 = phi1 * r2 + phi2 * r1
    np.testing.assert_allclose([mo.autocorr[k][z, z] for k in range(3)], [r1, r2, r3], rtol=1e-12)
    assert abs(mo.corr[w, p] - 1) < 1e-12 and abs(mo.corr[w, d]) < 1e-12
    vd = mo.variance_decomposition                       # shocks e, u, v (v has zero variance)
    np.testing.assert_allclose(vd[[w, d, z, p]], [[0, 100, 0], [100, 0, 0], [0, 100, 0], [0, 100, 0]], atol=1e-10)
    # conditional variance decompositions are finite also for the non-stationary variables
    cv = mo.conditional_variance_decomposition
    assert np.all(np.isfinite(cv))
    np.testing.assert_allclose(cv[names.index("x")], [[1, 0, 0]] * 2, atol=1e-14)
    np.testing.assert_allclose(cv[names.index("q"), 0], [se2 / (se2 + su2), su2 / (se2 + su2), 0], rtol=1e-12)
    # filtered moments are finite for every variable (Dynare: all variables are stationary once filtered)
    hp = moments.theoretical_moments(cs.m, cs.dr(1), order=1, hp_filter=1600, ar=2)
    assert hp.stationary.all() and np.all(np.isfinite(hp.var)) and np.all(np.isfinite(hp.mean))


def test_unit_root_second_order_pruning_is_rejected(cases):
    """Dynare errors out on unit roots in the pruned state space at order 2; so do we."""
    cs = cases["mom_collard"]
    m = cs.m
    # build a decision rule with a unit root in the states by hand: a = 1 * a(-1) in ghx
    dr = dict(cs.dynare_dr(2))
    ghx = np.zeros_like(dr["ghx"])
    ghx[moments._Rules(m, dr).index_states[0], 0] = 1.0   # first state carries a unit root
    dr["ghx"] = ghx
    with pytest.raises(NotImplementedError):
        moments.theoretical_moments(m, dr, order=2, pruning=True)


# ------------------------------------------------------------------------------------ simulated moments
@pytest.mark.parametrize("name,case", [pytest.param(n, c, id=f"{n}-{c}") for n in MODELS for c in SIM_CASES
                                       if not (n == "mom_unitroot" and c.startswith("s2"))])
def test_simulated_moments_against_dynare(cases, name, case):
    """Moments of the path Dynare simulated (oo_.endo_simul), including the one-shock-at-a-time variance decomposition."""
    cs = cases[name]
    kw, sub = SIM_CASES[case]
    if kw["order"] == 2 and not cs.has(2):
        pytest.skip("k-order library not available")
    r = cs.r
    key = str(r[f"{case}_simkey"])
    endo, exo = r[f"sim_{key}_endo"].T, r[f"sim_{key}_exo"]
    ivar = r[f"{case}_ivar"].astype(int) - 1
    hp = bool(kw.get("hp_filter"))
    tol = dict(rtol=1e-8, atol=1e-9) if hp else dict(rtol=1e-9, atol=1e-11)
    mo = moments.simulated_moments(endo, ivar=ivar, model=cs.m, drop=30, shocks=exo, dr=cs.dr(kw["order"]), **kw)
    assert mo.var_names == [cs.m.endo_names[i] for i in ivar]
    for f in ("mean", "var", "skewness", "kurtosis"):
        close(getattr(mo, f), r[f"{case}_{f}"], what=f, **tol)
    close(mo.corr, r[f"{case}_contemporaneous_correlation"], what="corr", **tol)
    close(np.stack(mo.autocorr, axis=2), r[f"{case}_autocorr"], what="autocorr", **tol)
    close(mo.variance_decomposition, r[f"{case}_variance_decomposition"], what="vd", **tol)
    close(mo.std ** 2, np.diag(r[f"{case}_var"]), what="std", **tol)


@needs_lib
@pytest.mark.parametrize("name,case", [("mom_collard", "s1"), ("mom_aux", "s2p"), ("mom_schorfheide", "s1_hp")])
def test_stoch_simul_moments_reproduces_dynare_simulation(cases, name, case):
    """The convenience function with Dynare's shocks gives Dynare's simulated path and moments."""
    cs = cases[name]
    kw, _ = SIM_CASES[case]
    r = cs.r
    key = str(r[f"{case}_simkey"])
    kw = dict(kw, periods=160, drop=30)
    mo = moments.stoch_simul_moments(cs.m, None, cs.dr(kw["order"]), shocks=r[f"sim_{key}_exo"], **kw)
    assert mo.kind == "simulated" and mo.var_names == cs.m.orig_endo_names
    tol = dict(rtol=1e-8, atol=1e-9) if kw.get("hp_filter") else dict(rtol=1e-9, atol=1e-11)
    close(mo.var, r[f"{case}_var"], **tol)
    close(mo.variance_decomposition, r[f"{case}_variance_decomposition"], **tol)
    # random shocks: reproducible with a seeded generator, statistically close to the theoretical moments
    a = moments.stoch_simul_moments(cs.m, None, cs.dr(1), order=1, periods=300, drop=50, rng=np.random.default_rng(1))
    b = moments.stoch_simul_moments(cs.m, None, cs.dr(1), order=1, periods=300, drop=50, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a.var, b.var)
    with pytest.raises(ValueError):
        moments.stoch_simul_moments(cs.m, None, cs.dr(1), order=1, periods=40, drop=50)


# ------------------------------------------------------------------------------------ text output
def _lines(t):
    return [ln.rstrip() for ln in t.splitlines() if ln.strip()]


@pytest.mark.parametrize("name,case", [("mom_collard", "th1_hp"), ("mom_collard", "th2"), ("mom_unitroot", "th1"),
                                       ("mom_aux", "th1"), ("mom_aux", "th2p"), ("mom_aux", "th2p_sub"),
                                       ("mom_schorfheide", "th1"), ("mom_collard", "s1_hp"), ("mom_collard", "s2p")])
def test_text_output_matches_dynare_tables(cases, name, case):
    """Moments.format() reproduces the tables stoch_simul prints (title, layout, rounding)."""
    cs = cases[name]
    if f"{case}_text" not in cs.r:
        pytest.skip("no printed reference")
    kw, sub = (TH_CASES.get(case) or SIM_CASES[case])
    if kw["order"] == 2 and not cs.has(2):
        pytest.skip("k-order library not available")
    vl = SUBSET[name] if sub else None
    if case in SIM_CASES:
        key = str(cs.r[f"{case}_simkey"])
        mo = moments.stoch_simul_moments(cs.m, vl, cs.dr(kw["order"]), shocks=cs.r[f"sim_{key}_exo"],
                                         periods=160, drop=30, **kw)
    else:
        mo = moments.theoretical_moments(cs.m, cs.dr(kw["order"]), vl, **kw)
    mine, theirs = _lines(mo.format()), _lines(str(cs.r[f"{case}_text"]))
    if mine != theirs:
        # Dynare prints a numerical zero (+-1e-17) with its sign, and widens the table by one column when
        # such noise is negative. The sign of that noise depends on the solver's rounding, so fall back to
        # comparing the tokens, with zeros of either sign treated as the same number.
        mine, theirs = ([[x.lstrip("-") if set(x) <= set("-0.") else x for x in ln.split()] for ln in side]
                        for side in (mine, theirs))
    assert mine == theirs


# ------------------------------------------------------------------------------------ building blocks
def test_lyapunov_symm_matches_scipy_and_projects_unit_roots():
    rng = np.random.default_rng(3)
    n = 9
    A = rng.standard_normal((n, n))
    A *= 0.9 / max(abs(np.linalg.eigvals(A)))               # stable, complex eigenvalues
    Bm = rng.standard_normal((n, n))
    Q = Bm @ Bm.T
    X, u = moments.lyapunov_symm(A, Q)
    assert u.shape == (n, 0)
    np.testing.assert_allclose(X, linalg.solve_discrete_lyapunov(A, Q), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(X, A @ X @ A.T + Q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(moments.disclyap_fast(A, Q)[0], X, rtol=1e-11, atol=1e-12)
    # real-eigenvalue case, scalar case
    A2 = np.diag([0.5, -0.3, 0.9])
    np.testing.assert_allclose(moments.lyapunov_symm(A2, np.eye(3))[0], np.diag(1 / (1 - np.diag(A2) ** 2)))
    np.testing.assert_allclose(moments.lyapunov_symm(np.array([[0.5]]), np.array([[2.0]]))[0], [[2 / 0.75]])
    # a unit root: the solution lives on the stable subspace, the unit-root Schur vector is returned
    S = rng.standard_normal((4, 4))
    Sinv = np.linalg.inv(S)
    Au = S @ linalg.block_diag(1.0, [[0.5, 0.3], [-0.3, 0.5]], 0.4) @ Sinv
    Xu, uu = moments.lyapunov_symm(Au, np.eye(4))
    assert uu.shape == (4, 1)
    P = np.eye(4) - uu @ uu.T                              # projector on the stable Schur subspace
    np.testing.assert_allclose(Xu, P @ Xu @ P, atol=1e-12)
    np.testing.assert_allclose(Xu, P @ Au @ Xu @ Au.T @ P + P @ np.eye(4) @ P, atol=1e-10)
    # an explosive root is projected out as well (|lambda| > 1 - 1e-6)
    assert moments.lyapunov_symm(np.diag([1.2, 0.5]), np.eye(2))[1].shape == (2, 1)
    assert moments.disclyap_fast(np.diag([1.2, 0.5]), np.eye(2))[1]       # error flag: no convergence


def test_filters_against_direct_formulas():
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.standard_normal((60, 2)), axis=0)
    # HP: trend minimises sum (y - tau)^2 + lambda sum (second differences of tau)^2
    T, lam = len(y), 1600.0
    Dm = np.diff(np.eye(T), 2, axis=0)
    trend_ref = np.linalg.solve(np.eye(T) + lam * Dm.T @ Dm, y)
    tr, cy = moments.hp_filter(y, lam)
    np.testing.assert_allclose(tr, trend_ref, rtol=1e-9)
    np.testing.assert_allclose(cy, y - trend_ref, rtol=1e-8, atol=1e-9)
    # Baxter-King: symmetric weights summing to zero, kills constants and linear trends
    t = np.arange(80.0)
    out = moments.baxter_king_filter(np.column_stack([np.ones(80), t, np.sin(2 * np.pi * t / 12)]), 6, 32, 12)
    assert out.shape == (80 - 24, 3)
    np.testing.assert_allclose(out[:, :2], 0, atol=1e-12)
    assert np.max(np.abs(out[:, 2])) > 0.9                  # business-cycle frequency passes
    # one-sided HP: causal (no look-ahead) and reduces to the data minus a smooth trend
    t1, c1 = moments.one_sided_hp_filter(y, 1600.0)
    y_alt = y.copy()
    y_alt[40:] += 10.0
    t2, _ = moments.one_sided_hp_filter(y_alt, 1600.0)
    np.testing.assert_allclose(t1[:40], t2[:40], rtol=1e-12)


def test_variance_decomposition_with_correlated_shocks(cases):
    """Shares come from the lower Cholesky factor of Sigma_e (order of varexo matters) and sum to 100;
    checked against a brute-force MA(infinity) sum, independent of Dynare."""
    cs = cases["mom_collard"]
    m = cs.m
    sol = pydyn.first_order(m)
    mo = moments.theoretical_moments(m, sol, order=1, ar=1)
    np.testing.assert_allclose(mo.variance_decomposition.sum(axis=1), 100, rtol=1e-12)
    L = np.linalg.cholesky(m.Sigma_e)
    assert abs(m.Sigma_e[0, 1]) > 0
    contrib = []
    for i in range(2):
        v, Gk = np.zeros(m.n_endo), np.eye(m.n_endo)
        for _ in range(4000):
            f = Gk @ sol.H @ L[:, i]
            v += f * f
            Gk = sol.G @ Gk
        contrib.append(v)
    contrib = np.column_stack(contrib)
    ref = 100 * contrib / contrib.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(mo.variance_decomposition, ref, rtol=1e-8, atol=1e-8)
    # reversing the shock order changes the attribution (the first shock gets the common component)
    P = np.array([[0, 1], [1, 0]])
    dr_sw = dict(sol.as_dr())
    dr_sw["ghu"] = sol.ghu @ P
    sw = moments.theoretical_moments(m, dr_sw, Sigma_e=P @ m.Sigma_e @ P, order=1, ar=1)
    np.testing.assert_allclose(sw.var, mo.var, rtol=1e-12)
    assert np.max(np.abs(sw.variance_decomposition[:, ::-1] - mo.variance_decomposition)) > 0.1


def test_conditional_variance_decomposition_with_measurement_error(cases):
    cs = cases["mom_aux"]
    assert cs.m.varobs == ["c", "z", "k"] and cs.m.measurement_errors == pytest.approx({"c": 0.0025, "z": 9e-6})
    dr = cs.dr(1)
    cvd, cvd_me = moments.conditional_variance_decomposition(cs.m, dr, [1, 3], ["z", "c", "r"])
    np.testing.assert_allclose(cvd.sum(axis=2), 1, rtol=1e-12)
    assert cvd_me.shape == (2, 2, 3)                         # observed among the requested: z, c (in this order)
    np.testing.assert_allclose(cvd_me.sum(axis=2), 1, rtol=1e-12)
    with np.errstate(all="ignore"):                          # the ME rescales all shocks of a row by one factor
        ratio = cvd_me[:, :, :2] / cvd[:2]
    np.testing.assert_allclose(ratio[:, :, 0], ratio[:, :, 1], rtol=1e-10)
    assert np.all(ratio[:, :, 0] < 1) and np.all(cvd_me[:, :, 2] > 0)
    with pytest.raises(ValueError):
        moments.conditional_variance_decomposition(cs.m, dr, [0, 1], ["z"])
    with pytest.raises(ValueError):
        moments.conditional_variance_decomposition(cs.m, dr, [4, 2], ["z"])
    # zero measurement error: no ME table
    assert moments.conditional_variance_decomposition(cs.m, dr, [1], ["z"], H=np.zeros(3))[1] is None


@needs_lib
def test_pruned_second_order_closed_form(tmp_path):
    """x is AR(1), y = x + x^2/2 (static). With the pruned state space y = x + x^2/2 with Gaussian x, so
    E y = s2/2, Var y = s2 + s2^2/2, Cov(y_t, y_t-k) = rho^k s2 + rho^2k s2^2/2 (s2 = Var x); Dynare's
    order-2 moments without pruning keep the first-order variance and only correct the mean."""
    shutil.copy(REF / "mom_quad.mod", tmp_path)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "mom_quad.mod"))
    rho, s2 = 0.8, 0.1 ** 2 / (1 - 0.8 ** 2)
    dr = moments.solve(m, 2)
    pr = moments.theoretical_moments(m, dr, order=2, pruning=True, ar=3)
    var_y = s2 + 0.5 * s2 ** 2
    acov = [rho ** k * s2 + 0.5 * rho ** (2 * k) * s2 ** 2 for k in (1, 2, 3)]
    np.testing.assert_allclose(pr.mean, [0.0, 0.5 * s2], atol=1e-15)
    np.testing.assert_allclose(pr.variance, [s2, var_y], rtol=1e-12)
    np.testing.assert_allclose([a[1, 1] for a in pr.autocorr], np.array(acov) / var_y, rtol=1e-11)
    np.testing.assert_allclose([a[0, 0] for a in pr.autocorr], rho ** np.arange(1, 4), rtol=1e-11)
    np.testing.assert_allclose(pr.corr[0, 1], s2 / np.sqrt(s2 * var_y), rtol=1e-11)
    un = moments.theoretical_moments(m, dr, order=2, pruning=False, ar=1)
    np.testing.assert_allclose(un.mean, pr.mean, atol=1e-15)
    np.testing.assert_allclose(un.variance, [s2, s2], rtol=1e-12)
    np.testing.assert_allclose(un.mean_correction, [0.0, 0.5 * s2], atol=1e-15)


def test_unsupported_requests_raise(cases):
    cs = cases["mom_collard"]
    with pytest.raises(ValueError):
        moments.theoretical_moments(cs.m, cs.dr(1), order=1, one_sided_hp_filter=1600)
    with pytest.raises(ValueError):
        moments.theoretical_moments(cs.m, cs.dr(1), order=1, hp_filter=1600, bandpass_filter=(6, 32))
    with pytest.raises(NotImplementedError):
        moments.theoretical_moments(cs.m, cs.dr(1), order=3)
    with pytest.raises(TypeError):
        moments.theoretical_moments(cs.m, cs.dr(1), order=1, no_such_option=1)
    if HAVE_LIB:
        with pytest.raises(NotImplementedError):
            moments.theoretical_moments(cs.m, cs.dr(2), order=2, hp_filter=1600)
