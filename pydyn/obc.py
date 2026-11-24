# This file is part of pydyn. Released under the GNU General Public License, version 3 or later; see LICENSE.
# No warranty.
#
# The news-shock representation of an occasionally binding constraint as a linear complementarity problem, the
# homogenised mixed-integer formulation in ``solve_news`` and the role of the matrix M follow
#   T. D. Holden, "Existence and uniqueness of solutions to dynamic models with occasionally binding constraints"
#   and "Computation of solutions to dynamic models with occasionally binding constraints",
# and the constraint set of DynareOBC's Core/InnerProblem/FormOptimizer.m (GPL-3.0-or-later). The code here is
# written from those descriptions for pydyn's own data structures; it is not a translation of DynareOBC.
"""Occasionally binding constraints by news shocks (``obc_method='news'``).

The model is specified exactly as for OccBin: an ``occbin_constraints`` block and ``bind``/``relax`` equation
tags.  For each constraint the post-processor (``constraint_objects``) derives a complementary pair

* the gap ``g``: the residual of the alternative-regime ("bind") equation, zero while that regime holds;
* the wedge ``y``: the residual of the reference-regime ("relax") equation, zero while that regime holds;

both normalised to be non-negative.  With wedges announced ``horizon`` periods ahead the constrained path
satisfies ``g = q + M y >= 0, y >= 0, y'g = 0``.  See ``claude/obc-shared-contract.md`` in the project notes.

Order 1 only.  Each period is solved under perfect foresight with no future shocks expected, as in OccBin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .expr import NAMESPACE, Translator, tokenize
from .occbin import Occbin, OccbinResult, _split_comparison

__all__ = ["OBCError", "OBCClassError", "ConstraintObject", "NewsOBC", "RegimeOBC", "constraint_objects", "solve_news",
           "solve_active_set", "m_diagnostics", "verify_lcp"]


class OBCError(Exception):
    """Failure of a news-shock solve. ``status`` is 'no_solution', 'horizon_too_short' or 'not_converged';
    ``period`` is the 1-based shock period."""

    def __init__(self, message, status="no_solution", period=None):
        super().__init__(message)
        self.status, self.period = status, period


class OBCClassError(ValueError):
    """The constraint cannot be written as one complementary pair (see ``ConstraintObject.cls``)."""


@dataclass
class ConstraintObject:
    name: str
    param: int                       # index of occbin_<name>_bind
    equations: list                  # rows of the dynamic model that change with the parameter
    cls: str                         # 'complementarity', 'multi_equation' or 'inconsistent'
    why: str = ""
    ref: np.ndarray | None = None    # affine reference residual: [y(-1) | y | y(+1) | u | const]
    alt: np.ndarray | None = None    # affine alternative residual, same layout
    s_g: float = 1.0                 # g = s_g * alt
    s_y: float = 1.0                 # y = s_y * ref
    g_ss: float = np.nan             # gap at the steady state (must be > 0)
    fit: dict = field(default_factory=dict)   # coefficients and residuals of the entry and exit tests
    approx: float = 0.0              # largest relative residual of the two tests (0 when they hold exactly)


# --------------------------------------------------------------------------------------------- post-processor
def _condition_fn(model, src):
    def resolve(name, shift):
        if name in model._ie:
            if shift != 0:
                raise NotImplementedError("leads/lags in occbin constraints")
            return f"z[:, {model._ie[name]}]"
        if name in model._ip:
            return f"params[{model._ip[name]}]"
        return None

    tr = Translator(resolve, steady=lambda v: f"ss[{model._ie[v]}]", shiftable=set(model._ie))
    return eval("lambda z, params, ss: " + tr(src), dict(NAMESPACE))  # noqa: S307 - model expression


def _affine_condition(ob: Occbin, expr: str, enter: bool):
    """Affine form (layout of ConstraintObject.ref) of a condition written so that it holds when the
    returned function is negative (``enter=True``, bind) or positive (``enter=False``, relax).  Depends on the
    parameters only, so it is kept on the Occbin instance (which is built per parameter set)."""
    cache = ob.__dict__.setdefault("_affine_cache", {})
    key = (expr, bool(enter))
    if key not in cache:
        cache[key] = _affine_condition_uncached(ob, expr, enter)
    return None if cache[key] is None else cache[key].copy()


def _affine_condition_uncached(ob: Occbin, expr: str, enter: bool):
    parts = _split_comparison(expr)
    if parts is None:
        return None
    lhs, op, rhs = parts
    f = _condition_fn(ob.model, f"({lhs}) - ({rhs})")
    sign = 1.0 if op in ("<", "<=") else -1.0          # 'lhs < rhs'  <=>  lhs - rhs < 0
    if not enter:
        sign = -sign                                   # 'lhs > rhs'  <=>  lhs - rhs > 0
    n, nx, ys = ob.n, ob.n_exo, ob.ys

    def val(z):
        return float(np.ravel(np.asarray(f(z[None, :], ob.params, ys), float))[0])

    out = np.zeros(3 * n + nx + 1)
    out[-1] = sign * val(ys)
    for i in sorted(_used_variables(ob.model, expr)):     # the other variables have derivative exactly zero
        h = 1e-6 * max(1.0, abs(ys[i]))
        zp, zm = ys.copy(), ys.copy()
        zp[i] += h
        zm[i] -= h
        out[n + i] = sign * (val(zp) - val(zm)) / (2 * h)
    return out


def _used_variables(model, src):
    """Indices of the endogenous variables named in a condition (all of them if it cannot be tokenised)."""
    try:
        return {model._ie[v] for kind, v in tokenize(src) if kind == "id" and v in model._ie}
    except Exception:  # noqa: BLE001
        return set(range(model.n_endo))


def _fit(target, a, b):
    """Least squares target ~ ca*a + cb*b; returns (ca, cb, relative residual)."""
    X = np.column_stack([a, b])
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    res = np.linalg.norm(X @ coef - target) / max(np.linalg.norm(target), 1e-300)
    return float(coef[0]), float(coef[1]), float(res)


def constraint_objects(ob: Occbin, tol: float = 1e-6, approx_tol: float = 0.05) -> list:
    """The ConstraintObjects of ``ob`` (see ``_constraint_objects``), computed once per Occbin instance and
    tolerance pair: they depend on the parameters only, and ``Occbin`` is built per parameter set."""
    cache = ob.__dict__.setdefault("_constraint_cache", {})
    key = (tol, approx_tol)
    if key not in cache:
        cache[key] = _constraint_objects(ob, tol, approx_tol)
    return list(cache[key])


def _constraint_objects(ob: Occbin, tol: float = 1e-6, approx_tol: float = 0.05) -> list:
    """Build one ConstraintObject per constraint of the ``occbin_constraints`` block (contract section 3).

    The entry and exit tests compare the linearised bind/relax conditions with the linearised equation
    residuals.  They hold exactly (relative residual below ``tol``) when the conditions are linear in the same
    terms as the equations.  When the equations carry a term the conditions lack (e.g. ``inom = inomlb*exp(mp)``
    with ``bind inomnot <= inomlb``) the two agree only up to that term; such constraints are accepted when the
    relative residual is below ``approx_tol`` and flagged through ``ConstraintObject.approx``.  OccBin switches
    on the condition and the news-shock methods on the linearised residual, so their paths then differ by a
    term of that order."""
    n = ob.n
    C0, B0, A0, J0, D0 = ob.regime_matrices(tuple([False] * ob.k))
    ref_all = np.column_stack([C0, B0, A0, J0, D0])
    out = []
    for j, (name, c) in enumerate(zip(ob.names, ob.model.occbin_constraints)):
        reg = tuple(i == j for i in range(ob.k))
        alt_all = np.column_stack(ob.regime_matrices(reg))
        scale = np.maximum(np.abs(ref_all).max(axis=1), np.abs(alt_all).max(axis=1))
        rows = [int(r) for r in np.nonzero(np.abs(alt_all - ref_all).max(axis=1) > 1e-12 * np.maximum(scale, 1.0))[0]]
        co = ConstraintObject(name=name, param=ob.switch[j], equations=rows, cls="complementarity")
        if len(rows) == 0:
            co.cls, co.why = "inconsistent", "no equation changes between the regimes"
        elif len(rows) > 1:
            co.cls, co.why = "multi_equation", f"{len(rows)} equations change between the regimes"
        if co.cls != "complementarity":
            out.append(co)
            continue
        e = rows[0]
        co.ref, co.alt = ref_all[e].copy(), alt_all[e].copy()
        try:
            d_b = _affine_condition(ob, c["bind"], enter=True)
            relax = c.get("relax") or None
            d_r = _affine_condition(ob, relax, enter=False) if relax else None
        except NotImplementedError as err:
            co.cls, co.why = "inconsistent", str(err)
            out.append(co)
            continue
        if d_b is None or (relax and d_r is None):
            co.cls, co.why = "inconsistent", "bind/relax condition is not a single comparison"
            out.append(co)
            continue
        if d_r is None:
            d_r = d_b          # no relax condition: OccBin leaves when the bind condition fails, i.e. d_b >= 0
        # entry: inside the reference regime (ref = 0) the bind condition is a statement about the gap
        a, b, res_b = _fit(d_b, co.alt, co.ref)
        # exit: inside the alternative regime (alt = 0) the relax condition is a statement about the wedge
        cc, ee, res_r = _fit(d_r, co.ref, co.alt)
        co.fit = dict(entry=(a, b, res_b), exit=(cc, ee, res_r))
        co.approx = max(res_b, res_r) if max(res_b, res_r) > tol else 0.0
        if res_b > approx_tol or abs(a) < tol:
            co.cls, co.why = "inconsistent", "the bind condition is not equivalent to the alternative equation's residual changing sign"
        elif res_r > approx_tol or abs(cc) < tol:
            co.cls, co.why = "inconsistent", "the relax condition is not equivalent to the reference equation's residual changing sign"
        else:
            co.s_g = float(np.sign(a))           # bind  <=>  d_b < 0  <=>  g < 0
            co.s_y = -float(np.sign(cc))         # relax <=>  d_r > 0  <=>  y < 0
            co.g_ss = co.s_g * co.alt[-1]
            if not co.g_ss > 0:
                co.cls, co.why = "inconsistent", f"the gap is not strictly positive at the steady state ({co.g_ss:.3g})"
        out.append(co)
    return out


# --------------------------------------------------------------------------------------------------- solvers
def verify_lcp(M, q, y, n_s, tol=1e-8):
    """True if y >= 0, q + M y >= 0 on every row, and complementarity holds on the first n_s rows."""
    g = q + M @ y
    scale = max(1.0, float(np.abs(q).max(initial=0.0)))
    return bool(np.all(y >= -tol * scale) and np.all(g >= -tol * scale)
                and np.all(np.minimum(np.abs(y), np.abs(g[:n_s])) <= tol * scale))


def solve_active_set(M, q, n_s, tol=1e-8, maxit=60):
    """Guess-and-verify on the binding set: start from the periods where q < 0, solve the square system on the
    guessed set, drop negative wedges, add violated periods.  Returns y or None (no guarantee of success)."""
    Ms = M[:n_s]
    S = q[:n_s] < -tol
    seen = set()
    for _ in range(maxit):
        key = S.tobytes()
        if key in seen:
            return None
        seen.add(key)
        y = np.zeros(n_s)
        if S.any():
            try:
                y[S] = np.linalg.solve(Ms[np.ix_(S, S)], -q[:n_s][S])
            except np.linalg.LinAlgError:
                return None
        g = q + M @ y
        drop = S & (y < -tol)
        add = (~S) & (g[:n_s] < -tol)
        if not drop.any() and not add.any():
            return y if verify_lcp(M, q, y, n_s, tol) else None
        S = (S & ~drop) | add
    return None


def solve_news(M, q, n_s, omega=1000.0, tol=1e-8, horizons=None, n_c=1):
    """Solve ``y >= 0, q + M y >= 0, y'(q + M y)[:n_s] = 0`` as a mixed-integer linear programme.

    Homogenised formulation (Holden): maximise alpha subject to ``0 <= yn <= z``, ``alpha q + M yn >= 0`` and
    ``alpha q_s + M_s yn <= omega (1 - z)`` with z binary; then ``y = yn / alpha``.  alpha > 0 at the optimum if
    and only if a solution exists with wedges confined to the horizon.  ``horizons`` lists the numbers of leading
    periods (per constraint) in which wedges are allowed, tried in order; the first that works is returned, so
    shorter binding spells are preferred.  Returns (y, status)."""
    q = np.asarray(q, float)
    if np.all(q >= -tol * max(1.0, np.abs(q).max(initial=0.0))):
        return np.zeros(n_s), "ok"
    qn = np.abs(q).max()
    qs = q / qn
    msc = np.abs(M).max()
    Mn = M / msc                                   # y = (msc-scaled) -> true wedge = yn / alpha * qn / msc
    T = n_s // n_c
    horizons = [T] if horizons is None else list(horizons)
    nv = 2 * n_s + 1                               # [yn (n_s) | alpha | z (n_s)]
    cobj = np.zeros(nv)
    cobj[n_s] = -1.0
    I = np.eye(n_s)
    A1 = np.hstack([I, np.zeros((n_s, 1)), -I])                                  # yn - z <= 0
    A2 = np.hstack([Mn, qs[:, None], np.zeros((M.shape[0], n_s))])               # alpha q + M yn >= 0
    A3 = np.hstack([Mn[:n_s], qs[:n_s, None], omega * I])                        # ... <= omega (1 - z)
    cons = [LinearConstraint(A1, -np.inf, 0.0), LinearConstraint(A2, 0.0, np.inf),
            LinearConstraint(A3, -np.inf, omega)]
    integrality = np.r_[np.zeros(n_s + 1), np.ones(n_s)]
    for Tss in horizons:
        ub_y = np.zeros(n_s)
        for c in range(n_c):
            ub_y[c * T:c * T + Tss] = 1.0
        bounds = Bounds(np.zeros(nv), np.r_[ub_y, 1e8, ub_y])
        res = milp(cobj, constraints=cons, integrality=integrality, bounds=bounds,
                   options=dict(mip_rel_gap=0.0))
        if res.x is None or not res.success:
            continue
        alpha = res.x[n_s]
        if alpha <= 1e-9:
            continue
        y = np.maximum(res.x[:n_s] / alpha, 0.0) * qn / msc
        z = res.x[n_s + 1:] > 0.5
        # polish: with the binding set known the wedges solve a square linear system exactly
        if z.any():
            try:
                yp = np.zeros(n_s)
                yp[z] = np.linalg.solve(M[:n_s][np.ix_(z, z)], -q[:n_s][z])
                if verify_lcp(M, q, yp, n_s, tol):
                    y = yp
            except np.linalg.LinAlgError:
                pass
        if verify_lcp(M, q, y, n_s, max(tol, 1e-7)):
            return y, "ok"
    return None, "no_solution"


def _is_p_exact(A):
    """P-matrix test by the recursion of Tsatsomeros and Li (2000): A is P iff a11 > 0 and both A[1:,1:] and
    the Schur complement of a11 are P.  2^n small steps instead of 2^n determinants."""
    A = np.asarray(A, float)
    if not A[0, 0] > 0:
        return False
    if A.shape[0] == 1:
        return True
    B = A[1:, 1:]
    return _is_p_exact(B) and _is_p_exact(B - np.outer(A[1:, 0], A[0, 1:]) / A[0, 0])


def _is_p_minors(A):
    """Reference implementation by enumeration of principal minors (tests only)."""
    n = A.shape[0]
    return all(np.linalg.det(A[np.ix_(idx, idx)]) > 0 for r in range(1, n + 1) for idx in combinations(range(n), r))


def m_diagnostics(Ms, n_c=1, max_exact=16):
    """Existence and uniqueness diagnostics on the square matrix M (Holden's theory paper).

    * ``is_s``: M is an S-matrix (some y > 0 has M y > 0); necessary for a solution to exist for every q.
    * ``is_p``: True / False / None.  A P-matrix (all principal minors positive) gives exactly one solution
      for every q.  Decided exactly up to ``max_exact`` rows; above that True only when the symmetric part
      is positive definite (sufficient), False when a negative principal minor is found among the leading
      and diagonal ones, otherwise None (undecided).
    * ``largest_p``: the largest T' (per constraint) for which the leading block is a P-matrix by the exact
      test, capped at ``max_exact // n_c``.
    """
    Ms = np.asarray(Ms, float)
    N = Ms.shape[0]
    T = N // n_c
    res = linprog(np.r_[np.zeros(N), -1.0], A_ub=np.hstack([-Ms, np.ones((N, 1))]), b_ub=np.zeros(N),
                  bounds=[(0, 1)] * N + [(None, 1)])
    is_s = bool(res.status == 0 and -res.fun > 1e-9)
    pd = bool(np.all(np.linalg.eigvalsh((Ms + Ms.T) / 2) > 0))

    def block(t):
        idx = np.concatenate([np.arange(c * T, c * T + t) for c in range(n_c)])
        return Ms[np.ix_(idx, idx)]

    largest = 0
    for t in range(1, min(T, max_exact // n_c) + 1):
        if _is_p_exact(block(t)):
            largest = t
        else:
            break
    if N <= max_exact:
        is_p = largest == T
    elif pd:
        is_p = True
    elif largest < min(T, max_exact // n_c) or np.any(np.diag(Ms) <= 0) or \
            any(np.linalg.det(block(t)) <= 0 for t in range(1, T + 1)):
        is_p = False
    else:
        is_p = None
    return dict(is_s=is_s, is_p=is_p, symmetric_part_positive_definite=pd, largest_p=largest,
                min_diag=float(np.diag(Ms).min()))


def _functional_rows(P, aC, aB, aA, N):
    """Rows ``r_t`` (t = 0..N) of the response of the linear functional  a_C y(t-1) + a_B y(t) + a_A y(t+1)  to a
    state carried forward by P:  r_0 = a_B + a_A P,  r_1 = a_C + a_B P + a_A P^2,  r_{t+1} = r_t P.  ``aC``, ``aB``, ``aA``
    have shape (m, n); returns (N + 1, m, n)."""
    m, n = aB.shape
    R = np.empty((N + 1, m, n))
    R[0] = aB + aA @ P
    if N >= 1:
        r = aC + aB @ P + (aA @ P) @ P
        R[1] = r
        for t in range(2, N + 1):
            r = r @ P
            R[t] = r
    return R


def _wedge_vectors(G, FE, T):
    """Columns ``G^s F e_w`` (wedge block w, news period s): the response in period 0 of the wedge announced for
    period s.  Returns (n, n_w * T), column w*T + s."""
    n, nw = FE.shape
    V = np.empty((n, nw, T))
    v = FE
    for s in range(T):
        V[:, :, s] = v
        v = G @ v
    return V.reshape(n, nw * T)


def _response_matrix(R, V, aA, rows, T):
    """M[(i, t), (w, s)] = response of functional i in period t (< rows) to a unit wedge w announced for period s.

    The response of the path to a wedge at s + 1 equals the response to a wedge at s shifted by one period, plus
    the free response P^t v_{s+1} of the state jump v_{s+1} = G^(s+1) F e in period 0.  Hence along each
    diagonal  M[t, s+1] = M[t-1, s] + r_t v_{s+1}  (t >= 1) and  M[0, s+1] = a_A v_s + r_0 v_{s+1};  first column
    M[t, 0] = r_t v_0.  One matrix product for R V and a loop of vector additions over t: no n x n product per
    (t, s) pair."""
    m, n = R.shape[1], R.shape[2]
    nw = V.shape[1] // T
    RV = (R[:rows].reshape(rows * m, n) @ V).reshape(rows, m, nw, T)
    AV = (aA @ V).reshape(m, nw, T)
    M = np.empty_like(RV)
    M[0] = RV[0]
    M[0, :, :, 1:] += AV[:, :, :-1]
    for t in range(1, rows):
        M[t, :, :, 0] = RV[t, :, :, 0]
        M[t, :, :, 1:] = M[t - 1, :, :, :-1] + RV[t, :, :, 1:]
    return M.transpose(1, 0, 2, 3).reshape(m * rows, nw * T)


def _full_responses(P, V, N, T):
    """Responses (N+1, n, n_w*T) of the path to every unit wedge, by one forward recursion over t."""
    n = P.shape[0]
    nw = V.shape[1] // T
    K = V.reshape(n, nw, T)
    out = np.empty((N + 1, n, nw * T))
    Y = np.zeros((n, nw, T))
    for t in range(N + 1):
        Y = (P @ Y.reshape(n, -1)).reshape(n, nw, T)
        if t < T:
            Y[:, :, t:] += K[:, :, :T - t]
        out[t] = Y.reshape(n, nw * T)
    return out


def _wedge_path(P, G, FE, init_term, w, T, upto):
    """Path y_0..y_upto of  y_t = P y_{t-1} + u_t  with  u_t = sum_d G^d F E w_{t+d}  (wedges in periods < T), starting from
    P y_{-1} + (shock term) = ``init_term`` in period 0.  ``w``: (n_w * T) wedges, block-major."""
    n, nw = FE.shape
    wt = w.reshape(nw, T)
    H = np.empty((T, n))
    h = np.zeros(n)
    for t in range(T - 1, -1, -1):
        h = FE @ wt[:, t] + G @ h
        H[t] = h
    Y = np.empty((upto + 1, n))
    y = init_term + H[0]
    Y[0] = y
    for t in range(1, upto + 1):
        y = P @ y
        if t < T:
            y = y + H[t]
        Y[t] = y
    return Y


def _map_regime(col):
    """``Occbin.map_regime`` of a 0/1 integer column, without the Python loop over its elements."""
    ch = np.flatnonzero(col[1:] != col[:-1])
    if ch.size == 0:
        return [int(col[0])], [1]
    return [int(col[0])] + col[ch + 1].tolist(), [1] + (ch + 2).tolist()


def _regime_history(y, T, n_c):
    """Regime history entry of one shock period from the wedges ``y`` (constraint-major blocks of length T)."""
    if not y.any():
        return [([0], [1]) for _ in range(n_c)]
    scale = max(1.0, float(np.abs(y).max()))
    flag = (y > 1e-9 * scale).astype(np.int64)
    return [_map_regime(flag[c * T:(c + 1) * T]) for c in range(n_c)]


# ------------------------------------------------------------------------------------------------- the model
class NewsOBC:
    """News-shock solution of a model with an ``occbin_constraints`` block.

    ``horizon``: number of periods in which wedges may be non-zero.  ``check_ahead``: further periods over
    which the gap must stay non-negative.  Raises OBCClassError if a constraint is not a complementary pair.
    """

    def __init__(self, model_or_occbin, params=None, horizon=40, check_ahead=40):
        self.ob = ob = model_or_occbin if isinstance(model_or_occbin, Occbin) else Occbin(model_or_occbin, params)
        self.constraints = constraint_objects(ob)
        bad = [c for c in self.constraints if c.cls != "complementarity"]
        if bad:
            raise OBCClassError("; ".join(f"constraint '{c.name}' is {c.cls}: {c.why}" for c in bad)
                                + ". The news-shock methods need one complementary pair per constraint; "
                                  "use obc_method='occbin' for this model.")
        self.T, self.L = int(horizon), int(check_ahead)
        self.n_c = len(self.constraints)
        self._build(self.T + self.L)

    # ----------------------------------------------------------------------------------------- M and Q
    def _build(self, N):
        """M and the affine map of q.  Everything follows from the rows r_t of ``_functional_rows`` and the news
        vectors G^s F e; the (N+1, n, n_c*T) array of wedge responses is only built if asked for (``Yw``)."""
        ob, n, nx, T, nc = self.ob, self.ob.n, self.ob.n_exo, self.T, self.n_c
        C0, B0, A0, _J0, _D0 = ob.regime_matrices(tuple([False] * ob.k))
        lu = lu_factor(B0 + A0 @ ob.P)
        G = -lu_solve(lu, A0)                           # G = -F A0 with F = (B0 + A0 P)^-1, F never formed
        E = np.zeros((n, nc))
        for ci, co in enumerate(self.constraints):
            E[co.equations[0], ci] = co.s_y             # ref_resid = s_y * y
        FE = lu_solve(lu, E)
        self.N, self._G, self._FE, self._Yw = N, G, FE, None
        V = _wedge_vectors(G, FE, T)
        a = np.array([co.s_g * co.alt for co in self.constraints])          # (nc, 3n + nx + 1)
        aC, aB, aA, aJ, a0 = a[:, :n], a[:, n:2 * n], a[:, 2 * n:3 * n], a[:, 3 * n:3 * n + nx], a[:, -1]
        R = _functional_rows(ob.P, aC, aB, aA, N)                           # (N+1, nc, n)
        self.M = _response_matrix(R, V, aA, N, T)       # (nc*N, nc*T)
        self._rows_s = np.concatenate([np.arange(c * N, c * N + T) for c in range(nc)])
        self._rows_tail = np.setdiff1d(np.arange(nc * N), self._rows_s)
        self._order = np.r_[self._rows_s, self._rows_tail]
        self.Mo = self.M[self._order]                   # complementarity rows first, then check-ahead rows
        self.Ms = self.Mo[:nc * T]
        # q = q0 + Qs init + Qu shock: the initial state enters through r_{t+1}, the shock through r_t Q, and
        # through the shock term of the residual in the first period
        Qinit = R[1:N + 1].transpose(1, 0, 2).reshape(nc * N, n)
        Qsh = (R[:N] @ ob.Q).transpose(1, 0, 2)         # (nc, N, nx)
        Qsh[:, 0, :] += aJ
        Qm = np.hstack([Qinit, Qsh.reshape(nc * N, nx)])
        q0 = np.repeat(a0, N)
        self.q0, self.Qm = q0[self._order], Qm[self._order]
        self._V0 = V
        self._prepare_step()

    @property
    def Yw(self):
        """Response of every variable (N+1 periods) to a unit wedge of each constraint at each horizon: array
        (N+1, n, n_c*T).  Built on first use; the solver itself does not need it."""
        if self._Yw is None:
            self._Yw = _full_responses(self.ob.P, self._V0, self.N, self.T)
        return self._Yw

    def _prepare_step(self):
        """Objects for the per-period step: the response of the next state to the wedges, and q as a map from the
        state variables (the variables with a non-negligible column in P or in the map of the initial state)
        and the shocks only."""
        ob, n = self.ob, self.ob.n
        # self._V0 (n, n_c*T): next state = P x + Q u + V0 y
        scale = max(1.0, float(np.abs(ob.P).max()), float(np.abs(self.Qm[:, :n]).max()))
        used = np.maximum(np.abs(ob.P).max(axis=0), np.abs(self.Qm[:, :n]).max(axis=0)) > 1e-14 * scale
        self._st = np.flatnonzero(used)
        self._Ps = np.ascontiguousarray(ob.P[:, self._st])
        self._Qx = np.ascontiguousarray(np.hstack([self.Qm[:, self._st], self.Qm[:, n:]]))
        self._PQ = None

    def _gap_of(self, Y, const=True, shock=None):
        """Gaps along paths Y of shape (N+1, n, ...) -> (n_c*N, ...): g_c(t) for t = 0..N-1 (period N is only
        used as the lead).  The lag at t=0 is the initial state and enters through ``_gap_init``."""
        n, nx, N = self.ob.n, self.ob.n_exo, Y.shape[0] - 1
        out = np.zeros((self.n_c * N,) + Y.shape[2:])
        for ci, co in enumerate(self.constraints):
            a = co.s_g * co.alt
            aC, aB, aA, aJ, a0 = a[:n], a[n:2 * n], a[2 * n:3 * n], a[3 * n:3 * n + nx], a[-1]
            for t in range(N):
                v = np.tensordot(aB, Y[t], axes=(0, 0)) + np.tensordot(aA, Y[t + 1], axes=(0, 0))
                if t > 0:
                    v = v + np.tensordot(aC, Y[t - 1], axes=(0, 0))
                if const:
                    v = v + a0
                if shock is not None and t == 0:
                    v = v + aJ @ shock
                out[ci * N + t] = v
        return out

    def base_path(self, init, shock):
        """Reference-regime path (N+1, n) from state ``init`` with ``shock`` in the first period."""
        ob = self.ob
        Y = np.empty((self.N + 1, ob.n))
        y = np.asarray(init, float)
        for t in range(self.N + 1):
            y = ob.P @ y + (ob.Q @ shock if t == 0 else 0.0)
            Y[t] = y
        return Y

    def _q(self, init, shock):
        return self.q0 + self.Qm @ np.r_[np.asarray(init, float), np.asarray(shock, float)]

    def q_of(self, init, shock):
        """(q, base path): gaps with no wedges, ordered complementarity rows first."""
        return self._q(init, shock), self.base_path(init, shock)

    def q_direct(self, init, shock):
        """q computed from the base path itself (check of the affine map used by ``q_of``)."""
        n = self.ob.n
        Y = self.base_path(init, shock)
        q = self._gap_of(Y, const=True, shock=shock)
        for ci, co in enumerate(self.constraints):       # lag of the first period is the initial state
            q[ci * self.N] += co.s_g * co.alt[:n] @ np.asarray(init, float)
        return q[self._order]

    def stacked_M(self, pad=200):
        """M from one stacked linear system over a long horizon (independent check of ``_build``)."""
        ob, n, T, N = self.ob, self.ob.n, self.T, self.N
        H = N + pad
        C0, B0, A0, _J, _D = ob.regime_matrices(tuple([False] * ob.k))
        big = np.zeros((H * n, H * n))
        for t in range(H):
            big[t * n:(t + 1) * n, t * n:(t + 1) * n] = B0
            if t > 0:
                big[t * n:(t + 1) * n, (t - 1) * n:t * n] = C0
            if t < H - 1:
                big[t * n:(t + 1) * n, (t + 1) * n:(t + 2) * n] = A0
            else:
                big[t * n:(t + 1) * n, t * n:(t + 1) * n] += A0 @ ob.P      # terminal: y_{H} = P y_{H-1}
        rhs = np.zeros((H * n, self.n_c * T))
        for ci, co in enumerate(self.constraints):
            for s in range(T):
                rhs[s * n + co.equations[0], ci * T + s] = co.s_y
        sol = np.linalg.solve(big, rhs).reshape(H, n, self.n_c * T)
        return self._gap_of(sol[:N + 1], const=False)[self._order]

    # ------------------------------------------------------------------------------------------- solving
    def _wedges(self, q, method, omega, tol):
        """Wedges for the gaps q (complementarity rows first): (y, info)."""
        n_s = self.n_c * self.T
        info = dict(used="none")
        if np.all(q >= -tol * max(1.0, np.abs(q).max())):
            return np.zeros(n_s), info
        y = solve_active_set(self.Mo, q, n_s, tol) if method in ("news", "active_set") else None
        info["used"] = "active_set"
        if y is None:
            if method == "active_set":
                raise OBCError("the active-set iteration did not converge", "not_converged")
            T = self.T
            hz = sorted({max(1, T // 4), max(1, T // 2), T})
            y, status = solve_news(self.Mo, q, n_s, omega=omega, tol=tol, horizons=hz, n_c=self.n_c)
            info["used"] = "milp"
            if y is None:
                # is it the check-ahead rows that fail?
                y2, _ = solve_news(self.Mo[:n_s], q[:n_s], n_s, omega=omega, tol=tol, horizons=[T], n_c=self.n_c)
                if y2 is not None:
                    raise OBCError("a constraint still binds after the news horizon: increase obc_horizon",
                                   "horizon_too_short")
                raise OBCError("no solution returns to the reference regime within the news horizon",
                               "no_solution")
        return y, info

    def _step(self, init, shock, method="news", omega=1000.0, tol=1e-8):
        """One shock period without the path: (y, next state (deviations), info).  The path is only needed
        in the last shock period of a simulation (see ``_path``)."""
        init = np.asarray(init, float)
        shock = np.asarray(shock, float)
        x = np.concatenate([init[self._st], shock])
        q = self.q0 + self._Qx @ x
        y, info = self._wedges(q, method, omega, tol)
        nxt = self._Ps @ init[self._st] + self.ob.Q @ shock
        if y.any():
            nxt = nxt + self._V0 @ y
        return y, nxt, info

    def _path(self, init, shock, y, upto=None):
        """Path (upto+1, n; default N+1 rows) from the initial state and shock with wedges y."""
        ob = self.ob
        return _wedge_path(ob.P, self._G, self._FE, ob.P @ np.asarray(init, float) + ob.Q @ np.asarray(shock, float),
                           y, self.T, self.N if upto is None else upto)

    def solve_period(self, init, shock, method="news", omega=1000.0, tol=1e-8):
        """Wedges and path for one shock period. Returns (y, path (N+1, n) in deviations, info)."""
        q = self._q(init, shock)
        y, info = self._wedges(q, method, omega, tol)
        return y, self._path(init, shock, y), info

    def simulate(self, shocks, periods=100, init=None, method="news", exo_pos=None, omega=1000.0) -> OccbinResult:
        """Simulation under a sequence of unanticipated shocks, with OccBin's timing: every shock period is
        solved with no future shocks expected; after the last shock the perfect-foresight path is followed."""
        ob, n, T = self.ob, self.ob.n, self.T
        shocks_in = np.atleast_2d(np.asarray(shocks, float))
        if exo_pos is None:
            exo_idx = tuple(range(ob.n_exo))
        else:
            exo_idx = tuple(ob.model._ix[p] if isinstance(p, str) else int(p)
                            for p in np.atleast_1d(np.asarray(exo_pos, dtype=object)))
        S = shocks_in.shape[0]
        sh = np.zeros((S, ob.n_exo))
        sh[:, list(exo_idx)] = shocks_in
        n_periods = max(periods, S)
        need = n_periods - S + 1
        if need > self.N:
            self._build(need)
        y0 = np.zeros(n) if init is None or np.size(init) == 0 else np.asarray(init, float).ravel().copy()
        init_orig = y0.copy()
        piecewise = np.zeros((n_periods, n))
        history, wedges, used = [], np.zeros((S, self.n_c * T)), []
        path = None
        for sp in range(S):
            y_prev = y0
            try:
                y, y0, info = self._step(y_prev, sh[sp], method=method, omega=omega)
            except OBCError as err:
                err.period = sp + 1
                raise
            wedges[sp] = y
            used.append(info["used"])
            history.append(_regime_history(y, T, self.n_c))
            piecewise[sp] = y0
            if sp == S - 1:                      # the path is only needed once, after the last shock period
                path = self._path(y_prev, sh[sp], y, upto=n_periods - S + 1)
        piecewise[S:] = path[1:n_periods - S + 1]
        linear = np.empty((n_periods, n))
        yl = init_orig
        for t in range(n_periods):
            yl = ob.P @ yl + (ob.Q @ sh[t] if t < S else 0.0)
            linear[t] = yl
        res = OccbinResult(piecewise=piecewise + ob.ys, linear=linear + ob.ys, shocks=shocks_in,
                           regime_history=history, ys=ob.ys, exo_pos=exo_idx)
        res.news = dict(method=method, wedges=wedges, solver=used, horizon=T)
        return res

    def diagnostics(self, max_exact=16):
        return m_diagnostics(self.Ms, n_c=self.n_c, max_exact=max_exact)


# --------------------------------------------------------------------------- general regime formulation
class RegimeOBC:
    """News-shock solution for any model OccBin accepts, including constraints that switch several equations
    or whose bind/relax conditions are not the mirror image of the switched equation.

    Every equation that a constraint switches gets its own wedge on the reference version.  In a period in
    which the constraint is in its alternative regime the wedges are whatever makes the alternative versions
    hold; otherwise they are zero.  A regime sequence is accepted when OccBin's own test passes: the bind
    condition is false wherever the reference regime is assumed and the relax condition is false wherever the
    alternative regime is assumed (conditions evaluated in levels, as OccBin does).  The result is therefore
    OccBin's piecewise-linear solution; what changes is how it is found: the responses to wedges are computed
    once per parameter set, each guess costs one small linear solve, and a mixed-integer programme over the
    regime indicators is the fallback when the guess-and-verify iteration cycles.

    With one equation per constraint and mirror-image conditions this is the LCP of ``NewsOBC``, which should
    be preferred there because the theory on M (existence, uniqueness) applies to it.
    """

    milp_time_limit = 30.0

    def __init__(self, model_or_occbin, params=None, horizon=40, check_ahead=40):
        self.ob = ob = model_or_occbin if isinstance(model_or_occbin, Occbin) else Occbin(model_or_occbin, params)
        self.T, self.L, self.n_c = int(horizon), int(check_ahead), ob.k
        n, nx = ob.n, ob.n_exo
        ref_all = np.column_stack(ob.regime_matrices(tuple([False] * ob.k)))
        self.cols, self.alt_rows = [], []                 # (constraint, equation) per wedge block; affine alt rows
        per_reg = []
        for j in range(ob.k):
            alt_all = np.column_stack(ob.regime_matrices(tuple(i == j for i in range(ob.k))))
            scale = np.maximum(np.maximum(np.abs(ref_all).max(axis=1), np.abs(alt_all).max(axis=1)), 1.0)
            rows = [int(r) for r in np.nonzero(np.abs(alt_all - ref_all).max(axis=1) > 1e-12 * scale)[0]]
            if not rows:
                raise OBCClassError(f"constraint '{ob.names[j]}': no equation changes between the regimes")
            per_reg.append((alt_all, rows))
            for r in rows:
                self.cols.append((j, r))
                self.alt_rows.append(alt_all[r].copy())
        if ob.k == 2:                                     # an equation's version may depend on one constraint only
            both = np.column_stack(ob.regime_matrices((True, True)))
            expect = ref_all.copy()
            for alt_all, rows in per_reg:
                expect[rows] = alt_all[rows]
            shared = set(per_reg[0][1]) & set(per_reg[1][1])
            if shared or np.abs(both - expect).max() > 1e-10 * max(1.0, np.abs(both).max()):
                raise OBCClassError("an equation's version depends on both constraints jointly; the news-shock "
                                    "methods need each switched equation to depend on one constraint. "
                                    "Use obc_method='occbin' for this model.")
        self._cond = []                                   # linearised conditions, for the integer programme
        for c in ob.model.occbin_constraints:
            d_b = _affine_condition(ob, c["bind"], enter=True)
            relax = c.get("relax") or None
            d_r = _affine_condition(ob, relax, enter=False) if relax else d_b
            self._cond.append((d_b, d_r))
        # variables that the bind/relax conditions read: only their paths are needed to check a regime guess
        used = set()
        for c in ob.model.occbin_constraints:
            for src in (c["bind"], c.get("relax") or ""):
                used |= _used_variables(ob.model, src)
        self._cv = np.array(sorted(used), dtype=int)
        self._build(self.T + self.L)

    def _build(self, N):
        ob, n, nx, T = self.ob, self.ob.n, self.ob.n_exo, self.T
        C0, B0, A0, _J0, _D0 = ob.regime_matrices(tuple([False] * ob.k))
        lu = lu_factor(B0 + A0 @ ob.P)
        G = -lu_solve(lu, A0)
        nb = len(self.cols)
        E = np.zeros((n, nb))
        for b, (_j, row) in enumerate(self.cols):
            E[row, b] = 1.0
        FE = lu_solve(lu, E)
        self.N, self._G, self._FE, self._Yw = N, G, FE, None
        V = _wedge_vectors(G, FE, T)
        self._V0 = V
        a = np.array(self.alt_rows)                                          # (nb, 3n + nx + 1)
        aC, aB, aA, aJ = a[:, :n], a[:, n:2 * n], a[:, 2 * n:3 * n], a[:, 3 * n:3 * n + nx]
        # alternative residuals in periods 0..T-1: a = a0 + Am [init; shock] + R w
        Rr = _functional_rows(ob.P, aC, aB, aA, T)                          # (T+1, nb, n)
        self.R = _response_matrix(Rr, V, aA, T, T)                          # (nb*T, nb*T)
        Qsh = (Rr[:T] @ ob.Q).transpose(1, 0, 2)                            # (nb, T, nx)
        Qsh[:, 0, :] += aJ
        self.Am = np.hstack([Rr[1:T + 1].transpose(1, 0, 2).reshape(nb * T, n), Qsh.reshape(nb * T, nx)])
        self.a0 = np.repeat(a[:, -1], T)
        # paths of the variables that the conditions read, periods 0..N-1: y_t = P^(t+1) init + P^t Q u + wedges
        cv, k = self._cv, len(self._cv)
        Ecv = np.zeros((k, n))
        Ecv[np.arange(k), cv] = 1.0
        Z = np.zeros((k, n))
        Rc = _functional_rows(ob.P, Z, Ecv, Z, N)                           # E_cv P^t
        Wcv = _response_matrix(Rc, V, Z, N, T)                              # (k*N, nb*T), row j*N + t
        self._Wcv = np.ascontiguousarray(Wcv.reshape(k, N, nb * T).transpose(1, 0, 2).reshape(N * k, nb * T))
        self._Bcv = np.ascontiguousarray(np.hstack([Rc[1:N + 1].reshape(N * k, n), (Rc[:N] @ ob.Q).reshape(N * k, nx)]))
        self._prepare_step()

    @property
    def Yw(self):
        """Responses (N+1, n, n_blocks*T) of the path to a unit wedge on each block and period; built on first use."""
        if self._Yw is None:
            self._Yw = _full_responses(self.ob.P, self._V0, self.N, self.T)
        return self._Yw

    def _prepare_step(self):
        ob, n, N = self.ob, self.ob.n, self.N
        scale = max(1.0, float(np.abs(ob.P).max()), float(np.abs(self.Am[:, :n]).max()))
        used = np.maximum(np.abs(ob.P).max(axis=0), np.abs(self.Am[:, :n]).max(axis=0)) > 1e-14 * scale
        self._st = np.flatnonzero(used)
        self._Ps = np.ascontiguousarray(ob.P[:, self._st])
        self._Ax = np.ascontiguousarray(np.hstack([self.Am[:, self._st], self.Am[:, n:]]))
        self._Bx = np.ascontiguousarray(np.hstack([self._Bcv[:, self._st], self._Bcv[:, n:]]))
        self._zbuf = np.tile(ob.ys, (N, 1))

    def base_path(self, init, shock):
        """Reference-regime path (N+1, n) from state ``init`` with ``shock`` in the first period."""
        ob = self.ob
        Y = np.empty((self.N + 1, ob.n))
        y = np.asarray(init, float)
        for t in range(self.N + 1):
            y = ob.P @ y + (ob.Q @ shock if t == 0 else 0.0)
            Y[t] = y
        return Y

    # -------------------------------------------------------------------------------------------- pieces
    def _index(self, z):
        """Positions (in the stacked wedge vector) of the periods in which each block's constraint binds."""
        T = self.T
        return np.concatenate([b * T + np.nonzero(z[:T, j])[0] for b, (j, _r) in enumerate(self.cols)]).astype(int)

    def _wedges(self, z, a):
        w = np.zeros(len(self.cols) * self.T)
        S = self._index(z)
        if S.size:
            w[S] = np.linalg.solve(self.R[np.ix_(S, S)], -a[S])
        return w

    def _violations(self, z, pcv):
        """Regime indicators that fail OccBin's test along the path of the condition variables ``pcv`` (N, k)."""
        ob = self.ob
        zl = self._zbuf
        zl[:, self._cv] = pcv + ob.ys[self._cv]
        args = (zl, ob.params, ob.ys)
        N = self.N
        binding, relax = np.zeros((N, self.n_c), bool), np.zeros((N, self.n_c), bool)
        for j, (bind, _eb, rel, _er) in enumerate(ob._cons):
            binding[:, j] = np.broadcast_to(np.asarray(bind(*args), float) != 0, (N,))
            relax[:, j] = ~binding[:, j] if rel is None else np.broadcast_to(np.asarray(rel(*args), float) != 0, (N,))
        return binding & ~z, relax & z

    def _path(self, init, shock, w, upto=None):
        ob = self.ob
        return _wedge_path(ob.P, self._G, self._FE, ob.P @ np.asarray(init, float) + ob.Q @ np.asarray(shock, float),
                           w, self.T, self.N if upto is None else upto)

    def solve_period(self, init, shock, maxit=60, method="news"):
        """Regime indicators (N, n_c), wedges and path (N+1, n; deviations) for one shock period."""
        init, shock = np.asarray(init, float), np.asarray(shock, float)
        z, w, _nxt, info = self._step(init, shock, maxit=maxit, method=method)
        return z, w, self._path(init, shock, w), info

    def _step(self, init, shock, maxit=60, method="news"):
        """Regime indicators, wedges, next state and info for one shock period (no full path)."""
        T, N, k = self.T, self.N, len(self._cv)
        init, shock = np.asarray(init, float), np.asarray(shock, float)
        xs = np.concatenate([init[self._st], shock])
        Ycv = (self._Bx @ xs).reshape(N, k)
        a = self.a0 + self._Ax @ xs
        z = np.zeros((N, self.n_c), bool)
        seen, info = set(), dict(used="none", iterations=0)
        ok = beyond = False
        if method != "milp":
            for it in range(maxit):
                if z[T:].any():
                    beyond = True                           # the iteration wants a regime change after the horizon
                    break
                try:
                    w = self._wedges(z, a)
                except np.linalg.LinAlgError:
                    break
                pcv = Ycv + (self._Wcv @ w).reshape(N, k)
                vb, vr = self._violations(z, pcv)
                if not vb.any() and not vr.any():
                    ok = True
                    info.update(used="guess_verify" if it else "none", iterations=it)
                    break
                key = z.tobytes()
                if key in seen:
                    break
                seen.add(key)
                z = (z & ~vr) | vb
        if not ok and beyond and method != "milp":
            raise OBCError("a constraint still binds after the news horizon: increase obc_horizon", "horizon_too_short")
        if not ok and method == "guess_verify":
            raise OBCError("the guess-and-verify iteration cycles", "not_converged")
        if not ok:
            Y = self.base_path(init, shock)
            try:
                z = self._milp(Y, a)
            except OBCError as err:
                if beyond:
                    raise OBCError("a constraint still binds after the news horizon: increase obc_horizon",
                                   "horizon_too_short") from err
                raise
            info["used"] = "milp"
            w = self._wedges(z, a)
            pcv = Ycv + (self._Wcv @ w).reshape(N, k)
            vb, vr = self._violations(z, pcv)
            if vb[T:].any():
                raise OBCError("a constraint still binds after the news horizon: increase obc_horizon",
                               "horizon_too_short")
            if vb.any() or vr.any():
                raise OBCError("the regime sequence found by the integer programme (linearised conditions) fails "
                               "OccBin's test on the conditions in levels", "not_converged")
        nxt = self._Ps @ init[self._st] + self.ob.Q @ shock
        if w.any():
            nxt = nxt + self._V0 @ w
        return z, w, nxt, info

    def _milp(self, Y, a):
        """Regime indicators from a mixed-integer programme on the linearised conditions: minimise the number
        of periods spent in the alternative regimes subject to the regime logic (big-M)."""
        T, N, n, nb, nc = self.T, self.N, self.ob.n, len(self.cols), self.n_c
        nw, nz = nb * T, nc * T
        Yw = self.Yw
        # linearised conditions along the path: d = d0 + D w   (rows: constraint-major, period-minor, N periods)
        Db, Dr, b0, r0 = (np.zeros((nc * N, nw)), np.zeros((nc * N, nw)), np.zeros(nc * N), np.zeros(nc * N))
        for j, (d_b, d_r) in enumerate(self._cond):
            for d, D, d0 in ((d_b, Db, b0), (d_r, Dr, r0)):
                g = d[n:2 * n]
                D[j * N:(j + 1) * N] = np.tensordot(g, Yw[:N], axes=(0, 1))
                d0[j * N:(j + 1) * N] = Y[:N] @ g + d[-1]
        sc = max(np.abs(a).max(), np.abs(b0).max(), np.abs(r0).max(), 1e-12)
        diag = np.abs(np.diag(self.R))
        W = 1e3 * sc / max(float(np.median(diag[diag > 0])) if (diag > 0).any() else 1.0, 1e-12)
        Om = 1e3 * sc
        eps = 1e-9 * sc
        rows, lo, hi = [], [], []

        def add(Aw, Az, lower, upper):
            rows.append(np.hstack([Aw, Az]))
            lo.append(np.broadcast_to(lower, (Aw.shape[0],)).copy())
            hi.append(np.broadcast_to(upper, (Aw.shape[0],)).copy())

        Zb = np.zeros((nw, nz))                              # block (wedge) -> its constraint's indicator
        for b, (j, _r) in enumerate(self.cols):
            Zb[b * T:(b + 1) * T, j * T:(j + 1) * T] = np.eye(T)
        Iw = np.eye(nw)
        add(Iw, -W * Zb, -np.inf, 0.0)                       #  w <= W z
        add(Iw, W * Zb, 0.0, np.inf)                         # -w <= W z
        add(self.R, Om * Zb, -np.inf, Om - a)                #  alt residual <= Om (1 - z)
        add(self.R, -Om * Zb, -Om - a, np.inf)               #  alt residual >= -Om (1 - z)
        for j in range(nc):
            sel = np.zeros((T, nz))
            sel[:, j * T:(j + 1) * T] = np.eye(T)
            r_in = slice(j * N, j * N + T)
            r_out = slice(j * N + T, (j + 1) * N)
            add(Db[r_in], Om * sel, -b0[r_in] - eps, np.inf)         # reference regime: bind condition false
            add(Dr[r_in], Om * sel, -np.inf, Om - r0[r_in] + eps)    # alternative regime: relax condition false
            if N > T:
                add(Db[r_out], np.zeros((N - T, nz)), -b0[r_out] - eps, np.inf)
        A = np.vstack(rows)
        cons = LinearConstraint(A, np.concatenate(lo), np.concatenate(hi))
        cobj = np.r_[np.zeros(nw), np.ones(nz)]
        res = milp(cobj, constraints=[cons], integrality=np.r_[np.zeros(nw), np.ones(nz)],
                   bounds=Bounds(np.r_[np.full(nw, -W), np.zeros(nz)], np.r_[np.full(nw, W), np.ones(nz)]),
                   options=dict(time_limit=self.milp_time_limit))
        if res.status == 1 and res.x is None:
            raise OBCError(f"the integer programme over regime sequences did not finish in {self.milp_time_limit:g} s "
                           "(this big-M formulation is slow with several constraints or long horizons)", "not_converged")
        if res.x is None or not res.success:
            # distinguish a horizon that is too short from no solution: drop the rows beyond the horizon
            raise OBCError("no regime sequence within the news horizon satisfies the model (linearised conditions)",
                           "no_solution")
        z = np.zeros((N, nc), bool)
        z[:T] = (res.x[nw:] > 0.5).reshape(nc, T).T
        return z

    def simulate(self, shocks, periods=100, init=None, method="news", exo_pos=None) -> OccbinResult:
        ob, n, T = self.ob, self.ob.n, self.T
        shocks_in = np.atleast_2d(np.asarray(shocks, float))
        if exo_pos is None:
            exo_idx = tuple(range(ob.n_exo))
        else:
            exo_idx = tuple(ob.model._ix[p] if isinstance(p, str) else int(p)
                            for p in np.atleast_1d(np.asarray(exo_pos, dtype=object)))
        S = shocks_in.shape[0]
        sh = np.zeros((S, ob.n_exo))
        sh[:, list(exo_idx)] = shocks_in
        n_periods = max(periods, S)
        if n_periods - S + 1 > self.N:
            self._build(n_periods - S + 1)
        y0 = np.zeros(n) if init is None or np.size(init) == 0 else np.asarray(init, float).ravel().copy()
        init_orig = y0.copy()
        piecewise = np.zeros((n_periods, n))
        history, used, wedges = [], [], np.zeros((S, len(self.cols) * T))
        path = None
        for sp in range(S):
            y_prev = y0
            try:
                z, w, y0, info = self._step(y_prev, sh[sp], method=method)
            except OBCError as err:
                err.period = sp + 1
                raise
            wedges[sp] = w
            used.append(info["used"])
            history.append([_map_regime(z[:, j].astype(np.int64)) for j in range(self.n_c)])
            piecewise[sp] = y0
            if sp == S - 1:                      # the path is only needed once, after the last shock period
                path = self._path(y_prev, sh[sp], w, upto=n_periods - S + 1)
        piecewise[S:] = path[1:n_periods - S + 1]
        linear = np.empty((n_periods, n))
        yl = init_orig
        for t in range(n_periods):
            yl = ob.P @ yl + (ob.Q @ sh[t] if t < S else 0.0)
            linear[t] = yl
        res = OccbinResult(piecewise=piecewise + ob.ys, linear=linear + ob.ys, shocks=shocks_in,
                           regime_history=history, ys=ob.ys, exo_pos=exo_idx)
        res.news = dict(method=method, formulation="regime", wedges=wedges, solver=used, horizon=T,
                        blocks=list(self.cols))
        return res
