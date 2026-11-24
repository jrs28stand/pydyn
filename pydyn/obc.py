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
from scipy.linalg.lapack import dgesv as _dgesv
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .expr import NAMESPACE, Translator, tokenize
from .occbin import OBC_SELECTIONS, Occbin, OccbinResult, _split_comparison

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
    """Compiled condition ``lambda z, params, ss``; depends on the model's equations only, so kept on the model."""
    cache = model.__dict__.setdefault("_obc_condition_fns", {})
    if src not in cache:
        cache[src] = _condition_fn_uncached(model, src)
    return cache[src]


def _condition_fn_uncached(model, src):
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


def solve_active_set(M, q, n_s, tol=1e-8, maxit=60, start=None):
    """Guess-and-verify on the binding set: start from the periods where q < 0 (or from the boolean vector
    ``start``), solve the square system on the guessed set, drop negative wedges, add violated periods.
    Returns y or None (no guarantee of success).  Only the complementarity rows are touched inside the loop; the
    check-ahead rows enter once, in the final verification."""
    Ms = M[:n_s]
    qs = q[:n_s]
    S = qs < -tol if start is None else np.array(start, bool)
    seen = set()
    for _ in range(maxit):
        key = S.tobytes()
        if key in seen:
            return None
        seen.add(key)
        y = np.zeros(n_s)
        idx = np.flatnonzero(S)
        if idx.size:
            _lu, _piv, x, info = _dgesv(Ms[idx[:, None], idx], -qs[idx], overwrite_a=True, overwrite_b=True)
            if info != 0:
                return None
            y[idx] = x
        g = qs + Ms @ y
        drop = S & (y < -tol)
        add = (~S) & (g < -tol)
        if not drop.any() and not add.any():
            return y if verify_lcp(M, q, y, n_s, tol) else None
        S = (S & ~drop) | add
    return None


def solve_news(M, q, n_s, omega=1000.0, tol=1e-8, horizons=None, n_c=1, time_limit=None):
    """Solve ``y >= 0, q + M y >= 0, y'(q + M y)[:n_s] = 0`` as a mixed-integer linear programme.

    Homogenised formulation (Holden): maximise alpha subject to ``0 <= yn <= z``, ``alpha q + M yn >= 0`` and
    ``alpha q_s + M_s yn <= omega (1 - z)`` with z binary; then ``y = yn / alpha``.  alpha > 0 at the optimum if
    and only if a solution exists with wedges confined to the horizon.  ``horizons`` lists the numbers of leading
    periods (per constraint) in which wedges are allowed, tried in order; the first that works is returned, so
    shorter binding spells are preferred.  Returns (y, status); status is 'ok', 'no_solution' or, when the
    ``time_limit`` stopped a solve before it found anything, 'time_limit' (then no solution is proved)."""
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
    limit_hit = False
    for Tss in horizons:
        ub_y = np.zeros(n_s)
        for c in range(n_c):
            ub_y[c * T:c * T + Tss] = 1.0
        bounds = Bounds(np.zeros(nv), np.r_[ub_y, 1e8, ub_y])
        opts = dict(mip_rel_gap=0.0)
        if time_limit is not None:
            opts["time_limit"] = float(time_limit)
        res = milp(cobj, constraints=cons, integrality=integrality, bounds=bounds, options=opts)
        if res.x is None or not res.success:
            limit_hit = limit_hit or res.status == 1
            continue
        alpha = res.x[n_s]
        if alpha <= 1e-9:
            limit_hit = limit_hit or res.status == 1      # the incumbent is the trivial point alpha = 0
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
    return None, "time_limit" if limit_hit else "no_solution"


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


def _linear_path(ob, st, init, sh, n_periods):
    """Unconstrained path y_t = P y_{t-1} + Q u_t (deviations) for n_periods; the recursion runs on the state
    variables ``st`` only and the other variables follow from them."""
    n = ob.n
    QU = np.zeros((n_periods, n))
    QU[:len(sh)] = sh @ ob.Q.T
    Pss = ob.P[np.ix_(st, st)]
    xs = np.asarray(init, float)[st]
    prev = np.empty((n_periods, len(st)))
    for t in range(n_periods):
        prev[t] = xs
        xs = Pss @ xs + QU[t, st]
    return prev @ ob.P[:, st].T + QU


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
SELECTIONS = OBC_SELECTIONS


def check_selection(selection, n_c=1):
    """Validate an ``obc_selection`` value (and that a non-default one is used with one constraint only)."""
    if not isinstance(selection, str) or selection not in SELECTIONS:
        raise ValueError(f"unknown obc_selection {selection!r}: use one of {', '.join(map(repr, SELECTIONS))}")
    if selection != "occbin" and n_c != 1:
        raise ValueError(f"obc_selection={selection!r} is implemented for models with one constraint only (this model has "
                         f"{n_c}); use obc_selection='occbin', which returns the solution that the guess-and-verify "
                         "iteration reaches")
    return selection


def pick_spell(cands, selection):
    """The candidate ``(l, k, w)`` (spell start, number of binding periods, wedges) chosen by ``selection``: the fewest
    binding periods (``'shortest'``), the most (``'longest'``) or the smallest sum of squared wedges
    (``'min_wedge'``); ties go to the earliest start."""
    if selection == "shortest":
        key = lambda c: (c[1], c[0])                      # noqa: E731
    elif selection == "longest":
        key = lambda c: (-c[1], c[0])                     # noqa: E731
    else:
        key = lambda c: (float(np.sum(np.square(c[2]))), c[0], c[1])    # noqa: E731
    return min(cands, key=key)


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
    spell_l_max = 8      # latest start (period index) of a single spell that ``single_spells`` looks for

    def single_spells(self, q, l_max=None, tol=1e-8):
        """All solutions of the complementarity problem with one binding spell {l, ..., l+k-1}, l <= l_max (one constraint):
        a list of (l, k, y) with y the wedges (length n_s).  On the spell the wedges solve the square system
        ``M[S, S] y = -q[S]`` and must be positive; the gap must be non-negative on every row, check-ahead rows
        included.  More than one entry means that the state has several solutions."""
        if self.n_c != 1:
            raise ValueError("single_spells needs a model with one constraint")
        T = self.T
        scale = max(1.0, float(np.abs(q).max()))
        out = []
        for l in range(min(self.spell_l_max if l_max is None else l_max, T - 1) + 1):
            for k in range(1, T - l + 1):
                sl = slice(l, l + k)
                try:
                    yk = np.linalg.solve(self.Mo[sl, sl], -q[sl])
                except np.linalg.LinAlgError:
                    continue
                if yk.min() <= tol * scale:               # a spell of exactly k periods needs positive wedges
                    continue
                if (q + self.Mo[:, sl] @ yk).min() < -tol * scale:
                    continue
                y = np.zeros(T)
                y[sl] = yk
                out.append((l, k, y))
        return out

    def _select(self, q, y, selection, info, tol):
        """Replace the default solution ``y`` (which binds somewhere) by the single-spell solution chosen by
        ``selection``; the default stands, flagged, if there is none.  Fills ``info``."""
        cands = self.single_spells(q, tol=tol)
        info["n_single_spell"] = len(cands)
        if not cands:
            info["selection_fallback"] = True
            return y
        l, k, ys = pick_spell(cands, selection)
        info["spell"] = (l, k)
        return ys

    def _wedges(self, q, method, omega, tol, start=None, selection="occbin"):
        """Wedges for the gaps q (complementarity rows first): (y, info).  ``start``: boolean guess of the
        binding set for the active-set iteration (tried first; the standard start follows if it fails).
        ``selection``: which solution to return when there are several (see ``pick_spell``); the default is the
        one the iteration reaches, the others look for single-spell solutions wherever something binds."""
        n_s = self.n_c * self.T
        info = dict(used="none")
        qmin = q.min()
        if qmin >= 0.0 or qmin >= -tol * max(1.0, np.abs(q).max()):
            return np.zeros(n_s), info
        y = None
        if method in ("news", "active_set"):
            if start is not None and start.any():
                y = solve_active_set(self.Mo, q, n_s, tol, start=start)
            if y is None:
                y = solve_active_set(self.Mo, q, n_s, tol)
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
        if selection != "occbin" and y.any():
            y = self._select(q, y, selection, info, tol)
        return y, info

    def _step(self, init, shock, method="news", omega=1000.0, tol=1e-8, start=None, selection="occbin"):
        """One shock period without the path: (y, next state (deviations), info).  The path is only needed
        in the last shock period of a simulation (see ``_path``)."""
        init = np.asarray(init, float)
        shock = np.asarray(shock, float)
        x = np.concatenate([init[self._st], shock])
        q = self.q0 + self._Qx @ x
        y, info = self._wedges(q, method, omega, tol, start, selection)
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

    def simulate(self, shocks, periods=100, init=None, method="news", exo_pos=None, omega=1000.0,
                 warm_start=False, selection="occbin") -> OccbinResult:
        """Simulation under a sequence of unanticipated shocks, with OccBin's timing: every shock period is
        solved with no future shocks expected; after the last shock the perfect-foresight path is followed.

        ``selection`` (``obc_selection``): ``'occbin'`` (default) returns the solution that the active-set iteration
        reaches from the no-binding guess; ``'shortest'``, ``'longest'`` and ``'min_wedge'`` choose, in every period in
        which something binds, among the single-spell solutions at the state (``single_spells``): the fewest
        binding periods, the most, the smallest sum of squared wedges.  If none exists the default solution stands
        and the period is flagged in ``res.news['selection_fallback']``; ``res.news['n_single_spell']`` holds the
        number found per shock period (-1: not looked for).  One constraint only.

        ``warm_start``: start each active-set iteration from the previous period's binding set moved forward
        by one period (exact when no new shock arrives); the standard start is the fallback.  Off by default:
        with a fresh shock every period it needed more iterations than the standard start (docs/OBC_PERF.md),
        and where M is not a P-matrix the two starts could in principle end in different solutions."""
        ob, n, T = self.ob, self.ob.n, self.T
        check_selection(selection, self.n_c)
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
        n_spell, fallback = np.full(S, -1), np.zeros(S, bool)
        path, start = None, None
        for sp in range(S):
            y_prev = y0
            try:
                y, y0, info = self._step(y_prev, sh[sp], method=method, omega=omega, start=start, selection=selection)
            except OBCError as err:
                err.period = sp + 1
                raise
            wedges[sp] = y
            used.append(info["used"])
            n_spell[sp], fallback[sp] = info.get("n_single_spell", -1), info.get("selection_fallback", False)
            if warm_start:
                start = np.zeros(self.n_c * T, bool)
                if y.any():
                    flag = (y > 1e-9 * max(1.0, float(np.abs(y).max()))).reshape(self.n_c, T)
                    start.reshape(self.n_c, T)[:, :-1] = flag[:, 1:]
            history.append(_regime_history(y, T, self.n_c))
            piecewise[sp] = y0
            if sp == S - 1:                      # the path is only needed once, after the last shock period
                path = self._path(y_prev, sh[sp], y, upto=n_periods - S + 1)
        piecewise[S:] = path[1:n_periods - S + 1]
        linear = _linear_path(ob, self._st, init_orig, sh, n_periods)
        res = OccbinResult(piecewise=piecewise + ob.ys, linear=linear + ob.ys, shocks=shocks_in,
                           regime_history=history, ys=ob.ys, exo_pos=exo_idx)
        res.news = dict(method=method, wedges=wedges, solver=used, horizon=T, selection=selection,
                        n_single_spell=n_spell, selection_fallback=fallback)
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

    milp_time_limit = 30.0       # seconds for each integer programme; 0 switches the integer-programme fallbacks off

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
        self._jb = np.array([j for j, _r in self.cols], dtype=int)      # constraint of each wedge block
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
        self.N, self._G, self._FE, self._Yw, self._Yb, self._tie_cache = N, G, FE, None, None, 0
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

    @property
    def Yb(self):
        """Responses (N+1, n, n+nx) of the no-wedge path to the initial state and the shock; built on first use."""
        if self.__dict__.get("_Yb") is None:
            ob, n, nx = self.ob, self.ob.n, self.ob.n_exo
            Yb = np.zeros((self.N + 1, n, n + nx))
            y = np.hstack([np.eye(n), np.zeros((n, nx))])
            for t in range(self.N + 1):
                y = ob.P @ y
                if t == 0:
                    y[:, n:] = ob.Q
                Yb[t] = y
            self._Yb = Yb
        return self._Yb

    @property
    def _tie(self):
        """Tied-wedge structure (``_find_tie``); computed on first use, since it needs the full responses."""
        if self.__dict__.get("_tie_cache", 0) == 0:
            self._tie_cache = self._find_tie()
        return self._tie_cache

    # ------------------------------------------------------------------ reduction to an LCP (tied wedges)
    def _find_tie(self):
        """Structure that lets a one-constraint, several-equation model be written as an LCP.

        If all but one of the switched equations respond to the wedges only in the same period and with
        constant coefficients (``alt_i(t) = sum_b C[i, b] w_b(t) + (terms the wedges do not move)``), then in an
        alternative-regime period those equations tie the other wedges to the wedge of the remaining
        ("primary") equation.  Returns None when the structure is absent."""
        nb, T = len(self.cols), self.T
        if self.n_c != 1 or nb < 2:
            return None
        Rb = self.R.reshape(nb, T, nb, T)
        eye = np.eye(T, dtype=bool)
        C = np.zeros((nb, nb))
        diagonal = np.zeros(nb, bool)
        scale = max(np.abs(self.R).max(), 1e-300)
        for i in range(nb):
            ok = True
            for b in range(nb):
                blk = Rb[i, :, b, :]
                c = float(np.mean(np.diag(blk)))
                if np.abs(blk[~eye]).max(initial=0.0) > 1e-9 * scale or np.abs(np.diag(blk) - c).max() > 1e-9 * scale:
                    ok = False
                    break
                C[i, b] = c
            diagonal[i] = ok
        if diagonal.sum() != nb - 1:
            return None
        pb = int(np.nonzero(~diagonal)[0][0])
        sec = [i for i in range(nb) if i != pb]
        Co = C[np.ix_(sec, sec)]
        if abs(np.linalg.det(Co)) < 1e-12:
            return None
        d = np.zeros(nb)                                    # wedge direction per unit of the primary wedge
        d[pb] = 1.0
        d[sec] = -np.linalg.solve(Co, C[sec, pb])
        Coi = np.linalg.inv(Co)                             # offsets: w_sec(t) = -Coi a_sec(t) (+ d_sec y)
        # composite responses over all N periods
        Yc = sum(d[b] * self.Yw[:, :, b * T:(b + 1) * T] for b in range(nb))          # (N+1, n, T)
        n, nx, N = self.ob.n, self.ob.n_exo, self.N
        a = self.alt_rows[pb]
        aC, aB, aA, aJ = a[:n], a[n:2 * n], a[2 * n:3 * n], a[3 * n:3 * n + nx]

        def apply(Y):
            out = np.zeros((N,) + Y.shape[2:])
            for t in range(N):
                v = aB @ Y[t] + aA @ Y[t + 1]
                if t > 0:
                    v = v + aC @ Y[t - 1]
                out[t] = v
            return out

        Mblocks = [apply(self.Yw[:, :, b * T:(b + 1) * T]) for b in range(nb)]   # alt_p response per wedge block
        Mraw = sum(d[b] * Mblocks[b] for b in range(nb))    # ... and to the composite (tied) wedge
        Qraw = apply(self.Yb)
        Qraw[0, :n] += aC
        Qraw[0, n:] += aJ
        tie = dict(primary=pb, secondary=sec, d=d, Coi=Coi, Mblocks=Mblocks, Mraw=Mraw, Qraw=Qraw,
                   q0raw=float(a[-1]), mirror=False, s_g=1.0, s_y=1.0)
        # conditions as mirror images of the primary equation, tested on the affine functions themselves
        ref_all = np.column_stack(self.ob.regime_matrices(tuple([False] * self.ob.k)))
        refs = [ref_all[row] for _j, row in self.cols]
        d_b, d_r = self._cond[0]
        Xb = np.column_stack([a] + refs)                                    # bind: k1 alt_p + reference residuals
        cb, *_ = np.linalg.lstsq(Xb, d_b, rcond=None)
        Xr = np.column_stack([refs[pb]] + list(self.alt_rows))              # relax: c ref_p + alternative residuals
        cr, *_ = np.linalg.lstsq(Xr, d_r, rcond=None)
        okb = np.linalg.norm(Xb @ cb - d_b) <= 1e-8 * np.linalg.norm(d_b) and abs(cb[0]) > 1e-10
        okr = np.linalg.norm(Xr @ cr - d_r) <= 1e-8 * np.linalg.norm(d_r) and abs(cr[0]) > 1e-10
        if okb and okr:
            tie["s_g"], tie["s_y"] = float(np.sign(cb[0])), -float(np.sign(cr[0]))
            tie["mirror"] = bool(tie["s_g"] * tie["q0raw"] > 0)
        return tie

    def lcp(self, init, shock, tol=1e-8):
        """(M, q, G, info) of the equivalent complementarity problem for this state, or None when the model does
        not have the structure.

        With y_t >= 0 the (normalised) wedge on the primary equation and z_t = 1 in alternative-regime periods,
        the primary gap is ``g = q + M y + G z`` and a solution has ``g >= 0, y >= 0, y'g = 0, y_t > 0 only if
        z_t = 1, g_t = 0 if z_t = 1``.  G collects the state-dependent offsets of the tied wedges; when it is
        zero this is a standard LCP in (M, q) and Holden's results on M apply.  The structure requires
        (i) the tie (``_find_tie``), (ii) the bind condition to equal a multiple of the primary alternative
        equation plus reference residuals, and (iii) the relax condition to equal a multiple of the primary
        reference residual plus alternative residuals.  The full wedge vector is ``info['D'] @ y + offsets``."""
        tie = self._tie
        if tie is None:
            return None
        T, N, nb = self.T, self.N, len(self.cols)
        x = np.r_[np.asarray(init, float), np.asarray(shock, float)]
        a = self.a0 + self.Am @ x
        base = tie["q0raw"] + tie["Qraw"] @ x               # alt_p on the no-wedge path, N periods
        s_g, s_y = tie["s_g"], tie["s_y"]
        M = s_g * s_y * tie["Mraw"]
        q = s_g * base
        G = np.zeros((N, T))
        asec = np.array([a[i * T:(i + 1) * T] for i in tie["secondary"]])      # (n_sec, T)
        f = -tie["Coi"] @ asec                                                  # offsets of the tied wedges
        for o, blk in enumerate(tie["secondary"]):
            G += s_g * tie["Mblocks"][blk] * f[o][None, :]
        if np.abs(G).max() <= tol * max(np.abs(q).max(), 1e-300):
            G = None
        return M, q, G, dict(a=a, x=x)

    def diagnostics(self, max_exact=16):
        """Existence/uniqueness diagnostics on the M of the equivalent LCP (see ``m_diagnostics``). They are
        Holden's conditions when the offsets G vanish (e.g. no shock enters the tied equations); None when the
        structure is absent."""
        if self._tie is None or not self._tie.get("mirror"):
            return None
        return m_diagnostics((self._tie["s_g"] * self._tie["s_y"] * self._tie["Mraw"])[:self.T], n_c=1,
                             max_exact=max_exact)

    def single_spell_solutions(self, init, shock, l_max=8, tol=1e-7):
        """All solutions with one binding spell {l..l+k-1}, l <= l_max, as (l, k) pairs. More than one entry
        means multiple equilibria at this state."""
        T, out = self.T, []
        got = self.lcp(init, shock) if (self._tie is not None and self._tie.get("mirror")) else None
        if got is not None:
            M, q, G, _info = got
            for l in range(min(l_max, T - 1) + 1):
                for k in range(1, T - l + 1):
                    idx = np.arange(l, l + k)
                    qq = q if G is None else q + G[:, idx].sum(axis=1)
                    y = np.zeros(T)
                    try:
                        y[idx] = np.linalg.solve(M[np.ix_(idx, idx)], -qq[idx])
                    except np.linalg.LinAlgError:
                        continue
                    if verify_lcp(M, qq, y, T, tol):
                        out.append((l, k))
            return out
        x = np.r_[np.asarray(init, float), np.asarray(shock, float)]
        Y = np.tensordot(self.Yb, x, axes=(2, 0))
        a = self.a0 + self.Am @ x
        for l in range(min(l_max, T - 1) + 1):
            for k in range(1, T - l + 1):
                z = np.zeros((self.N, self.n_c), bool)
                z[l:l + k, 0] = True
                try:
                    w = self._wedges(z, a)
                except np.linalg.LinAlgError:
                    continue
                vb, vr = self._violations(z, Y + self.Yw @ w)
                if not vb.any() and not vr.any():
                    out.append((l, k))
        return out

    def _solve_lcp(self, init, shock):
        """Solve through the equivalent complementarity problem. Returns (z, w, path) or None if the structure is
        absent; raises OBCError('no_solution') -- with a proof when the linear relaxation is infeasible."""
        if self._tie is None or not self._tie.get("mirror"):
            return None
        M, q, G, info = self.lcp(init, shock)
        T, N = self.T, self.N
        proof = ("no solution exists that returns to the reference regime within the news horizon: the constraint "
                 "cannot be kept satisfied in every period by any non-negative wedges (linear programme infeasible)")
        zsol = None
        no_milp = "integer programme not attempted (milp_time_limit = 0)"
        if G is None:
            y = solve_active_set(M, q, T)
            if y is None and self.milp_time_limit > 0:
                y, _status = solve_news(M, q, T, horizons=sorted({max(1, T // 4), max(1, T // 2), T}))
            if y is None:
                feas = linprog(np.zeros(T), A_ub=-M, b_ub=q, bounds=[(0, None)] * T)
                raise OBCError(proof if feas.status == 2 else
                               no_milp if self.milp_time_limit <= 0 else
                               "no solution within the news horizon (the linear complementarity problem has none)",
                               "no_solution")
            zsol = y > 1e-9 * max(1.0, float(np.abs(y).max()))
        else:
            # relaxation without bounds: y >= 0, 0 <= z <= 1, q + M y + G z >= 0
            feas = linprog(np.zeros(2 * T), A_ub=-np.hstack([M, G]), b_ub=q,
                           bounds=[(0, None)] * T + [(0, 1)] * T)
            if feas.status == 2:
                raise OBCError(proof, "no_solution")
            if self.milp_time_limit <= 0:
                raise OBCError(no_milp, "no_solution")
            sc = max(np.abs(q).max(), 1e-12)
            dg = np.abs(np.diag(M[:T]))
            U = 1e3 * sc / max(float(dg[dg > 0].min()) if (dg > 0).any() else 1.0, 1e-12)
            Om = 1e3 * sc
            I, Z = np.eye(T), np.zeros((T, T))
            cons = [LinearConstraint(np.hstack([I, -U * I]), -np.inf, 0.0),                    # y <= U z
                    LinearConstraint(np.hstack([M, G]), -q, np.inf),                           # g >= 0
                    LinearConstraint(np.hstack([M[:T], G[:T] + Om * I]), -np.inf, Om - q[:T])]  # g <= Om (1 - z)
            res = milp(np.r_[np.zeros(T), np.ones(T)], constraints=cons,
                       integrality=np.r_[np.zeros(T), np.ones(T)],
                       bounds=Bounds(np.zeros(2 * T), np.r_[np.full(T, U), np.ones(T)]),
                       options=dict(time_limit=self.milp_time_limit))
            if res.x is None:
                raise OBCError("no solution found within the news horizon by the integer programme (the linear "
                               "relaxation is feasible, so this is not a proof)" if res.status != 1 else
                               f"the integer programme did not finish in {self.milp_time_limit:g} s",
                               "no_solution" if res.status != 1 else "not_converged")
            zsol = res.x[T:] > 0.5
        z = np.zeros((N, 1), bool)
        z[:T, 0] = zsol
        w = self._wedges(z, info["a"])
        Y = np.tensordot(self.Yb, info["x"], axes=(2, 0))
        return z, w, Y + self.Yw @ w

    # -------------------------------------------------------------------------------------------- pieces
    def _index(self, z):
        """Positions (in the stacked wedge vector) of the periods in which each block's constraint binds."""
        return np.flatnonzero(z[:self.T][:, self._jb].T.ravel())

    def _wedges(self, z, a):
        w = np.zeros(len(self.cols) * self.T)
        S = self._index(z)
        if S.size:
            _lu, _piv, x, info = _dgesv(self.R[S[:, None], S], -a[S], overwrite_a=True, overwrite_b=True)
            if info != 0:
                raise np.linalg.LinAlgError("Singular matrix")
            w[S] = x
        return w

    def _violations(self, z, pcv):
        """Regime indicators that fail OccBin's test along the path of the condition variables ``pcv`` (N, k)."""
        ob = self.ob
        zl = self._zbuf
        if pcv.shape[1] == ob.n and len(self._cv) != ob.n:      # a full path (N+1, n) is accepted too
            pcv = pcv[:self.N][:, self._cv]
        zl[:, self._cv] = pcv + ob.ys[self._cv]
        args = (zl, ob.params, ob.ys)
        N = self.N
        binding, relax = np.empty((N, self.n_c), bool), np.empty((N, self.n_c), bool)
        for j, (bind, _eb, rel, _er) in enumerate(ob._cons):
            b = np.asarray(bind(*args), float) != 0
            binding[:, j] = b
            relax[:, j] = ~binding[:, j] if rel is None else np.asarray(rel(*args), float) != 0
        return binding & ~z, relax & z

    def _path(self, init, shock, w, upto=None):
        ob = self.ob
        return _wedge_path(ob.P, self._G, self._FE, ob.P @ np.asarray(init, float) + ob.Q @ np.asarray(shock, float),
                           w, self.T, self.N if upto is None else upto)

    def solve_period(self, init, shock, maxit=60, method="news", selection="occbin"):
        """Regime indicators (N, n_c), wedges and path (N+1, n; deviations) for one shock period."""
        init, shock = np.asarray(init, float), np.asarray(shock, float)
        z, w, _nxt, info = self._step(init, shock, maxit=maxit, method=method, selection=selection)
        return z, w, self._path(init, shock, w), info

    def _iterate(self, z, Ycv, a, maxit):
        """OccBin's guess-and-verify iteration from the guess ``z``: (ok, beyond, z, w, iterations).  ``beyond``:
        the iteration wants a regime change after the news horizon."""
        T, N, k = self.T, self.N, len(self._cv)
        seen = set()
        for it in range(maxit):
            if z[T:].any():
                return False, True, z, None, it
            try:
                w = self._wedges(z, a)
            except np.linalg.LinAlgError:
                return False, False, z, None, it
            pcv = Ycv + (self._Wcv @ w).reshape(N, k) if w.any() else Ycv
            vb, vr = self._violations(z, pcv)
            if not vb.any() and not vr.any():
                return True, False, z, w, it
            key = z.tobytes()
            if key in seen:
                return False, False, z, None, it
            seen.add(key)
            z = (z & ~vr) | vb
        return False, False, z, None, maxit

    spell_l_max = 8      # latest start (period index) of a single spell that ``spells`` looks for

    def spells(self, Ycv, a, l_max=None):
        """Regime sequences with one alternative-regime spell {l, ..., l+k-1}, l <= l_max, that pass OccBin's test (one
        constraint): a list of (l, k, w) with w the wedges.  Each is a solution of the piecewise-linear model; more
        than one entry means that the state has several."""
        if self.n_c != 1:
            raise ValueError("spells needs a model with one constraint")
        T, N, k_cv = self.T, self.N, len(self._cv)
        out = []
        for l in range(min(self.spell_l_max if l_max is None else l_max, T - 1) + 1):
            for k in range(1, T - l + 1):
                z = np.zeros((N, 1), bool)
                z[l:l + k, 0] = True
                try:
                    w = self._wedges(z, a)
                except np.linalg.LinAlgError:
                    continue
                vb, vr = self._violations(z, Ycv + (self._Wcv @ w).reshape(N, k_cv))
                if not vb.any() and not vr.any():
                    out.append((l, k, w))
        return out

    def _step(self, init, shock, maxit=60, method="news", z0=None, selection="occbin"):
        """Regime indicators, wedges, next state and info for one shock period (no full path).  ``z0``: a
        starting guess (N, n_c) tried before the reference regime everywhere.  ``selection``: as in
        ``NewsOBC.simulate``; the single-spell solutions are those of ``spells``."""
        T, N, k = self.T, self.N, len(self._cv)
        init, shock = np.asarray(init, float), np.asarray(shock, float)
        xs = np.concatenate([init[self._st], shock])
        Ycv = (self._Bx @ xs).reshape(N, k)
        a = self.a0 + self._Ax @ xs
        z = np.zeros((N, self.n_c), bool)
        info = dict(used="none", iterations=0)
        ok = beyond = False
        if method != "milp":
            if z0 is not None and z0.any():
                ok, beyond, zg, w, it = self._iterate(z0.copy(), Ycv, a, maxit)
                if ok:
                    z = zg
                    info.update(used="guess_verify" if it else "none", iterations=it)
            if not ok:
                ok, beyond, z, w, it = self._iterate(z, Ycv, a, maxit)
                if ok:
                    info.update(used="guess_verify" if it else "none", iterations=it)
        if not ok and beyond and method != "milp":
            raise OBCError("a constraint still binds after the news horizon: increase obc_horizon", "horizon_too_short")
        if not ok and method == "guess_verify":
            raise OBCError("the guess-and-verify iteration cycles", "not_converged")
        if not ok and method != "milp":
            got = self._solve_lcp(init, shock)               # raises with a proof when no solution exists
            if got is not None:
                z, w, path = got
                vb, vr = self._violations(z, path)
                if not vb.any() and not vr.any():
                    info["used"] = "lcp"
                    if selection == "occbin" or not w.any():
                        return z, w, path[0], info
                    ok = True                                  # the chosen solution is looked for below
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
        if selection != "occbin" and w.any():
            cands = self.spells(Ycv, a)
            info["n_single_spell"] = len(cands)
            if cands:
                l, ks, w = pick_spell(cands, selection)
                z = np.zeros((N, self.n_c), bool)
                z[l:l + ks, 0] = True
                info["spell"] = (l, ks)
            else:
                info["selection_fallback"] = True
        nxt = self._Ps @ init[self._st] + self.ob.Q @ shock
        if w.any():
            nxt = nxt + self._V0 @ w
        return z, w, nxt, info

    def _milp(self, Y, a):
        """Regime indicators from a mixed-integer programme on the linearised conditions: minimise the number
        of periods spent in the alternative regimes subject to the regime logic (big-M)."""
        if self.milp_time_limit <= 0:
            raise OBCError("integer programme not attempted (milp_time_limit = 0)", "no_solution")
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

    def simulate(self, shocks, periods=100, init=None, method="news", exo_pos=None, warm_start=False,
                 selection="occbin") -> OccbinResult:
        """As ``NewsOBC.simulate`` (including ``selection``).  ``warm_start``: start the guess-and-verify iteration from
        the previous period's regime indicators moved forward by one period (the reference regime everywhere is the
        fallback)."""
        ob, n, T = self.ob, self.ob.n, self.T
        check_selection(selection, self.n_c)
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
        n_spell, fallback = np.full(S, -1), np.zeros(S, bool)
        path, z0 = None, None
        for sp in range(S):
            y_prev = y0
            try:
                z, w, y0, info = self._step(y_prev, sh[sp], method=method, z0=z0, selection=selection)
            except OBCError as err:
                err.period = sp + 1
                raise
            if warm_start:
                z0 = np.zeros_like(z)
                z0[:-1] = z[1:]
            wedges[sp] = w
            used.append(info["used"])
            n_spell[sp], fallback[sp] = info.get("n_single_spell", -1), info.get("selection_fallback", False)
            history.append([_map_regime(z[:, j].astype(np.int64)) for j in range(self.n_c)])
            piecewise[sp] = y0
            if sp == S - 1:                      # the path is only needed once, after the last shock period
                path = self._path(y_prev, sh[sp], w, upto=n_periods - S + 1)
        piecewise[S:] = path[1:n_periods - S + 1]
        linear = _linear_path(ob, self._st, init_orig, sh, n_periods)
        res = OccbinResult(piecewise=piecewise + ob.ys, linear=linear + ob.ys, shocks=shocks_in,
                           regime_history=history, ys=ob.ys, exo_pos=exo_idx)
        res.news = dict(method=method, formulation="regime", wedges=wedges, solver=used, horizon=T,
                        blocks=list(self.cols), selection=selection, n_single_spell=n_spell, selection_fallback=fallback)
        return res
