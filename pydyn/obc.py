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
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .expr import NAMESPACE, Translator
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
    returned function is negative (``enter=True``, bind) or positive (``enter=False``, relax)."""
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
    for i in range(n):
        h = 1e-6 * max(1.0, abs(ys[i]))
        zp, zm = ys.copy(), ys.copy()
        zp[i] += h
        zm[i] -= h
        out[n + i] = sign * (val(zp) - val(zm)) / (2 * h)
    return out


def _fit(target, a, b):
    """Least squares target ~ ca*a + cb*b; returns (ca, cb, relative residual)."""
    X = np.column_stack([a, b])
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    res = np.linalg.norm(X @ coef - target) / max(np.linalg.norm(target), 1e-300)
    return float(coef[0]), float(coef[1]), float(res)


def constraint_objects(ob: Occbin, tol: float = 1e-6, approx_tol: float = 0.05) -> list:
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
        """Response of every variable (N+1 periods) to a unit wedge of each constraint at each horizon."""
        ob, n, T = self.ob, self.ob.n, self.T
        C0, B0, A0, _J0, _D0 = ob.regime_matrices(tuple([False] * ob.k))
        F = np.linalg.inv(B0 + A0 @ ob.P)
        G = -F @ A0
        self.N = N
        # Yw[t, :, c*T+s]: deviation at period t of a unit wedge y_c(s)
        Yw = np.zeros((N + 1, n, self.n_c * T))
        for ci, co in enumerate(self.constraints):
            e = np.zeros(n)
            e[co.equations[0]] = co.s_y                 # ref_resid = s_y * y
            k = [F @ e]
            for _ in range(T - 1):
                k.append(G @ k[-1])                     # k[d] = G^d F e
            for s in range(T):
                y = np.zeros(n)
                col = ci * T + s
                for t in range(N + 1):
                    y = ob.P @ y + (k[s - t] if t <= s else 0.0)
                    Yw[t, :, col] = y
        self.Yw = Yw
        self.M = self._gap_of(Yw, const=False)          # (n_c*N, n_c*T)
        self._rows_s = np.concatenate([np.arange(c * N, c * N + T) for c in range(self.n_c)])
        self._rows_tail = np.setdiff1d(np.arange(self.n_c * N), self._rows_s)
        self._order = np.r_[self._rows_s, self._rows_tail]
        self.Mo = self.M[self._order]                   # complementarity rows first, then check-ahead rows
        self.Ms = self.Mo[:self.n_c * T]
        # q is affine in the initial state and the shock: q = q0 + Qs init + Qu shock (depends on parameters only)
        nx = ob.n_exo
        Yb = np.zeros((N + 1, n, n + nx))
        y = np.hstack([np.eye(n), np.zeros((n, nx))])
        for t in range(N + 1):
            y = ob.P @ y
            if t == 0:
                y[:, n:] = ob.Q
            Yb[t] = y
        Qm = self._gap_of(Yb, const=False)
        q0 = np.zeros(self.n_c * N)
        for ci, co in enumerate(self.constraints):
            a = co.s_g * co.alt
            q0[ci * N:(ci + 1) * N] = a[-1]
            Qm[ci * N, :n] += a[:n]                       # lag of the first period is the initial state
            Qm[ci * N, n:] += a[3 * n:3 * n + nx]         # the shock enters the first period's residual directly
        self.q0, self.Qm = q0[self._order], Qm[self._order]

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

    def q_of(self, init, shock):
        """(q, base path): gaps with no wedges, ordered complementarity rows first."""
        Y = self.base_path(init, shock)
        return self.q0 + self.Qm @ np.r_[np.asarray(init, float), np.asarray(shock, float)], Y

    def q_direct(self, init, shock):
        """q computed from the base path itself (check of the affine map used by ``q_of``)."""
        n = self.ob.n
        Y = self.base_path(init, shock)
        q = self._gap_of(Y, const=True, shock=shock)
        for ci, co in enumerate(self.constraints):       # lag of the first period is the initial state
            q[ci * self.N] += co.s_g * co.alt[:n] @ np.asarray(init, float)
        return q[self._order]

    def stacked_M(self):
        """M from one stacked linear system over a long horizon (independent check of ``_build``)."""
        ob, n, T, N = self.ob, self.ob.n, self.T, self.N
        H = N + 200
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
    def solve_period(self, init, shock, method="news", omega=1000.0, tol=1e-8):
        """Wedges and path for one shock period. Returns (y, path (N+1, n) in deviations, info)."""
        q, Y = self.q_of(init, shock)
        n_s = self.n_c * self.T
        info = dict(used="none")
        if np.all(q >= -tol * max(1.0, np.abs(q).max())):
            y = np.zeros(n_s)
        else:
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
        path = Y + self.Yw @ y
        return y, path, info

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
            try:
                y, path, info = self.solve_period(y0, sh[sp], method=method, omega=omega)
            except OBCError as err:
                err.period = sp + 1
                raise
            wedges[sp] = y
            used.append(info["used"])
            scale = max(1.0, float(np.abs(y).max(initial=0.0)))
            history.append([Occbin.map_regime((y[c * T:(c + 1) * T] > 1e-9 * scale).astype(int))
                            for c in range(self.n_c)])
            y0 = path[0].copy()
            piecewise[sp] = y0
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
        self._build(self.T + self.L)

    def _build(self, N):
        ob, n, nx, T = self.ob, self.ob.n, self.ob.n_exo, self.T
        C0, B0, A0, _J0, _D0 = ob.regime_matrices(tuple([False] * ob.k))
        F = np.linalg.inv(B0 + A0 @ ob.P)
        G = -F @ A0
        self.N, nb = N, len(self.cols)
        Yw = np.zeros((N + 1, n, nb * T))
        for b, (_j, row) in enumerate(self.cols):
            e = np.zeros(n)
            e[row] = 1.0
            k = [F @ e]
            for _ in range(T - 1):
                k.append(G @ k[-1])
            for s in range(T):
                y = np.zeros(n)
                for t in range(N + 1):
                    y = ob.P @ y + (k[s - t] if t <= s else 0.0)
                    Yw[t, :, b * T + s] = y
        self.Yw = Yw
        # alternative residuals in periods 0..T-1: a = a0 + Am [init; shock] + R w
        Yb = np.zeros((N + 1, n, n + nx))
        y = np.hstack([np.eye(n), np.zeros((n, nx))])
        for t in range(N + 1):
            y = ob.P @ y
            if t == 0:
                y[:, n:] = ob.Q
            Yb[t] = y
        self.Yb = Yb
        R = np.zeros((nb * T, nb * T))
        Am = np.zeros((nb * T, n + nx))
        a0 = np.zeros(nb * T)
        for b, a in enumerate(self.alt_rows):
            aC, aB, aA, aJ = a[:n], a[n:2 * n], a[2 * n:3 * n], a[3 * n:3 * n + nx]
            for t in range(T):
                for Y, out in ((Yw, R), (Yb, Am)):
                    v = aB @ Y[t] + aA @ Y[t + 1]
                    if t > 0:
                        v = v + aC @ Y[t - 1]
                    out[b * T + t] = v
            a0[b * T:(b + 1) * T] = a[-1]
            Am[b * T, :n] += aC
            Am[b * T, n:] += aJ
        self.R, self.Am, self.a0 = R, Am, a0
        self._tie = self._find_tie()

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
        if G is None:
            y = solve_active_set(M, q, T)
            if y is None:
                y, _status = solve_news(M, q, T, horizons=sorted({max(1, T // 4), max(1, T // 2), T}))
            if y is None:
                feas = linprog(np.zeros(T), A_ub=-M, b_ub=q, bounds=[(0, None)] * T)
                raise OBCError(proof if feas.status == 2 else
                               "no solution within the news horizon (the linear complementarity problem has none)",
                               "no_solution")
            zsol = y > 1e-9 * max(1.0, float(np.abs(y).max()))
        else:
            # relaxation without bounds: y >= 0, 0 <= z <= 1, q + M y + G z >= 0
            feas = linprog(np.zeros(2 * T), A_ub=-np.hstack([M, G]), b_ub=q,
                           bounds=[(0, None)] * T + [(0, 1)] * T)
            if feas.status == 2:
                raise OBCError(proof, "no_solution")
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
        T = self.T
        return np.concatenate([b * T + np.nonzero(z[:T, j])[0] for b, (j, _r) in enumerate(self.cols)]).astype(int)

    def _wedges(self, z, a):
        w = np.zeros(len(self.cols) * self.T)
        S = self._index(z)
        if S.size:
            w[S] = np.linalg.solve(self.R[np.ix_(S, S)], -a[S])
        return w

    def _violations(self, z, path):
        binding, relax, _eb, _er = self.ob.check(path[:self.N] + self.ob.ys)
        return binding & ~z, relax & z

    def solve_period(self, init, shock, maxit=60, method="news"):
        """Regime indicators (N, n_c), wedges and path (N+1, n; deviations) for one shock period."""
        T, N = self.T, self.N
        x = np.r_[np.asarray(init, float), np.asarray(shock, float)]
        Y = np.tensordot(self.Yb, x, axes=(2, 0))
        a = self.a0 + self.Am @ x
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
                path = Y + self.Yw @ w
                vb, vr = self._violations(z, path)
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
        if not ok and method != "milp":
            got = self._solve_lcp(init, shock)               # raises with a proof when no solution exists
            if got is not None:
                z, w, path = got
                vb, vr = self._violations(z, path)
                if not vb.any() and not vr.any():
                    info["used"] = "lcp"
                    return z, w, path, info
        if not ok:
            try:
                z = self._milp(Y, a)
            except OBCError as err:
                if beyond:
                    raise OBCError("a constraint still binds after the news horizon: increase obc_horizon",
                                   "horizon_too_short") from err
                raise
            info["used"] = "milp"
            w = self._wedges(z, a)
            path = Y + self.Yw @ w
            vb, vr = self._violations(z, path)
            if vb[T:].any():
                raise OBCError("a constraint still binds after the news horizon: increase obc_horizon",
                               "horizon_too_short")
            if vb.any() or vr.any():
                raise OBCError("the regime sequence found by the integer programme (linearised conditions) fails "
                               "OccBin's test on the conditions in levels", "not_converged")
        return z, w, path, info

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
            try:
                z, w, path, info = self.solve_period(y0, sh[sp], method=method)
            except OBCError as err:
                err.period = sp + 1
                raise
            wedges[sp] = w
            used.append(info["used"])
            history.append([Occbin.map_regime(z[:, j].astype(int)) for j in range(self.n_c)])
            y0 = path[0].copy()
            piecewise[sp] = y0
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
