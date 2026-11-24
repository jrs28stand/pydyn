# This file is part of pydyn. Released under the GNU General Public License, version 3 or later; see LICENSE.
# No warranty.
"""Inversion filter with the news-shock solver (``obc_method='news'``) in place of OccBin's.

The inversion filter (occbin_filter.inversion_filter) only needs, for one period, the solution from a state and
a shock vector together with the first period's rule ``y_t - ys = T (y_{t-1} - ys) + R u_t + C``.  For a given
binding set the news-shock solution is affine in [state; shock], so T, R and C come from the same maps as the
path and are cached per binding set (and per parameter vector, since the solver object is rebuilt with the model).

Constraints that are one complementary pair each use ``NewsOBC`` (Holden's LCP); every other model OccBin accepts
(a constraint that switches several equations, conditions that are not the mirror image of the switched equation) uses
``RegimeOBC``: for a given regime sequence the wedges are affine in [state; shock], so the first period's rule follows
in the same way.  ``NewsFilterSolver(ob)`` picks the class as ``Occbin.simulate(obc_method='news')`` does.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .obc import NewsOBC, OBCError, RegimeOBC, constraint_objects, solve_active_set, solve_news
from .occbin import Occbin
from .occbin_filter import _Solver, as_regime

__all__ = ["NewsFilterSolver", "LCPFilterSolver", "RegimeFilterSolver"]


class _OnePeriod:
    __slots__ = ("piecewise", "ss", "regimes", "regime_history")


class NewsFilterSolver(_Solver):
    """Solver of the inversion filter for ``obc_method='news'``: ``NewsFilterSolver(ob, horizon=40, check_ahead=40,
    selection='occbin')`` returns a :class:`LCPFilterSolver` when every constraint is one complementary pair and a
    :class:`RegimeFilterSolver` otherwise (the choice of ``Occbin.simulate``)."""

    # Seconds the integer programme may take at a trial point where the cheap checks (active-set iteration, proof of
    # non-existence by a linear programme) did not decide.  0 (default): none, the trial point has failed (error 313,
    # as OccBin's own failures: at every such point seen OccBin failed too, see docs/OBC_ESTIMATION.md).  A positive
    # value runs the integer programme with that limit; a time-out is a failure as well.
    milp_time_limit = 0.0

    def __new__(cls, ob: Occbin, *args, **kw):
        if cls is NewsFilterSolver:
            lcp = all(c.cls == "complementarity" for c in constraint_objects(ob))
            cls = LCPFilterSolver if lcp else RegimeFilterSolver
        return object.__new__(cls)

    def __init__(self, ob: Occbin, horizon=40, check_ahead=40, selection="occbin", milp_time_limit=None):
        super().__init__(ob)
        from .obc import check_selection
        self.selection = check_selection(selection, len(ob.names))
        if milp_time_limit is not None:
            self.milp_time_limit = float(milp_time_limit)
        self._maps = {}
        self._tol = 1e-8
        self.calls = self.fallbacks = self.proved_infeasible = self.timeouts = self.skipped = self.rescued = 0
        self.errors = {}                     # error code -> number of failed calls

    def _fail(self, code):
        self.errors[code] = self.errors.get(code, 0) + 1
        return code, None

    def _finish(self, S, s0, regime, x):
        ob = self.ob
        res = _OnePeriod()
        state = S @ x + s0
        res.piecewise = (state + ob.ys)[None, :]
        res.ss = [(S[:, :ob.n], S[:, ob.n:], s0)]
        res.regimes = [regime]
        res.regime_history = [regime]
        return 0, res

    def _x(self, shocks, init, exo_pos):
        ob = self.ob
        u = np.zeros(ob.n_exo)
        sh = np.asarray(shocks, float).ravel()
        if exo_pos is None:
            u[:sh.size] = sh
        else:
            u[np.asarray(exo_pos, int)] = sh
        return u, np.r_[np.asarray(init, float).ravel(), u]


class LCPFilterSolver(NewsFilterSolver):
    """One complementary pair per constraint (``NewsOBC``)."""

    def __init__(self, ob: Occbin, horizon=40, check_ahead=40, selection="occbin", milp_time_limit=None):
        super().__init__(ob, horizon, check_ahead, selection, milp_time_limit)
        self.news = NewsOBC(ob, horizon=horizon, check_ahead=check_ahead)
        nb = self.news
        self._S0 = np.hstack([ob.P, ob.Q])
        self._Yw0 = nb._V0                                # effect of each wedge on the next state

    def _map(self, on):
        key = on.tobytes()
        got = self._maps.get(key)
        if got is None:
            nb = self.news
            idx = np.nonzero(on)[0]
            sol = -np.linalg.solve(nb.Mo[np.ix_(idx, idx)], np.column_stack([nb.Qm[idx], nb.q0[idx]]))
            W, c = sol[:, :-1], sol[:, -1]
            S = self._S0 + self._Yw0[:, idx] @ W
            s0 = self._Yw0[:, idx] @ c
            T = nb.T
            hist = [Occbin.map_regime(on[j * T:(j + 1) * T].astype(int)) for j in range(nb.n_c)]
            got = self._maps[key] = (S, s0, as_regime(hist))
        return got

    def solve(self, shocks, init, sim: dict, init_regime=None, exo_pos=None, **kw):
        ob, nb, n = self.ob, self.news, self.ob.n
        self.calls += 1
        u, x = self._x(shocks, init, exo_pos)
        f0 = self.fallbacks
        q = nb.q0 + nb.Qm @ x
        n_s = nb.n_c * nb.T
        scale = max(1.0, float(np.abs(q).max()))
        if np.all(q >= -self._tol * scale):
            on = np.zeros(n_s, bool)
            S, s0 = self._S0, np.zeros(n)
            regime = as_regime([Occbin.map_regime(np.zeros(nb.T, int)) for _ in range(nb.n_c)])
        else:
            y = solve_active_set(nb.Mo, q, n_s, self._tol)
            if y is None:
                self.fallbacks += 1
                # cheap proof of non-existence first (the filter's root finder also tries implausible shocks)
                feas = linprog(np.zeros(n_s), A_ub=-nb.Mo, b_ub=q, bounds=[(0, None)] * n_s)
                if feas.status == 2:
                    self.proved_infeasible += 1
                    return self._fail(313)
                T = nb.T
                if self.milp_time_limit > 0:
                    y, st = solve_news(nb.Mo, q, n_s, tol=self._tol, n_c=nb.n_c, horizons=[T],
                                       time_limit=self.milp_time_limit)
                    if y is None and st == "time_limit":
                        self.timeouts += 1
                else:
                    self.skipped += 1
            if y is None:
                return self._fail(313)
            if self.fallbacks > f0:
                self.rescued += 1                        # the integer programme found what the cheap checks did not
            if self.selection != "occbin" and y.any():
                y = nb._select(q, y, self.selection, {}, self._tol)
            on = y > 1e-9 * max(1.0, float(np.abs(y).max()))
            S, s0, regime = self._map(on)
        return self._finish(S, s0, regime, x)


class RegimeFilterSolver(NewsFilterSolver):
    """General formulation (``RegimeOBC``): any model OccBin accepts.  For a regime sequence ``z`` the wedges on the
    binding periods are ``w = -R[S,S]^-1 (a0 + Am [state; shock])[S]`` (S: binding positions), so the next state is
    ``[P Q] x + V0[:, S] w``: affine in x, and the first period's rule (T, R, C) is read off and cached per S."""

    def __init__(self, ob: Occbin, horizon=40, check_ahead=40, selection="occbin", milp_time_limit=None):
        super().__init__(ob, horizon, check_ahead, selection, milp_time_limit)
        self.rb = rb = RegimeOBC(ob, horizon=horizon, check_ahead=check_ahead)
        rb.milp_time_limit = self.milp_time_limit        # fail fast: the filter's root finder tries implausible shocks
        self._S0 = np.hstack([ob.P, ob.Q])

    def _map(self, S, z):
        key = S.tobytes()
        got = self._maps.get(key)
        if got is None:
            rb, T = self.rb, self.rb.T
            if S.size:
                sol = -np.linalg.solve(rb.R[np.ix_(S, S)], np.column_stack([rb.Am[S], rb.a0[S]]))
                W, c = sol[:, :-1], sol[:, -1]
                M = self._S0 + rb._V0[:, S] @ W
                s0 = rb._V0[:, S] @ c
            else:
                M, s0 = self._S0, np.zeros(self.ob.n)
            regime = as_regime([Occbin.map_regime(z[:T, j].astype(int)) for j in range(rb.n_c)])
            got = self._maps[key] = (M, s0, regime)
        return got

    def solve(self, shocks, init, sim: dict, init_regime=None, exo_pos=None, **kw):
        rb = self.rb
        self.calls += 1
        u, x = self._x(shocks, init, exo_pos)
        try:
            z, w, _nxt, info = rb._step(x[:self.ob.n], u, selection=self.selection)
        except OBCError as err:
            self.fallbacks += 1
            if err.status == "not_converged":          # the integer programme hit its time limit
                self.timeouts += 1
            elif err.status == "no_solution":
                self.proved_infeasible += "linear programme infeasible" in str(err)
                self.skipped += "not attempted" in str(err)
            return self._fail(313)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return self._fail(311)
        if info["used"] not in ("none", "guess_verify"):
            self.fallbacks += 1
            self.rescued += 1
        S, s0, regime = self._map(rb._index(z), z)
        return self._finish(S, s0, regime, x)
