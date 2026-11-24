# This file is part of pydyn. Released under the GNU General Public License, version 3 or later; see LICENSE.
# No warranty.
"""Inversion filter with the news-shock solver (``obc_method='news'``) in place of OccBin's.

The inversion filter (occbin_filter.inversion_filter) only needs, for one period, the solution from a state and
a shock vector together with the first period's rule ``y_t - ys = T (y_{t-1} - ys) + R u_t + C``.  For a given
binding set the news-shock solution is affine in [state; shock], so T, R and C come from the same maps as the
path and are cached per binding set (and per parameter vector, since the solver object is rebuilt with the model).

One complementary pair per constraint (``NewsOBC``) for now.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .obc import NewsOBC, OBCError, solve_active_set, solve_news
from .occbin import Occbin
from .occbin_filter import _Solver, as_regime

__all__ = ["NewsFilterSolver"]


class _OnePeriod:
    __slots__ = ("piecewise", "ss", "regimes", "regime_history")


class NewsFilterSolver(_Solver):
    milp_time_limit = 1.0        # seconds per fallback solve; a time-out counts as a failed trial point

    def __init__(self, ob: Occbin, horizon=40, check_ahead=40):
        super().__init__(ob)
        self.news = NewsOBC(ob, horizon=horizon, check_ahead=check_ahead)
        nb = self.news
        self._S0 = np.hstack([ob.P, ob.Q])
        self._Yw0 = nb.Yw[0]
        self._maps = {}
        self._tol = 1e-8
        self.calls = self.fallbacks = self.proved_infeasible = 0

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
        u = np.zeros(ob.n_exo)
        sh = np.asarray(shocks, float).ravel()
        if exo_pos is None:
            u[:sh.size] = sh
        else:
            u[np.asarray(exo_pos, int)] = sh
        x = np.r_[np.asarray(init, float).ravel(), u]
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
                    return 313, None
                T = nb.T
                y, _st = solve_news(nb.Mo, q, n_s, tol=self._tol, n_c=nb.n_c, horizons=[T], time_limit=self.milp_time_limit)
            if y is None:
                return 313, None
            on = y > 1e-9 * max(1.0, float(np.abs(y).max()))
            S, s0, regime = self._map(on)
        res = _OnePeriod()
        state = S @ x + s0
        res.piecewise = (state + ob.ys)[None, :]
        res.ss = [(S[:, :n], S[:, n:], s0)]
        res.regimes = [regime]
        res.regime_history = [regime]
        return 0, res
