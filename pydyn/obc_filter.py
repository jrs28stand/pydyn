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

    # Why a trial point failed (``causes``, cumulative, and ``period_causes``, since ``begin_period``):
    #   horizon_too_short  the wedges that meet the complementarity rows inside the news horizon cannot meet the check-ahead rows
    #                      (a linear programme on the first n_s rows is feasible, the one on all rows is not), or RegimeOBC says so
    #   no_solution        proved: not even the complementarity rows can be met (or RegimeOBC: no solution)
    #   undecided          the cheap checks did not decide and no integer programme was run (the default); counted as a failure.  Not
    #                      proved to be a horizon problem, but it is the usual form it takes (soak/repro/finite_vs_inf_news_304.py: horizon
    #                      40 fails with 168 undecided trial points, horizon 80 reproduces OccBin to 1e-11); obc_horizon_max retries these too
    #   milp_timeout       the integer programme reached ``milp_time_limit``
    #   numerical          a linear-algebra failure (error 311)
    # Dynare's numeric code of a failed trial point stays 313 (311); the cause is extra information.
    CAUSES = ("horizon_too_short", "no_solution", "undecided", "milp_timeout", "numerical")

    def __init__(self, ob: Occbin, horizon=40, check_ahead=40, selection="occbin", milp_time_limit=None, horizon_max=None):
        super().__init__(ob)
        from .obc import check_selection
        self.horizon, self.check_ahead = int(horizon), int(check_ahead)
        if horizon_max is not None and int(horizon_max) < self.horizon:
            raise ValueError(f"obc_horizon_max={horizon_max} is below the horizon {horizon}")
        self.horizon_max = None if horizon_max is None else int(horizon_max)
        self.causes, self.period_causes, self.retries, self.retried_ok, self._big = {}, {}, 0, 0, {}
        self.selection = check_selection(selection, len(ob.names))
        if milp_time_limit is not None:
            self.milp_time_limit = float(milp_time_limit)
        self._maps = {}
        self._tol = 1e-8
        self.calls = self.fallbacks = self.proved_infeasible = self.timeouts = self.skipped = self.rescued = 0
        self.errors = {}                     # error code -> number of failed calls
        self.last_cause = None

    def _fail(self, code, cause=None):
        self.errors[code] = self.errors.get(code, 0) + 1
        if cause is not None:
            self.causes[cause] = self.causes.get(cause, 0) + 1
            self.period_causes[cause] = self.period_causes.get(cause, 0) + 1
            self.last_cause = cause
        return code, None

    def begin_period(self, t):
        """Called by the inversion filter at the start of period ``t`` (1-based): the per-period failure causes restart."""
        self.period, self.period_causes = t, {}

    def failure_info(self):
        """The cause of the inversion filter's failure in the current period: dict(period, primary, counts, horizon, advice) or None.
        ``primary`` is ``horizon_too_short`` if any trial point of the period failed for that reason (the actionable one), else the
        most frequent cause."""
        c = dict(self.period_causes)
        if not c:
            return None
        primary = "horizon_too_short" if c.get("horizon_too_short") else max(c, key=c.get)
        adv = {"horizon_too_short": f"news horizon {self.horizon} too short at period {getattr(self, 'period', '?')}: try obc_horizon >= "
                                    f"{2 * self.horizon} (or set obc_horizon_max to retry with a doubled horizon)",
               "no_solution": "the trial shocks have no equilibrium at this state (proved by a linear programme)",
               "undecided": f"active-set iteration did not decide and no integer programme was run (obc_milp_time_limit = 0); a truncated horizon "
                            f"({self.horizon}) can cause this: try obc_horizon >= {2 * self.horizon} or obc_horizon_max",
               "milp_timeout": "the integer programme reached obc_milp_time_limit", "numerical": "linear-algebra failure"}[primary]
        return dict(period=getattr(self, "period", None), primary=primary, counts=c, horizon=self.horizon, advice=adv)

    def _retry(self, shocks, init, sim, kw):
        """``obc_horizon_max``: after a trial point failed with horizon_too_short, solve it again with the horizon doubled, up to the
        maximum (the larger solver is built once and kept with this solver, i.e. per parameter vector).  None if not possible."""
        if self.horizon_max is None:
            return None
        h = self.horizon
        while h < self.horizon_max:
            h = min(2 * h, self.horizon_max)
            big = self._big.get(h)
            if big is None:
                big = self._big[h] = type(self)(self.ob, horizon=h, check_ahead=max(h, self.check_ahead), selection=self.selection,
                                                milp_time_limit=self.milp_time_limit)
            self.retries += 1
            flag, res = big.solve(shocks, init, sim, **kw)
            if flag == 0:
                self.retried_ok += 1
                return 0, res
            if big.last_cause not in ("horizon_too_short", "undecided"):
                break
        return None

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

    def __init__(self, ob: Occbin, horizon=40, check_ahead=40, selection="occbin", milp_time_limit=None, horizon_max=None):
        super().__init__(ob, horizon, check_ahead, selection, milp_time_limit, horizon_max)
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
            y = solve_active_set(nb.Mo, q, n_s, self._tol, scale=getattr(nb, "_mscale", None))
            cause = None
            if y is None:
                self.fallbacks += 1
                # cheap proof of non-existence first (the filter's root finder also tries implausible shocks)
                feas = linprog(np.zeros(n_s), A_ub=-nb.Mo, b_ub=q, bounds=[(0, None)] * n_s)
                if feas.status == 2:
                    self.proved_infeasible += 1
                    head = linprog(np.zeros(n_s), A_ub=-nb.Mo[:n_s], b_ub=q[:n_s], bounds=[(0, None)] * n_s)
                    cause = "horizon_too_short" if head.status == 0 else "no_solution"     # (b) / (a) of docs/OBC_HO_PERF.md
                    return self._fail_or_retry(cause, shocks, init, sim, init_regime, exo_pos, kw)
                T = nb.T
                if self.milp_time_limit > 0:
                    y, st = solve_news(nb.Mo, q, n_s, tol=self._tol, n_c=nb.n_c, horizons=[T],
                                       time_limit=self.milp_time_limit)
                    if y is None and st == "time_limit":
                        self.timeouts += 1
                        cause = "milp_timeout"
                else:
                    self.skipped += 1
                    cause = "undecided"
            if y is None:
                return self._fail_or_retry(cause or "undecided", shocks, init, sim, init_regime, exo_pos, kw)
            if self.fallbacks > f0:
                self.rescued += 1                        # the integer programme found what the cheap checks did not
            if self.selection != "occbin" and y.any():
                y = nb._select(q, y, self.selection, {}, self._tol)
            on = y > 1e-9 * max(1.0, float(np.abs(y).max()))
            S, s0, regime = self._map(on)
        return self._finish(S, s0, regime, x)


def _fail_or_retry(self, cause, shocks, init, sim, init_regime, exo_pos, kw):
    flag, res = self._fail(313, cause)
    if cause in ("horizon_too_short", "undecided"):
        got = self._retry(shocks, init, sim, dict(init_regime=init_regime, exo_pos=exo_pos, **kw))
        if got is not None:
            self.errors[313] -= 1                    # the point was solved with the larger horizon: not a failure
            self.causes[cause] -= 1
            self.period_causes[cause] -= 1
            return got
    return flag, res


NewsFilterSolver._fail_or_retry = _fail_or_retry


class RegimeFilterSolver(NewsFilterSolver):
    """General formulation (``RegimeOBC``): any model OccBin accepts.  For a regime sequence ``z`` the wedges on the
    binding periods are ``w = -R[S,S]^-1 (a0 + Am [state; shock])[S]`` (S: binding positions), so the next state is
    ``[P Q] x + V0[:, S] w``: affine in x, and the first period's rule (T, R, C) is read off and cached per S."""

    def __init__(self, ob: Occbin, horizon=40, check_ahead=40, selection="occbin", milp_time_limit=None, horizon_max=None):
        super().__init__(ob, horizon, check_ahead, selection, milp_time_limit, horizon_max)
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
            cause = {"horizon_too_short": "horizon_too_short", "no_solution": "no_solution", "not_converged": "milp_timeout"}.get(err.status, "undecided")
            if "not attempted" in str(err):
                cause = "undecided"
            return self._fail_or_retry(cause, shocks, init, sim, init_regime, exo_pos, kw)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return self._fail(311, "numerical")
        if info["used"] not in ("none", "guess_verify"):
            self.fallbacks += 1
            self.rescued += 1
        S, s0, regime = self._map(rb._index(z), z)
        return self._finish(S, s0, regime, x)
