# This file is part of pydyn. Released under the GNU General Public License, version 3 or later; see LICENSE.
# No warranty.
#
# The idea of indexing the solution by the number of slack periods l before a single binding spell and its
# length k, and of precomputing per (l, k) the maps from the state to the next state and to the quantities that
# verify the guess, is from G. Boehl, "Efficient solution and computation of models with occasionally binding
# constraints" (JEDC 2022) and his pydsge package (MIT licence). The code is written for pydyn's M/q objects.
"""Single-spell fast path for the news-shock LCP (one constraint).

For a binding set that is one contiguous spell S = {l, ..., l+k-1} the wedges are y_S = -M_SS^{-1} q_S, and q is
affine in x = [state; shock].  So for each (l, k) everything is affine in x:

* the verification vector  v(x) = V x + v0 >= 0  (the k wedges and the N-k gaps outside the spell: the complete
  LCP test, not a subset of periods), and
* the next state  s'(x) = S x + s0.

Both are built on first use and cached per parameter set.  A period then costs one matrix-vector product per
candidate (l, k) and one for the next state.  Candidates: nothing binds; last period's spell shifted by one
(exact when no new shock arrives); the spell found by the general solver, which is then cached.

Unlike pydsge, a candidate is accepted only if the whole LCP test passes, and when no single spell works the
general solver of ``NewsOBC`` takes over (several spells, integer programme).
"""
from __future__ import annotations

import numpy as np

from .obc import NewsOBC, OBCError
from .occbin import Occbin, OccbinResult

__all__ = ["LKNews"]


class LKNews(NewsOBC):
    """``NewsOBC`` with the cached single-spell fast path. One constraint only."""

    def __init__(self, model_or_occbin, params=None, horizon=40, check_ahead=40):
        super().__init__(model_or_occbin, params, horizon, check_ahead)
        if self.n_c != 1:
            raise ValueError("the single-spell fast path handles one constraint")
        self._prepare()

    def _build(self, N):
        super()._build(N)
        if getattr(self, "_lk", None) is not None:
            self._prepare()

    def _prepare(self):
        n, nx = self.ob.n, self.ob.n_exo
        self._lk = {}
        # next state with no wedges: P init + Q shock
        self._S0 = np.hstack([self.ob.P, self.ob.Q])
        self._Yw0 = self.Yw[0]                                  # (n, T): effect of each wedge on the next state
        self._tol = 1e-8
        self.stats = dict(none=0, shifted=0, cached=0, general=0, built=0)

    def _entry(self, l, k):
        """(V, v0, S, s0) for the spell {l..l+k-1}; V x + v0 >= 0 is the full LCP test."""
        e = self._lk.get((l, k))
        if e is None:
            T = self.T
            idx = np.arange(l, l + k)
            rhs = np.column_stack([self.Qm[idx], self.q0[idx]])
            sol = -np.linalg.solve(self.Mo[np.ix_(idx, idx)], rhs)       # y_S = W x + c
            W, c = sol[:, :-1], sol[:, -1]
            Mcol = self.Mo[:, idx]
            V = self.Qm + Mcol @ W                                         # gaps on all rows
            v0 = self.q0 + Mcol @ c
            V[idx], v0[idx] = W, c                                         # on the spell the test is y >= 0
            S = self._S0 + self._Yw0[:, idx] @ W
            s0 = self._Yw0[:, idx] @ c
            e = self._lk[(l, k)] = (V, v0, S, s0)
            self.stats["built"] += 1
        return e

    def _try(self, l, k, x, scale):
        V, v0, S, s0 = self._entry(l, k)
        if np.all(V @ x + v0 >= -self._tol * scale):
            return S @ x + s0
        return None

    def step(self, init, shock, prev=None):
        """Next state (deviations) and the spell (l, k); k = 0 when nothing binds. ``prev`` is last period's spell."""
        x = np.r_[init, shock]
        q = self.q0 + self.Qm @ x
        scale = max(1.0, float(np.abs(q).max()))
        if np.all(q >= -self._tol * scale):
            self.stats["none"] += 1
            return self._S0 @ x, (0, 0)
        T = self.T
        cands = []
        if prev is not None and prev[1] > 0:
            l, k = prev
            cands.append((l - 1, k) if l > 0 else (0, k - 1))
        neg = np.nonzero(q[:T] < -self._tol * scale)[0]
        if neg.size:
            cands.append((int(neg[0]), int(neg[-1] - neg[0] + 1)))          # the spell where q itself is negative
        for j, (l, k) in enumerate(cands):
            if k <= 0 or l + k > T or (j and (l, k) == cands[0]):
                continue
            nxt = self._try(l, k, x, scale)
            if nxt is not None:
                self.stats["shifted" if (j == 0 and prev is not None and prev[1] > 0) else "cached"] += 1
                return nxt, (l, k)
        # general solver; remember its answer if it is a single spell
        y, path, _info = self.solve_period(init, shock)
        self.stats["general"] += 1
        on = np.nonzero(y > 1e-9 * max(1.0, float(np.abs(y).max(initial=0.0))))[0]
        if on.size and on[-1] - on[0] + 1 == on.size:
            return path[0].copy(), (int(on[0]), int(on.size))
        return path[0].copy(), None if on.size else (0, 0)

    def simulate(self, shocks, periods=100, init=None, method="news", exo_pos=None, omega=1000.0) -> OccbinResult:
        if method != "news":
            return super().simulate(shocks, periods=periods, init=init, method=method, exo_pos=exo_pos, omega=omega)
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
        history, prev, last_state = [], None, y0
        for sp in range(S):
            last_state = y0
            try:
                y0, lk = self.step(y0, sh[sp], prev)
            except OBCError as err:
                err.period = sp + 1
                raise
            if lk is None:                                     # several spells: take the history from the wedges
                yw, _p, _i = self.solve_period(last_state, sh[sp])
                history.append([Occbin.map_regime((yw > 1e-9 * max(1.0, float(np.abs(yw).max()))).astype(int))])
            else:
                l, k = lk
                ind = np.zeros(T, int)
                ind[l:l + k] = 1
                history.append([Occbin.map_regime(ind)])
            prev = lk
            piecewise[sp] = y0
        if n_periods > S:                                      # the perfect-foresight path after the last shock
            _y, path, _info = self.solve_period(last_state, sh[S - 1])
            piecewise[S:] = path[1:n_periods - S + 1]
        linear = np.empty((n_periods, n))
        yl = init_orig
        for t in range(n_periods):
            yl = ob.P @ yl + (ob.Q @ sh[t] if t < S else 0.0)
            linear[t] = yl
        res = OccbinResult(piecewise=piecewise + ob.ys, linear=linear + ob.ys, shocks=shocks_in,
                           regime_history=history, ys=ob.ys, exo_pos=exo_idx)
        res.news = dict(method="news", formulation="lcp_single_spell", horizon=T, stats=dict(self.stats))
        return res
