# This file is part of pydyn. Released under the GNU General Public License, version 3 or later; see LICENSE.
# No warranty.
#
# New design (no external reference): the structure of pydyn.obc_ho.HigherOrderNews (pruned base simulation, first-order bound
# offset, expected path with future shocks at their distribution, quasi-Monte-Carlo averaging over perturbed paths; after T. D. Holden
# and DynareOBC) combined with pydyn.obc.RegimeOBC (one wedge per switched equation, acceptance by OccBin's bind/relax test in levels).
"""Order 2 and cubature for the regime formulation (``obc_formulation='regime'``): see docs/OBC_REGIME_HO.md.

Per period the state-dependent inputs of ``RegimeOBC``'s search are the alternative residuals ``a`` of every switched equation (T periods)
and the path ``Ycv`` of the variables the bind/relax conditions read (N periods), both on the no-wedge path.  At order 2 they are
evaluated on the order-2 expected path of the unconstrained model (``HigherOrderNews.expected_path``); the response to the wedges stays
first order (``R``, ``Wcv``, ``V0`` of ``RegimeOBC``).  Both are affine in ``z = (y1 + y2, vec(x1 x1'), o_pre)``, precomputed per parameter set.
"""
from __future__ import annotations

import warnings

import numpy as np

from .obc import OBCError, RegimeOBC, check_selection
from .obc_ho import CubatureFailure, HigherOrderNews
from .occbin import Occbin

__all__ = ["HigherOrderRegime"]


class HigherOrderRegime(HigherOrderNews):
    formulation = "regime"

    def __init__(self, model, order=2, horizon=40, check_ahead=40, params=None, Sigma_e=None,
                 cubature=False, periods_of_uncertainty=16, qmc_level=11, regions="all", taper=True, seed=0, pruning_cutoff=0.01,
                 max_cubature_dimension=128, cubature_milp_time_limit=1.0, cubature_failure_tolerance=0.0, m_diagnostic=False,
                 obc_formulation="regime", milp_time_limit=None):
        from .korder import k_order
        self.model = model
        self.order = int(order)
        if self.order not in (1, 2):
            raise NotImplementedError("order 3 is not implemented in this prototype")
        self.ob = ob = Occbin(model, params)
        self.rb = rb = RegimeOBC(ob, horizon=horizon, check_ahead=check_ahead)
        if milp_time_limit is not None:
            rb.milp_time_limit = float(milp_time_limit)
        self.news = None
        self.T, self.N, self.n = rb.T, rb.N, ob.n
        self.Sigma = np.array(model.Sigma_e if Sigma_e is None else Sigma_e, float)
        dr = k_order(model, order=max(self.order, 1), ys=ob.ys, params=ob.params, pruning=True, Sigma_e=self.Sigma)
        self.dr = dr
        self.ov, self.sv = np.asarray(dr["order_var"]), np.asarray(dr["state_var"])
        pos = {v: i for i, v in enumerate(self.ov)}
        self.ist = np.array([pos[v] for v in self.sv])
        self.ghx, self.ghu = dr["ghx"], dr["ghu"]
        self.ghxs, self.ghus = self.ghx[self.ist], self.ghu[self.ist]
        if self.order == 2:
            self.ghxx, self.ghxu, self.ghuu = dr["ghxx"], dr["ghxu"], dr["ghuu"]
            self.ghs2 = dr["ghs2"].ravel()
            self.c2 = 0.5 * (self.ghuu @ self.Sigma.ravel() + self.ghs2)
            ns_ = len(self.ist)
            self._iu, self._ju = np.triu_indices(ns_)
            self.ghxx_s = self._fold(self.ghxx.reshape(self.n, ns_, ns_))
        self.SigS = self.ghus @ self.Sigma @ self.ghus.T
        self.n_c, self.nb, self.nws = rb.n_c, len(rb.cols), len(rb.cols) * rb.T
        self._build_maps()
        self.cubature = bool(cubature)
        self.pruning_cutoff, self.max_cubature_dimension = pruning_cutoff, int(max_cubature_dimension)
        self.pou, self.qmc_level, self.regions, self.taper, self.seed = (int(periods_of_uncertainty), int(qmc_level), regions,
                                                                         bool(taper), seed)
        self.milp_time_limit = float(cubature_milp_time_limit)
        self.failure_tolerance = float(cubature_failure_tolerance)
        if not 0.0 <= self.failure_tolerance < 1.0:
            raise ValueError("cubature_failure_tolerance must be in [0, 1)")
        self.m_diag = None
        if self.cubature:
            self._init_cubature()
        self.trace = None                      # a list: simulate appends per-period inputs and outputs (for the tests)
        self._dropped_last = self._relevant_last = 0
        self.stats = dict(regime=0, cubature_points=0, relevant_points=0)

    # ------------------------------------------------------------------------------------ precomputed maps
    def _functionals(self):
        """(aC, aB, aA) of every functional of the expected path that the search needs, declaration order: the alternative residual of
        each switched equation (rows 0..nb-1) and the condition variables (rows nb..nb+k-1; aB only)."""
        rb, n = self.rb, self.n
        a = np.array(rb.alt_rows)
        aC, aB, aA = a[:, :n], a[:, n:2 * n], a[:, 2 * n:3 * n]
        k = len(rb._cv)
        E = np.zeros((k, n))
        E[np.arange(k), rb._cv] = 1.0
        Z = np.zeros((k, n))
        return np.vstack([aC, Z]), np.vstack([aB, E]), np.vstack([aA, Z])

    def _build_maps(self):
        rb, N = self.rb, self.N
        aC, aB, aA = self._functionals()
        self._fC, self._fB, self._fA = aC, aB, aA
        self._Gall, self._c0all = self._affine_maps(aC, aB, aA, N)
        a = np.array(rb.alt_rows)
        self._aCb, self._aJb, self._a0b = a[:, :self.n], a[:, 3 * self.n:3 * self.n + self.ob.n_exo], a[:, -1]
        self._nz = self._Gall.shape[1]
        self._z = np.empty(self._nz)

    def _affine_maps(self, aC, aB, aA, H):
        """Affine maps of functionals ``f_t = aC E_{t-1} + aB E_t + aA E_{t+1}``, t = 0..H-1, of the expected path:
        ``f = c0 + Gz z`` (row f*H + t; the lag of period 0 and the shock are added by the caller).  Generalises
        ``HigherOrderNews._build_gap_maps`` (the same adjoint recursion) to any list of functionals."""
        n, ist, ov = self.n, self.ist, self.ov
        ns, two = len(ist), self.order == 2
        F, ghx = self.ghxs, self.ghx
        K = H + 1
        m = aB.shape[0]
        m3 = 3 * m
        Ay = np.empty((m3, n)); Ao = np.empty((m3, n))
        for j, A in enumerate((aB, aA, aC)):
            Ay[j::3] = A[:, ov]
            Ao[j::3] = A
        rY = np.zeros((K, m3, n)); rY[0] = Ay
        rO = np.empty((K, m3, n)); rO[0] = Ao
        if two:
            nn = ns * ns
            rW = np.zeros((K, m3, nn)); rc = np.zeros((K, m3))
            Gbar = (Ay @ self.ghxx).reshape(m3, ns, ns)
            Gs = self.ghxx[ist]
            Bm = Ay @ ghx
            D = np.zeros((m3, ns, ns))
            zc = np.zeros(ns)
            S = np.zeros((ns, ns))
        Vy = Ay @ ghx
        P = self.ob.P
        for k in range(1, K):
            rY[k][:, ist] = Vy
            rO[k] = rO[k - 1] @ P
            if two:
                dc = 0.5 * (self.ghxx @ S.ravel()) + self.c2
                rW[k] = 0.5 * (Gbar + D).reshape(m3, nn)
                rc[k] = Ay @ (ghx @ zc + dc)
                D = (Bm @ Gs).reshape(m3, ns, ns) + np.matmul(np.matmul(F.T, D), F)
                Gbar = np.matmul(np.matmul(F.T, Gbar), F)
                Bm = Bm @ F
                zc = F @ zc + dc[ist]
                S = F @ S @ F.T + self.SigS
            Vy = Vy @ F

        def comb(r):
            out = []
            for c in range(m):
                G = r[:H, 3 * c] + r[1:K, 3 * c + 1]
                G[1:] += r[:H - 1, 3 * c + 2]
                out.append(G)
            return np.vstack(out)
        cols = [comb(rY)]
        c0 = np.zeros(m * H)
        if two:
            cols.append(comb(rW))
            c0 = comb(rc[:, :, None])[:, 0]
        cols.append(comb(rO))
        if two:
            cols[1] = self._fold(cols[1].reshape(cols[1].shape[0], ns, ns))
        return np.ascontiguousarray(np.hstack(cols)), c0

    def _zvec(self, y1, y2, o_pre):
        n, ist = self.n, self.ist
        z = self._z
        if self.order == 2:
            np.add(y1, y2, out=z[:n])
            x1 = y1[ist]
            z[n:n + self._iu.size] = x1[self._iu] * x1[self._ju]
        else:
            z[:n] = y1
        z[-n:] = o_pre
        return z

    def inputs_from_state(self, y1, y2, o_pre, prev_total, shock):
        """(a, Ycv, base) from the precomputed maps: ``a`` (nb*T) alternative residuals of the switched equations on the no-wedge path,
        ``Ycv`` (N, k) the condition variables' path, ``base`` (nb, N) the residual paths over all N periods."""
        rb, N, T, nb = self.rb, self.N, self.T, self.nb
        v = self._Gall @ self._zvec(y1, y2, o_pre)
        v += self._c0all
        A = v[:nb * N].reshape(nb, N)
        A[:, 0] += self._aCb @ prev_total + self._aJb @ shock
        A += self._a0b[:, None]
        Ycv = v[nb * N:].reshape(len(rb._cv), N).T
        return A[:, :T].ravel(), Ycv, A

    def inputs_direct(self, y1, y2, o_pre, prev_total, shock):
        """The same from the expected path itself (slow reference implementation for the tests)."""
        rb, N, T, nb, n = self.rb, self.N, self.T, self.nb, self.n
        E = self.expected_path(y1, y2, o_pre)
        A = np.empty((nb, N))
        for b in range(nb):
            A[b] = self._a0b[b] + E[:N] @ self._fB[b] + E[1:N + 1] @ self._fA[b]
            A[b, 1:] += E[:N - 1] @ self._fC[b]
            A[b, 0] += self._aCb[b] @ prev_total + self._aJb[b] @ shock
        return A[:, :T].ravel(), E[:N][:, rb._cv], A

    # ------------------------------------------------------------------------------------------ one period
    def solve_inputs(self, a, Ycv, A, z0=None, selection="occbin", maxit=60):
        """The regime search of ``RegimeOBC`` on given inputs: (z, w, info)."""
        rb, N, k, n = self.rb, self.N, len(self.rb._cv), self.n
        self.stats["regime"] += 1

        def lcp_fn():
            if rb._tie is None or not rb._tie.get("mirror"):
                return None
            tie = rb._tie
            base = A[tie["primary"]]
            M, q, G = rb.lcp_from(a, base)
            z, w = rb._solve_lcp_core(M, q, G, a)
            return z, w, Ycv + (rb._Wcv @ w).reshape(N, k)

        def base_fn():
            Y = np.zeros((N + 1, n))
            Y[:N, rb._cv] = Ycv
            return Y

        z, w, info, _ = rb._search(Ycv, a, maxit, "news", z0, selection, lcp_fn, base_fn)
        return z, w, info

    # ----------------------------------------------------------------------------------------- simulation
    def simulate(self, shocks, init=None, selection="occbin"):
        """Simulate under unanticipated shocks; the same dict as ``HigherOrderNews.simulate`` (levels, declaration order), with the wedges of all
        switched equations (``nb * T`` per period, block-major) and the regime history (``regimes``)."""
        ob, n, rb = self.ob, self.n, self.rb
        check_selection(selection, self.n_c)
        shocks = np.atleast_2d(np.asarray(shocks, float))
        S = shocks.shape[0]
        ys = ob.ys
        y1, y2, o = np.zeros(n), np.zeros(n), np.zeros(n)
        if init is not None:
            y1[:] = (np.asarray(init, float) - ys)[self.ov]
        prev_total = np.empty(n)
        prev_total[self.ov] = y1 + y2
        prev_total += o
        with_b, without_b, wedges = np.empty((S, n)), np.empty((S, n)), np.zeros((S, self.nws))
        dropped, relevant = np.zeros(S, int), np.zeros(S, int)
        regimes, used = [], []
        for t in range(S):
            u = shocks[t]
            x1, x2 = y1[self.ist], y2[self.ist]
            if self.order == 2:
                y2 = self.ghx @ x2 + 0.5 * (self.ghxx_s @ (x1[self._iu] * x1[self._ju])
                                            + self.ghuu @ np.multiply.outer(u, u).ravel()
                                            + 2 * self.ghxu @ np.multiply.outer(x1, u).ravel() + self.ghs2)
            y1 = self.ghx @ x1 + self.ghu @ u
            o_pre = ob.P @ o
            a, Ycv, A = self.inputs_from_state(y1, y2, o_pre, prev_total, u)
            try:
                if self.cubature and self.R.shape[1] > 0:
                    w, z, info = self.news_wedges(a, Ycv, A, selection)
                else:
                    z, w, info = self.solve_inputs(a, Ycv, A, selection=selection)
            except OBCError as err:
                err.period = t + 1
                raise
            if self.trace is not None:
                self.trace.append(dict(y1=y1.copy(), y2=y2.copy(), o_pre=o_pre.copy(), prev=prev_total.copy(), u=u.copy(), z=z, w=w.copy()))
            o = o_pre + rb._V0 @ w
            base = np.empty(n)
            base[self.ov] = y1 + y2
            without_b[t] = ys + base
            with_b[t] = ys + base + o
            prev_total = base + o
            wedges[t] = w
            regimes.append(None if z is None else z[:self.T].copy())
            used.append(info.get("used"))
            dropped[t], relevant[t] = self._dropped_last, self._relevant_last
        if dropped.any():
            warnings.warn(f"cubature_failure_tolerance={self.failure_tolerance:g}: {dropped.sum()} of {relevant.sum()} relevant perturbed "
                          f"points ({100 * dropped.sum() / relevant.sum():.1f}%) were dropped in {int((dropped > 0).sum())} of {S} periods; "
                          f"the remaining points are reweighted by 1/(points - dropped)", UserWarning, stacklevel=2)
        return dict(with_bound=with_b, without_bound=without_b, wedges=wedges, dropped=dropped, relevant=relevant, regimes=regimes, solver=used)

    def _init_cubature(self):
        raise NotImplementedError("cubature for the regime formulation: step 3")
