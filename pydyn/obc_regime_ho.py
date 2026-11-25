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

from .obc import OBCError, RegimeOBC, _functional_rows, _solve_checked, check_selection, constraint_objects
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
    def _gap_functionals(self):
        """(nc, 3n + nx + 1): the gap-like functional of each constraint, used to build the cubature root and the region split: ``s_g * alt``
        of ``NewsOBC`` for a complementary pair (so the root of an LCP-class model is the LCP's), else the sign-normalised alternative
        residual of the primary switched equation (the tied one when ``_find_tie`` applies, else the first)."""
        rb = self.rb
        cos = constraint_objects(self.ob)
        rows = []
        for j, co in enumerate(cos):
            if co.cls == "complementarity":
                rows.append(co.s_g * co.alt)
                continue
            blocks = [b for b, (jj, _r) in enumerate(rb.cols) if jj == j]
            pb = blocks[0]
            if rb.n_c == 1 and rb._tie is not None:
                pb = rb._tie["primary"]
            r = np.array(rb.alt_rows[pb], float)
            rows.append(r if r[-1] >= 0 else -r)
        return np.array(rows)

    def _functionals(self):
        """(aC, aB, aA) of every functional of the expected path that the search needs, declaration order: the alternative residual of
        each switched equation (rows 0..nb-1), the condition variables (nb..nb+k-1; aB only) and the gap-like functional of each
        constraint (last n_c rows; used for the cubature root and regions only)."""
        rb, n = self.rb, self.n
        a = np.vstack([np.array(rb.alt_rows), self._gap_functionals()])
        k = len(rb._cv)
        E = np.zeros((k, n))
        E[np.arange(k), rb._cv] = 1.0
        Z = np.zeros((k, n))
        nb = self.nb
        aC, aB, aA = a[:, :n], a[:, n:2 * n], a[:, 2 * n:3 * n]
        self._gapfun = a[nb:]
        return (np.vstack([aC[:nb], Z, aC[nb:]]), np.vstack([aB[:nb], E, aB[nb:]]), np.vstack([aA[:nb], Z, aA[nb:]]))

    def _build_maps(self):
        rb, N = self.rb, self.N
        aC, aB, aA = self._functionals()
        self._fC, self._fB, self._fA = aC, aB, aA
        self._Gall, self._c0all = self._affine_maps(aC, aB, aA, N)
        a = np.array(rb.alt_rows)
        self._order = self._news_order()
        self._aCb, self._aJb, self._a0b = a[:, :self.n], a[:, 3 * self.n:3 * self.n + self.ob.n_exo], a[:, -1]
        g = self._gapfun
        self._aCg, self._aJg, self._a0g = g[:, :self.n], g[:, 3 * self.n:3 * self.n + self.ob.n_exo], g[:, -1]
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
        kk = len(rb._cv)
        Ycv = v[nb * N:(nb + kk) * N].reshape(kk, N).T
        self._last = (v, prev_total, shock)
        return A[:, :T].ravel(), Ycv, A

    def gap_path(self, order_rows=True):
        """The gap-like path (n_c * N), in ``NewsOBC``'s row order (complementarity rows first), of the last ``inputs_from_state``."""
        v, prev_total, shock = self._last
        N, nb, kk = self.N, self.nb, len(self.rb._cv)
        q = v[(nb + kk) * N:].reshape(self.n_c, N).copy()
        q[:, 0] += self._aCg @ prev_total + self._aJg @ shock
        q += self._a0g[:, None]
        q = q.ravel()
        return q[self._order] if order_rows else q

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

    # ------------------------------------------------------------------------------------------------ cubature
    def _news_order(self):
        """Row order of ``NewsOBC``'s gap path: the complementarity rows (first T periods of every constraint), then the check-ahead rows."""
        T, N, nc = self.T, self.N, self.n_c
        rows_s = np.concatenate([np.arange(c * N, c * N + T) for c in range(nc)])
        tail = np.setdiff1d(np.arange(nc * N), rows_s)
        return np.r_[rows_s, tail]

    def _init_cubature(self):
        self.R, self._lam = self._conditional_root()
        self.xi = self._points(self.R.shape[1])
        self._ro_cache = None

    def _conditional_root(self):
        """``R`` with ``R R'`` = covariance of the first-order effect, on the gap-like path of each constraint, of the shocks of the next ``pou``
        periods (cosine taper, eigenvalue pruning: the code of ``HigherOrderNews.conditional_root``, on the same rows in the same order), and
        the effects ``Lam_o`` of the same shocks on every other acceptance quantity (alternative residuals over N periods, condition-variable
        paths), kept for ``_other_roots``.  Returns (R, (Lam_g, Lam_o))."""
        ob, rb, n, nx, N, nc, nb = self.ob, self.rb, self.n, self.ob.n_exo, self.N, self.n_c, self.nb
        P, Q = ob.P, ob.Q
        g = self._gapfun
        Rg = _functional_rows(P, g[:, :n], g[:, n:2 * n], g[:, 2 * n:3 * n], N)
        Qg = (Rg[:N] @ Q).transpose(1, 0, 2)
        Qg[:, 0, :] += g[:, 3 * n:3 * n + nx]
        Gq_all = Qg.reshape(nc * N, nx)                                   # constraint-major rows c*N + h
        a = np.array(rb.alt_rows)
        Ra = _functional_rows(P, a[:, :n], a[:, n:2 * n], a[:, 2 * n:3 * n], N)
        Qa = (Ra[:N] @ Q).transpose(1, 0, 2)
        Qa[:, 0, :] += a[:, 3 * n:3 * n + nx]
        Sa = Qa.reshape(nb * N, nx)                                       # rows b*N + t
        kk = len(rb._cv)
        Ecv = np.zeros((kk, n))
        Ecv[np.arange(kk), rb._cv] = 1.0
        Z = np.zeros((kk, n))
        Rc = _functional_rows(P, Z, Ecv, Z, N)
        Sc = (Rc[:N] @ Q).reshape(N * kk, nx)                             # rows t*k + j
        w, V = np.linalg.eigh(self.Sigma)
        keep = w > 1e-14 * max(w.max(), 1e-300)
        L = V[:, keep] * np.sqrt(w[keep])
        bg, bo = [], []
        for i in range(1, self.pou + 1):
            wi = 0.5 * (1 + np.cos(np.pi * (i - 1) / self.pou)) if self.taper else 1.0
            B = np.zeros((nc * N, L.shape[1]))
            Ba = np.zeros((nb * N, L.shape[1]))
            Bc = np.zeros((N * kk, L.shape[1]))
            if i < N:
                for c in range(nc):
                    B[c * N + i:(c + 1) * N] = np.sqrt(wi) * (Gq_all[c * N:c * N + N - i] @ L)
                for b in range(nb):
                    Ba[b * N + i:(b + 1) * N] = np.sqrt(wi) * (Sa[b * N:b * N + N - i] @ L)
                Bc[i * kk:] = np.sqrt(wi) * (Sc[:(N - i) * kk] @ L)
            bg.append(B[self._order])
            bo.append(np.vstack([Ba, Bc]))
        Lg = np.hstack(bg) if bg else np.zeros((nc * N, 0))
        Lo = np.hstack(bo) if bo else np.zeros((nb * N + N * kk, 0))
        R = Lg
        if R.shape[1] > 0 and self.pruning_cutoff is not None:
            ev, U = np.linalg.eigh(R @ R.T)
            ev[ev < self.pruning_cutoff * ev.max()] = 0.0
            ev[:max(0, len(ev) - self.max_cubature_dimension)] = 0.0
            kp = np.sqrt(np.maximum(ev, 0.0)) > np.sqrt(np.finfo(float).eps)
            R = U[:, kp] * np.sqrt(ev[kp])
        return R, (Lg, Lo)

    def _other_roots(self):
        """Perturbation roots of the other acceptance quantities along the retained directions of ``R``:  Ro = Cov(other, gap) pinv(R')
        (for an LCP-class model the alternative residuals are +-gap, so this is +-R).  Recomputed when ``R`` is replaced.  Rows: the
        alternative residuals of every block over N periods (b*N + t), then the condition-variable paths (t*k + j)."""
        c = getattr(self, "_ro_cache", None)
        if c is not None and c[0] is self.R:
            return c[1]
        Lg, Lo = self._lam
        Ro = (Lo @ (Lg.T @ np.linalg.pinv(self.R.T))) if self.R.shape[1] else np.zeros((Lo.shape[0], 0))
        self._ro_cache = (self.R, Ro)
        return Ro

    def _viol_many(self, Z, pcv):
        """OccBin's test for many paths at once: Z (P, N, n_c) regime guesses, pcv (P, N, k) condition-variable paths -> (bind & ~z, relax & z)."""
        ob, rb = self.ob, self.rb
        P, N = Z.shape[0], self.N
        vb = np.empty((P, N, self.n_c), bool)
        vr = np.empty((P, N, self.n_c), bool)
        step = max(1, 200000 // N)
        for s0 in range(0, P, step):
            sl = slice(s0, min(P, s0 + step))
            m = sl.stop - sl.start
            zl = np.tile(ob.ys, (m * N, 1))
            zl[:, rb._cv] = pcv[sl].reshape(m * N, -1) + ob.ys[rb._cv]
            for j, ev in enumerate(ob._cons):
                ib, _eb, ir, _er = ev(zl, ob.params, ob.ys)
                b = np.broadcast_to(np.asarray(ib, bool), (m * N,)).reshape(m, N)
                r = (~b) if ir is None else np.broadcast_to(np.asarray(ir, bool), (m * N,)).reshape(m, N)
                vb[sl, :, j] = b & ~Z[sl, :, j]
                vr[sl, :, j] = r & Z[sl, :, j]
        return vb, vr

    def _guess_verify_many(self, pa, pY, maxit=60):
        """OccBin's guess-and-verify iteration (``RegimeOBC._iterate``) on P perturbed points at once, from the reference regime:
        pa (P, nb*T), pY (P, N, k).  Points are grouped by their guess so that each distinct regime sequence needs one linear solve with
        many right-hand sides.  Returns (W (P, nb*T), solved (P,) bool); an unsolved point is one the iteration could not settle (it cycles,
        wants a regime change beyond the horizon, or has a singular block): the caller runs the full search on it."""
        rb, T, N, nc, k = self.rb, self.T, self.N, self.n_c, len(self.rb._cv)
        P = pa.shape[0]
        Z = np.zeros((P, N, nc), bool)
        W = np.zeros((P, self.nws))
        state = np.zeros(P, np.int8)                       # 0 active, 1 solved, 2 not settled
        seen = [set() for _ in range(P)]
        if "_rscale" not in rb.__dict__:
            rb._rscale = float(np.abs(rb.R).max())
        for it in range(maxit):
            act = np.flatnonzero(state == 0)
            if not act.size:
                break
            bey = Z[act][:, T:].any(axis=(1, 2))
            state[act[bey]] = 2
            act = act[~bey]
            if not act.size:
                break
            Wa = np.zeros((act.size, self.nws))
            keys = [Z[i, :T].tobytes() for i in act]
            groups = {}
            for pos, key in enumerate(keys):
                groups.setdefault(key, []).append(pos)
            bad = np.zeros(act.size, bool)
            for key, poss in groups.items():
                poss = np.array(poss)
                S = rb._index(Z[act[poss[0]]])
                if S.size:
                    x = _solve_checked(rb.R[S[:, None], S], -pa[act[poss]][:, S].T, rb._rscale)
                    if x is None:
                        bad[poss] = True
                        continue
                    Wa[np.ix_(poss, S)] = x.T
            state[act[bad]] = 2
            pcv = pY[act] + (Wa @ rb._Wcv.T).reshape(act.size, N, k)
            vb, vr = self._viol_many(Z[act], pcv)
            okp = ~(vb.any(axis=(1, 2)) | vr.any(axis=(1, 2))) & ~bad
            W[act[okp]] = Wa[okp]
            state[act[okp]] = 1
            for pos in np.flatnonzero(~okp & ~bad):
                i = act[pos]
                key = Z[i].tobytes()
                if key in seen[i]:
                    state[i] = 2
                else:
                    seen[i].add(key)
                    Z[i] = (Z[i] & ~vr[pos]) | vb[pos]
        state[state == 0] = 2
        return W, state == 1

    def _relevant(self, pY):
        """Points whose reference-regime path (no wedges anywhere) fails OccBin's bind test: the paths that need a regime change."""
        P, N = pY.shape[0], self.N
        vb, _vr = self._viol_many(np.zeros((P, N, self.n_c), bool), pY)
        return vb.any(axis=(1, 2))

    @staticmethod
    def _cls_of(err):
        return "b" if err.status == "horizon_too_short" else ("c" if err.status == "not_converged" else "a")

    def news_wedges(self, a, Ycv, A, selection="occbin"):
        """Wedges for one period with cubature: the average over the perturbed acceptance paths ``(a, Ycv) + Ro xi_j`` (``xi`` the quasi-Monte-Carlo
        points), a point being *relevant* when its reference-regime path fails OccBin's bind test; non-relevant points contribute zero.  P points,
        F failed points (class a/b/c as in ``HigherOrderNews``): (sum over solved relevant units of size * w) / (P - F); ``CubatureFailure`` unless
        F <= ``cubature_failure_tolerance`` * (number of relevant points).  Returns (w, z, info) with z None."""
        self._dropped_last = self._relevant_last = 0
        rb, nb, N, T, k = self.rb, self.nb, self.N, self.T, len(self.rb._cv)
        npts = self.xi.shape[1]
        Ro = self._other_roots()
        RA, RY = Ro[:nb * N], Ro[nb * N:]
        A0 = A.reshape(-1)
        Y0 = Ycv.reshape(-1)
        pY_all = (Y0[:, None] + RY @ self.xi).T.reshape(npts, N, k)          # perturbed condition paths of every point
        rel = np.flatnonzero(self._relevant(pY_all))
        self.stats["cubature_points"] += npts
        self.stats["relevant_points"] += rel.size
        self._relevant_last = rel.size
        if rel.size == 0:
            return np.zeros(self.nws), None, dict(used="none")
        nrel, tol = rel.size, self.failure_tolerance
        counts = dict(a=0, b=0, c=0)
        nfail = 0
        limit_fail = tol * nrel
        acc = np.zeros(self.nws)
        unexamined = 0
        old_limit = rb.milp_time_limit
        rb.milp_time_limit = self.milp_time_limit

        def full_search(Ap, Yp):                                              # one unit through the complete search (tied LCP, integer programme)
            ap = Ap[:, :T].ravel()
            try:
                z, w, _info = self.solve_inputs(ap, Yp, Ap, selection=selection)
                return w, None
            except OBCError as err:
                return None, self._cls_of(err)

        def failed(cls, size):
            nonlocal nfail
            counts[cls] += size
            nfail += size
            return nfail > limit_fail

        try:
            if self.regions == "all":
                xr = self.xi[:, rel]
                pA = (A0[:, None] + RA @ xr).T.reshape(nrel, nb, N)
                pa = pA[:, :, :T].reshape(nrel, nb * T)
                W, ok = self._guess_verify_many(pa, pY_all[rel])
                acc = W[ok].sum(axis=0)
                self.stats["batch_fail"] = self.stats.get("batch_fail", 0) + int((~ok).sum())
                bad = np.flatnonzero(~ok)
                for i, j in enumerate(bad):
                    w, cls = full_search(pA[j], pY_all[rel][j])
                    if cls is None:
                        acc += w
                    elif failed(cls, 1):
                        unexamined = bad.size - i - 1
                        break
            else:
                xr = self.xi[:, rel]
                if self.regions == 1:
                    groups = [np.arange(nrel)]
                else:
                    kreg = int(self.regions)
                    if nrel <= kreg:
                        groups = [np.array([j]) for j in range(nrel)]
                    else:
                        q = self.gap_path()
                        Rg = self.R
                        groups = self._split_regions(q[:, None] + Rg @ xr, kreg)
                for i, g in enumerate(groups):
                    xm = xr[:, g].mean(axis=1)
                    Ap = (A0 + RA @ xm).reshape(nb, N)
                    Yp = (Y0 + RY @ xm).reshape(N, k)
                    if not self._relevant(Yp[None])[0]:                       # the region's mean point needs no wedge
                        continue
                    w, cls = full_search(Ap, Yp)
                    if cls is None:
                        acc += g.size * w
                    elif failed(cls, g.size):
                        unexamined = len(groups) - i - 1
                        break
        finally:
            rb.milp_time_limit = old_limit
        if nfail > limit_fail:
            self.stats["no_solution"] = self.stats.get("no_solution", 0) + 1
            status = "no_solution" if counts["a"] else ("horizon_too_short" if counts["b"] else "not_converged")
            raise CubatureFailure(status, nfail, unexamined > 0, nrel, npts, counts, None, tol, self.milp_time_limit)
        if nfail:
            self._dropped_last = nfail
            self.stats["dropped_points"] = self.stats.get("dropped_points", 0) + nfail
            self.stats["dropped_periods"] = self.stats.get("dropped_periods", 0) + 1
        return acc / (npts - nfail), None, dict(used="cubature")
