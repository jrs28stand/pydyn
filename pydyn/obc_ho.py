# This file is part of pydyn. Released under the GNU General Public License, version 3 or later; see LICENSE.
# No warranty.
#
# The decomposition into a pruned base simulation plus a first-order bound offset, the expected ("return") path
# from the pruned system and the integration over future shocks follow T. D. Holden, "Computation of solutions to
# dynamic models with occasionally binding constraints", and the structure of DynareOBC's SimulateModel.m,
# ExpectedReturn.m, PerformCubature.m and CacheConditionalCovariancesAndAugmentedStateTransitionMatrices.m
# (GPL-3.0-or-later). Written for pydyn's data structures; not a translation.
"""News-shock OBC solution at order 2 and with integration over future shocks (prototype / executable spec).

State carried from period to period (deviations from the deterministic steady state):

* ``y1``: first-order component of the pruned simulation that ignores the bound;
* ``y2``: second-order component (zero at order 1), including the risk correction ghs2/2;
* ``o``:  the bound offset, the accumulated first-order effect of the news shocks.

The variables are ``ys + y1 + y2 + o``.  Each period:

1. base step: y1, y2 from the pruned decision rules and this period's shock; o_pre = P o (first-order propagation);
2. expected path E_t[y_{t+k}], k = 0..N, with future shocks at their distribution (not at zero): linear in
   (y1 states, y2 states, E[y1s y1s']), see ``expected_path``;
3. q = gap along that path; solve the same LCP as at order 1 (same M) for the news shocks y;
   with ``cubature`` the solution is averaged over perturbed paths q + R xi (``conditional_root``);
4. o = o_pre + V0 y.

One complementary pair per constraint (the class NewsOBC handles), one constraint for cubature.
"""
from __future__ import annotations

import warnings

import numpy as np

from .obc import NewsOBC, OBCError, solve_active_set, solve_news
from .occbin import Occbin

__all__ = ["HigherOrderNews"]

_SOBOL_CACHE: dict = {}                                          # (dim, level) -> read-only (dim, 2^(level+1)-1) normals


class HigherOrderNews:
    def __init__(self, model, order=2, horizon=40, check_ahead=40, params=None, Sigma_e=None,
                 cubature=False, periods_of_uncertainty=16, qmc_level=10, regions=1, taper=True, seed=0, pruning_cutoff=0.01,
                 max_cubature_dimension=128):
        from .korder import k_order
        self.model = model
        self.order = int(order)
        if self.order not in (1, 2):
            raise NotImplementedError("order 3 is not implemented in this prototype")
        self.ob = ob = Occbin(model, params)
        self.news = nb = NewsOBC(ob, horizon=horizon, check_ahead=check_ahead)
        self.T, self.N, self.n = nb.T, nb.N, ob.n
        self.Sigma = np.array(model.Sigma_e if Sigma_e is None else Sigma_e, float)
        dr = k_order(model, order=max(self.order, 1), ys=ob.ys, params=ob.params, pruning=True, Sigma_e=self.Sigma)
        self.dr = dr
        self.ov, self.sv = np.asarray(dr["order_var"]), np.asarray(dr["state_var"])
        pos = {v: i for i, v in enumerate(self.ov)}
        self.ist = np.array([pos[v] for v in self.sv])          # rows of the states in DR order
        self.ghx, self.ghu = dr["ghx"], dr["ghu"]
        self.ghxs, self.ghus = self.ghx[self.ist], self.ghu[self.ist]
        if self.order == 2:
            self.ghxx, self.ghxu, self.ghuu = dr["ghxx"], dr["ghxu"], dr["ghuu"]
            self.ghs2 = dr["ghs2"].ravel()
            self.c2 = 0.5 * (self.ghuu @ self.Sigma.ravel() + self.ghs2)        # E of the shock-quadratic terms
        self.SigS = self.ghus @ self.Sigma @ self.ghus.T
        co = nb.constraints[0]
        a = co.s_g * co.alt
        n, nx = self.n, ob.n_exo
        self.aC, self.aB, self.aA, self.aJ, self.a0 = a[:n], a[n:2 * n], a[2 * n:3 * n], a[3 * n:3 * n + nx], a[-1]
        self._build_gap_maps()
        self.cubature = bool(cubature)
        self.pruning_cutoff, self.max_cubature_dimension = pruning_cutoff, int(max_cubature_dimension)
        self.pou, self.qmc_level, self.regions, self.taper, self.seed = (int(periods_of_uncertainty), int(qmc_level),
                                                                         regions, bool(taper), seed)
        if self.cubature:
            if nb.n_c != 1:
                raise NotImplementedError("cubature with more than one constraint")
            self.R = self.conditional_root()
            self.xi = self._points(self.R.shape[1])
        self.stats = dict(lcp=0, cubature_points=0, relevant_points=0)

    # ------------------------------------------------------------------------------------ precomputed maps
    def _build_gap_maps(self):
        """The gap path q is affine in z = (y1, y2, vec(x1 x1'), o_pre) (x1 = y1[ist]): the expected path is linear
        in them (the covariance recursion W_k = F^k W_0 F'^k + S_k with F = ghxs is linear in W_0 and its shock part
        S_k does not depend on the state), and so is the gap. Computed once per parameter set:

            q = q0 + Gz @ z,  then  q[0] += aC.prev_total + aJ.u.

        Rows of ``expected_path``/``gap_path`` are contracted with the gap coefficients on the fly, so no
        (N+1) n x n_s^2 path map is stored."""
        N, n, ist, ov = self.N, self.n, self.ist, self.ov
        ns, two = len(ist), self.order == 2
        F, ghx = self.ghxs, self.ghx
        aB, aA, aC = self.aB[ov], self.aA[ov], self.aC[ov]            # DR order, for the y-parts
        oB, oA, oC = self.aB, self.aA, self.aC                        # declaration order, for the o-part
        K = N + 1
        rB = {}; rA = {}; rC = {}
        names = ["y1"] + (["y2", "W"] if two else []) + ["o"]
        for nm in names:
            rB[nm], rA[nm], rC[nm] = ({} for _ in range(3))
        cB = {nm: [] for nm in names}; cA = {nm: [] for nm in names}; cC = {nm: [] for nm in names}
        const = [[], [], []]
        A1 = np.eye(n)
        if two:
            A2, AW, a_k = np.eye(n), np.zeros((n, ns * ns)), np.zeros(n)
            Cj = self.ghxx.reshape(n, ns, ns)                         # <Cj_i, W0> = i-th row of ghxx @ vec(W_j)
            S = np.zeros((ns, ns))
        PO = np.eye(n)
        for k in range(K):
            if k > 0:
                if two:
                    sj = self.ghxx @ S.ravel()                         # state-independent shock part of W_{k-1}
                    dW = 0.5 * Cj.reshape(n, ns * ns)
                    dc = 0.5 * sj + self.c2
                    A2n = ghx @ A2[ist]
                    AW = ghx @ AW[ist] + dW
                    a_k = ghx @ a_k[ist] + dc
                    A2 = A2n
                    S = F @ S @ F.T + self.SigS
                    Cj = np.matmul(np.matmul(F.T, Cj), F)
                A1 = ghx @ A1[ist]
                PO = self.ob.P @ PO
            blocks = {"y1": (aB @ A1, aA @ A1, aC @ A1), "o": (oB @ PO, oA @ PO, oC @ PO)}
            if two:
                blocks["y2"] = (aB @ A2, aA @ A2, aC @ A2)
                blocks["W"] = (aB @ AW, aA @ AW, aC @ AW)
                const[0].append(aB @ a_k); const[1].append(aA @ a_k); const[2].append(aC @ a_k)
            for nm, (b, a, c) in blocks.items():
                cB[nm].append(b); cA[nm].append(a); cC[nm].append(c)
        cols = []
        for nm in names:
            B, A, C = (np.array(cB[nm]), np.array(cA[nm]), np.array(cC[nm]))
            G = B[:N] + A[1:K]
            G[1:] += C[:N - 1]
            cols.append(G)
        q0 = np.full(N, float(self.a0))
        if two:
            B, A, C = (np.array(c) for c in const)
            g = B[:N] + A[1:K]
            g[1:] += C[:N - 1]
            q0 += g
        self._q0 = q0
        self._Gz = np.ascontiguousarray(np.hstack(cols))
        self._nz = self._Gz.shape[1]
        self._z = np.empty(self._nz)

    def gap_from_state(self, y1, y2, o_pre, prev_total, shock):
        """q from the precomputed maps; equals ``gap_path(expected_path(...), ...)`` up to rounding."""
        n, ist = self.n, self.ist
        z = self._z
        z[:n] = y1
        if self.order == 2:
            z[n:2 * n] = y2
            x1 = y1[ist]
            ns = x1.size
            z[2 * n:2 * n + ns * ns] = np.multiply.outer(x1, x1).ravel()
        z[-n:] = o_pre
        q = self._Gz @ z
        q += self._q0
        q[0] += self.aC @ prev_total + self.aJ @ shock
        return q

    # --------------------------------------------------------------------------------------- expected path
    def expected_path(self, y1, y2, o_pre):
        """E_t of the total deviation (declaration order), periods 0..N. y1, y2 in DR order (this period's
        realised components), o_pre in declaration order."""
        N, n = self.N, self.n
        out = np.zeros((N + 1, n))
        m1, m2 = y1.copy(), y2.copy()
        W = np.outer(y1[self.ist], y1[self.ist])
        o = o_pre.copy()
        P = self.ob.P
        for k in range(N + 1):
            if k > 0:
                if self.order == 2:
                    m2 = self.ghx @ m2[self.ist] + 0.5 * (self.ghxx @ W.ravel()) + self.c2
                m1s = m1[self.ist]
                m1 = self.ghx @ m1s
                W = self.ghxs @ W @ self.ghxs.T + self.SigS
                o = P @ o
            tot = np.empty(n)
            tot[self.ov] = m1 + m2
            out[k] = tot + o
        return out

    def gap_path(self, path, prev_total, shock):
        """q: the gap along an expected path (N periods)."""
        N = self.N
        q = self.a0 + path[:N] @ self.aB + path[1:N + 1] @ self.aA
        q[1:] += path[:N - 1] @ self.aC
        q[0] += self.aC @ prev_total + self.aJ @ shock
        return q

    # ------------------------------------------------------------------------------------------- cubature
    def conditional_root(self):
        """R with R R' = covariance of the gap path caused by shocks arriving in the next ``pou`` periods,
        to first order, with DynareOBC's cosine taper w_i = (1 + cos(pi (i-1)/pou))/2 (shock i periods ahead)."""
        nb, N, nx = self.news, self.N, self.ob.n_exo
        Gq = nb.Qm[:, self.n:]                                   # gap at horizon h after a shock at horizon 0
        w, V = np.linalg.eigh(self.Sigma)
        L = V[:, w > 1e-14 * max(w.max(), 1e-300)] * np.sqrt(w[w > 1e-14 * max(w.max(), 1e-300)])
        blocks = []
        for i in range(1, self.pou + 1):
            wi = 0.5 * (1 + np.cos(np.pi * (i - 1) / self.pou)) if self.taper else 1.0
            B = np.zeros((N, L.shape[1]))
            if i < N:
                B[i:] = np.sqrt(wi) * (Gq[:N - i] @ L)
            blocks.append(B)
        R = np.hstack(blocks) if blocks else np.zeros((N, 0))
        if R.shape[1] == 0 or self.pruning_cutoff is None:
            return R
        # DynareOBC's ObtainRootConditionalCovariance: drop directions with variance below the cut-off share
        # of the largest, keep at most max_cubature_dimension of them.
        ev, U = np.linalg.eigh(R @ R.T)
        ev[ev < self.pruning_cutoff * ev.max()] = 0.0
        ev[:max(0, len(ev) - self.max_cubature_dimension)] = 0.0
        k = np.sqrt(np.maximum(ev, 0.0)) > np.sqrt(np.finfo(float).eps)
        return U[:, k] * np.sqrt(ev[k])

    def _points(self, dim):
        """DynareOBC's rule: the first 2^(level+1) - 1 Sobol points after the origin, unscrambled, mapped to
        normals. The set is symmetric, so its mean is exactly zero (with one region and every point relevant,
        cubature then equals no cubature, as in DynareOBC). Dynare's direction numbers differ from SciPy's
        beyond the second dimension, so individual points are not DynareOBC's."""
        key = (int(dim), int(self.qmc_level))
        xi = _SOBOL_CACHE.get(key)
        if xi is None:
            from scipy.stats import norm, qmc
            m = self.qmc_level + 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = qmc.Sobol(d=dim, scramble=False).random_base2(m)[1:]
            xi = np.ascontiguousarray(norm.ppf(u).T)             # (dim, 2^m - 1)
            xi.flags.writeable = False
            _SOBOL_CACHE[key] = xi
        return xi

    def _solve(self, q):
        nb, T = self.news, self.T
        self.stats["lcp"] += 1
        if np.all(q >= -1e-8 * max(1.0, np.abs(q).max())):
            return np.zeros(T)
        y = solve_active_set(nb.Mo, q, T)
        if y is None:
            y, _st = solve_news(nb.Mo, q, T, horizons=sorted({max(1, T // 4), max(1, T // 2), T}))
        if y is None:
            raise OBCError("no solution within the news horizon", "no_solution")
        return y

    def _cub_prep(self):
        """Quantities for the exact skipping rules, tied to the current (R, xi) objects."""
        if getattr(self, "_cc", None) is not None and self._cc[0] is self.R and self._cc[1] is self.xi:
            return self._cc
        xn = np.sqrt((self.xi * self.xi).sum(axis=0))
        order = np.argsort(xn, kind="stable")
        rown = np.sqrt((self.R * self.R).sum(axis=1))
        self._cc = (self.R, self.xi, xn[order], order, rown, float(xn.max()) if xn.size else 0.0)
        return self._cc

    def _relevant_points(self, q):
        """Indices (increasing) of the points q + R xi_j that fail the relevance test of the original code,
        ``not all(p >= -1e-8 max(1, max|p|))``. Exact skipping: by Cauchy-Schwarz p_h >= q_h - |R_h| |xi_j|, so
        points with |xi_j| <= min_h q_h/|R_h| (1 - 1e-9) are non-relevant (the margin dwarfs rounding), and
        min_h p_h >= -1e-8 implies non-relevant whatever max|p| is."""
        _, _, xn_sorted, order, rown, rmax = self._cub_prep()
        npts = self.xi.shape[1]
        cnt = 0
        if q.min() >= 0.0:
            pos = rown > 0
            rho = (q[pos] / rown[pos]).min() / (1.0 + 1e-9) if pos.any() else np.inf
            if rho >= rmax:
                return np.empty(0, dtype=np.intp)
            cnt = int(np.searchsorted(xn_sorted, rho, side="right"))
        if cnt == 0:
            pts = q[:, None] + self.R @ self.xi
            mn = pts.min(axis=0)
            cand = np.nonzero(mn < -1e-8)[0]
            sub, mnc = pts[:, cand], mn[cand]
            rel = mnc < -1e-8 * np.maximum(1.0, np.maximum(sub.max(axis=0), -mnc))      # max|p| = max(max p, -min p)
            return cand[rel]
        else:
            cand = order[cnt:]
            sub = q[:, None] + self.R @ self.xi[:, cand]
            keep = sub.min(axis=0) < -1e-8
            cand, sub = cand[keep], sub[:, keep]
        rel = sub.min(axis=0) < -1e-8 * np.maximum(1.0, np.abs(sub).max(axis=0))
        idx = cand[rel]
        return idx if cnt == 0 else np.sort(idx)

    def news_shocks(self, q):
        """Solution at q, or its average over the perturbed paths when cubature is on."""
        if not self.cubature or self.R.shape[1] == 0:
            return self._solve(q)
        npts = self.xi.shape[1]
        idx = self._relevant_points(q)
        self.stats["cubature_points"] += npts
        self.stats["relevant_points"] += idx.size
        if idx.size == 0:
            return np.zeros(self.T)
        if self.regions == 1:                                      # DynareOBC's default: one region, its mean point
            # mean of q + R xi_j over the relevant j  =  q + R mean(xi_j): an (r)-vector instead of an N x k array
            return (idx.size / npts) * self._solve(q + self.R @ self.xi[:, idx].mean(axis=1))
        pr = q[:, None] + self.R @ self.xi[:, idx]                 # perturbed paths of the relevant points only
        if self.regions == "all":                                  # plain quasi-Monte Carlo average
            acc = np.zeros(self.T)
            for j in range(idx.size):
                acc += self._solve(pr[:, j])
            return acc / npts
        raise NotImplementedError("regions must be 1 or 'all' in this prototype")

    # ----------------------------------------------------------------------------------------- simulation
    def simulate(self, shocks, init=None):
        """Simulate under unanticipated shocks. Returns a dict with levels (T, n) in declaration order:
        ``with_bound``, ``without_bound`` (the pruned simulation), and the news shocks per period."""
        ob, n = self.ob, self.n
        shocks = np.atleast_2d(np.asarray(shocks, float))
        S = shocks.shape[0]
        ys = ob.ys
        y1 = np.zeros(n)                                           # DR order
        y2 = np.zeros(n)
        o = np.zeros(n)                                            # declaration order
        if init is not None:
            y1[:] = (np.asarray(init, float) - ys)[self.ov]
        prev_total = np.empty(n)
        prev_total[self.ov] = y1 + y2
        prev_total += o
        with_b, without_b, wedges = np.empty((S, n)), np.empty((S, n)), np.zeros((S, self.T))
        for t in range(S):
            u = shocks[t]
            x1, x2 = y1[self.ist], y2[self.ist]
            if self.order == 2:
                y2 = self.ghx @ x2 + 0.5 * (self.ghxx @ np.multiply.outer(x1, x1).ravel()
                                            + self.ghuu @ np.multiply.outer(u, u).ravel()
                                            + 2 * self.ghxu @ np.multiply.outer(x1, u).ravel() + self.ghs2)
            y1 = self.ghx @ x1 + self.ghu @ u
            o_pre = ob.P @ o
            q = self.gap_from_state(y1, y2, o_pre, prev_total, u)
            try:
                y = self.news_shocks(q)
            except OBCError as err:
                err.period = t + 1
                raise
            o = o_pre + self.news._V0 @ y
            base = np.empty(n)
            base[self.ov] = y1 + y2
            without_b[t] = ys + base
            with_b[t] = ys + base + o
            prev_total = base + o
            wedges[t] = y
        return dict(with_bound=with_b, without_bound=without_b, wedges=wedges)
