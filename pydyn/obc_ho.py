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

from .obc import NewsOBC, OBCError, solve_active_set, solve_active_set_batch, solve_news
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
            ns_ = len(self.ist)                                         # x x' is symmetric: only its ns(ns+1)/2 distinct entries
            self._iu, self._ju = np.triu_indices(ns_)
            self.ghxx_s = self._fold(self.ghxx.reshape(self.n, ns_, ns_))
        self.SigS = self.ghus @ self.Sigma @ self.ghus.T
        n, nx, nc = self.n, ob.n_exo, nb.n_c
        self.n_c, self.nws = nc, nc * nb.T                          # number of constraints; length of the wedge vector
        a = np.array([co.s_g * co.alt for co in nb.constraints])    # (nc, 3n + nx + 1), one gap functional per constraint
        self.aC, self.aB, self.aA, self.aJ, self.a0 = a[:, :n], a[:, n:2 * n], a[:, 2 * n:3 * n], a[:, 3 * n:3 * n + nx], a[:, -1]
        self._row0 = np.argsort(nb._order)[np.arange(nc) * self.N]  # position of each constraint's first gap row in q
        self._build_gap_maps()
        self.cubature = bool(cubature)
        self.pruning_cutoff, self.max_cubature_dimension = pruning_cutoff, int(max_cubature_dimension)
        self.pou, self.qmc_level, self.regions, self.taper, self.seed = (int(periods_of_uncertainty), int(qmc_level),
                                                                         regions, bool(taper), seed)
        if self.cubature:
            self.R = self.conditional_root()
            self.xi = self._points(self.R.shape[1])
        self._fac_cache = {}
        self.stats = dict(lcp=0, cubature_points=0, relevant_points=0)

    # ------------------------------------------------------------------------------------ precomputed maps
    def _fold(self, A):
        """Coefficients on the distinct entries x_a x_b (a <= b) of a bilinear form sum_ab A[..., a, b] x_a x_b."""
        iu, ju = self._iu, self._ju
        F = A[..., iu, ju] + A[..., ju, iu]
        F[..., iu == ju] *= 0.5
        return F

    def _build_gap_maps(self):
        """The gap path q is affine in z = (y1 + y2, vec(x1 x1'), o_pre) (x1 = y1[ist]): the expected path is linear
        in them (the covariance recursion W_k = F^k W_0 F'^k + S_k with F = ghxs is linear in W_0 and its shock part
        S_k does not depend on the state; y1 and y2 enter the path through the same coefficients), and so is the gap.
        Computed once per parameter set:

            q = q0 + Gz @ z,  then  q[0] += aC.prev_total + aJ.u.

        Only the three gap coefficient vectors (aB, aA, aC) are propagated (adjoint form), never the full
        (N+1) n x n_s^2 path map: per k a handful of small matmuls."""
        N, n, ist, ov = self.N, self.n, self.ist, self.ov
        ns, two = len(ist), self.order == 2
        F, ghx = self.ghxs, self.ghx
        K = N + 1
        nc = self.n_c
        # rows 3c, 3c+1, 3c+2: (aB, aA, aC) of constraint c
        Ay = np.vstack([np.vstack([self.aB[c][ov], self.aA[c][ov], self.aC[c][ov]]) for c in range(nc)])   # DR order (y-parts)
        Ao = np.vstack([np.vstack([self.aB[c], self.aA[c], self.aC[c]]) for c in range(nc)])               # declaration order (o)
        m3 = 3 * nc
        rY = np.zeros((K, m3, n)); rY[0] = Ay
        rO = np.empty((K, m3, n)); rO[0] = Ao
        if two:
            nn = ns * ns
            rW = np.zeros((K, m3, nn)); rc = np.zeros((K, m3))
            # W-coefficients in adjoint form. With twist(X) = F' X F, G_i = ghxx[i] reshaped (ns x ns), Gbar = sum_i Ay_i G_i,
            # B_m = Ay ghx F^m and H(B) = sum_l B_l G_{ist_l}:
            #   rW_k = 0.5 (twist^(k-1)(Gbar) + D_k),  D_1 = 0,  D_(k+1) = H(B_(k-1)) + twist(D_k).
            # Per k only 3 n_c matrices of size ns x ns are carried (the direct recursion carries ns x ns^2 and n x ns x ns arrays).
            Gbar = (Ay @ self.ghxx).reshape(m3, ns, ns)
            Gs = self.ghxx[ist]                                       # (ns, ns^2): rows G_{ist_l}
            Bm = Ay @ ghx                                             # B_0, (m3, ns)
            D = np.zeros((m3, ns, ns))
            zc = np.zeros(ns)                                         # state rows of the constant of m2_k
            S = np.zeros((ns, ns))
        Vy = Ay @ ghx                                                 # (m3, ns): a ghx F^(k-1)
        P = self.ob.P
        for k in range(1, K):
            rY[k][:, ist] = Vy
            rO[k] = rO[k - 1] @ P
            if two:
                dc = 0.5 * (self.ghxx @ S.ravel()) + self.c2          # state-independent part of d_{k-1}
                rW[k] = 0.5 * (Gbar + D).reshape(m3, nn)
                rc[k] = Ay @ (ghx @ zc + dc)                          # a . a_k
                D = (Bm @ Gs).reshape(m3, ns, ns) + np.matmul(np.matmul(F.T, D), F)          # D_(k+1)
                Gbar = np.matmul(np.matmul(F.T, Gbar), F)
                Bm = Bm @ F
                zc = F @ zc + dc[ist]
                S = F @ S @ F.T + self.SigS
            Vy = Vy @ F
        order = self.news._order                                       # rows: complementarity rows first, as NewsOBC

        def comb(r):                                                  # q_h <- rB_h + rA_{h+1} + rC_{h-1}, per constraint
            out = []
            for c in range(nc):
                G = r[:N, 3 * c] + r[1:K, 3 * c + 1]
                G[1:] += r[:N - 1, 3 * c + 2]
                out.append(G)
            return np.vstack(out)[order]
        cols = [comb(rY)]
        q0 = np.repeat(self.a0, N).astype(float)
        if two:
            cols.append(comb(rW))
            q0 += comb(rc[:, :, None])[:, 0]
        q0 = q0[order]
        cols.append(comb(rO))
        self._q0 = q0
        if two:                                                       # fold the vec(x1 x1') block onto its distinct entries
            cols[1] = self._fold(cols[1].reshape(cols[1].shape[0], ns, ns))
        self._Gz = np.ascontiguousarray(np.hstack(cols))
        self._nz = self._Gz.shape[1]
        self._z = np.empty(self._nz)

    def gap_from_state(self, y1, y2, o_pre, prev_total, shock):
        """q from the precomputed maps; equals ``gap_path(expected_path(...), ...)`` up to rounding."""
        n, ist = self.n, self.ist
        z = self._z
        if self.order == 2:
            np.add(y1, y2, out=z[:n])
            x1 = y1[ist]
            ns = x1.size
            z[n:n + self._iu.size] = x1[self._iu] * x1[self._ju]
        else:
            z[:n] = y1
        z[-n:] = o_pre
        q = self._Gz @ z
        q += self._q0
        q[self._row0] += self.aC @ prev_total + self.aJ @ shock
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
        out = []
        for c in range(self.n_c):
            q = self.a0[c] + path[:N] @ self.aB[c] + path[1:N + 1] @ self.aA[c]
            q[1:] += path[:N - 1] @ self.aC[c]
            q[0] += self.aC[c] @ prev_total + self.aJ[c] @ shock
            out.append(q)
        return np.concatenate(out)[self.news._order]

    # ------------------------------------------------------------------------------------------- cubature
    def conditional_root(self):
        """R with R R' = covariance of the gap path caused by shocks arriving in the next ``pou`` periods,
        to first order, with DynareOBC's cosine taper w_i = (1 + cos(pi (i-1)/pou))/2 (shock i periods ahead)."""
        nb, N, nx = self.news, self.N, self.ob.n_exo
        nc = self.n_c
        Gq_all = nb.Qm[np.argsort(nb._order), self.n:]           # constraint-major rows (c*N + h): gap at horizon h after a shock now
        w, V = np.linalg.eigh(self.Sigma)
        L = V[:, w > 1e-14 * max(w.max(), 1e-300)] * np.sqrt(w[w > 1e-14 * max(w.max(), 1e-300)])
        blocks = []
        for i in range(1, self.pou + 1):
            wi = 0.5 * (1 + np.cos(np.pi * (i - 1) / self.pou)) if self.taper else 1.0
            B = np.zeros((nc * N, L.shape[1]))
            if i < N:
                for c in range(nc):
                    B[c * N + i:(c + 1) * N] = np.sqrt(wi) * (Gq_all[c * N:c * N + N - i] @ L)
            blocks.append(B[nb._order])
        R = np.hstack(blocks) if blocks else np.zeros((nc * N, 0))
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
            return np.zeros(self.nws)
        y = solve_active_set(nb.Mo, q, self.nws)
        if y is None:
            self.stats["milp"] = self.stats.get("milp", 0) + 1
            y, _st = solve_news(nb.Mo, q, self.nws, horizons=sorted({max(1, T // 4), max(1, T // 2), T}), n_c=self.n_c)
        if y is None:
            self.stats["no_solution"] = self.stats.get("no_solution", 0) + 1
            raise OBCError("no solution within the news horizon", "no_solution")
        return y

    def _solve_all(self, P):
        """Sum over the columns of P of the scalar solution ``_solve(P[:, j])``, computed in batch: columns that need no
        wedge are skipped as in ``_solve``; the rest go through ``solve_active_set_batch`` (the scalar iteration run in
        lockstep, factorisations shared by guessed binding set and cached across periods); columns on which that iteration
        fails take the scalar route (``_solve``, i.e. the MILP fallback)."""
        nb, T = self.news, self.T
        k = P.shape[1]
        self.stats["lcp"] += k
        need = ~np.all(P >= -1e-8 * np.maximum(1.0, np.abs(P).max(axis=0)), axis=0)
        if not need.any():
            return np.zeros(self.nws)
        Pn = P[:, need]
        Y, ok = solve_active_set_batch(nb.Mo, Pn, self.nws, scale=nb._mscale, cache=self._fac_cache, info=self.stats)
        acc = Y[:, ok].sum(axis=1)
        nfail = int((~ok).sum())
        self.stats["batch_fail"] = self.stats.get("batch_fail", 0) + nfail
        self.stats["lcp"] -= nfail                                 # _solve counts them again
        for j in np.nonzero(~ok)[0]:
            acc += self._solve(Pn[:, j])
        return acc

    def _cub_prep(self):
        """Quantities for the exact skipping rules, tied to the current (R, xi) objects."""
        if getattr(self, "_cc", None) is not None and self._cc[0] is self.R and self._cc[1] is self.xi:
            return self._cc
        xn = np.sqrt((self.xi * self.xi).sum(axis=0))
        order = np.argsort(xn, kind="stable")
        rown = np.sqrt((self.R * self.R).sum(axis=1))
        xa = np.vstack([self.xi, np.ones((1, self.xi.shape[1]))])       # augmented: q + R xi = [R q] @ [xi; 1]
        self._cc = (self.R, self.xi, xn[order], order, rown, float(xn.max()) if xn.size else 0.0, xa)
        return self._cc

    def _relevant_points(self, q):
        """Indices (increasing) of the points q + R xi_j that fail the relevance test of the original code,
        ``not all(p >= -1e-8 max(1, max|p|))``. Exact skipping: by Cauchy-Schwarz p_h >= q_h - |R_h| |xi_j|, so
        points with |xi_j| <= min_h q_h/|R_h| (1 - 1e-9) are non-relevant (the margin dwarfs rounding), and
        min_h p_h >= -1e-8 implies non-relevant whatever max|p| is."""
        _, _, xn_sorted, order, rown, rmax, xa = self._cub_prep()
        Ra = np.empty((self.R.shape[0], self.R.shape[1] + 1))
        Ra[:, :-1] = self.R
        Ra[:, -1] = q
        npts = self.xi.shape[1]
        cnt = 0
        if q.min() >= 0.0:
            pos = rown > 0
            rho = (q[pos] / rown[pos]).min() / (1.0 + 1e-9) if pos.any() else np.inf
            if rho >= rmax:
                return np.empty(0, dtype=np.intp)
            cnt = int(np.searchsorted(xn_sorted, rho, side="right"))
        if cnt == 0:
            pts = Ra @ xa
            mn = pts.min(axis=0)
            cand = np.nonzero(mn < -1e-8)[0]
            sub, mnc = pts[:, cand], mn[cand]
            rel = mnc < -1e-8 * np.maximum(1.0, np.maximum(sub.max(axis=0), -mnc))      # max|p| = max(max p, -min p)
            return cand[rel]
        else:
            cand = order[cnt:]
            sub = Ra @ xa[:, cand]
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
            return np.zeros(self.nws)
        if self.regions == 1:                                      # DynareOBC's default: one region, its mean point
            # mean of q + R xi_j over the relevant j  =  q + R mean(xi_j): an (r)-vector instead of an N x k array
            return (idx.size / npts) * self._solve(q + self.R @ self.xi[:, idx].mean(axis=1))
        pr = q[:, None] + self.R @ self.xi[:, idx]                 # perturbed paths of the relevant points only
        if self.regions == "all":                                  # plain quasi-Monte Carlo average
            return self._solve_all(pr) / npts
        k = int(self.regions)                                      # DynareOBC's CubatureRegions > 1
        if idx.size <= k:
            groups = [np.array([j]) for j in range(idx.size)]
        else:
            groups = self._split_regions(pr, k)
        xr = self.xi[:, idx]
        acc = np.zeros(self.nws)
        for g in groups:                                           # region mean point = q + R (mean of its xi)
            acc += (g.size / npts) * self._solve(q + self.R @ xr[:, g].mean(axis=1))
        return acc

    def _split_regions(self, pr, k):
        """DynareOBC's region rule (PerformCubature.m, CubatureClusteringEffort = 0): describe each relevant path
        by its direction and the direction of its negative part, reduce by principal components, then split the
        region with the largest within-sum-of-squares along its first principal axis at Otsu's threshold,
        until there are k regions. Returns the list of k index arrays (increasing) into the columns of ``pr``.

        Principal components from the eigendecomposition of the small Gram matrix X'X (2N x 2N) instead of the SVD of
        the tall X (m x 2N); region sums of squares kept per region and updated only for the two regions a split
        changes. Component signs are arbitrary in both versions and the split is symmetric under a sign change
        (the centred projection's Otsu criterion is symmetric in j <-> m - j)."""
        X = pr / np.sqrt((pr * pr).sum(axis=0))
        X = np.concatenate([X.T, np.minimum(0.0, X.T)], axis=1)
        X = X - X.mean(axis=0)
        ev, V = np.linalg.eigh(X.T @ X)
        ev, V = ev[::-1], V[:, ::-1]
        lat = np.maximum(ev, 0.0) / max(1, X.shape[0] - 1)
        keep = lat >= self.pruning_cutoff * lat[0] if self.pruning_cutoff is not None else np.ones(lat.size, bool)
        keep[self.max_cubature_dimension:] = False
        keep &= lat > np.finfo(float).eps
        Z = X @ V[:, keep]                                         # principal-component scores
        d = Z.shape[1]

        def sse(ix):
            Zi = Z[ix]
            return float(((Zi - Zi.mean(axis=0)) ** 2).sum())

        groups = [np.arange(Z.shape[0])]
        sses = [sse(groups[0])]
        for n in range(1, k):
            r = 0 if n == 1 else int(np.argmax(sses))
            pi_ = groups[r]
            if n == 1:
                proj = Z[:, 0]
            else:
                Zp = Z[pi_]
                Zc = Zp - Zp.mean(axis=0)
                C = (Zc.T @ Zc) / max(1, pi_.size - 1)
                ev2, V2 = np.linalg.eigh(C)
                proj = Zp @ V2[:, int(np.argmax(np.abs(ev2)))]
            proj = proj - proj.mean()
            srt = np.sort(proj)
            cs = np.cumsum(srt[:-1])
            j = np.arange(1, proj.size)
            cut = srt[int(np.argmax(cs * cs / (j * (proj.size - j))))]
            right = proj > cut
            groups[r], new = pi_[~right], pi_[right]
            groups.append(new)
            sses[r] = sse(groups[r])
            sses.append(sse(new) if new.size else 0.0)
        return groups

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
        with_b, without_b, wedges = np.empty((S, n)), np.empty((S, n)), np.zeros((S, self.nws))
        for t in range(S):
            u = shocks[t]
            x1, x2 = y1[self.ist], y2[self.ist]
            if self.order == 2:
                y2 = self.ghx @ x2 + 0.5 * (self.ghxx_s @ (x1[self._iu] * x1[self._ju])
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
