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

import hashlib
import time
import warnings

import numpy as np
from scipy.optimize import linprog

from .obc import NewsOBC, OBCError, m_diagnostics, solve_active_set, solve_active_set_batch, solve_news
from .occbin import Occbin

__all__ = ["HigherOrderNews", "CubatureFailure"]

_DIAG_CACHE: dict = {}                                           # (digest of M, n_c, level) -> M diagnostics (per parameter set)
_SOBOL_CACHE: dict = {}                                          # (dim, level) -> read-only (dim, 2^(level+1)-1) normals


class CubatureFailure(OBCError):
    """A perturbed (cubature) path without a usable solution, raised by ``HigherOrderNews.news_shocks``.

    Attributes: ``period`` (1-based, set by ``simulate``), ``n_failed`` (points failed, in units of cubature points: a failed
    region counts all its points), ``lower_bound`` (True when the search stopped at the first failure / when the tolerance was
    exceeded, so ``n_failed`` is only a lower bound), ``n_relevant``, ``n_points``, ``counts`` = {'a','b','c'} (see
    ``HigherOrderNews._classify``), ``n_active_set_failed`` (points on which the active-set iteration failed: an upper
    bound on the failures when the whole batch was examined, else None), ``tolerance``, ``advice``."""

    def __init__(self, status, n_failed, lower_bound, n_relevant, n_points, counts, n_active_set_failed, tolerance, limit):
        a, b, c = counts["a"], counts["b"], counts["c"]
        advice = []
        if b:
            advice.append("(b): the check-ahead rows cannot be met with wedges inside the news horizon; try a longer `horizon` "
                          "(not proved to help)")
        if a:
            advice.append("(a): proved no solution at this horizon (M is not an S-matrix or the path is outside its reach): "
                          "the model has reachable states without an equilibrium; set cubature_failure_tolerance > 0 only to "
                          "drop such points knowingly")
        if c:
            advice.append(f"(c): undecided at the MILP time limit ({limit:g} s); raise `cubature_milp_time_limit`")
        msg = (f"cubature: {'at least ' if lower_bound else ''}{n_failed} of {n_relevant} relevant perturbed points "
               f"({n_points} in total) have no usable solution: a={a} b={b} c={c}"
               + (f"; tolerance {tolerance:g}" if tolerance else "") + ". " + " ".join(advice))
        super().__init__(msg, status)
        self.n_failed, self.lower_bound, self.n_relevant, self.n_points = n_failed, lower_bound, n_relevant, n_points
        self.counts, self.n_active_set_failed, self.tolerance, self.advice = dict(counts), n_active_set_failed, tolerance, advice


class HigherOrderNews:
    def __init__(self, model, order=2, horizon=40, check_ahead=40, params=None, Sigma_e=None,
                 cubature=False, periods_of_uncertainty=16, qmc_level=11, regions="all", taper=True, seed=0, pruning_cutoff=0.01,
                 max_cubature_dimension=128, cubature_milp_time_limit=1.0, cubature_failure_tolerance=0.0, m_diagnostic="s"):
        """Defaults when ``cubature`` is on: the full quasi-Monte Carlo average (``regions="all"``) over 2^(qmc_level+1) - 1 = 4,095
        points (``qmc_level=11``); ``regions=1`` (DynareOBC's default; its single mean point does not price tail risk) and
        ``regions=k`` remain options. See docs/OBC_HO_PERF.md, "Cubature defaults and failure handling", for the failure semantics:
        ``cubature_milp_time_limit`` (s per point, MILP proof step), ``cubature_failure_tolerance`` (0 = raise on any failed point),
        ``m_diagnostic`` ("s": S-matrix test of M, "full": also the P-matrix tests, False: skip)."""
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
        self.milp_time_limit = float(cubature_milp_time_limit)
        self.failure_tolerance = float(cubature_failure_tolerance)
        if not 0.0 <= self.failure_tolerance < 1.0:
            raise ValueError("cubature_failure_tolerance must be in [0, 1)")
        self.m_diag = None
        if self.cubature:
            self.R = self.conditional_root()
            self.xi = self._points(self.R.shape[1])
            if m_diagnostic:
                self.m_diag = self._m_diagnostics(full=(m_diagnostic == "full"))
                if not self.m_diag["is_s"]:
                    warnings.warn("M is not an S-matrix: some paths can have no solution; cubature points may fail "
                                  "(see HigherOrderNews.m_diag)", UserWarning, stacklevel=2)
        self._dropped_last = self._relevant_last = 0
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

    def _m_diagnostics(self, full=False):
        """``m_diagnostics`` of the complementarity block of M, cached per parameter set (key: digest of M). Level "s" (default) runs
        only the S-matrix LP and one leading P-block (2-6 ms at horizon 20-60); "full" adds the exact P-matrix tests (0.2-0.3 s at
        T = 32). Returns the dict of ``m_diagnostics`` plus ``seconds`` and ``level``."""
        nb = self.news
        key = (hashlib.sha1(np.ascontiguousarray(nb.Ms).tobytes()).hexdigest(), nb.n_c, bool(full))
        d = _DIAG_CACHE.get(key)
        if d is None:
            t = time.perf_counter()
            d = m_diagnostics(nb.Ms, n_c=nb.n_c, **({} if full else dict(max_exact=nb.n_c)))
            d["level"] = "full" if full else "s"
            if not full:
                d["is_p"] = None                                     # not decided at this level
            d["seconds"] = time.perf_counter() - t
            if len(_DIAG_CACHE) > 32:
                _DIAG_CACHE.clear()
            _DIAG_CACHE[key] = d
        return d

    # ------------------------------------------------------------------------------------- failure classification
    def _lp_gap(self, p, rows):
        """min s  s.t.  y >= 0, p[rows] + M[rows] y + s >= 0, s >= 0 (one LP). s* = 0 iff y >= 0 with p + M y >= 0 on ``rows`` exists."""
        M = self.news.Mo[rows]
        m, k = M.shape
        res = linprog(np.r_[np.zeros(k), 1.0], A_ub=np.hstack([-M, -np.ones((m, 1))]), b_ub=p[rows],
                      bounds=[(0, None)] * (k + 1), method="highs")
        return float(res.fun) if res.status == 0 else None

    def _classify(self, p):
        """A point on which the active-set iteration failed: (y, None) if the MILP finds a solution, else (None, cls).

        cls 'a': proved no solution at this horizon. Either the LP {y >= 0, p + M y >= 0 on the first n_s rows} (a necessary
        condition of any LCP solution, complementarity dropped) is infeasible by more than 1e-6 max(1, |p|) (so is the LP on
        all N rows); or the MILP (horizons T/4, T/2, T) terminated with alpha = 0 at every horizon and also on the first n_s rows.
        cls 'b': the LP on all N rows is infeasible (or the MILP on all rows finds no solution) but the check-ahead rows are the
        cause: the LP on the first n_s rows alone is feasible (resp. the MILP on the first n_s rows finds a solution). This is the
        diagnosis NewsOBC reports as 'horizon_too_short'. It does NOT establish that a longer horizon has a solution (that needs
        M and the gap path for a longer horizon, which are not built for perturbed points), nor, in the LP case, that the
        complementarity conditions can be met on the first n_s rows (LP feasibility is necessary, not sufficient).
        cls 'c': undecided: the MILP hit its time limit (``cubature_milp_time_limit``, shared by all MILP calls for this point).
        The LP is tried first (about 1-2 ms); the MILP is reached only for points whose LP is feasible."""
        nb, T, n_s = self.news, self.T, self.nws
        N = nb.Mo.shape[0]
        scale = max(1.0, float(np.abs(p).max()))
        tol = 1e-6 * scale
        s_head = self._lp_gap(p, slice(0, n_s))                      # first n_s rows (the horizon proper) first: 1 LP decides 'a'
        if s_head is not None and s_head > tol:
            return None, "a"
        s_full = self._lp_gap(p, slice(0, N)) if s_head is not None else None
        if s_full is not None and s_full > tol:
            return None, "b"
        hz = sorted({max(1, T // 4), max(1, T // 2), T})
        t_end = time.perf_counter() + self.milp_time_limit
        undecided = False
        for h in hz:
            left = t_end - time.perf_counter()
            if left <= 0:
                undecided = True
                break
            self.stats["milp"] = self.stats.get("milp", 0) + 1
            y, st = solve_news(nb.Mo, p, n_s, horizons=[h], n_c=self.n_c, time_limit=left)
            if y is not None:
                return y, None
            undecided = undecided or st == "time_limit"
        if undecided:
            return None, "c"
        left = max(t_end - time.perf_counter(), 1e-3)
        y2, st2 = solve_news(nb.Mo[:n_s], p[:n_s], n_s, horizons=[T], n_c=self.n_c, time_limit=left)
        return None, ("b" if y2 is not None else "a")

    def _solve_unit(self, p):
        """(y, cls) for one cubature unit (a point or a region mean point): zeros if nothing to do, the scalar active set,
        else ``_classify``. cls is None when solved."""
        self.stats["lcp"] += 1
        if np.all(p >= -1e-8 * max(1.0, np.abs(p).max())):
            return np.zeros(self.nws), None
        y = solve_active_set(self.news.Mo, p, self.nws)
        if y is not None:
            return y, None
        return self._classify(p)

    def _batch_columns(self, P):
        """Sum of the solutions of the columns of P that the batch active-set solver solves, and the indices of the columns it
        does not solve (those need ``_classify``). Columns needing no wedge contribute zero. Returns (sum, failed, n_failed_as)."""
        nb = self.news
        self.stats["lcp"] += P.shape[1]
        need = ~np.all(P >= -1e-8 * np.maximum(1.0, np.abs(P).max(axis=0)), axis=0)
        if not need.any():
            return np.zeros(self.nws), np.empty(0, np.intp)
        cols = np.flatnonzero(need)
        Pn = P[:, cols]
        Y, ok = solve_active_set_batch(nb.Mo, Pn, self.nws, scale=nb._mscale, cache=self._fac_cache, info=self.stats)
        nfail = int((~ok).sum())
        self.stats["batch_fail"] = self.stats.get("batch_fail", 0) + nfail
        self.stats["lcp"] -= nfail                                  # counted again if the MILP is reached
        return Y[:, ok].sum(axis=1), cols[~ok]

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
        """Solution at q; with cubature, the average over the perturbed paths q + R xi_j (see docs/OBC_HO_PERF.md, "Cubature
        defaults and failure handling"). With P points, F failed points (no usable solution), the result is
        (sum over solved relevant units of size * y) / (P - F): non-relevant points contribute y = 0, failed points are removed from
        the sample. F = 0 gives the plain average sum / P. F > 0 is only allowed when ``cubature_failure_tolerance`` > 0 and
        F <= tolerance * (number of relevant points); otherwise ``CubatureFailure`` is raised."""
        self._dropped_last = self._relevant_last = 0
        if not self.cubature or self.R.shape[1] == 0:
            return self._solve(q)
        npts = self.xi.shape[1]
        idx = self._relevant_points(q)
        self.stats["cubature_points"] += npts
        self.stats["relevant_points"] += idx.size
        self._relevant_last = idx.size
        if idx.size == 0:
            return np.zeros(self.nws)
        nrel, tol = idx.size, self.failure_tolerance
        counts = dict(a=0, b=0, c=0)
        nfail = 0                                                    # failed points so far (region: all its points)
        n_as = None                                                  # active-set failures among all units, when known
        limit_fail = tol * nrel                                      # failures allowed (0 = none)
        acc = np.zeros(self.nws)

        def failed(cls, size):
            nonlocal nfail
            counts[cls] += size
            nfail += size
            return nfail > limit_fail                                # True: stop (raise)

        unexamined = 0                                               # units not looked at when the search stopped early
        if self.regions == "all":                                    # plain quasi-Monte Carlo average
            pr = q[:, None] + self.R @ self.xi[:, idx]
            acc, bad = self._batch_columns(pr)
            n_as = int(bad.size)
            for i, j in enumerate(bad):
                y, cls = self._classify(pr[:, j])
                if cls is None:
                    acc += y
                elif failed(cls, 1):
                    unexamined = bad.size - i - 1
                    break
        else:
            xr = self.xi[:, idx]
            if self.regions == 1:                                    # DynareOBC's default: one region, its mean point
                groups = [np.arange(nrel)]
            else:                                                    # DynareOBC's CubatureRegions > 1
                k = int(self.regions)
                if nrel <= k:
                    groups = [np.array([j]) for j in range(nrel)]
                else:
                    groups = self._split_regions(q[:, None] + self.R @ xr, k)
            for i, g in enumerate(groups):                           # region mean point = q + R (mean of its xi)
                y, cls = self._solve_unit(q + self.R @ xr[:, g].mean(axis=1))
                if cls is None:
                    acc += g.size * y
                elif failed(cls, g.size):
                    unexamined = len(groups) - i - 1
                    break
        if nfail > limit_fail:
            self.stats["no_solution"] = self.stats.get("no_solution", 0) + 1
            status = "no_solution" if counts["a"] else ("horizon_too_short" if counts["b"] else "not_converged")
            raise CubatureFailure(status, nfail, unexamined > 0, nrel, npts, counts, n_as, tol, self.milp_time_limit)
        if nfail:
            self._dropped_last = nfail
            self.stats["dropped_points"] = self.stats.get("dropped_points", 0) + nfail
            self.stats["dropped_periods"] = self.stats.get("dropped_periods", 0) + 1
        return acc / (npts - nfail)

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
        ``with_bound``, ``without_bound`` (the pruned simulation), the news shocks per period (``wedges``), and per period the
        number of relevant cubature points (``relevant``) and of failed points dropped under ``cubature_failure_tolerance`` (``dropped``)."""
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
        dropped, relevant = np.zeros(S, int), np.zeros(S, int)
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
            dropped[t], relevant[t] = self._dropped_last, self._relevant_last
        if dropped.any():
            warnings.warn(f"cubature_failure_tolerance={self.failure_tolerance:g}: {dropped.sum()} of {relevant.sum()} relevant perturbed "
                          f"points ({100 * dropped.sum() / relevant.sum():.1f}%) were dropped in {int((dropped > 0).sum())} of {S} periods; "
                          f"the remaining points are reweighted by 1/(points - dropped)", UserWarning, stacklevel=2)
        return dict(with_bound=with_b, without_bound=without_b, wedges=wedges, dropped=dropped, relevant=relevant)
