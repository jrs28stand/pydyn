# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 2013-2026 Dynare Team, https://www.dynare.org), modified in
# 2026: matlab/ep/solve_stochastic_perfect_foresight_model_0.m, ep_problem_0.m, setup_integration_nodes.m,
# gauss_hermite_weights_and_nodes.m, cartesian_product_of_sets.m, get_block_world_nbr.m.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""The stochastic extended path with the full tree of future innovations (``options_.ep.stochastic.algo = 0``) and the
integration rules ``Tensor-Gaussian-Quadrature`` and ``Unscented``.

For ``order = k`` the shocks of the ``k`` first periods ahead are integrated over a tree of ``nnodes`` branches per period; later
periods are perfect foresight with the shocks at their mean.  The stacked system is the one of ``ep_problem_0.m``, assembled here
with the same index bookkeeping (the unknowns are the endogenous variables of the nodes of the tree in the order Dynare uses), and
solved with the trust region of ``dynare_solve`` (``solve_algo = 9``).

Not implemented (refused by the caller): the sparse tree (``algo = 1``), ``hybrid`` corrections, the Stroud cubatures, models whose
maximum lag or lead differs from one.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from . import extended_path as _ep


class SEPError(Exception):
    pass


def gauss_hermite(n):
    """gauss_hermite_weights_and_nodes.m (probabilists' normalisation: weights sum to one, nodes scaled by sqrt(2))."""
    b = np.sqrt(np.arange(1, n) / 2.0)
    J = np.diag(b, 1) + np.diag(b, -1)
    vals, vecs = np.linalg.eigh(J) if n > 1 else (np.zeros(1), np.ones((1, 1)))
    idx = np.argsort(vals, kind="stable")
    nodes = np.sqrt(2.0) * vals[idx]
    weights = vecs[0, idx] ** 2
    return nodes, weights


def integration_nodes(algorithm, nodes_per_shock, Omega):
    """setup_integration_nodes.m: ``(nodes (nnodes x n), weights (nnodes,))`` for the shocks with a positive variance;
    ``Omega`` is the upper Cholesky factor of their covariance matrix."""
    n = Omega.shape[0]
    if algorithm == "Tensor-Gaussian-Quadrature":
        nodes0, weights0 = gauss_hermite(int(nodes_per_shock))
        nodes0 = np.tile(nodes0[:, None], (1, n)) @ Omega
        grids = np.meshgrid(*[nodes0[:, i] for i in range(n)], indexing="ij")
        nodes = np.column_stack([g.ravel(order="F") for g in grids])
        wgrids = np.meshgrid(*[weights0] * n, indexing="ij")
        weights = np.prod(np.column_stack([g.ravel(order="F") for g in wgrids]), axis=1)
        return nodes, weights
    if algorithm == "Unscented":
        k = 3
        C = np.sqrt(n + k) * Omega.T
        nodes = np.vstack([np.zeros((1, n)), -C, C])
        weights = np.concatenate([[k / (n + k)], np.full(2 * n, 1.0 / (2 * (n + k)))])
        return nodes, weights
    raise SEPError(f"the integration algorithm '{algorithm}' is not implemented (implemented: Tensor-Gaussian-Quadrature, Unscented)")


class Tree:
    """The state of ``pfm`` for ``algo = 0``: ``setup_stochastic_perfect_foresight_model_solver.m`` and the first call of
    ``solve_stochastic_perfect_foresight_model_0.m``."""

    def __init__(self, model, params, ss, Sigma_e, order, periods, nodes, weights):
        n = model.n_endo
        if int(model.info["maximum_lag"]) != 1 or int(model.info["maximum_lead"]) != 1:
            raise SEPError("the stochastic extended path is implemented for models with maximum lag and lead equal to one")
        lli = np.asarray(model.lead_lag_incidence) > 0                       # n x 3 (lag, current, lead)
        if not lli[:, 1].all():
            raise SEPError("a variable does not appear at the current period in the dynamic model")
        self.model, self.params, self.ss = model, params, np.asarray(ss, float)
        self.n, self.order, self.periods = n, int(order), int(periods)
        self.pos = np.flatnonzero(np.diag(Sigma_e) > 0)
        # make sure that there is a node equal to zero, and put it first
        k = np.flatnonzero(np.abs(nodes).sum(axis=1) < 1e-12)
        if k.size == 0:
            raise SEPError("solve_stochastic_perfect_foresight_model_0: there is no node equal to zero")
        k = int(k[0])
        perm = [k] + [i for i in range(len(weights)) if i != k]
        self.nodes, self.weights = nodes[perm], weights[perm]
        nn = self.nnodes = len(self.weights)
        W = self.world_nbr = nn ** self.order
        if self.periods <= self.order:
            raise SEPError("ep.periods must exceed the order of the stochastic extended path")
        flat = np.flatnonzero(lli.ravel(order="F"))                           # find(lead_lag_incidence')
        nyp, nyf = int(lli[:, 0].sum()), int(lli[:, 2].sum())
        self.nyp, self.nyf = nyp, nyf
        self.i_cols_j = flat
        self.i_cols_1 = np.flatnonzero(lli[:, 1:3].ravel(order="F")) + n
        self.i_cols_T = np.flatnonzero(lli[:, 0:2].ravel(order="F"))
        self.i_cols_Ap0 = flat[:nyp]
        self.i_cols_As0 = flat[nyp:nyp + n]
        self.i_cols_Af0 = flat[nyp + n:nyp + n + nyf] - n
        S = sum(nn ** i for i in range(self.order))
        self.S = S
        self.dimension = n * (S + (self.periods - self.order) * W)
        if self.dimension > 2_000_000:
            raise SEPError(f"the tree of future innovations has {self.dimension} unknowns ({nn} nodes per period, order "
                           f"{self.order}, {self.periods} periods): too large for pydyn")
        self.icA = np.concatenate([np.flatnonzero(lli[:, 0]), np.flatnonzero(lli[:, 1]) + W * n,
                                   np.flatnonzero(lli[:, 2]) + 2 * W * n])
        # i_upd_r, i_upd_y
        dim, P = self.dimension, self.periods
        upd_r = np.zeros(dim, dtype=np.int64)
        upd_y = np.zeros(dim, dtype=np.int64)
        upd_r[:n] = np.arange(n)
        upd_y[:n] = n + np.arange(n)
        i1, i2, n1, n2 = n, 2 * n, n, 2 * n
        for i in range(2, P + 1):
            for j in range(1, nn ** min(i - 1, self.order) + 1):
                upd_r[i1:i2] = np.arange(n1, n2) + (j - 1) * n * P
                upd_y[i1:i2] = np.arange(n1, n2) + n + (j - 1) * n * (P + 2)
                i1 = i2
                i2 += n
            n1 = n2
            n2 += n
        self.i_upd_r, self.i_upd_y = upd_r, upd_y
        self.nblocks1 = sum(nn ** i for i in range(self.order + 1))

    # ep_problem_0.m
    def problem(self, y, endo_simul, x):
        """``(res, A)`` of ep_problem_0 at the unknowns ``y``; ``endo_simul`` is ``(periods+2) x n`` (its first row is the initial
        condition, the others the guess), ``x`` the ``(periods+2) x n_exo`` exogenous variables."""
        m, n, order, P, nn, W = self.model, self.n, self.order, self.periods, self.nnodes, self.world_nbr
        params, ss, nodes, weights = self.params, self.ss, self.nodes, self.weights
        Y = np.tile(np.asarray(endo_simul, float).reshape(-1), (W, 1)).T.copy(order="F")      # repmat(endo_simul(:), 1, W)
        Yf = Y.reshape(-1, order="F")
        Yf[self.i_upd_y] = y
        Y = Yf.reshape(Y.shape, order="F")
        a1r, a1c, a1v = [], [], []                      # A1 as triplets (duplicates add up, as A1(rows, cols) = A1(rows, cols) + ...)

        def put(rows_, cols_, block):
            ir, ic = np.nonzero(block)
            a1r.append(rows_[ir])
            a1c.append(cols_[ic])
            a1v.append(block[ir, ic])

        res = np.zeros((n, P, W), order="F")
        i_cols_p = np.arange(n)
        i_cols_s = n + np.arange(n)
        i_cols_f = 2 * n + np.arange(n)
        i_rows = np.arange(n)
        i_cols_Ap, i_cols_As, i_cols_Af = self.i_cols_Ap0.copy(), self.i_cols_As0.copy(), self.i_cols_Af0.copy()
        i_cols_1, i_cols_j = self.i_cols_1, self.i_cols_j
        n3 = 3 * n

        def evaluate(z, xi):
            d1 = np.asarray(m.dynamic_resid(z, xi, params, ss), float)
            jac = np.asarray(m.dynamic_jac(z, xi, params, ss), float)[:, :n3]
            return d1, jac

        for i in range(1, order + 2):
            i_w_p = 0
            for j in range(1, nn ** (i - 1) + 1):
                innovation = x.copy()
                if i > 1:
                    innovation[i, self.pos] = nodes[(j - 1) % nn]
                if i <= order:
                    for k in range(1, nn + 1):
                        z = np.concatenate([Y[i_cols_p, i_w_p], Y[i_cols_s, j - 1], Y[i_cols_f, (j - 1) * nn + k - 1]])
                        d1, jac = evaluate(z, innovation[i])
                        if i == 1:
                            cols = np.concatenate([i_cols_As - n, i_cols_Af])
                            put(i_rows, cols, weights[k - 1] * jac[:, i_cols_1])
                        else:
                            cols = np.concatenate([i_cols_Ap, i_cols_As, i_cols_Af])
                            put(i_rows, cols, weights[k - 1] * jac[:, i_cols_j])
                        res[:, i - 1, j - 1] += weights[k - 1] * d1
                        i_cols_Af = i_cols_Af + n
                else:
                    z = np.concatenate([Y[i_cols_p, i_w_p], Y[i_cols_s, j - 1], Y[i_cols_f, j - 1]])
                    d1, jac = evaluate(z, innovation[i])
                    if i == 1:
                        cols = np.concatenate([i_cols_As - n, i_cols_Af])
                        put(i_rows, cols, jac[:, i_cols_1])
                    else:
                        cols = np.concatenate([i_cols_Ap, i_cols_As, i_cols_Af])
                        put(i_rows, cols, jac[:, i_cols_j])
                    res[:, i - 1, j - 1] = d1
                    i_cols_Af = i_cols_Af + n
                i_rows = i_rows + n
                if j % nn == 0:
                    i_w_p += 1
                if i > 1:
                    if j % nn == 0:
                        i_cols_Ap = i_cols_Ap + n
                    i_cols_As = i_cols_As + n
            i_cols_p = i_cols_p + n
            i_cols_s = i_cols_s + n
            i_cols_f = i_cols_f + n
        rows, cols, vals = [], [], []
        for j in range(1, W + 1):
            i_rows_y = np.arange(n3) + (order + 1) * n
            offset_c = n * (self.S + j - 1)
            offset_r = (j - 1) * n
            for i in range(order + 2, P + 1):
                d1, jac = evaluate(Y[i_rows_y, j - 1], x[i])
                sub = jac[:, self.i_cols_T] if i == P else jac[:, i_cols_j]
                ir, ic = np.nonzero(sub.T)[::-1]                                 # find(): column-major order
                rows.append(offset_r + ir)
                cols.append(offset_c + self.icA[ic])
                vals.append(sub[ir, ic])
                res[:, i - 1, j - 1] = d1
                i_rows_y = i_rows_y + n
                offset_c += W * n
                offset_r += W * n
        nrow2 = n * (P - order - 1) * W
        if rows:
            A2 = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(nrow2, self.dimension))
        else:
            A2 = sparse.coo_matrix((0, self.dimension))
        A1 = sparse.coo_matrix((np.concatenate(a1v), (np.concatenate(a1r), np.concatenate(a1c))),
                               shape=(n * self.nblocks1, self.dimension))
        A = sparse.vstack([A1.tocsr(), A2.tocsr()], format="csc")
        return res.reshape(-1, order="F")[self.i_upd_r], A

    def solve(self, endo_simul, exo_simul, y, o, log=None):
        """solve_stochastic_perfect_foresight_model_0.m (without the first-call bookkeeping done in ``__init__``): returns
        ``(errorflag, endo_simul, y)``."""
        n = self.n
        endo_simul = np.asarray(endo_simul, float)
        if y is None:
            y = np.tile(self.ss, self.dimension // n)
        tolf, tolx = o["dynatol.f"], o["dynatol.x"]

        def fun(v):
            return self.problem(v, endo_simul, exo_simul)

        with np.errstate(all="ignore"):
            f0 = fun(y)[0]
            if np.all(np.isfinite(f0)) and np.max(np.abs(f0)) < tolf:
                err, ynew = False, np.asarray(y, float)
            else:
                ynew, err = _ep.trust_region(fun, y, tolf, tolx, int(o["simul.maxit"]),
                                             o.get("trust_region_initial_step_bound_factor", 1.0))
        out = endo_simul.copy()
        out[1] = ynew[:n]
        return bool(err), out, ynew


def run(model, params, ss, xss, Sigma_e, ep, samplesize, y0, shocks, o, log=None):
    """The loop of ``extended_path.m`` for ``order > 0`` with the full tree.  ``ep``: the dict of ``options_.ep`` (``periods``,
    ``order``, ``IntegrationAlgorithm``, ``nodes``).  Homotopy is off (extended_path_initialization.m)."""
    n, nx = model.n_endo, model.n_exo
    lag, lead = int(model.info["maximum_lag"]), int(model.info["maximum_lead"])
    periods = int(ep["periods"])
    pos = np.flatnonzero(np.diag(Sigma_e) > 0)
    if pos.size == 0:
        raise SEPError("the stochastic extended path needs at least one shock with a positive variance")
    Omega = np.linalg.cholesky(np.asarray(Sigma_e, float)[np.ix_(pos, pos)]).T
    nodes, weights = integration_nodes(ep["IntegrationAlgorithm"], ep["nodes"], Omega)
    tree = Tree(model, params, ss, Sigma_e, ep["order"], periods, nodes, weights)
    rows = lag + periods + lead
    exo_full = np.tile(np.asarray(xss, float)[None, :], (rows, 1))
    path = np.full((samplesize + lag, n), np.nan)
    path[0] = y0
    status = True
    y = None
    endo = None
    t = 1
    while t <= samplesize:
        t += 1
        exo = exo_full.copy()
        exo[1] = shocks[t - 2]
        if t > 2:
            init = np.vstack([endo[1:], np.asarray(ss, float)[None, :]])
        else:
            init = np.vstack([y0[None, :], np.tile(np.asarray(ss, float)[None, :], (periods + lead, 1))])
        err, endo, y = tree.solve(init, exo, y, o, log)
        if err:
            if log:
                log(f"extended_path: No convergence of the (stochastic) perfect foresight solver (in period {t})!")
            status = False
            break
        path[t - 1] = endo[1]
    exo_sim = np.vstack([np.tile(np.asarray(xss, float)[None, :], (lag, 1)), shocks])
    return _ep.EPResult(path, exo_sim, status, shocks)
