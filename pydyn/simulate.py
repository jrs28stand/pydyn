# This file is part of pydyn.
#
# It is a Python port of matlab/stochastic_solver/simult_.m and irf.m from Dynare 7.2
# (Copyright (C) 2001-2026 Dynare Team, https://www.dynare.org), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Simulation and impulse responses from perturbation decision rules (orders 1 to 3).

Ports of Dynare's ``simult_.m`` and ``irf.m``. Decision rules are dicts as returned by
``pydyn.k_order`` (or ``FirstOrderSolution.as_dr()``): arrays ``ghx``, ``ghu``, ``ghxx`` ... with
rows in decision-rule order, plus ``ys``, ``order_var`` and ``state_var``.
"""
from __future__ import annotations

import numpy as np


def _as_dr(dr):
    return dr.as_dr() if hasattr(dr, "as_dr") else dr


def simulate(dr, shocks, order=None, pruning=False, y0=None) -> np.ndarray:
    """Simulate levels for a (T, n_exo) shock array; returns (T, n_endo) in declaration order.

    ``y0`` is the initial level vector (default: steady state). At order 2 with ``pruning`` this is
    the Kim et al. (2008) scheme; at order 3 with ``pruning`` the Andreasen et al. (2018) scheme.
    """
    dr = _as_dr(dr)
    order = dr.get("order", 1) if order is None else order
    if order > dr.get("order", 1):
        raise ValueError(f"decision rule only available up to order {dr.get('order', 1)}")
    ys, ov, sv = dr["ys"], dr["order_var"], dr["state_var"]
    shocks = np.atleast_2d(np.asarray(shocks, float))
    T, n = shocks.shape[0], len(ys)
    y0 = ys.copy() if y0 is None else np.asarray(y0, float)
    out = np.empty((T, n))
    ghx, ghu = dr["ghx"], dr["ghu"]
    kron = np.kron

    if order == 1:
        x = y0[sv] - ys[sv]
        for t in range(T):
            yh = ghx @ x + ghu @ shocks[t]
            out[t, ov] = ys[ov] + yh
            x = out[t, sv] - ys[sv]
        return out

    ghxx, ghxu, ghuu, ghs2 = dr["ghxx"], dr["ghxu"], dr["ghuu"], dr["ghs2"].ravel()
    if order == 2:
        const = ys[ov] + 0.5 * ghs2
        x = y0[sv] - ys[sv]          # state used in the full recursion
        x1 = x.copy()                # first-order state (pruning)
        for t in range(T):
            u = shocks[t]
            xq = x1 if pruning else x
            quad = 0.5 * ghxx @ kron(xq, xq) + 0.5 * ghuu @ kron(u, u) + ghxu @ kron(xq, u)
            out[t, ov] = const + ghx @ x + ghu @ u + quad
            if pruning:
                y1 = np.empty(n)
                y1[ov] = ghx @ x1 + ghu @ u
                x1 = y1[sv]
            x = out[t, sv] - ys[sv]
        return out

    if order != 3:
        raise NotImplementedError("simulation above third order")
    ghxxx, ghxxu, ghxuu, ghuuu = dr["ghxxx"], dr["ghxxu"], dr["ghxuu"], dr["ghuuu"]
    ghxss, ghuss = dr["ghxss"], dr["ghuss"]
    pos = {v: i for i, v in enumerate(ov)}
    ipred = np.array([pos[v] for v in sv])           # rows of the states in DR order
    if pruning:
        x1 = y0[sv] - ys[sv]
        x2 = np.zeros_like(x1)
        x3 = np.zeros_like(x1)
        for t in range(T):
            u = shocks[t]
            yu = kron(x1, u)
            y3 = ghx @ x3 + ghxx @ kron(x1, x2) + ghxu @ kron(x2, u) + (
                ghxxx @ kron(kron(x1, x1), x1) + ghuuu @ kron(kron(u, u), u)
                + 3 * (ghxxu @ kron(x1, yu) + ghxuu @ kron(yu, u) + ghxss @ x1 + ghuss @ u)) / 6
            y2 = ghx @ x2 + 0.5 * (ghxx @ kron(x1, x1) + ghuu @ kron(u, u) + 2 * ghxu @ yu + ghs2)
            y1 = ghx @ x1 + ghu @ u
            out[t, ov] = ys[ov] + y1 + y2 + y3
            x1, x2, x3 = y1[ipred], y2[ipred], y3[ipred]
        return out
    x = y0[sv] - ys[sv]
    for t in range(T):
        u = shocks[t]
        xu = kron(x, u)
        yh = (ghx @ x + ghu @ u
              + 0.5 * (ghxx @ kron(x, x) + 2 * ghxu @ xu + ghuu @ kron(u, u) + ghs2)
              + (ghxxx @ kron(kron(x, x), x) + ghuuu @ kron(kron(u, u), u)
                 + 3 * (ghxxu @ kron(x, xu) + ghxuu @ kron(xu, u) + ghxss @ x + ghuss @ u)) / 6)
        out[t, ov] = ys[ov] + yh
        x = yh[ipred]
    return out


def irf(dr, impulse, Sigma_e=None, periods=40, order=None, pruning=False, drop=100, replic=50,
        rng=None, draws=None) -> np.ndarray:
    """Impulse responses, shape (periods, n_endo), deviations as Dynare reports them.

    At order 1 this is the exact linear response. At higher orders it is Dynare's simulation-based
    generalised IRF: the average over ``replic`` random shock paths of the difference between the
    path with ``impulse`` added in period ``drop + 1`` and the path without it. ``draws`` (shape
    (periods + drop, n_shocks_with_variance, replic), standard normal) can be given to make the
    result reproducible; otherwise ``rng`` (a numpy Generator) is used.
    """
    dr = _as_dr(dr)
    order = dr.get("order", 1) if order is None else order
    impulse = np.asarray(impulse, float)
    n_exo = impulse.size
    if order == 1:
        ex = np.zeros((periods, n_exo))
        ex[0] = impulse
        return simulate(dr, ex, order=1) - dr["ys"]
    if Sigma_e is None:
        raise ValueError("Sigma_e is needed for impulse responses above first order")
    Sigma_e = np.asarray(Sigma_e, float)
    active = np.nonzero(np.diag(Sigma_e) != 0)[0]
    chol_u = np.linalg.cholesky(Sigma_e[np.ix_(active, active)]).T     # upper factor, as chol() in MATLAB
    if draws is None:
        rng = np.random.default_rng() if rng is None else rng
        draws = rng.standard_normal((periods + drop, len(active), replic))
    else:
        draws = np.asarray(draws, float)
        replic = draws.shape[2]
        if draws.shape[0] != periods + drop:
            raise ValueError("draws must have periods + drop rows")
    acc = np.zeros((periods, len(dr["ys"])))
    for j in range(replic):
        ex1 = np.zeros((periods + drop, n_exo))
        ex1[:, active] = draws[:, :, j] @ chol_u
        ex2 = ex1.copy()
        ex2[drop] += impulse
        y1 = simulate(dr, ex1, order=order, pruning=pruning)
        y2 = simulate(dr, ex2, order=order, pruning=pruning)
        acc += y2[drop:] - y1[drop:]
    return acc / replic
