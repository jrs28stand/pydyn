# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 1996-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: matlab/stochastic_solver/stochastic_solvers.m (the
# "exogenous deterministic variables" section) and matlab/stochastic_solver/simult_varexo_det.m.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Deterministic exogenous variables (``varexo_det``) in stochastic models.

Dynare's stochastic solvers treat the variables declared with ``varexo_det`` as known in advance:
``stochastic_solvers.m`` adds to the decision rules the effects of the deviations of ``x_det(t+j-1)``
from their steady state, ``ghud{j}`` (first order) and, at order 2 with Dynare's default second-order
solver, ``ghxud{j}``, ``ghuud{j}`` and ``ghudud{j,k}``, for ``j = 1 .. exo_det_length``, the last
period with a deterministic shock on a ``varexo_det`` variable in the ``shocks`` blocks.
``simult_varexo_det.m`` simulates with them (Dynare's ``forecast`` uses it for the mean path).

Everything else in ``stoch_simul`` (moments, IRFs, ``simult``) ignores ``varexo_det``: they stay at
their steady-state values (``initval``), which is what ``pydyn.Model`` evaluates by default.
"""
from __future__ import annotations

import numpy as np

from .perturbation import dr_ordering


def exo_det_length(model) -> int:
    """``dr.exo_det_length``: the last period of a deterministic shock on a varexo_det variable."""
    return max((p2 for _, p1, p2, _ in model.det_shocks_exo_det), default=0)


def _dense_derivs(model, order, ys):
    n, nx, nxd = model.n_endo, model.n_exo, model.n_exo_det
    nk = 3 * n + nx + nxd
    x = np.concatenate([np.zeros(nx) if not hasattr(model, "exo_steady") else model.exo_steady(), model.exo_det_steady()])
    rows, cols, vals = model.dynamic_full_derivs(order, np.tile(ys, 3), x, model.params, ys)
    if order == 1:
        g = np.zeros((n, nk))
        g[rows, cols[:, 0]] = vals
        return g
    H = np.zeros((n, nk, nk))
    H[rows, cols[:, 0], cols[:, 1]] = vals
    H[rows, cols[:, 1], cols[:, 0]] = vals
    return H


def decision_rules(model, dr, length=None):
    """Add ``ghud`` (and, if ``dr`` holds second-order rules, ``ghxud``, ``ghuud``, ``ghudud``) to a
    copy of the decision-rule dict ``dr`` (``pydyn.first_order(m).as_dr()`` or ``pydyn.k_order``),
    as ``stochastic_solvers.m`` does. ``ghud`` etc. are lists indexed from 0 (Dynare's ``ghud{1}`` is
    ``ghud[0]``); ``ghudud[(j, k)]`` holds Dynare's ``ghudud{j+1,k+1}`` for ``j <= k``.

    ``length`` overrides ``exo_det_length``. Order 3 is refused: Dynare uses its k-order solver there,
    which computes no varexo_det terms."""
    if model.n_exo_det == 0:
        raise ValueError("the model has no varexo_det variables")
    dr = dict(dr.as_dr() if hasattr(dr, "as_dr") else dr)
    order = int(dr.get("order", 1))
    if order > 2:
        raise NotImplementedError("varexo_det: Dynare computes no deterministic-exogenous terms at order 3 "
                                  "(k_order_solver)")
    lli = model.lead_lag_incidence
    if not lli[:, 0].any():
        raise ValueError("var_exo_det not implemented for purely forward models")
    if not lli[:, 2].any():
        raise ValueError("var_exo_det not implemented for purely backwards models")
    n, nx, nxd = model.n_endo, model.n_exo, model.n_exo_det
    order_var, _, cnt = dr_ordering(lli)
    order_var = np.asarray(dr.get("order_var", order_var), int)
    nstatic, npred, nboth, nfwrd = cnt["nstatic"], cnt["npred"], cnt["nboth"], cnt["nfwrd"]
    nspred, nsfwrd = npred + nboth, nboth + nfwrd
    ys = np.asarray(dr["ys"], float)
    ghx = np.asarray(dr["ghx"], float)
    ghu = np.asarray(dr["ghu"], float)
    gx = ghx[nstatic + npred:, :]                       # dr.gx: rows of the forward-looking variables
    Gy = ghx[nstatic:nstatic + nspred, :]               # dr.Gy (hx)
    g1 = _dense_derivs(model, 1, ys)
    icurr = np.nonzero(lli[order_var, 1])[0]            # icurr_dr
    f1 = g1[:, 2 * n + order_var[nstatic + npred + np.arange(nsfwrd)]]
    f0 = g1[:, n + order_var[icurr]]
    fudet = g1[:, 3 * n + nx:]
    M1 = np.linalg.inv(f0 + np.hstack([np.zeros((n, nstatic)), f1 @ gx, np.zeros((n, nsfwrd - nboth))]))
    M2 = M1 @ f1
    L = exo_det_length(model) if length is None else int(length)
    ghud = []
    if L > 0:
        ghud.append(-M1 @ fudet)
    for i in range(1, L):
        ghud.append(-M2 @ ghud[i - 1][n - nsfwrd:, :])
    dr["exo_det_length"] = L
    dr["ghud"] = ghud
    if order == 2 and L > 0:
        H = _dense_derivs(model, 2, ys)
        kk1 = np.concatenate([order_var[nstatic + np.arange(nspred)], n + order_var[icurr],
                              2 * n + order_var[nstatic + npred + np.arange(nsfwrd)], 3 * n + np.arange(nx + nxd)])
        K = len(kk1)
        G2 = H[:, kk1][:, :, kk1].reshape(n, K * K)
        k0 = np.nonzero(lli[order_var, 1])[0]
        kf = np.arange(n - nfwrd - nboth, n)
        kp = nstatic + np.arange(nspred)
        ghxx, ghuu = np.asarray(dr["ghxx"], float), np.asarray(dr["ghuu"], float)
        hu = ghu[kp, :]
        hud = ghud[0][kp, :]
        Z = np.zeros
        zx = np.vstack([np.eye(nspred), ghx[k0, :], gx @ Gy, Z((nx + nxd, nspred))])
        zu = np.vstack([Z((nspred, nx)), ghu[k0, :], gx @ hu, Z((nx + nxd, nx))])
        zud = np.vstack([Z((nspred, nxd)), ghud[0], gx[:, :nspred] @ hud, Z((nx, nxd)), np.eye(nxd)])
        Eud = np.eye(nxd)

        def zudi_(i, extra=None):
            hudi = ghud[i][kp, :]
            mid = gx[:, :nspred] @ hudi if extra is None else gx[:, :nspred] @ hudi + extra
            return np.vstack([Z((nspred, nxd)), ghud[i], mid, Z((nx + nxd, nxd))])

        ghxud = [-M1 @ (G2 @ np.kron(zx, zud) + f1 @ ghxx[kf, :] @ np.kron(ghx[kp, :], ghud[0][kp, :]))]
        for i in range(1, L):
            R2 = G2 @ np.kron(zx, zudi_(i))
            ghxud.append(-M2 @ (ghxud[i - 1][kf, :] @ np.kron(Gy, Eud)
                                + ghxx[kf, :] @ np.kron(ghx[kp, :], ghud[i][kp, :])) - M1 @ R2)
        ghuud = [-M1 @ (G2 @ np.kron(zu, zud) + f1 @ ghxx[kf, :] @ np.kron(ghu[kp, :], ghud[0][kp, :]))]
        for i in range(1, L):
            R2 = G2 @ np.kron(zu, zudi_(i))
            ghuud.append(-M2 @ ghxud[i - 1][kf, :] @ np.kron(hu, Eud) - M1 @ R2)
        ghudud = {(0, 0): -M1 @ (G2 @ np.kron(zud, zud)) - M2 @ ghxx[kf, :] @ np.kron(hud, hud)}
        for i in range(1, L):
            hudi = ghud[i][nstatic:nstatic + nspred, :]
            zudi = zudi_(i, ghud[i - 1][kf, :])
            R2 = G2 @ np.kron(zudi, zudi)
            ghudud[(i, i)] = (-M2 @ (ghudud[(i - 1, i - 1)][kf, :] + 2 * ghxud[i - 1][kf, :] @ np.kron(hudi, Eud)
                                     + ghxx[kf, :] @ np.kron(hudi, hudi)) - M1 @ R2)
            R2 = G2 @ np.kron(zud, zudi)
            ghudud[(0, i)] = (-M2 @ (ghxud[i - 1][kf, :] @ np.kron(hud, Eud) + ghxx[kf, :] @ np.kron(hud, hudi))
                              - M1 @ R2)
            for j in range(1, i):
                hudj = ghud[j][kp, :]
                zudj = zudi_(j)
                R2 = G2 @ np.kron(zudj, zudi)
                ghudud[(j, i)] = (-M2 @ (ghudud[(j - 1, i - 1)][kf, :] + ghxud[j - 1][kf, :] @ np.kron(hudi, Eud)
                                         + ghxud[i - 1][kf, :] @ np.kron(hudj, Eud) + ghxx[kf, :] @ np.kron(hudj, hudi))
                                  - M1 @ R2)
        dr.update(ghxud=ghxud, ghuud=ghuud, ghudud=ghudud)
    return dr


def simulate(model, dr, ex, ex_det, y0=None, order=None):
    """``simult_varexo_det.m`` (the path, without confidence bands): ``ex`` (horizon, n_exo) stochastic
    shocks, ``ex_det`` (horizon + 1, n_exo_det) values (levels) of the varexo_det variables, the first
    row being the initial period as in Dynare. ``dr`` from :func:`decision_rules`. Returns the
    (horizon + 1, n_endo) path in declaration order, the first row being ``y0`` (default: steady state).
    """
    order = int(dr.get("order", 1)) if order is None else int(order)
    if order > 2:
        raise NotImplementedError("simult_varexo_det.m: Forecasting with varexo_det does not support order>2.")
    if dr.get("pruning") and order == 2:
        raise NotImplementedError("simult_varexo_det.m: Forecasting with varexo_det does not support pruning.")
    ex = np.atleast_2d(np.asarray(ex, float))
    ex_det = np.atleast_2d(np.asarray(ex_det, float))
    horizon = ex.shape[0]
    if ex_det.shape[0] != horizon + 1:
        raise ValueError("simult_varexo_det: Size mismatch: number of forecasting periods for stochastic "
                         "exogenous and deterministic exogenous don't match")
    _, _, cnt = dr_ordering(model.lead_lag_incidence)
    nstatic, nspred = cnt["nstatic"], cnt["npred"] + cnt["nboth"]
    ov = np.asarray(dr["order_var"], int)
    ys = np.asarray(dr["ys"], float)
    yss = ys[ov]
    L = dr["exo_det_length"]
    xdss = model.exo_det_steady()
    Y = np.zeros((horizon + 1, model.n_endo))
    Y[0] = ys if y0 is None else np.asarray(y0, float)
    for i in range(1, horizon + 1):                  # Dynare's i = maximum_lag + 1 .. (here maximum_lag = 1)
        tempx = (Y[i - 1][ov] - yss)[nstatic:nstatic + nspred]
        tempu = ex[i - 1]
        y = yss + dr["ghx"] @ tempx + dr["ghu"] @ tempu
        if order == 2:
            y = y + np.ravel(dr["ghs2"]) / 2 + 0.5 * (dr["ghxx"] @ np.kron(tempx, tempx)
                                                      + dr["ghuu"] @ np.kron(tempu, tempu)) \
                + dr["ghxu"] @ np.kron(tempx, tempu)
        # Dynare (1-based): j = 1 .. min(maximum_lag + exo_det_length + 1 - i, exo_det_length), ex_det(i+j-1)
        for j in range(min(L + 1 - i, L)):
            tempud = ex_det[i + j] - xdss
            y = y + dr["ghud"][j] @ tempud
            if order == 2:
                y = y + dr["ghxud"][j] @ np.kron(tempx, tempud) + dr["ghuud"][j] @ np.kron(tempu, tempud) \
                    + 0.5 * dr["ghudud"][(j, j)] @ np.kron(tempud, tempud)
                for k in range(j):
                    y = y + dr["ghudud"][(k, j)] @ np.kron(ex_det[i + k] - xdss, tempud)
        Y[i][ov] = y
    return Y
