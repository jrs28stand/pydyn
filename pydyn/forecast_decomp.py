# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2 (matlab/shock_decomposition/shock_decomposition.m, the branch of
# options_.shock_decomp.forecast_type; Copyright (C) Dynare Team, GPL-3.0-or-later).
"""Shock decomposition of a forecast: ``shock_decomposition(forecast_type=unconditional|conditional)``.

The decomposition runs over the smoothed sample followed by the forecast.  In the forecast periods the "smoothed"
column (the last one) holds the forecast of the requested variables in deviation from the steady state (zero for the
other variables); the shocks are zero for an unconditional forecast and the forecast's controlled exogenous variables
for a conditional one."""
from __future__ import annotations

import numpy as np


class ForecastDecompositionError(Exception):
    """A MATLAB error of Dynare's routine (``message`` is MATLAB's text)."""


def forecast_decomposition(smoothed, shocks, G, H, kind, i_var, forecast_means, ys, init_state=0,
                           controlled_exo=None, max_lag=1):
    """The array ``z`` (n_endo, n_exo + 2, gend) of ``shock_decomposition.m`` with a forecast type.

    ``smoothed`` (T, n_endo): smoothed variables in deviation from the mean, declaration order; ``shocks`` (T, n_exo):
    smoothed shocks; ``G`` (n_endo, n_endo) and ``H`` (n_endo, n_exo): ``dr.ghx(inv_order_var, :)`` spread to
    declaration order and ``dr.ghu(inv_order_var, :)`` (one lag); ``kind``: "unconditional" or "conditional";
    ``i_var``: 0-based indices of the variables; ``forecast_means``: {index: forecast path} for each of them (the
    conditional paths start at the initial period, which is dropped); ``ys``: steady state at the smoother's
    parameters (declaration order); ``controlled_exo``: (n_exo paths) of the conditional forecast's controlled
    exogenous variables.
    """
    smoothed = np.asarray(smoothed, float)
    shocks = np.asarray(shocks, float)
    T, n = smoothed.shape
    nx = shocks.shape[1]
    i0 = int(i_var[0])
    L = len(forecast_means[i0])
    gend = T + L - (1 if kind == "conditional" else 0)
    eps = np.zeros((nx, gend))
    z = np.zeros((n, nx + 2, gend))
    eps[:, :T] = shocks.T
    z[:, nx + 1, :T] = smoothed.T
    for i in i_var:
        path = np.asarray(forecast_means[int(i)], float).ravel()
        if kind == "conditional":
            path = path[1:]
        if path.size != gend - T:
            raise ForecastDecompositionError("Unable to perform assignment because the size of the left and right sides are not the same.")
        z[int(i), nx + 1, T:] = path - ys[int(i)]
    if kind == "conditional":
        cp = len(controlled_exo[0])
        if T + cp > gend:
            raise ForecastDecompositionError("out of bound; value %d out of bound %d" % (T + cp, gend))
        for i in range(nx):
            c = np.asarray(controlled_exo[i], float).ravel()
            if c.size != cp:
                raise ForecastDecompositionError("Unable to perform assignment because the size of the left and right sides are not the same.")
            eps[i, T:T + cp] = c
    for t in range(gend):
        if t > 0:
            z[:, :nx, t] = G @ z[:, :nx, t - 1]
        if t + 1 > init_state:
            z[:, :nx, t] += H * eps[:, t][None, :]
        z[:, nx, t] = z[:, nx + 1, t] - z[:, :nx, t].sum(axis=1)
    return z
