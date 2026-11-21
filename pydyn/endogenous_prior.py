# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 2011-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: matlab/estimation/endogenous_prior.m and its use in
# matlab/estimation/dsge_likelihood.m.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Endogenous prior of Christiano, Trabandt and Walentin (2011), Dynare's ``endogenous_prior`` option.

The log density of the sample variances of the observed variables (after the last missing
observation), given the variances the model implies at ``theta``, with a Newey-West estimate (two
lags, Bartlett weights) of the covariance of the variance estimates. It is added to the log
posterior kernel (``dsge_likelihood.m``: ``fval = likelihood - lnprior - lnpriormom``); stationary
models only.
"""
from __future__ import annotations

import numpy as np


def lnpriormom(Y, Pstar, mf, H):
    """``endogenous_prior.m``. ``Y`` (n_obs, T) data used by the filter (from the first period after
    the last missing observation), ``Pstar`` the stationary state covariance, ``mf`` the positions of
    the observed variables in the state vector, ``H`` the measurement-error covariance matrix."""
    Y = np.asarray(Y, float).T
    Tsamp, n = Y.shape
    Yd = Y - Y.mean(axis=0)
    Fhat = np.sum(Yd * Yd, axis=0) / Tsamp
    h = (Yd * Yd - Fhat).T                                  # hmat (n, T)
    C0 = np.zeros((n, n))
    for t in range(Tsamp):
        C0 = C0 + 1 / Tsamp * np.outer(h[:, t], h[:, t])
    C1 = np.zeros((n, n))
    for t in range(1, Tsamp):
        C1 = C1 + 1 / (Tsamp - 1) * np.outer(h[:, t], h[:, t - 1])
    C2 = np.zeros((n, n))
    for t in range(2, Tsamp):
        C2 = C2 + 1 / (Tsamp - 2) * np.outer(h[:, t], h[:, t - 2])
    Shat = C0 + (1 - 1 / 3) * (C1 + C1.T) + (1 - 2 / 3) * (C2 + C2.T)
    mf = np.asarray(mf, int)
    Ftheta = np.diag(Pstar[np.ix_(mf, mf)] + np.asarray(H, float) * np.ones((len(mf), len(mf))))
    d = Fhat - Ftheta
    return float(0.5 * n * np.log(Tsamp / (2 * np.pi)) - 0.5 * np.log(np.linalg.det(Shat))
                 - 0.5 * Tsamp * d @ np.linalg.solve(Shat, d))
