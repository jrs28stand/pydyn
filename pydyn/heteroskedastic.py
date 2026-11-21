# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 1996-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: matlab/kalman/get_Qvec_heteroskedastic_filter.m and the
# "set heteroskedastic shocks" section of matlab/estimation/dynare_estimation_init.m.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""``heteroskedastic_shocks`` blocks and the ``heteroskedastic_filter`` estimation option.

With ``heteroskedastic_filter``, the covariance matrix of the structural shocks changes over the
sample: ``Qvalue`` (``values``: standard deviations) replaces the variance of a shock in given
periods and ``Qscale`` (``scales``) multiplies its standard deviation. Periods are those of the .mod
file, counted from ``first_obs`` (Dynare's ``options_.first_obs``); the arrays have ``nobs + 1``
columns. The Kalman filter uses ``Qvec[:, :, t + 1]`` in the prediction step from period ``t`` to
``t + 1`` (1-based ``t``), never switches to the steady-state filter, and the initial covariance uses
the constant ``Sigma_e``.
"""
from __future__ import annotations

import numpy as np

from .dates import Date, is_date_value


def blocks(model):
    """``(Qvalue_orig, Qscale_orig)``: lists of ``(exo index, first period, last period, value)`` in the
    order of the .mod file; a block with ``overwrite`` discards the earlier ones."""
    values, scales = [], []
    for st in model.statements:
        if st.get("statementName") != "heteroskedastic_shocks":
            continue
        if st.get("overwrite"):
            values, scales = [], []
        for key, sub, out in (("shocks_values", "values", values), ("shocks_scales", "scales", scales)):
            for d in st.get(key, []):
                if d["var"] not in model._ix:
                    raise ValueError(f"heteroskedastic_shocks: {d['var']} is not a stochastic shock")
                for v in d[sub]:
                    p1, p2 = (Date.parse(x) if is_date_value(x) else int(x) for x in (v["period1"], v["period2"]))
                    out.append((model._ix[d["var"]], p1, p2, float(model._value(v["value"]))))
    return values, scales


def has_blocks(model) -> bool:
    return any(st.get("statementName") == "heteroskedastic_shocks" for st in model.statements)


def q_arrays(model, nobs, first_obs=1, first_date=None):
    """``M_.heteroskedastic_shocks.Qvalue`` and ``Qscale`` (n_exo, nobs + 1), NaN where not set.
    Periods given as dates are located with ``first_date``, the date of the first observation used
    (pydyn.dates; Dynare: ``dataset_.dates(1)``)."""
    values, scales = blocks(model)
    Qvalue = np.full((model.n_exo, nobs + 1), np.nan)
    Qscale = np.full((model.n_exo, nobs + 1), np.nan)
    for out, entries in ((Qvalue, values), (Qscale, scales)):
        for i, p1, p2, v in entries:
            if isinstance(p1, Date):
                if first_date is None:
                    raise ValueError("heteroskedastic_shocks: periods are dates but the data have none")
                if p1.freq != first_date.freq:
                    raise ValueError('heteroskedastic_shocks: dates in "periods" do not have the same frequency as '
                                     'those in the dataset')
                k = np.arange(p1 - first_date, p2 - first_date + 1)          # 0-based
                k = k[(k >= 0) & (k <= nobs - 1)]
            else:
                periods = np.arange(p1, p2 + 1)
                periods = periods[(periods >= first_obs) & (periods <= nobs + first_obs)]
                k = periods - first_obs
            out[i, k] = v ** 2
    return Qvalue, Qscale


def qvec(Q, Qvalue, Qscale):
    """``get_Qvec_heteroskedastic_filter.m``: (n_exo, n_exo, nobs + 1)."""
    Q = np.asarray(Q, float)
    k_n = Qvalue.shape[1]
    isqdiag = bool(np.all(np.abs(Q - np.diag(np.diag(Q))) < 1e-14))
    out = np.repeat(Q[:, :, None], k_n, axis=2)
    for k in range(k_n):
        Qk = out[:, :, k]
        inx = ~np.isnan(Qvalue[:, k])
        if inx.any():
            if isqdiag:
                ii = np.flatnonzero(inx)
                Qk[np.ix_(ii, ii)] = np.diag(Qvalue[ii, k])
            else:
                for s in np.flatnonzero(inx):
                    if Q[s, s] > 1e-14:
                        tmpscale = np.sqrt(Qvalue[s, k] / Q[s, s])
                        Qk[s, :] = Qk[s, :] * tmpscale
                        Qk[:, s] = Qk[:, s] * tmpscale
                    else:
                        Qk[s, s] = Qvalue[s, k]
        inx = ~np.isnan(Qscale[:, k])
        if inx.any():
            if isqdiag:
                ii = np.flatnonzero(inx)
                Qk[np.ix_(ii, ii)] = Qk[np.ix_(ii, ii)] * np.diag(Qscale[ii, k])
            else:
                for s in np.flatnonzero(inx):
                    tmpscale = np.sqrt(Qscale[s, k])
                    Qk[s, :] = Qk[s, :] * tmpscale
                    Qk[:, s] = Qk[:, s] * tmpscale
    return out


def rqr(R, Qv):
    """``R Q_t R'`` for every slice, (n_states, n_states, nobs + 1)."""
    return np.einsum("ij,jkt,lk->ilt", R, Qv, R)
