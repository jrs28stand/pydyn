# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2: matlab/get_companion_matrix.m, get_difference_order.m, get_aux_variable_id.m,
# matlab/+var_expectation/update_parameters.m (Copyright (C) 2018-2025 Dynare Team, GPL-3.0-or-later).
"""Companion matrices of VAR and trend component models, and the reduced form parameters of VAR expectations.

Pure numerics on plain arrays: the interface with ``M_``/``oo_`` is :mod:`pydyn.results_var`."""
from __future__ import annotations

import numpy as np


class VarModelError(Exception):
    """A MATLAB ``error()`` of Dynare's functions (the message is Dynare's)."""


class AuxVars:
    """``M_.aux_vars`` (from the JSON of the preprocessor), looked up by the 1-based index of the endogenous variable."""

    def __init__(self, raw_aux, orig_endo_nbr):
        self.orig = int(orig_endo_nbr)
        self.by_endo = {int(a["endo_index"]): a for a in raw_aux}

    def info(self, endo_index):
        """``M_.aux_vars(get_aux_variable_id(var))`` or None (get_aux_variable_id returns 0)."""
        if endo_index <= self.orig:
            return None
        return self.by_endo.get(int(endo_index))

    def difference_order(self, endo_index):
        """get_difference_order.m"""
        n = 0
        a = self.info(endo_index)
        while a is not None:
            if a["type"] == 8:
                n += 1
                v = int(a["orig_index"])
                a = None if v <= self.orig else self.info(v)
            elif a["type"] == 9:
                a = self.info(int(a["orig_index"]))
            else:
                break
        return n


def companion_var(AR, constant):
    """The companion matrix of a VAR(p): [1 0; c H] with a constant, H without.  ``AR`` is n x n x p, ``constant`` n."""
    n, _, p = AR.shape
    isconstant = bool(np.any(np.abs(constant) > 0))
    k = int(isconstant)
    C = np.zeros((n * p + k, n * p + k))
    C[k:k + n, k:k + n] = AR[:, :, 0]
    for i in range(2, p + 1):
        C[k:k + n, k + (i - 1) * n:k + i * n] = AR[:, :, i - 1]
        C[k + (i - 1) * n:k + i * n, k + (i - 2) * n:k + (i - 1) * n] = np.eye(n)
    if isconstant:
        C[0, 0] = 1
        for i in range(n):
            C[1 + i, 0] = constant[i]
    return C, isconstant


def companion_trend_component(AR, A0, A0star, n, target_eqn, targets):
    """The companion matrix of a trend component model (VAR(p+1) in levels of [X; Z]).

    ``AR`` (m x m x p), ``A0`` (m x m), ``A0star`` (m x q) are those of the error correction equations;
    ``target_eqn`` are the (1-based) equation numbers of the trends in the model, ``targets`` the boolean vector."""
    p = AR.shape[2]
    q = int(np.sum(targets))
    m = n - q
    tg = np.asarray(target_eqn, int) - 1
    ec = np.flatnonzero(~np.asarray(targets, bool))
    B = np.zeros((m + q, m + q, p + 1))
    B[np.ix_(ec, ec, [0])] = (np.eye(m) + A0 + AR[:, :, 0])[:, :, None]
    # MATLAB: B(ecm, targets) = -A0star assigns to the first page of B
    B[np.ix_(ec, tg, [0])] = (-np.asarray(A0star).reshape(m, q))[:, :, None]
    B[np.ix_(tg, tg, [0])] = np.eye(q)[:, :, None]
    for i in range(2, p + 1):
        B[np.ix_(ec, ec, [i - 1])] = (AR[:, :, i - 1] - AR[:, :, i - 2])[:, :, None]
    B[np.ix_(ec, ec, [p])] = (-AR[:, :, p - 1])[:, :, None]
    C = np.zeros((n * (p + 1), n * (p + 1)))
    for i in range(1, p + 1):
        C[0:n, (i - 1) * n:i * n] = B[:, :, i - 1]
        C[i * n:(i + 1) * n, (i - 1) * n:i * n] = np.eye(n)
    C[0:n, p * n:(p + 1) * n] = B[:, :, p]
    return C, B


def _isint(x):
    return float(x) == round(float(x))


def var_expectation_parameters(C, alpha, discount, horizon, timeshift):
    """The reduced form parameters of a VAR expectation (var_expectation.update_parameters.m): the row vector
    ``alpha`` (weights of the expression on the state vector of the companion form) times the discounted sum of
    powers of the companion matrix ``C``."""
    n = C.shape[0]
    h = np.atleast_1d(np.asarray(horizon, float))
    bad = True
    if h.size == 1:
        bad = not (np.isfinite(h[0]) and _isint(h[0]) and h[0] > 0)
    elif h.size == 2:
        if np.isfinite(h[0]) and _isint(h[0]) and h[0] >= 0:
            if np.isinf(h[1]) or (_isint(h[1]) and h[1] > h[0]):
                bad = False
    if bad:
        raise VarModelError("horizon must be an integer scalar or an integer vector with two elements.")
    mp = np.linalg.matrix_power
    shift = int(-timeshift)
    if h.size == 1:
        hh = int(h[0])
        if hh == 1:
            par = discount * (alpha @ C)
        else:
            par = alpha @ mp(discount * C, hh)
        if timeshift < 0:
            par = par @ mp(C, shift)
        return par
    eye = np.eye(n)
    if h[0] == 0 and np.isinf(h[1]):
        par = np.linalg.solve((eye - discount * C).T, alpha.T).T          # alpha/(eye-discount*C)
        if timeshift < 0:
            par = par @ mp(C, shift)
        return par
    D = discount * C
    if h[0] > 0 and np.isinf(h[1]):
        tmp1 = np.zeros((n, n))
        for k in range(int(h[0])):
            tmp1 = tmp1 + mp(D, k)
        tmp1 = alpha @ tmp1
        if timeshift < 0:
            tmp1 = tmp1 @ mp(C, shift)
        tmp2 = np.linalg.solve((eye - D).T, alpha.T).T
        if timeshift < 0:
            tmp2 = tmp2 @ mp(C, shift)
        return tmp2 - tmp1
    tmp = np.zeros((n, n))
    for k in range(int(h[0]), int(h[1]) + 1):
        tmp = tmp + mp(D, k)
    par = alpha @ tmp
    if timeshift < 0:
        par = par @ mp(C, shift)
    return par


# ---------------------------------------------------------------------------------------- get_ar_ec_matrices.m
def _find_lhs_in_aux(aux, auxvar, lhs):
    if auxvar <= aux.orig:
        return -1
    av = aux.by_endo[int(auxvar)]
    if av["type"] in (8, 10):
        if av["endo_index"] in lhs:
            return int(av["endo_index"])
        return _find_lhs_in_aux(aux, av["orig_index"], lhs)
    if av["orig_index"] in lhs:
        return int(av["orig_index"])
    return _find_lhs_in_aux(aux, av["orig_index"], lhs)


def _find_var_no_lag(aux, auxvar):
    if auxvar <= aux.orig:
        raise VarModelError("Shouldn't arrive here")
    av = aux.by_endo[int(auxvar)]
    if av["type"] in (8, 10):
        return int(av["endo_index"])
    if av["orig_index"] <= aux.orig:
        return int(av["orig_index"])
    return _find_var_no_lag(aux, av["orig_index"])


def _find_lag_for_var(aux, auxvar, lag, ndiffs, rhsvars):
    if auxvar <= aux.orig:
        assert lag > 0
        return lag, ndiffs
    av = aux.by_endo[int(auxvar)]
    if av["type"] == 8:
        ndiffs += 1
    olag = abs(av.get("orig_lead_lag", 0))
    if av["endo_index"] in rhsvars:
        if av["type"] in (8, 9):
            lag = lag + olag
    elif av["orig_index"] in rhsvars:
        if av["orig_index"] <= aux.orig:
            lag = lag + olag
        else:
            lag, ndiffs = _find_lag_for_var(aux, av["orig_index"], lag + max(1, olag), ndiffs, rhsvars)
    else:
        if av["type"] == 8:
            lag, ndiffs = _find_lag_for_var(aux, av["orig_index"], lag, ndiffs, rhsvars)
        else:
            lag, ndiffs = _find_lag_for_var(aux, av["orig_index"], lag + max(1, olag), ndiffs, rhsvars)
    assert lag > 0
    return lag, ndiffs


def ar_ec_matrices(g1, lhs, vars_at_eq, max_lag, aux, endo_nbr):
    """get_ar_ec_matrices.m: ``g1`` = -(Jacobian rows of the equations of the model at ones), ``lhs`` the (1-based)
    left-hand side variables, ``vars_at_eq`` a list of (vars, lags) per equation.  Returns (ar, ec, ar_idx, ec_idx)."""
    lhs = [int(v) for v in lhs]
    ecRhsVars = []
    rhs = []
    for vars_, lags_ in vars_at_eq:
        r = dict(vars=[int(v) for v in vars_], lags=[int(v) for v in lags_], ar=[], ec=[], ecv=[])
        for v in r["vars"]:
            if v <= aux.orig:
                if v in lhs:
                    r["ar"].append(lhs.index(v) + 1)
                    r["ec"].append(-1)
                    r["ecv"].append(-1)
                else:
                    ecRhsVars = sorted(set(ecRhsVars) | {v})
                    r["ecv"].append(v)
                    r["ar"].append(-1)
                    r["ec"].extend([k + 1 for k, x in enumerate(r["ecv"]) if x == v])
            else:
                li = _find_lhs_in_aux(aux, v, lhs)
                if li >= 1:
                    r["ar"].append(lhs.index(li) + 1 if li in lhs else 0)
                    r["ec"].append(-1)
                    r["ecv"].append(-1)
                else:
                    vi = _find_var_no_lag(aux, v)
                    ecRhsVars = sorted(set(ecRhsVars) | {vi})
                    r["ecv"].append(vi)
                    r["ar"].append(-1)
                    r["ec"].extend([k + 1 for k, x in enumerate(r["ecv"]) if x == vi])
        rhs.append(r)
    # reorderECvars
    reordered = []
    for i in range(len(lhs)):
        av = aux.by_endo.get(lhs[i])
        if av is not None:
            var = [x for x in ecRhsVars if x == av["orig_index"]]
            if var:
                reordered.append(var[0])
                for r in rhs:
                    for k, x in enumerate(r["ecv"]):
                        if x == var[0]:
                            r["ec"][k] = len(reordered)
    ecRhsVars = reordered
    nl = len(lhs)
    ar = np.zeros((nl, nl, int(max(max_lag))))
    ec = np.zeros((nl, len(ecRhsVars), 1))
    for i in range(nl):
        r = rhs[i]
        for j, var in enumerate(r["vars"]):
            col = var if r["lags"][j] == -1 else var + endo_nbr
            if np.any(g1[:, col - 1]):
                if r["ar"][j] > 0:
                    lag, _ = _find_lag_for_var(aux, var, -r["lags"][j], 0, lhs)
                    ar[i, r["ar"][j] - 1, lag - 1] += g1[i, col - 1]
                elif r["ec"][j] > 0:
                    lag, _ = _find_lag_for_var(aux, var, -r["lags"][j], 0, ecRhsVars)
                    if lag == 1:
                        ec[i, r["ec"][j] - 1, lag - 1] += g1[i, col - 1]
                else:
                    raise VarModelError("Shouldn't arrive here")
    return ar, ec, np.array(lhs, float), np.array(ecRhsVars, float)
