# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 2012-2025 Dynare Team, https://www.dynare.org), modified in
# 2026: matlab/backward/simul_backward_model.m, simul_backward_nonlinear_model.m, simul_backward_linear_model.m,
# matlab/+backward_model/initialize.m, simul_nonlinear_model.m, simul_linear_model.m, dynamic_model.m (and the
# dynamic_set_auxiliary_series.m that the preprocessor writes, for the auxiliary variables of leads, lags and diff()).
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Simulation of purely backward-looking models (``simul_backward_model``).

Nonlinear models: in every period the model's dynamic equations are solved for the current endogenous variables with
:func:`pydyn.steady.dynare_solve` (the port of Dynare's ``dynare_solve``; ``options_.solve_algo`` as Dynare), starting
from the previous period's values, exactly as ``simul_nonlinear_model.m`` does.  Linear models: one inversion of the
Jacobian block of the current variables, then ``y_t = -A0^{-1} (c + A1 y_{t-1} + B x_t)`` (``simul_linear_model.m``).

The functions here work on numpy arrays.  The statement-level glue (dseries in and out, ``oo_.endo_simul``,
``oo_.exo_simul``) lives in :mod:`pydyn.results_bwd`.

``options_.solve_algo`` 12 and 14 in Dynare solve the equations block by block with the preprocessor's block
decomposition (``setup_time_recursive_block_simul``); pydyn solves the same equations with its port of
``dynare_solve`` (algorithm 14: the block trust-region; algorithm 12: Newton's method, then scipy's hybrid method), so the
paths agree with Dynare's up to the solver tolerance, not iterate by iterate.
"""
from __future__ import annotations

import importlib

import numpy as np

_pf = importlib.import_module(__package__ + ".perfect_foresight")      # the package attribute is a function


class BackwardError(Exception):
    """The simulation cannot be run (the model is not backward, the data are inconsistent, ...)."""


class BackwardResult:
    """``endo``: ``n_endo x (n0 + T)`` (the columns of ``oo_.endo_simul``), ``exo``: ``(n0 + T) x n_exo``
    (``oo_.exo_simul``); ``errorflag`` is True when the solver failed in some period (the columns after the last
    solved period are cut from ``endo``, as Dynare does); ``message`` describes the failure."""

    def __init__(self, endo, exo, errorflag=False, message="", failed_period=None):
        self.endo, self.exo, self.errorflag, self.message, self.failed_period = endo, exo, errorflag, message, failed_period


def check_backward(model):
    """Dynare's tests: no lead; at least one lag (a static model is simulated by simul_static_model)."""
    if model.info["maximum_lead"]:
        raise BackwardError("the model is not backward or static")
    if model.n_exo_det:
        raise BackwardError("simul_backward_model: models with varexo_det are not supported")


def draw_innovations(rng, Sigma_e, T):
    """Dynare's innovations when none are given: ``randn(T, k) * chol(Sigma_e(pos, pos))`` for the shocks with positive
    variance (``k`` of them); the other columns are zero.  ``rng`` is the OctaveRandom stream, which the caller has
    reset to Dynare's default seed if ``options_.bnlms.set_dynare_seed_to_default`` (the default)."""
    Sigma_e = np.asarray(Sigma_e, float)
    nx = Sigma_e.shape[0]
    var = np.diag(Sigma_e)
    pos = np.flatnonzero(var > 0)
    if pos.size == 0:
        return np.zeros((T, nx))
    cov = Sigma_e[np.ix_(pos, pos)]
    try:
        upper = np.linalg.cholesky(cov).T                   # MATLAB's chol returns the upper factor
    except np.linalg.LinAlgError:
        raise BackwardError("chol: the covariance matrix of the structural innovations is not positive definite") from None
    z = rng.randn((T, pos.size))
    out = np.zeros((T, nx))
    out[:, pos] = z @ upper
    return out


def _aux_rows(model):
    """(index, equation row) of each auxiliary variable that is not a Lagrange multiplier."""
    idx = [a["endo_index"] - 1 for a in model.raw.get("aux_vars", []) if a.get("type") != 6]
    if not idx:
        return [], []
    from .optimal_policy import _aux_rows as rows_of
    return idx, [int(r) for r in rows_of(model, idx)]


def aux_series(model, Y, X, params, ss):
    """The auxiliary variables of ``Y`` (``n_endo x n0``, NaN where unknown) computed from the original variables, period
    by period, from their defining equations (what ``dynamic_set_auxiliary_series`` does): in period ``t`` the equations
    of the auxiliary variables, taken at ``(y_{t-1}, y_t)``, are linear in the unknown ``y_t[aux]``; the lagged
    auxiliary values come from the previous columns.  Where a value needs an observation before the sample, it is NaN."""
    idx, rows = _aux_rows(model)
    Y = np.array(Y, float)
    if not idx:
        return Y
    n = model.n_endo
    nan = np.full(n, np.nan)
    for t in range(Y.shape[1]):
        lag = Y[:, t - 1] if t > 0 else nan
        cur = Y[:, t].copy()
        cur[idx] = np.nan
        # one auxiliary variable at a time, in the order of declaration (as the preprocessor's file does): its defining
        # equation is affine in itself; a value that depends on something unknown stays NaN
        for _ in range(2):
            for i, row in zip(idx, rows):
                cur0 = cur[i]
                cur[i] = 0.0 if not np.isfinite(cur0) else cur0
                with np.errstate(all="ignore"):
                    yy = np.concatenate([lag, cur, nan])
                    r = float(model.dynamic_resid(yy, X[t], params, ss)[row])
                    d = float(model.dynamic_jac(yy, X[t], params, ss)[row, n + i])
                cur[i] = cur[i] - r / d if np.isfinite(r) and np.isfinite(d) and d != 0.0 else np.nan
        Y[:, t] = cur
    return Y


def histval_matrices(model, values, params, ss):
    """``M_.endo_histval`` (``n_endo x maximum_lag``) and ``M_.exo_histval`` (``n_exo x maximum_lag``) as the driver the
    preprocessor writes builds them: a series of zeros for the declared variables over ``orig_maximum_lag_with_diffs_expanded``
    periods ending in 0, the values of the ``histval`` block (``values``: ``{(name, lag): value}``), the auxiliary variables
    (``dynamic_set_auxiliary_series``), then the last ``maximum_lag`` periods, NaN set to 0."""
    info = model.info
    n, nx = model.n_endo, model.n_exo
    n_orig = int(model.raw.get("orig_endo_nbr", n))
    L = int(info.get("orig_maximum_lag_with_diffs_expanded", info.get("orig_maximum_lag", 1)))
    ML = max(int(info["maximum_lag"]), 1)
    L = max(L, ML, 1)
    Y = np.full((n, L), np.nan)
    Y[:n_orig] = 0.0
    for a in model.raw.get("aux_vars", []):
        if a.get("type") == 6:                                   # Lagrange multipliers: columns of zeros
            Y[a["endo_index"] - 1] = 0.0
    X = np.zeros((L, nx))
    ie, ix = model._ie, model._ix
    for (name, lag), v in values.items():
        c = L - 1 + int(lag)
        if not 0 <= c < L:
            raise BackwardError(f"histval: {name}({lag}) is outside the {L} periods of initial conditions of the model")
        if name in ie and ie[name] < n_orig:
            Y[ie[name], c] = v
        elif name in ix:
            X[c, ix[name]] = v
    Y = aux_series(model, Y, X, params, ss)
    endo = np.where(np.isnan(Y[:, L - ML:]), 0.0, Y[:, L - ML:])
    exo = X[L - ML:].T.copy()
    return endo, exo


def _dynare_solve_options(model, o):
    algo = o.get("solve_algo")
    return _pf.steady_command_algo(model) if algo is None else int(algo)


def simulate(model, endo0, exo0, T, innovations, params, ss, o, linear=False, rng=None, log=None):
    """The simulation of ``T`` periods after the ``n0`` initial periods ``endo0`` (``n_endo x n0``, NaN for the variables
    without a value) and ``exo0`` (``n0 x n_exo``); ``innovations`` is ``T x n_exo``.  ``o`` holds Dynare's options
    (``solve_algo``, ``dynatol.f``, ``dynatol.x``, ``simul.maxit``; see perfect_foresight._resolve_options)."""
    n, nx = model.n_endo, model.n_exo
    n0 = endo0.shape[1]
    innovations = np.asarray(innovations, float).reshape(T, nx)
    endo = np.full((n, n0 + T), np.nan)
    endo[:, :n0] = endo0
    exo = np.vstack([np.full((n0, nx), np.nan) if exo0 is None else np.asarray(exo0, float).reshape(n0, nx), innovations])
    ss = np.asarray(ss, float)
    if linear:
        return _simulate_linear(model, endo, exo, n0, T, params, ss)
    o = dict(o)
    algo = _dynare_solve_options(model, o)
    if algo in (12, 14):
        # Dynare solves the blocks of the preprocessor's decomposition: the recursive blocks are evaluated, so no
        # random restart of the initial guess (whose entries may be NaN for the auxiliary variables) takes place
        o["solve_algo"], o["_rng"] = 14, None
    else:
        o["solve_algo"], o["_rng"] = algo, rng
    nan = np.full(n, np.nan)
    for it in range(n0, n0 + T):
        y_ = endo[:, it - 1]
        try:
            with np.errstate(all="ignore"):
                z, ok = _pf._solve_period(model, y_, y_, nan, exo[it], params, ss, o)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            z, ok = None, False
            msg = str(exc)
        else:
            msg = f"Nonlinear solver routine failed in period {it - n0 + 1}."
        if not ok:
            if log:
                log(f"Newton failed on iteration i = {it - n0 + 1}: {msg}")
            return BackwardResult(endo[:, :it], exo, True, msg, it - n0 + 1)
        endo[:, it] = z
    return BackwardResult(endo, exo)


def _simulate_linear(model, endo, exo, n0, T, params, ss):
    n, nx = model.n_endo, model.n_exo
    y = np.concatenate([np.zeros(2 * n), np.full(n, np.nan)])
    x = np.zeros(nx)
    with np.errstate(all="ignore"):
        cst = np.asarray(model.dynamic_resid(y, x, params, ss), float)
        jac = model.dynamic_jac(y, x, params, ss)
    try:
        A0inv = np.linalg.inv(jac[:, n:2 * n])
    except np.linalg.LinAlgError:
        return BackwardResult(endo, exo, True, "the Jacobian of the current variables is singular")
    A1 = jac[:, :n]
    B = jac[:, 3 * n:]
    for it in range(n0, n0 + T):
        endo[:, it] = -_sparse_dot(A0inv, cst + _sparse_dot(A1, endo[:, it - 1]) + _sparse_dot(B, exo[it]))
    return BackwardResult(endo, exo)


def _sparse_dot(A, v):
    """``A * v`` as MATLAB/Octave compute it for a sparse ``A`` (Dynare's Jacobian is sparse): the structural zeros of
    ``A`` are not multiplied, so a NaN in ``v`` (an auxiliary variable whose lag lies before the sample) does not spread
    through the zeros of a row."""
    with np.errstate(all="ignore"):
        return np.where(A != 0.0, A * v[None, :], 0.0).sum(axis=1)


def static_simulate(model, exo, params, o, rng=None):
    """``backward_model.simul_static_model``: in each period the static model is solved from zeros with dynare_solve;
    a period that fails is NaN.  Returns ``(endo n x T, failed periods)``."""
    from .steady import dynare_solve
    n, T = model.n_endo, exo.shape[0]
    algo = _dynare_solve_options(model, o)
    if algo not in _pf.PORTED_SOLVE_ALGOS:
        raise BackwardError(f"solve_algo={algo} is not available in pydyn's port of dynare_solve")
    endo = np.full((n, T), np.nan)
    failed = []
    for t in range(T):
        def fun(y, t=t):
            return np.asarray(model.static_resid(y, exo[t], params), float), np.asarray(model.static_jac(y, exo[t], params), float)
        with np.errstate(all="ignore"):
            z, err, _ = dynare_solve(fun, np.zeros(n), o["dynatol.f"], o["dynatol.x"], o["simul.maxit"], algo, stream=rng)
        if err:
            failed.append(t + 1)
        else:
            endo[:, t] = z
    return endo, failed


def invert(model, constraints, cnames, exo, ex_names, free, y_init, params, ss, o, static=False, rng=None):
    """``backward_model.inversion`` (or ``static_model_inversion`` when ``static``).  ``constraints`` is the ``T x k`` data of
    the controlled variables ``cnames``; ``exo`` the ``Tx x n_exo`` exogenous data (all variables, in the order of
    ``model.exo_names``) whose first row is the period of the first constraint; ``free`` the names of the free innovations
    (in the order of the constraints); ``y_init`` the endogenous variables the period before the first constraint (unused
    for a static model).  Returns ``(Y n_endo x T, X T x n_exo)``."""
    from .steady import dynare_solve
    n, nx = model.n_endo, model.n_exo
    nfree_x = len(free)
    x_free = (np.arange(nx) if nfree_x >= nx else np.array([model.exo_names.index(f) for f in free], int))
    if nfree_x < n:
        y_ctrl = np.array([model.endo_names.index(c) for c in cnames], int)
        y_free = np.setdiff1d(np.arange(n), y_ctrl)
    else:
        y_ctrl, y_free = np.arange(n), np.zeros(0, int)
    nyfree, nyctrl = len(y_free), len(y_ctrl)
    J_id = np.concatenate([n + y_free, 3 * n + x_free]).astype(int)
    T = constraints.shape[0]
    Y = np.full((n, T if static else T + 1), np.nan)
    if not static:
        Y[:, 0] = y_init
    off = 0 if static else 1
    for i in range(nyctrl):
        Y[y_ctrl[i], off:] = constraints[:, i]
    X = np.array(exo, float)
    algo = 4 if static else _dynare_solve_options(model, o)
    if algo not in _pf.PORTED_SOLVE_ALGOS:
        raise BackwardError(f"solve_algo={algo} is not available in pydyn's port of dynare_solve")
    for t in range(T):
        ity = t + off
        ycur = Y[y_ctrl, ity]
        if static:
            def fun(z):
                y = np.full(3 * n, np.nan)
                y[n + y_ctrl] = ycur
                if nyfree:
                    y[n + y_free] = z[:nyfree]
                x = X[t].copy()
                x[x_free] = z[nyfree:]
                return (np.asarray(model.dynamic_resid(y, x, params, ss), float),
                        np.asarray(model.dynamic_jac(y, x, params, ss), float)[:, J_id])
            z0 = np.concatenate([Y[y_free, ity], np.zeros(nfree_x)])
        else:
            ylag = Y[:, ity - 1]

            def fun(z):
                y = np.concatenate([ylag, np.zeros(2 * n)])
                y[n + y_ctrl] = ycur
                if nyfree:
                    y[n + y_free] = z[:nyfree]
                x = X[t].copy()
                x[x_free] = z[nyfree:]
                return (np.asarray(model.dynamic_resid(y, x, params, ss), float),
                        np.asarray(model.dynamic_jac(y, x, params, ss), float)[:, J_id])
            z0 = np.concatenate([Y[y_free, ity - 1], np.zeros(nfree_x)])
        with np.errstate(all="ignore"):
            z, err, _ = dynare_solve(fun, z0, o["dynatol.f"], o["dynatol.x"], o["simul.maxit"], algo, stream=rng)
        if err:
            raise BackwardError("Nonlinear solver failed" if not static else "Enable to solve the system of equations")
        X[t, x_free] = z[nyfree:]
        if nyfree:
            Y[y_free, ity] = z[:nyfree]
    return (Y if static else Y[:, 1:]), X
