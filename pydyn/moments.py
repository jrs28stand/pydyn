# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 2001-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: matlab/moments/{th_autocovariances, disp_th_moments,
# disp_th_moments_pruned_state_space, disp_moments, conditional_variance_decomposition,
# display_*_variance_decomposition, get_filtered_time_series, add_filter_subtitle,
# check_measurement_error_requested_vars, compute_variance_decomposition}.m,
# matlab/matrix_solver/lyapunov_symm.m, matlab/get_lower_cholesky_covariance.m,
# matlab/+pruned_SS/{pruned_state_space_system, commutation, prodmom}.m (orders 2 and 3), matlab/sample_hp_filter.m,
# matlab/one_sided_hp_filter.m, the Baxter-King filter of dseries and mex/sources/disclyap_fast.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Moments of a solved DSGE model, with Dynare's definitions, defaults and conventions.

* theoretical moments at first order (``th_autocovariances``), including unit roots (variables that
  load on a unit-root Schur vector get NaN moments), HP and band-pass filtered moments computed in
  the frequency domain, unconditional and conditional variance decompositions, optional
  measurement errors;
* second order: the plain ``order=2`` moments (first-order covariances plus the ``ghs2`` mean
  correction) and the pruned-state-space moments (``pruning``); third order: pruned-state-space
  moments only (``order=3, pruning``), as in Dynare, without variance decompositions;
* moments of a simulated path (``disp_moments``), with the HP, one-sided HP and Baxter-King filters;
* ``stoch_simul_moments``: everything ``stoch_simul`` prints for a variable list, and
  ``Moments.format()`` to print it in a layout similar to Dynare's.

Conventions that follow Dynare and that are easy to miss:

* ``gamma_y`` is Dynare's cell array as a list: ``gamma_y[0]`` the variance matrix, ``gamma_y[i]`` the
  autocorrelation matrix of order ``i`` (entry ``(a, b)`` is corr(y_a,t, y_b,t-i)), ``gamma_y[ar+1]`` the
  variance decomposition (fractions) and, for ``order=2`` without pruning, ``gamma_y[ar+2]`` the mean
  correction.
* shocks with a non-diagonal ``Sigma_e`` are orthogonalised with the lower Cholesky factor in
  declaration order, so the variance decomposition depends on the order of ``varexo``.
* the unconditional variance decomposition of a model with one shock is returned with one row per
  requested variable (Dynare keeps one row per endogenous variable).
* simulated moments drop ``drop`` rows of the simulated path (Dynare's ``oo_.endo_simul`` has no
  initial-value column) and divide by ``T`` (variance, skewness, kurtosis) or use ``std`` with ``T-1``
  in the autocorrelation denominators, exactly as ``disp_moments.m`` does.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve

from .perturbation import QZ_CRITERIUM, dr_ordering


# ------------------------------------------------------------------------------------ linear algebra
def lyapunov_symm(a, b, qz_criterium=QZ_CRITERIUM, complex_threshold=1e-15):
    """Solve ``x = a x a' + b`` as ``lyapunov_symm.m`` (method 0, Schur based).

    Eigenvalues of ``a`` with modulus above ``2 - qz_criterium`` (unit roots, but also explosive
    roots) are projected out: the solution lives on the stable Schur subspace. Returns ``(x, u)``
    where ``u`` holds the Schur vectors of the removed roots (shape ``(n, 0)`` if none)."""
    a = np.atleast_2d(np.asarray(a, float))
    b = np.atleast_2d(np.asarray(b, float))
    n0 = a.shape[0]
    if n0 == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    if n0 == 1:                       # lyapunov_symm does not look for unit roots in this case
        return b / (1.0 - a * a), np.zeros((1, 0))
    thr = 2.0 - qz_criterium
    T, U, k = linalg.schur(a, output="real", sort=lambda re, im: abs(complex(re, im)) > thr)
    n = n0 - k
    u = U[:, :k]
    if n == 0:
        return np.zeros((n0, n0)), u
    T = T[k:, k:]
    U2 = U[:, k:]
    B = U2.T @ b @ U2
    x = np.zeros((n, n))
    i = n                                 # 1-based row index as in the MATLAB code
    while i >= 2:
        if abs(T[i - 1, i - 2]) < complex_threshold:
            if i == n:
                c = np.zeros(n)
            else:
                c = T[:i, :] @ (x[:, i:] @ T[i - 1, i:]) + T[i - 1, i - 1] * (T[:i, i:] @ x[i:, i - 1])
            q = np.eye(i) - T[:i, :i] * T[i - 1, i - 1]
            x[:i, i - 1] = np.linalg.solve(q, B[:i, i - 1] + c)
            x[i - 1, :i - 1] = x[:i - 1, i - 1]
            i -= 1
        else:
            if i == n:
                c = np.zeros(n)
                c1 = np.zeros(n)
            else:
                c = (T[:i, :] @ (x[:, i:] @ T[i - 1, i:]) + T[i - 1, i - 1] * (T[:i, i:] @ x[i:, i - 1])
                     + T[i - 1, i - 2] * (T[:i, i:] @ x[i:, i - 2]))
                c1 = (T[:i, :] @ (x[:, i:] @ T[i - 2, i:]) + T[i - 2, i - 2] * (T[:i, i:] @ x[i:, i - 2])
                      + T[i - 2, i - 1] * (T[:i, i:] @ x[i:, i - 1]))
            Ti = T[:i, :i]
            q = np.block([[np.eye(i) - Ti * T[i - 1, i - 1], -Ti * T[i - 1, i - 2]],
                          [-Ti * T[i - 2, i - 1], np.eye(i) - Ti * T[i - 2, i - 2]]])
            z = np.linalg.solve(q, np.concatenate([B[:i, i - 1] + c, B[:i, i - 2] + c1]))
            x[:i, i - 1] = z[:i]
            x[:i, i - 2] = z[i:]
            x[i - 1, :i - 1] = x[:i - 1, i - 1]
            x[i - 2, :i - 2] = x[:i - 2, i - 2]
            i -= 2
    if i == 1:
        c = T[0, :] @ (x[:, 1:] @ T[0, 1:]) + T[0, 0] * (T[0, 1:] @ x[1:, 0])
        x[0, 0] = (B[0, 0] + c) / (1.0 - T[0, 0] * T[0, 0])
    return U2 @ x @ U2.T, u


def disclyap_fast(g, v, tol=1e-16, max_iter=2000):
    """Doubling algorithm of Dynare's ``disclyap_fast`` MEX file. Returns ``(x, errorflag)``."""
    P0, A0 = np.array(v, float), np.array(g, float)
    matd, it = 1.0, 1
    with np.errstate(all="ignore"):
        while matd > tol and it < max_iter:
            P1 = P0 + A0 @ P0 @ A0.T
            A1 = A0 @ A0
            matd = np.max(np.abs(P1 - P0))
            P0, A0 = P1, A1
            it += 1
    if it == max_iter or np.isnan(matd):
        return np.full(P0.shape, np.nan), True
    return (P0 + P0.T) / 2.0, False


def get_lower_cholesky_covariance(Sigma_e, add_tiny_number_to_cholesky=1e-14):
    """Lower Cholesky factor that ignores zero-variance shocks (``get_lower_cholesky_covariance.m``)."""
    S = np.asarray(Sigma_e, float)
    nz = np.nonzero(np.sqrt(np.diag(S)) != 0)[0]
    out = np.zeros_like(S)
    sub = S[np.ix_(nz, nz)]
    try:
        out[np.ix_(nz, nz)] = np.linalg.cholesky(sub)
    except np.linalg.LinAlgError:
        print("Non-positive definite covariance matrix encountered. "
              "Using add_tiny_number_to_cholesky one the diagonal.")
        out[np.ix_(nz, nz)] = np.linalg.cholesky(sub + add_tiny_number_to_cholesky * np.eye(len(nz)))
    return out


def _commutation(n):
    """K with K kron(a, b) = kron(b, a) for a, b in R^n."""
    K = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            K[j * n + i, i * n + j] = 1.0
    return K


def _vecF(a):
    return np.asarray(a).ravel(order="F")


def _intersect_stable(a, b):
    """MATLAB ``[c, ia, ib] = intersect(a, b, 'stable')`` (0-based indices, first occurrences)."""
    c, ia, ib, seen = [], [], [], set()
    for i, v in enumerate(a):
        if v in seen:
            continue
        seen.add(v)
        if v in b:
            c.append(v)
            ia.append(i)
            ib.append(list(b).index(v))
    return np.array(c, int), np.array(ia, int), np.array(ib, int)


# ------------------------------------------------------------------------------------ filters
def hp_filter(y, lamb=1600.0):
    """``sample_hp_filter.m``: returns ``(trend, cycle)`` for a (T, n) array."""
    y = np.asarray(y, float)
    y2 = y.reshape(len(y), -1)
    T = y2.shape[0]
    s = float(lamb)
    D = sparse.diags([np.full(T - 2, s), np.full(T - 1, -4.0 * s), np.full(T, 1.0 + 6.0 * s),
                      np.full(T - 1, -4.0 * s), np.full(T - 2, s)], [-2, -1, 0, 1, 2], format="lil")
    D[0, 0] = 1.0 + s
    D[T - 1, T - 1] = 1.0 + s
    D[0, 1] = D[1, 0] = -2.0 * s
    D[T - 2, T - 1] = D[T - 1, T - 2] = -2.0 * s
    D[1, 1] = 1.0 + 5.0 * s
    D[T - 2, T - 2] = 1.0 + 5.0 * s
    D = D.tocsc()
    trend = np.column_stack([spsolve(D, y2[:, j]) for j in range(y2.shape[1])])
    return trend.reshape(y.shape), (y2 - trend).reshape(y.shape)


def one_sided_hp_filter(y, lamb=1600.0):
    """``one_sided_hp_filter.m`` (Kalman filter of the local linear trend), default initialisation."""
    y = np.asarray(y, float)
    y2 = y.reshape(len(y), -1)
    T, n = y2.shape
    F = np.array([[2.0, -1.0], [1.0, 0.0]])
    H = np.array([1.0, 0.0])
    Q = np.array([[1.0 / lamb, 0.0], [0.0, 0.0]])
    R = 1.0
    trend = np.full((T, n), np.nan)
    for k in range(n):
        x = np.array([2 * y2[0, k] - y2[1, k], 3 * y2[0, k] - 2 * y2[1, k]])
        P = 1e5 * np.eye(2)
        for j in range(T):
            S = H @ P @ H + R
            K = (F @ P @ H) / S
            x = F @ x + K * (y2[j, k] - H @ x)
            Tm = F - np.outer(K, H)
            P = Tm @ P @ Tm.T + Q + R * np.outer(K, K)
            trend[j, k] = x[1]
    return trend.reshape(y.shape), (y2 - trend).reshape(y.shape)


def baxter_king_filter(y, high_frequency=6, low_frequency=32, K=12):
    """Baxter-King filter as dseries' ``baxter_king_filter``: returns the ``T - 2K`` filtered rows."""
    y = np.asarray(y, float)
    y2 = y.reshape(len(y), -1)
    hf, lf = 2.0 * np.pi / high_frequency, 2.0 * np.pi / low_frequency
    lp = np.arange(1, K + 1)
    w = np.zeros(K + 1)
    w[1:] = (np.sin(lp * hf) - np.sin(lp * lf)) / (lp * np.pi)
    w[0] = (hf - lf) / np.pi
    target = 1.0 if low_frequency > 1000 else 0.0
    w = w + (target - (w[0] + 2 * w[1:].sum())) / (2 * K + 1)
    w = np.concatenate([w[:0:-1], w])
    T = y2.shape[0]
    out = np.empty((T - 2 * K, y2.shape[1]))
    for t in range(K, T - K):
        out[t - K] = w @ y2[t - K:t + K + 1]
    return out.reshape((T - 2 * K,) + y.shape[1:])


# ------------------------------------------------------------------------------------ options
_DEFAULTS = dict(order=1, ar=5, hp_filter=0, one_sided_hp_filter=0, bandpass=False, passband=(6, 32), bandpass_K=12,
                 filtered_theoretical_moments_grid=512, nodecomposition=False, nocorr=False, pruning=False,
                 periods=0, drop=100, conditional_variance_decomposition=(), contemporaneous_correlation=False,
                 qz_criterium=QZ_CRITERIUM, schur_vec_tol=1e-11, lyapunov_complex_threshold=1e-15,
                 lyapunov_doubling_tol=1e-16, add_tiny_number_to_cholesky=1e-14)
_FROM_MOD = {"bandpass.indicator": "bandpass", "bandpass.passband": "passband"}


def _options(m, overrides):
    """Dynare defaults, then the ``stoch_simul`` options of the .mod file, then ``overrides``."""
    o = dict(_DEFAULTS)
    for k, v in (getattr(m, "options", {}) or {}).get("stoch_simul", {}).items():
        k = _FROM_MOD.get(k, k)
        if k in o:
            o[k] = v
    if "bandpass_filter" in overrides:                 # Dynare option name: bandpass_filter=[high low]
        bp = overrides.pop("bandpass_filter")
        if bp is not None and bp is not False:
            o["bandpass"] = True
            if bp is not True:
                o["passband"] = tuple(bp)
    for k, v in overrides.items():
        if k == "steps":
            k = "conditional_variance_decomposition"
        if k not in o:
            raise TypeError(f"unknown option {k!r}")
        if v is not None:
            o[k] = v
    cv = o["conditional_variance_decomposition"]
    o["conditional_variance_decomposition"] = [int(x) for x in np.atleast_1d(cv)] if np.size(cv) else []
    o["passband"] = tuple(np.ravel(o["passband"]))
    o["hp_filter"] = float(o["hp_filter"])
    o["one_sided_hp_filter"] = float(o["one_sided_hp_filter"])
    o["ar"] = int(o["ar"])
    nf = sum([bool(o["hp_filter"]), bool(o["one_sided_hp_filter"]), bool(o["bandpass"])])
    if nf > 1:
        raise ValueError("can only use one of HP, one-sided HP, and bandpass filters")
    return o


def _filter_name(o):
    if o["hp_filter"]:
        return "hp"
    if o["one_sided_hp_filter"]:
        return "one-sided hp"
    if o["bandpass"]:
        return "bandpass"
    return ""


def _default_var_list(m):
    for st in m.statements:
        if st.get("statementName") == "stoch_simul" and st.get("symbol_list"):
            return list(st["symbol_list"])
    return list(m.orig_endo_names)


def _var_indices(m, var_list):
    names = list(m.endo_names)
    if var_list is None:
        var_list = _default_var_list(m)
    idx, seen = [], set()
    for v in var_list:
        if v not in names:
            raise ValueError(f"The following symbols are not endogenous variables: {v}")
        if v in seen:
            warnings.warn(f"{v} is specified twice in the variable list and is considered only once")
            continue
        seen.add(v)
        idx.append(names.index(v))
    return np.array(idx, int)


# ------------------------------------------------------------------------------------ decision rules
class _Rules:
    """Decision-rule arrays in Dynare's layout (rows in DR order)."""

    def __init__(self, m, dr):
        if hasattr(dr, "as_dr"):
            dr = dr.as_dr()
        self.dr = dr
        self.ys = np.asarray(dr["ys"], float)
        self.ghx = np.asarray(dr["ghx"], float)
        self.ghu = np.asarray(dr["ghu"], float)
        self.order_var = np.asarray(dr["order_var"], int)
        self.state_var = np.asarray(dr["state_var"], int)
        self.n = len(self.ys)
        self.nspred = self.ghx.shape[1]
        self.nexo = self.ghu.shape[1]
        _, _, cnt = dr_ordering(m.lead_lag_incidence)
        self.nstatic = cnt["nstatic"]
        if cnt["npred"] + cnt["nboth"] != self.nspred:
            raise ValueError("decision rule does not match the model")
        self.inv_order_var = np.empty(self.n, int)
        self.inv_order_var[self.order_var] = np.arange(self.n)
        self.index_states = self.nstatic + np.arange(self.nspred)
        self.has2 = all(k in dr for k in ("ghxx", "ghxu", "ghuu", "ghs2"))
        if self.has2:
            self.ghxx, self.ghxu, self.ghuu = (np.asarray(dr[k], float) for k in ("ghxx", "ghxu", "ghuu"))
            self.ghs2 = np.asarray(dr["ghs2"], float).ravel()


def solve(m, order=1):
    """Decision rules at ``order`` (1 or 2) as a dict: ``first_order`` or ``k_order``."""
    if order == 1:
        from .perturbation import first_order
        return first_order(m).as_dr()
    from .korder import k_order
    return k_order(m, order=order)


def _measurement_errors(m):
    """``(H_diag, varobs_id)``: measurement-error variances in ``varobs`` order, observed indices."""
    varobs = list(getattr(m, "varobs", []) or [])
    me = getattr(m, "measurement_errors", {}) or {}
    H = np.array([float(me.get(v, 0.0)) for v in varobs])
    vid = np.array([list(m.endo_names).index(v) for v in varobs], int)
    return H, vid


# ------------------------------------------------------------------------------------ results
@dataclass
class Moments:
    """Moments of a variable list, named as Dynare's ``oo_`` fields where there is one."""
    kind: str                                   # 'theoretical', 'pruned' or 'simulated'
    order: int
    pruning: bool
    filter: str                                 # '', 'hp', 'one-sided hp' or 'bandpass'
    filter_param: object
    var_names: list
    var_index: np.ndarray                       # declaration-order indices of var_names
    exo_names: list
    stationary: np.ndarray                      # bool: moments are finite (not on a unit-root Schur vector)
    ar: int
    mean: np.ndarray                            # oo_.mean
    std: np.ndarray
    variance: np.ndarray                        # diag of var
    var: np.ndarray                             # oo_.var (variance matrix)
    corr: np.ndarray | None = None              # oo_.contemporaneous_correlation
    autocorr: list = field(default_factory=list)  # oo_.autocorr, ar matrices
    gamma_y: list = field(default_factory=list)   # oo_.gamma_y (theoretical moments only)
    mean_correction: np.ndarray | None = None
    skewness: np.ndarray | None = None
    kurtosis: np.ndarray | None = None
    variance_decomposition: np.ndarray | None = None        # percent, (nvar, nexo)
    variance_decomposition_me: np.ndarray | None = None     # percent, (nobs, nexo + 1)
    conditional_steps: list = field(default_factory=list)
    conditional_variance_decomposition: np.ndarray | None = None      # fractions (nvar, nsteps, nexo)
    conditional_variance_decomposition_me: np.ndarray | None = None   # (nobs, nsteps, nexo + 1)
    me_var_names: list = field(default_factory=list)

    def format(self) -> str:
        return format_moments(self)


def _stationary_mask(nvar, stationary):
    mask = np.zeros(nvar, bool)
    mask[np.asarray(stationary, int)] = True
    return mask


# ------------------------------------------------------------------------------------ variance decompositions
def _compute_variance_decomposition(R, Sigma, o, A, aa, ghu_s, stationary, idx_stat, nvars, var_stationary,
                                    warn=True):
    """``compute_variance_decomposition.m``: fractions of the variance due to each (Cholesky-orthogonalised) shock."""
    if R.nexo == 1:
        return np.ones((nvars, 1))
    cs = get_lower_cholesky_covariance(Sigma, o["add_tiny_number_to_cholesky"])
    b1 = ghu_s @ cs
    b2 = R.ghu[idx_stat] @ cs
    out = np.full((nvars, R.nexo), np.nan)
    total = 0.0
    for i in range(R.nexo):
        Vx, _ = lyapunov_symm(A, np.outer(b1[:, i], b1[:, i]), o["qz_criterium"], o["lyapunov_complex_threshold"])
        vx2 = np.diag(aa @ Vx @ aa.T + np.outer(b2[:, i], b2[:, i]))
        out[stationary, i] = vx2
        total = total + vx2
    if warn and not o["pruning"]:
        with np.errstate(all="ignore"):
            rel = np.abs(total - var_stationary) / var_stationary
            if np.nanmax(rel) > 1e-4 and np.nanmax(np.abs(total - var_stationary)) > 1e-7:
                warnings.warn("Aggregate variance and sum of variances by shocks differ by more than 0.01 %")
    with np.errstate(all="ignore"):
        out[stationary, :] = out[stationary, :] / total[:, None]
    return out


def conditional_variance_decomposition(m, dr, steps, var_list=None, *, Sigma_e=None, H=None, varobs_id=None):
    """``conditional_variance_decomposition.m``.

    Returns ``(cvd, cvd_me)``: ``cvd`` has shape (nvar, nsteps, nexo) in fractions, ``cvd_me`` has one
    more shock (the measurement error, last) for the requested variables that are observed, or is
    ``None`` when there is no measurement error. ``dr`` is a decision-rule dict or solution object.
    The decomposition always uses the first-order rules, also at order 2."""
    R = dr if isinstance(dr, _Rules) else _Rules(m, dr)
    if var_list is not None and len(var_list) > 0 and not isinstance(var_list[0], str):
        ivar = np.asarray(var_list, int)            # declaration-order indices
    else:
        ivar = _var_indices(m, var_list)
    Sigma = np.asarray(m.Sigma_e if Sigma_e is None else Sigma_e, float)
    steps = [int(s) for s in np.atleast_1d(steps)]
    if any(s <= 0 for s in steps):
        raise ValueError("Conditional variance decomposition: All periods must be strictly positive")
    if any(b <= a for a, b in zip(steps[:-1], steps[1:])):
        raise ValueError("conditional variance decomposition: periods must be strictly increasing")
    n, nexo = R.n, R.nexo
    trans = np.zeros((n, n))
    trans[:, R.index_states] = R.ghx
    impulse = R.ghu
    diag_sigma = np.allclose(Sigma, np.diag(np.diag(Sigma)), rtol=0, atol=0)
    B = impulse * np.sqrt(np.diag(Sigma))[None, :] if diag_sigma else impulse @ np.linalg.cholesky(Sigma)
    cv = np.zeros((n, len(steps), nexo))
    for i in range(nexo):
        BB = np.outer(B[:, i], B[:, i])
        V = np.zeros((n, n))
        j = 0
        for h in range(1, steps[-1] + 1):
            V = trans @ V @ trans.T + BB
            if h == steps[j]:
                cv[R.order_var, j, i] = np.diag(V)
                j += 1
    cv = cv[ivar]
    tot = cv.sum(axis=2)
    with np.errstate(all="ignore"):
        cvd = cv / tot[:, :, None]
    cvd_me = None
    if H is None or varobs_id is None:
        Hd, vid = _measurement_errors(m)
        H = Hd if H is None else H
        varobs_id = vid if varobs_id is None else varobs_id
    H = np.asarray(H, float)
    if np.any(H != 0):
        obs_pos, isub, iobs = _intersect_stable(ivar, np.asarray(varobs_id, int))
        cvd_me = np.zeros((len(obs_pos), len(steps), nexo + 1))
        with np.errstate(all="ignore"):
            for i in range(nexo):
                cvd_me[:, :, i] = cv[isub, :, i] / (tot[isub, :] + H[iobs][:, None])
        cvd_me[:, :, nexo] = 1.0 - cvd_me[:, :, :nexo].sum(axis=2)
    return cvd, cvd_me


# ------------------------------------------------------------------------------------ theoretical moments
def _th_autocovariances(R, Sigma, ivar, o, nodecomposition):
    """``th_autocovariances.m`` for orders 1 and 2 (order 2 only adds the mean correction).

    Returns ``(gamma, stationary_vars)``; ``gamma`` is a list laid out as Dynare's ``Gamma_y`` cell."""
    nar = o["ar"]
    local_order = o["order"]
    filtered = bool(o["hp_filter"]) or bool(o["bandpass"])
    if local_order >= 3:
        raise NotImplementedError("Theoretical moments not implemented above 2nd order")
    if local_order > 1 and filtered:
        raise NotImplementedError("Theoretical filtered moments not implemented above 1st order")
    nvar = len(ivar)
    ns = R.nspred
    ghx, ghu = R.ghx, R.ghu
    inv = R.inv_order_var
    ghu_s = ghu[R.index_states]
    A = ghx[R.index_states, :]
    gamma = [None] * (nar + 2)
    Vx = None
    if filtered:
        idx_stat = inv[ivar]
        stationary = np.arange(nvar)
    else:
        Vx, u = lyapunov_symm(A, ghu_s @ Sigma @ ghu_s.T, o["qz_criterium"], o["lyapunov_complex_threshold"])
        idx_stat = inv[ivar]
        stationary = np.arange(nvar)
        if u.shape[1] > 0:
            x = np.abs(ghx @ u)
            stationary = np.nonzero(np.all(x[inv[ivar]] < o["schur_vec_tol"], axis=1))[0]
            idx_stat = inv[ivar[stationary]]
    mean_corr = None
    if local_order == 2:
        if not R.has2:
            raise ValueError("second-order decision rules (ghxx, ghxu, ghuu, ghs2) are needed")
        sig = _vecF(Sigma)
        mean_corr = np.full(nvar, np.nan)
        if ns > 0:
            ist = R.index_states
            Ex = (R.ghs2[ist] + R.ghxx[ist] @ _vecF(Vx) + R.ghuu[ist] @ sig) / 2.0
            try:
                Ex = np.linalg.solve(np.eye(ns) - ghx[ist, :], Ex)
            except np.linalg.LinAlgError:          # unit root among the states: MATLAB warns and returns Inf/NaN
                Ex = np.full(ns, np.nan)
            mean_corr[stationary] = ghx[idx_stat] @ Ex + (R.ghs2[idx_stat] + R.ghxx[idx_stat] @ _vecF(Vx)
                                                        + R.ghuu[idx_stat] @ sig) / 2.0
        else:
            mean_corr[stationary] = (R.ghs2[idx_stat] + R.ghuu[idx_stat] @ sig) / 2.0
    aa = ghx[idx_stat]
    bb = ghu[idx_stat]
    if not filtered:
        v = np.full((nvar, nvar), np.nan)
        v[np.ix_(stationary, stationary)] = aa @ Vx @ aa.T + bb @ Sigma @ bb.T
        v = np.where(np.abs(v) < 1e-12, 0.0, v)
        gamma[0] = v
        if nar > 0:
            vxy = A @ Vx @ aa.T + ghu_s @ Sigma @ bb.T
            sy = np.sqrt(np.diag(v))[stationary]
            sy = np.outer(sy, sy)
            with np.errstate(all="ignore"):
                for i in range(1, nar + 1):
                    g = np.full((nvar, nvar), np.nan)
                    g[np.ix_(stationary, stationary)] = (aa @ vxy) / sy
                    gamma[i] = g
                    vxy = A @ vxy
        if not nodecomposition and R.nexo > 0 and len(stationary) > 0:
            vd = _compute_variance_decomposition(R, Sigma, o, A, aa, ghu_s, stationary, idx_stat, nvar,
                                                 np.diag(v[np.ix_(stationary, stationary)]))
            gamma[nar + 1] = vd
    else:
        ngrid = int(o["filtered_theoretical_moments_grid"])
        freqs = np.arange(ngrid) * (2 * np.pi / ngrid)
        tneg = np.exp(-1j * freqs)
        if o["bandpass"]:
            gain = np.zeros(ngrid)
            low, high = o["passband"][1], max(2, o["passband"][0])
            gain[(freqs >= 2 * np.pi / low) & (freqs <= 2 * np.pi / high)] = 1.0
            gain[(freqs <= -2 * np.pi / low + 2 * np.pi) & (freqs >= -2 * np.pi / high + 2 * np.pi)] = 1.0
        else:
            lam = o["hp_filter"]
            gain = 4 * lam * (1 - np.cos(freqs)) ** 2 / (1 + 4 * lam * (1 - np.cos(freqs)) ** 2)
        active = gain != 0
        IA, IE = np.eye(ns), np.eye(R.nexo)

        def filtered_acov(Sig, b1, b2):
            """real(ifft of the filtered spectral density) * 2 pi, shape (ngrid, nvar, nvar)."""
            f_hp = np.zeros((ngrid, nvar, nvar), complex)
            ga = np.nonzero(active)[0]
            if ns > 0:
                M = IA[None] - A[None] * tneg[ga, None, None]
                X = np.linalg.solve(M, np.broadcast_to(b1, (len(ga),) + b1.shape))
            else:
                X = np.zeros((len(ga), 0, R.nexo))
            top = np.concatenate([X, np.broadcast_to(IE, (len(ga), R.nexo, R.nexo))], axis=1)
            f_om = top @ Sig @ np.conj(np.transpose(top, (0, 2, 1))) / (2 * np.pi)
            W = np.concatenate([aa[None] * tneg[ga, None, None], np.broadcast_to(bb if b2 is None else b2,
                                                                                 (len(ga),) + (aa.shape[0], R.nexo))],
                               axis=2)
            g = W @ f_om @ np.conj(np.transpose(W, (0, 2, 1)))
            f_hp[ga] = gain[ga, None, None] ** 2 * g
            return np.real(np.fft.ifft(f_hp, axis=0)) * (2 * np.pi)

        im = filtered_acov(Sigma, ghu_s, None)
        gamma[0] = im[0].copy()
        if nar > 0:
            sy = np.sqrt(np.diag(gamma[0]))
            sy = np.outer(sy, sy)
            with np.errstate(all="ignore"):
                for i in range(1, nar + 1):
                    gamma[i] = im[i] / sy
        if not nodecomposition and R.nexo > 0:
            if R.nexo == 1:
                gamma[nar + 1] = np.ones((nvar, 1))
            else:
                cs = get_lower_cholesky_covariance(Sigma, o["add_tiny_number_to_cholesky"])
                SS = cs @ cs.T
                vv = np.diag(filtered_acov(SS, ghu_s, ghu[idx_stat])[0])
                vd = np.zeros((nvar, R.nexo))
                for i in range(R.nexo):
                    ii = filtered_acov(np.outer(cs[:, i], cs[:, i]), ghu_s, ghu[idx_stat])
                    with np.errstate(all="ignore"):
                        vd[:, i] = np.abs(np.diag(ii[0])) / vv
                gamma[nar + 1] = vd
    if mean_corr is not None:
        gamma.append(mean_corr)
    return gamma, stationary


def _pruned_order2(R, Sigma, ivar, o):
    """Order-2 pruned-state-space moments (``pruned_state_space_system.m`` with order 2).

    Returns ``(E_y, Var_y, Corr_y, Corr_yi)`` for the variables ``ivar``."""
    nar = o["ar"]
    if not R.has2:
        raise ValueError("second-order decision rules (ghxx, ghxu, ghuu, ghs2) are needed")
    x, u = R.nspred, R.nexo
    indx = R.index_states
    indy = R.inv_order_var[ivar]
    Yss = R.ys[R.order_var]
    hx, gx = R.ghx[indx], R.ghx[indy]
    hu, gu = R.ghu[indx], R.ghu[indy]
    hxx, gxx = R.ghxx[indx], R.ghxx[indy]
    hxu, gxu = R.ghxu[indx], R.ghxu[indy]
    huu, guu = R.ghuu[indx], R.ghuu[indy]
    hss, gss = R.ghs2[indx], R.ghs2[indy]
    E_uu = Sigma
    Var_z1, Schur_u = lyapunov_symm(hx, hu @ E_uu @ hu.T, o["qz_criterium"], o["lyapunov_complex_threshold"])
    if Schur_u.shape[1] > 0:
        raise NotImplementedError("pruned_state_space_system: the pruned state space currently does not "
                                  "support unit roots at higher order.")
    E_xfxf = Var_z1
    vS = _vecF(E_uu)
    hx_hx, hx_hu, hu_hu = np.kron(hx, hx), np.kron(hx, hu), np.kron(hu, hu)
    I_xx = np.eye(x * x)
    K_xx = _commutation(x)
    invIx_hx = np.linalg.inv(np.eye(x) - hx)
    # fourth moments of the Gaussian shocks (Isserlis); rows/columns indexed like kron(u, u)
    M4 = (np.einsum("ij,kl->ijkl", E_uu, E_uu) + np.einsum("ik,jl->ijkl", E_uu, E_uu)
          + np.einsum("il,jk->ijkl", E_uu, E_uu))
    V22 = M4.reshape(u * u, u * u) - np.outer(vS, vS)
    E_xfxf_uu = np.kron(E_xfxf, E_uu.T)

    nz = 2 * x + x * x
    ni = u + u * u + x * u
    id1, id2, id3 = slice(0, x), slice(x, 2 * x), slice(2 * x, nz)
    j1, j2, j3 = slice(0, u), slice(u, u + u * u), slice(u + u * u, ni)
    A = np.zeros((nz, nz))
    A[id1, id1] = hx
    A[id2, id2] = hx
    A[id2, id3] = 0.5 * hxx
    A[id3, id3] = hx_hx
    B = np.zeros((nz, ni))
    B[id1, j1] = hu
    B[id2, j2] = 0.5 * huu
    B[id2, j3] = hxu
    B[id3, j2] = hu_hu
    B[id3, j3] = (I_xx + K_xx) @ hx_hu
    c2 = 0.5 * hss + 0.5 * huu @ vS
    C = np.zeros((len(indy), nz))
    C[:, id1] = gx
    C[:, id2] = gx
    C[:, id3] = 0.5 * gxx
    D = np.zeros((len(indy), ni))
    D[:, j1] = gu
    D[:, j2] = 0.5 * guu
    D[:, j3] = gxu
    d = 0.5 * gss + 0.5 * guu @ vS
    Varinov = np.zeros((ni, ni))
    Varinov[j1, j1] = E_uu
    Varinov[j2, j2] = V22
    Varinov[j3, j3] = E_xfxf_uu
    E_xs = invIx_hx @ (0.5 * hxx @ _vecF(E_xfxf) + c2)
    Om_z = B @ Varinov @ B.T
    Var_z, err = disclyap_fast(A, Om_z, o["lyapunov_doubling_tol"])
    if err:
        print("PRUNED_STATE_SPACE_SYSTEM: error flag in disclyap_fast at order=2, use lyapunov_symm")
        Var_z, _ = lyapunov_symm(A, Om_z, 1.0, o["lyapunov_complex_threshold"])
    Var_z[id1, id2] = 0
    Var_z[id1, id3] = 0
    Var_z[id2, id1] = 0
    Var_z[id3, id1] = 0
    Var_y = C @ Var_z @ C.T + D @ Varinov @ D.T
    Var_y = np.where(np.abs(Var_y) < 1e-12, 0.0, Var_y)
    sd = np.sqrt(np.diag(Var_y))
    sy = np.outer(sd, sd)
    with np.errstate(all="ignore"):
        Corr_y = Var_y / sy
        tmp = A @ Var_z @ C.T + B @ Varinov @ D.T
        Ai = np.eye(nz)
        Corr_yi = []
        for _ in range(nar):
            Corr_yi.append((C @ Ai @ tmp) / sy)
            Ai = Ai @ A
    E_z = np.concatenate([np.zeros(x), E_xs, _vecF(E_xfxf)])
    E_y = Yss[indy] + C @ E_z + d
    return E_y, Var_y, Corr_y, Corr_yi


def _commutation_nm(n, m):
    """``pruned_SS.commutation(n, m)``: reshape(kron(vec(eye(n)), eye(m)), n*m, n*m) (column-major)."""
    return np.kron(np.eye(n).ravel(order="F")[:, None], np.eye(m)).reshape(n * m, n * m, order="F")


def _gaussian_moment_tensor(S, k):
    """E[u_{i1} ... u_{ik}] for u ~ N(0, S) (Isserlis), as a k-dimensional array (k even)."""
    letters = "abcdefghij"[:k]

    def matchings(idx):
        if not idx:
            yield []
            return
        a = idx[0]
        for j in range(1, len(idx)):
            rest = idx[1:j] + idx[j + 1:]
            for mt in matchings(rest):
                yield [(a, idx[j])] + mt

    out = np.zeros((S.shape[0],) * k)
    for mt in matchings(list(range(k))):
        spec = ",".join(letters[p] + letters[q] for p, q in mt) + "->" + letters
        out += np.einsum(spec, *([S] * len(mt)))
    return out


def _resF(a, r, c):
    return np.reshape(a, (r, c), order="F")


def _pruned_order3(R, Sigma, ivar, o):
    """Order-3 pruned-state-space moments (``pruned_state_space_system.m`` with order 3, no derivatives).

    Returns ``(E_y, Var_y, Corr_y, Corr_yi)`` for the variables ``ivar``. Literal port: vec/reshape are
    column-major as in MATLAB."""
    nar = o["ar"]
    dr = R.dr
    need = ("ghxx", "ghxu", "ghuu", "ghs2", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss")
    if not all(k in dr for k in need):
        raise ValueError("third-order decision rules (ghxxx, ghxxu, ghxuu, ghuuu, ghxss, ghuss) are needed")
    x, u = R.nspred, R.nexo
    indx = R.index_states
    indy = R.inv_order_var[ivar]
    ny = len(indy)
    Yss = R.ys[R.order_var]
    g = {k: np.asarray(dr[k], float) for k in need}
    g["ghs2"] = g["ghs2"].ravel()
    hx, gx = R.ghx[indx], R.ghx[indy]
    hu, gu = R.ghu[indx], R.ghu[indy]
    hxx, gxx = g["ghxx"][indx], g["ghxx"][indy]
    hxu, gxu = g["ghxu"][indx], g["ghxu"][indy]
    huu, guu = g["ghuu"][indx], g["ghuu"][indy]
    hss, gss = g["ghs2"][indx], g["ghs2"][indy]
    hxxx, gxxx = g["ghxxx"][indx], g["ghxxx"][indy]
    hxxu, gxxu = g["ghxxu"][indx], g["ghxxu"][indy]
    hxuu, gxuu = g["ghxuu"][indx], g["ghxuu"][indy]
    huuu, guuu = g["ghuuu"][indx], g["ghuuu"][indy]
    hxss, gxss = g["ghxss"][indx], g["ghxss"][indy]
    huss, guss = g["ghuss"][indx], g["ghuss"][indy]
    E_uu = np.asarray(Sigma, float)
    vE_uu = _vecF(E_uu)

    # first order
    Var_z1, Schur_u = lyapunov_symm(hx, hu @ E_uu @ hu.T, o["qz_criterium"], o["lyapunov_complex_threshold"])
    if Schur_u.shape[1] > 0:
        raise NotImplementedError("pruned_state_space_system: the pruned state space currently does not "
                                  "support unit roots at higher order.")
    E_xfxf = Var_z1
    vE_xfxf = _vecF(E_xfxf)
    hx_hx, hx_hu, hu_hu = np.kron(hx, hx), np.kron(hx, hu), np.kron(hu, hu)
    I_xx = np.eye(x * x)
    K_x_x = _commutation_nm(x, x)
    invIx_hx = np.linalg.inv(np.eye(x) - hx)
    QPuE4 = _gaussian_moment_tensor(E_uu, 4).ravel()          # E[kron(kron(kron(u,u),u),u)]
    E_xfxf_uu = np.kron(E_xfxf, E_uu.T)

    # second-order system: z = [xf; xs; kron(xf,xf)], inov = [u; kron(u,u)-E_uu(:); kron(xf,u)]
    i1 = np.arange(x)
    i2 = x + np.arange(x)
    i3 = 2 * x + np.arange(x * x)
    j1 = np.arange(u)
    j2 = u + np.arange(u * u)
    j3 = u + u * u + np.arange(x * u)
    nz2, ni2 = 2 * x + x * x, u + u * u + x * u
    A = np.zeros((nz2, nz2))
    A[np.ix_(i1, i1)] = hx
    A[np.ix_(i2, i2)] = hx
    A[np.ix_(i2, i3)] = 0.5 * hxx
    A[np.ix_(i3, i3)] = hx_hx
    B = np.zeros((nz2, ni2))
    B[np.ix_(i1, j1)] = hu
    B[np.ix_(i2, j2)] = 0.5 * huu
    B[np.ix_(i2, j3)] = hxu
    B[np.ix_(i3, j2)] = hu_hu
    B[np.ix_(i3, j3)] = (I_xx + K_x_x) @ hx_hu
    c_xs = 0.5 * hss + 0.5 * huu @ vE_uu
    Varinov = np.zeros((ni2, ni2))
    Varinov[np.ix_(j1, j1)] = E_uu
    Varinov[np.ix_(j2, j2)] = _resF(QPuE4, u * u, u * u) - np.outer(vE_uu, vE_uu)
    Varinov[np.ix_(j3, j3)] = E_xfxf_uu
    E_xs = invIx_hx @ (0.5 * hxx @ vE_xfxf + c_xs)
    Om_z = B @ Varinov @ B.T
    Var_z, err = disclyap_fast(A, Om_z, o["lyapunov_doubling_tol"])
    if err:
        print("PRUNED_STATE_SPACE_SYSTEM: error flag in disclyap_fast at order=2, use lyapunov_symm")
        Var_z, _ = lyapunov_symm(A, Om_z, 1.0, o["lyapunov_complex_threshold"])
    Var_z[np.ix_(i1, i2)] = 0
    Var_z[np.ix_(i1, i3)] = 0
    Var_z[np.ix_(i2, i1)] = 0
    Var_z[np.ix_(i3, i1)] = 0

    # objects for order 3
    K_u_xx = _commutation_nm(u, x * x)
    K_u_ux = _commutation_nm(u, u * x)
    K_xx_x = _commutation_nm(x * x, x)
    hx_hss2 = np.kron(hx, 0.5 * hss[:, None])
    hu_hss2 = np.kron(hu, 0.5 * hss[:, None])
    hx_hxx2 = np.kron(hx, 0.5 * hxx)
    hxx2_hu = np.kron(0.5 * hxx, hu)
    hx_hxu = np.kron(hx, hxu)
    hxu_hu = np.kron(hxu, hu)
    hx_huu2 = np.kron(hx, 0.5 * huu)
    hu_huu2 = np.kron(hu, 0.5 * huu)
    hx_hx_hx = np.kron(hx, hx_hx)
    hx_hx_hu = np.kron(hx_hx, hu)
    hu_hx_hx = np.kron(hu, hx_hx)
    hu_hu_hu = np.kron(hu_hu, hu)
    hx_hu_hu = np.kron(hx, hu_hu)
    hu_hx_hu = np.kron(hu, hx_hu)
    invIxx_hx_hx = np.linalg.inv(np.eye(x * x) - hx_hx)

    E_xsxs = Var_z[np.ix_(i2, i2)] + np.outer(E_xs, E_xs)
    E_xsxf_xf = Var_z[np.ix_(i2, i3)] + np.outer(E_xs, vE_xfxf)
    E_xf_xfxs = Var_z[np.ix_(i3, i2)] + np.outer(vE_xfxf, E_xs)
    E_xf_xfxf_xf = Var_z[np.ix_(i3, i3)] + np.outer(vE_xfxf, vE_xfxf)
    M = (hxx @ _resF(K_xx_x @ _vecF(E_xf_xfxs), x * x, x) @ hx.T
         + hxu @ np.kron(E_xs[:, None], E_uu) @ hu.T
         + 1 / 6 * hxxx @ _resF(E_xf_xfxf_xf, x ** 3, x) @ hx.T
         + 1 / 6 * huuu @ _resF(QPuE4, u ** 3, u) @ hu.T
         + 3 / 6 * hxxu @ np.kron(vE_xfxf[:, None], E_uu) @ hu.T
         + 3 / 6 * hxuu @ np.kron(E_xfxf, vE_uu[:, None]) @ hx.T
         + 3 / 6 * hxss @ E_xfxf @ hx.T
         + 3 / 6 * huss @ E_uu @ hu.T)
    E_xrdxf = _resF(invIxx_hx_hx @ _vecF(M), x, x)
    QPuE6 = _gaussian_moment_tensor(E_uu, 6).ravel()

    # third-order system
    # z = [xf; xs; kron(xf,xf); xrd; kron(xf,xs); kron(kron(xf,xf),xf)]
    # inov = [u; kron(u,u)-E_uu(:); kron(xf,u); kron(xs,u); kron(kron(xf,xf),u); kron(kron(xf,u),u); kron(kron(u,u),u)]
    i4 = nz2 + np.arange(x)
    i5 = i4[-1] + 1 + np.arange(x * x)
    i6 = i5[-1] + 1 + np.arange(x ** 3)
    j4 = ni2 + np.arange(x * u)
    j5 = j4[-1] + 1 + np.arange(x * x * u)
    j6 = j5[-1] + 1 + np.arange(x * u * u)
    j7 = j6[-1] + 1 + np.arange(u ** 3)
    nz = i6[-1] + 1
    ni = j7[-1] + 1
    A = np.zeros((nz, nz))
    A[np.ix_(i1, i1)] = hx
    A[np.ix_(i2, i2)] = hx
    A[np.ix_(i2, i3)] = 0.5 * hxx
    A[np.ix_(i3, i3)] = hx_hx
    A[np.ix_(i4, i1)] = 3 / 6 * hxss
    A[np.ix_(i4, i4)] = hx
    A[np.ix_(i4, i5)] = hxx
    A[np.ix_(i4, i6)] = 1 / 6 * hxxx
    A[np.ix_(i5, i1)] = hx_hss2
    A[np.ix_(i5, i5)] = hx_hx
    A[np.ix_(i5, i6)] = hx_hxx2
    A[np.ix_(i6, i6)] = hx_hx_hx
    B = np.zeros((nz, ni))
    B[np.ix_(i1, j1)] = hu
    B[np.ix_(i2, j2)] = 0.5 * huu
    B[np.ix_(i2, j3)] = hxu
    B[np.ix_(i3, j2)] = hu_hu
    B[np.ix_(i3, j3)] = (I_xx + K_x_x) @ hx_hu
    B[np.ix_(i4, j1)] = 3 / 6 * huss
    B[np.ix_(i4, j4)] = hxu
    B[np.ix_(i4, j5)] = 3 / 6 * hxxu
    B[np.ix_(i4, j6)] = 3 / 6 * hxuu
    B[np.ix_(i4, j7)] = 1 / 6 * huuu
    B[np.ix_(i5, j1)] = hu_hss2
    B[np.ix_(i5, j4)] = K_x_x @ hx_hu
    B[np.ix_(i5, j5)] = hx_hxu + K_x_x @ hxx2_hu
    B[np.ix_(i5, j6)] = hx_huu2 + K_x_x @ hxu_hu
    B[np.ix_(i5, j7)] = hu_huu2
    B[np.ix_(i6, j5)] = hx_hx_hu + np.kron(hx, K_x_x @ hx_hu) + hu_hx_hx @ K_u_xx
    B[np.ix_(i6, j6)] = hx_hu_hu + hu_hx_hu @ K_u_ux + np.kron(hu, K_x_x @ hx_hu) @ K_u_ux
    B[np.ix_(i6, j7)] = hu_hu_hu
    C = np.zeros((ny, nz))
    C[:, i1] = gx + 3 / 6 * gxss
    C[:, i2] = gx
    C[:, i3] = 0.5 * gxx
    C[:, i4] = gx
    C[:, i5] = gxx
    C[:, i6] = 1 / 6 * gxxx
    D = np.zeros((ny, ni))
    D[:, j1] = gu + 3 / 6 * guss
    D[:, j2] = 0.5 * guu
    D[:, j3] = gxu
    D[:, j4] = gxu
    D[:, j5] = 3 / 6 * gxxu
    D[:, j6] = 3 / 6 * gxuu
    D[:, j7] = 1 / 6 * guuu
    d = 0.5 * gss + 0.5 * guu @ vE_uu

    E4_u_uuu = _resF(QPuE4, u, u ** 3)
    E4_uuu_u = _resF(QPuE4, u ** 3, u)
    Vi = np.zeros((ni, ni))
    Vi[np.ix_(j1, j1)] = E_uu
    Vi[np.ix_(j1, j4)] = np.kron(E_xs[None, :], E_uu)
    Vi[np.ix_(j1, j5)] = np.kron(vE_xfxf[None, :], E_uu)
    Vi[np.ix_(j1, j7)] = E4_u_uuu
    Vi[np.ix_(j2, j2)] = _resF(QPuE4, u * u, u * u) - np.outer(vE_uu, vE_uu)
    Vi[np.ix_(j3, j3)] = E_xfxf_uu
    Vi[np.ix_(j4, j1)] = np.kron(E_xs[:, None], E_uu)
    Vi[np.ix_(j4, j4)] = np.kron(E_xsxs, E_uu)
    Vi[np.ix_(j4, j5)] = np.kron(E_xsxf_xf, E_uu)
    Vi[np.ix_(j4, j7)] = np.kron(E_xs[:, None], E4_u_uuu)
    Vi[np.ix_(j5, j1)] = np.kron(vE_xfxf[:, None], E_uu)
    Vi[np.ix_(j5, j4)] = np.kron(E_xf_xfxs, E_uu)
    Vi[np.ix_(j5, j5)] = np.kron(E_xf_xfxf_xf, E_uu)
    Vi[np.ix_(j5, j7)] = np.kron(vE_xfxf[:, None], E4_u_uuu)
    Vi[np.ix_(j6, j6)] = np.kron(E_xfxf, _resF(QPuE4, u * u, u * u))
    Vi[np.ix_(j7, j1)] = E4_uuu_u
    Vi[np.ix_(j7, j4)] = np.kron(E_xs[None, :], E4_uuu_u)
    Vi[np.ix_(j7, j5)] = np.kron(vE_xfxf[None, :], E4_uuu_u)
    Vi[np.ix_(j7, j7)] = _resF(QPuE6, u ** 3, u ** 3)
    Varinov = Vi

    def inovzlag(hxi):
        E = np.zeros((ni, nz))
        E[np.ix_(j6, i1)] = np.kron(hxi @ E_xfxf, vE_uu[:, None])
        E[np.ix_(j6, i4)] = np.kron(hxi @ E_xrdxf.T, vE_uu[:, None])
        E[np.ix_(j6, i5)] = np.kron(hxi @ _resF(K_xx_x @ _vecF(E_xsxf_xf), x, x * x), vE_uu[:, None])
        E[np.ix_(j6, i6)] = np.kron(hxi @ _resF(E_xf_xfxf_xf, x, x ** 3), vE_uu[:, None])
        return E

    E_inovzlag1 = inovzlag(np.eye(x))
    BinA = B @ E_inovzlag1 @ A.T
    Om_z = B @ Varinov @ B.T + BinA + BinA.T
    Var_z, err = disclyap_fast(A, Om_z, o["lyapunov_doubling_tol"])
    if err:
        print("PRUNED_STATE_SPACE_SYSTEM: error flag in disclyap_fast at order=3, use lyapunov_symm")
        Var_z, _ = lyapunov_symm(A, Om_z, 1.0, o["lyapunov_complex_threshold"])
    for a_, b_ in ((i1, i2), (i1, i3), (i2, i1), (i2, i4), (i2, i5), (i2, i6), (i3, i1), (i3, i4), (i3, i5),
                   (i3, i6), (i4, i2), (i4, i3), (i5, i2), (i5, i3), (i6, i2), (i6, i3)):
        Var_z[np.ix_(a_, b_)] = 0

    Var_y = (C @ Var_z @ C.T + D @ E_inovzlag1 @ C.T + C @ E_inovzlag1.T @ D.T + D @ Varinov @ D.T)
    Var_y = np.where(np.abs(Var_y) < 1e-12, 0.0, Var_y)
    sd = np.sqrt(np.diag(Var_y))
    sy = np.outer(sd, sd)
    Corr_yi = []
    with np.errstate(all="ignore"):
        Corr_y = Var_y / sy
        tmp = A @ E_inovzlag1.T @ D.T + B @ Varinov @ D.T
        Ai = np.eye(nz)
        hxi = np.eye(x)
        E_inovzlagi = E_inovzlag1
        Var_zi = Var_z
        for _ in range(nar):
            Var_zi = A @ Var_zi + B @ E_inovzlagi
            hxi = hx @ hxi
            E_inovzlagi = inovzlag(hxi)
            Var_yi = C @ Var_zi @ C.T + C @ Ai @ tmp + D @ E_inovzlagi @ C.T
            Corr_yi.append(Var_yi / sy)
            Ai = Ai @ A
    E_z = np.concatenate([np.zeros(x), E_xs, vE_xfxf, np.zeros(x), np.zeros(x * x), np.zeros(x ** 3)])
    E_y = Yss[indy] + C @ E_z + d
    return E_y, Var_y, Corr_y, Corr_yi


def _corr_from_var(var, sd):
    s2 = np.diag(var)
    i1 = np.nonzero(np.abs(s2) > 1e-12)[0]
    corr = np.full(var.shape, np.nan)
    with np.errstate(all="ignore"):
        corr[np.ix_(i1, i1)] = var[np.ix_(i1, i1)] / np.outer(sd[i1], sd[i1])
    return corr, i1


def theoretical_moments(m, dr=None, var_list=None, *, Sigma_e=None, H=None, **options) -> Moments:
    """Theoretical moments as ``disp_th_moments`` / ``disp_th_moments_pruned_state_space`` compute them.

    ``dr`` is a decision-rule dict (``pydyn.k_order`` or ``first_order(m).as_dr()``) or ``None``, in which
    case the model is solved at ``order``. ``var_list`` is a list of names (default: the ``stoch_simul``
    list of the .mod file, else the original endogenous variables). Options (Dynare names, defaults from
    the .mod file's ``stoch_simul`` statement, then Dynare's defaults): ``order`` (1 or 2), ``pruning``,
    ``ar`` (5), ``hp_filter``, ``bandpass_filter=(6, 32)``, ``bandpass_K`` (12),
    ``filtered_theoretical_moments_grid`` (512), ``nodecomposition``, ``nocorr``,
    ``conditional_variance_decomposition`` (or ``steps``), ``qz_criterium``, ``schur_vec_tol``.
    ``H`` overrides the measurement-error variances (in ``varobs`` order)."""
    o = _options(m, dict(options))
    if o["one_sided_hp_filter"]:
        raise ValueError("theoretical moments incompatible with one-sided HP filter. "
                         "Use simulated moments instead")
    order = o["order"]
    if order > 3:
        raise NotImplementedError("Theoretical moments are implemented up to third order (pruned)")
    if order >= 2 and m.max_deriv_order < 2:
        warnings.warn("using order = 1 because Hessian is equal to zero")
        order = o["order"] = 1
    if order == 3 and not o["pruning"]:
        # stoch_simul.m: "There is no code for theoretical moments at 3rd order without pruning"
        raise NotImplementedError("Dynare has no theoretical moments at third order without pruning; "
                                  "use pruning=True (pruned state space) or simulated moments (periods > 0)")
    if dr is None:
        dr = solve(m, order)
    R = _Rules(m, dr)
    Sigma = np.asarray(m.Sigma_e if Sigma_e is None else Sigma_e, float)
    ivar = _var_indices(m, var_list)
    nvar = len(ivar)
    nar = o["ar"]
    steps = o["conditional_variance_decomposition"]
    Hd, vid = _measurement_errors(m)
    if H is not None:
        Hd = np.asarray(H, float)
    me_present, obs_pos, isub, iobs = False, np.array([], int), np.array([], int), np.array([], int)
    if np.any(Hd != 0):
        obs_pos, isub, iobs = _intersect_stable(ivar, vid)
        me_present = len(obs_pos) > 0
    pruned = order >= 2 and bool(o["pruning"])
    if pruned and (o["hp_filter"] or o["bandpass"]):
        raise NotImplementedError(f"theoretical moments incompatible with filtering at order {order}. "
                                  "Use simulated moments instead")
    names = [m.endo_names[i] for i in ivar]
    res = Moments(kind="pruned" if pruned else "theoretical", order=order, pruning=pruned, filter=_filter_name(o),
                  filter_param=(o["hp_filter"] if o["hp_filter"] else o["passband"] if o["bandpass"] else None),
                  var_names=names, var_index=ivar, exo_names=list(m.exo_names), stationary=np.ones(nvar, bool),
                  ar=nar, mean=None, std=None, variance=None, var=None)
    if pruned:
        E_y, Var_y, Corr_y, Corr_yi = (_pruned_order3 if order == 3 else _pruned_order2)(R, Sigma, ivar, o)
        s2 = np.diag(Var_y)
        sd = np.sqrt(s2)
        res.mean, res.var, res.variance, res.std = E_y, Var_y, s2, sd
        res.gamma_y = [Var_y] + list(Corr_yi)
        i1 = np.nonzero(np.abs(s2) > 1e-12)[0]
        if len(i1) > 0:
            if not o["nocorr"]:
                res.corr = Corr_y
            res.autocorr = list(Corr_yi)
        if order == 2 and not o["nodecomposition"] and R.nexo > 1:     # Dynare: no decomposition at order 3
            ghu_s = R.ghu[R.index_states]
            A = R.ghx[R.index_states, :]
            idx = R.inv_order_var[ivar]
            vd = _compute_variance_decomposition(R, Sigma, o, A, R.ghx[idx], ghu_s, np.arange(nvar), idx, nvar, s2)
            res.gamma_y.append(vd)
            res.variance_decomposition = 100 * vd
            if me_present:
                res.variance_decomposition_me = _me_vd(res.variance_decomposition, res.var, isub, iobs, Hd, R.nexo)
            if steps:
                res.conditional_steps = steps
                res.conditional_variance_decomposition, res.conditional_variance_decomposition_me = \
                    conditional_variance_decomposition(m, R, steps, ivar, Sigma_e=Sigma, H=Hd, varobs_id=vid)
    else:
        gamma, stat = _th_autocovariances(R, Sigma, ivar, o, o["nodecomposition"])
        res.stationary = _stationary_mask(nvar, stat)
        res.gamma_y = gamma
        mean = R.ys[ivar].copy()
        mean[~res.stationary] = np.nan
        var = gamma[0]
        s2 = np.diag(var)
        with np.errstate(invalid="ignore"):
            sd = np.sqrt(s2)
        if order == 2:
            res.mean_correction = gamma[nar + 2]
            mean = mean + gamma[nar + 2]
        res.mean, res.var, res.variance, res.std = mean, var, s2, sd
        if len(stat) > 0:
            if not o["nodecomposition"]:
                vd = gamma[nar + 1]
                res.variance_decomposition = 100 * vd
                if me_present:
                    res.variance_decomposition_me = _me_vd(res.variance_decomposition, var, isub, iobs, Hd, R.nexo)
        if steps:
            res.conditional_steps = steps
            res.conditional_variance_decomposition, res.conditional_variance_decomposition_me = \
                conditional_variance_decomposition(m, R, steps, ivar, Sigma_e=Sigma, H=Hd, varobs_id=vid)
        i1 = np.nonzero(np.abs(s2) > 1e-12)[0]
        if len(i1) > 0:
            if not o["nocorr"] and len(stat) > 0:
                res.corr, _ = _corr_from_var(var, sd)
            if nar > 0 and len(stat) > 0:
                res.autocorr = [gamma[i] for i in range(1, nar + 1)]
    if me_present:
        res.me_var_names = [m.endo_names[i] for i in obs_pos]
    return res


def _me_vd(vd_percent, var, isub, iobs, Hd, nexo):
    v = np.diag(var)[isub]
    with np.errstate(all="ignore"):
        out = vd_percent[isub, :] * (v / (v + Hd[iobs]))[:, None]
    return np.column_stack([out, 100.0 - out.sum(axis=1)])


# ------------------------------------------------------------------------------------ simulated moments
def get_filtered_time_series(y, mean, o):
    """``get_filtered_time_series.m`` plus the removal of the band-pass edge rows done by the callers."""
    y = np.asarray(y, float)
    hp, os_, bp = o["hp_filter"], o["one_sided_hp_filter"], o["bandpass"]
    if sum([bool(hp), bool(os_), bool(bp)]) > 1:
        raise ValueError("You cannot use more than one filter at the same time")
    if hp:
        return hp_filter(y, hp)[1]
    if os_:
        return one_sided_hp_filter(y, os_)[1]
    if bp:
        return baxter_king_filter(y, o["passband"][0], o["passband"][1], o["bandpass_K"])
    return y - mean


def simulated_moments(y, var_names=None, *, ivar=None, drop=None, shocks=None, dr=None, model=None,
                      y0=None, **options) -> Moments:
    """Moments of a simulated path as ``disp_moments.m``.

    ``y`` has shape (T, n): either the full path as returned by ``pydyn.simulate`` (then pass ``ivar``, the
    column indices of the variables, and ``model`` for the names) or already restricted columns.
    ``drop`` rows are discarded first (default: the ``drop`` option, 100). Options as in
    ``theoretical_moments`` (``ar``, ``hp_filter``, ``one_sided_hp_filter``, ``bandpass_filter``,
    ``nodecomposition``); the variance decomposition simulating one shock at a time needs the shocks
    that generated ``y`` (``shocks``, shape (T, n_exo)) and the rules ``dr`` (plus ``order``,
    ``pruning`` and ``y0``, the initial level vector, default the steady state)."""
    o = _options(model, dict(options)) if model is not None else _options(_NoModel, dict(options))
    drop = o["drop"] if drop is None else drop
    y = np.asarray(y, float)
    if ivar is None:
        ivar = np.arange(y.shape[1])
    ivar = np.asarray(ivar, int)
    names = list(var_names) if var_names is not None else (
        [model.endo_names[i] for i in ivar] if model is not None else [f"v{i}" for i in ivar])
    ys = y[drop:][:, ivar]
    nvar = len(ivar)
    m_ = ys.mean(axis=0)
    yf = get_filtered_time_series(ys, m_, o)
    s2 = np.mean(yf * yf, axis=0)
    s = np.sqrt(s2)
    var = yf.T @ yf / yf.shape[0]
    with np.errstate(all="ignore"):
        skew = np.mean(yf ** 3, axis=0) / s2 ** 1.5
        kurt = np.mean(yf ** 4, axis=0) / (s2 * s2) - 3.0
    zero = np.nonzero(np.abs(s) < 1e-10)[0]
    skew[zero] = np.nan
    kurt[zero] = np.nan
    s[zero] = 0.0
    s2[zero] = 0.0
    var[zero, :] = 0.0
    var[:, zero] = 0.0
    corr = None
    if not o["nocorr"]:
        with np.errstate(all="ignore"):
            corr = (yf.T @ yf / yf.shape[0]) / np.outer(s, s)
        corr[zero, :] = np.nan
        corr[:, zero] = np.nan
    ar = o["ar"]
    ac = []
    T = yf.shape[0]
    with np.errstate(all="ignore"):
        for i in range(1, ar + 1):
            a1, a0 = yf[ar:], yf[ar - i:T - i]
            c = a1.T @ a0 / ((T - ar) * np.outer(np.std(a1, axis=0, ddof=1), np.std(a0, axis=0, ddof=1)))
            c[zero, :] = np.nan
            c[:, zero] = np.nan
            ac.append(c)
    res = Moments(kind="simulated", order=o["order"], pruning=bool(o["pruning"]), filter=_filter_name(o),
                  filter_param=(o["hp_filter"] if o["hp_filter"] else o["one_sided_hp_filter"] if
                                o["one_sided_hp_filter"] else o["passband"] if o["bandpass"] else None),
                  var_names=names, var_index=ivar,
                  exo_names=list(model.exo_names) if model is not None else [],
                  stationary=np.ones(nvar, bool), ar=ar, mean=m_, std=s, variance=s2, var=var, corr=corr,
                  autocorr=ac, skewness=skew, kurtosis=kurt)
    if not o["nodecomposition"] and shocks is not None:
        res.variance_decomposition = simulated_variance_decomposition(
            model, dr, shocks, ivar, drop, o, s2, zero, y0=y0)
    return res


class _NoModel:
    options: dict = {}


def simulated_variance_decomposition(m, dr, shocks, ivar, drop, o, s2, zero_idx, y0=None):
    """Percent of the simulated variance when feeding one (orthogonalised) shock at a time, as ``disp_moments``."""
    from .simulate import simulate
    Sigma = np.asarray(m.Sigma_e, float)
    nexo = Sigma.shape[0]
    nvar = len(ivar)
    if nexo == 1:
        return 100.0 * np.ones((nvar, 1))
    shocks = np.asarray(shocks, float)
    iv = np.nonzero(np.diag(Sigma) != 0)[0]
    chol_S = np.linalg.cholesky(Sigma[np.ix_(iv, iv)]).T         # MATLAB chol: upper
    shock_mat = np.zeros_like(shocks)
    shock_mat[:, iv] = np.linalg.solve(chol_S.T, shocks[:, iv].T).T       # shocks / chol_S
    out = np.zeros((nvar, nexo))
    for j in iv:
        tmp = np.zeros_like(shocks)
        tmp[:, j] = shock_mat[:, j]
        tmp[:, iv] = tmp[:, iv] @ chol_S
        yo = simulate(dr, tmp, order=o["order"], pruning=bool(o["pruning"]), y0=y0)[drop:][:, ivar]
        yo = get_filtered_time_series(yo, yo.mean(axis=0), o)
        out[:, j] = np.var(yo, axis=0, ddof=1) / s2 * 100.0
    out[zero_idx, :] = np.nan
    return out


def stoch_simul_moments(m, var_list=None, dr=None, *, shocks=None, rng=None, H=None, **options) -> Moments:
    """Everything ``stoch_simul`` reports for ``var_list``: theoretical moments if ``periods == 0``
    (the default), otherwise moments of a simulation of ``periods`` periods (Dynare's ``simult``: shocks
    drawn as ``randn(periods, n) @ chol(Sigma_e)`` unless ``shocks`` is given; ``rng`` is a numpy
    Generator). Results are in ``Moments``; ``Moments.format()`` prints them like Dynare."""
    o = _options(m, dict(options))
    if o["periods"] == 0:
        return theoretical_moments(m, dr, var_list, H=H, **options)
    from .simulate import simulate
    order = o["order"]
    if dr is None:
        dr = solve(m, order)
    _Rules(m, dr)                                 # checks that dr matches the model
    ivar = _var_indices(m, var_list)
    periods = int(o["periods"])
    if periods <= o["drop"]:
        raise ValueError("STOCH_SIMUL error: The horizon of simulation is shorter than the number of "
                         "observations to be dropped. Either increase periods or decrease drop.")
    Sigma = np.asarray(m.Sigma_e, float)
    if shocks is None:
        rng = np.random.default_rng() if rng is None else rng
        iv = np.nonzero(np.diag(Sigma) != 0)[0]
        shocks = np.zeros((periods, Sigma.shape[0]))
        shocks[:, iv] = rng.standard_normal((periods, len(iv))) @ np.linalg.cholesky(Sigma[np.ix_(iv, iv)]).T
    y = simulate(dr, shocks, order=order, pruning=bool(o["pruning"]))
    keys = {k: v for k, v in options.items() if k in _DEFAULTS or k in ("bandpass_filter", "steps")}
    return simulated_moments(y, ivar=ivar, shocks=shocks, dr=dr, model=m, **keys)


# ------------------------------------------------------------------------------------ text output
def _fmt(x, w, p):
    if np.isnan(x):
        s = "NaN"
    elif np.isinf(x):
        s = "Inf" if x > 0 else "-Inf"
    else:
        s = f"{x:.{p}f}"
        if s.startswith("-") and not s.strip("-0."):        # a value that rounds to zero prints as 0.00
            s = s[1:]
    return s.rjust(w)


def _dyntable(title, headers, labels, values, label_width, val_width, val_precis):
    """Port of ``dyntable.m``."""
    values = np.atleast_2d(np.asarray(values, float))
    lw = max(max(len(h) for h in [headers[0]] + list(labels)) + 2, label_width or 0)
    fin = values[np.isfinite(values)]
    if fin.size == 0:
        vlen = 4
    else:
        with np.errstate(divide="ignore"):
            vlen = max(int(np.ceil(np.max(np.log10(np.abs(fin))))) if np.any(fin != 0) else 1, 1) + val_precis + 1
    if np.any(values < 0):
        vlen += 1
    hlen = max((len(h) for h in headers[1:]), default=0)
    vw = max(max(hlen, vlen) + 2, val_width or 0)
    lines = []
    if title:
        lines += ["", "", title]
    lines.append(headers[0].ljust(lw) + "".join(h.rjust(vw) for h in headers[1:]))
    for lab, row in zip(labels, values):
        lines.append(lab.ljust(lw) + "".join(_fmt(v, vw, val_precis) for v in row))
    return "\n".join(lines)


def _g(x):
    return f"{x:g}"


def _filter_subtitle(title, mom):
    if mom.filter == "hp":
        return f"{title} (HP filter, lambda = {_g(mom.filter_param)})"
    if mom.filter == "one-sided hp":
        return f"{title} (One-sided HP filter, lambda = {_g(mom.filter_param)})"
    if mom.filter == "bandpass":
        return f"{title} (Bandpass filter, ({_g(mom.filter_param[0])} {_g(mom.filter_param[1])}))"
    return title


def format_moments(mom: Moments) -> str:
    """Text tables in the layout of Dynare's ``stoch_simul`` output."""
    out = []
    names = list(mom.var_names)
    lh = max(len(n) for n in names) + 2
    approx = "APPROXIMATED " if mom.order == 2 and mom.kind == "theoretical" else ""
    if mom.kind == "simulated":
        title = _filter_subtitle("MOMENTS OF SIMULATED VARIABLES", mom)
        z = np.column_stack([mom.mean, mom.std, mom.variance, mom.skewness, mom.kurtosis])
        out.append(_dyntable(title, ["VARIABLE", "MEAN", "STD. DEV.", "VARIANCE", "SKEWNESS", "KURTOSIS"],
                             names, z, lh, 16, 6))
        if mom.corr is not None:
            out.append(_dyntable(_filter_subtitle("CORRELATION OF SIMULATED VARIABLES", mom),
                                 ["VARIABLE"] + names, names, mom.corr, lh, 8, 4))
        if mom.autocorr:
            z = np.column_stack([np.diag(a) for a in mom.autocorr])
            out.append(_dyntable(_filter_subtitle("AUTOCORRELATION OF SIMULATED VARIABLES", mom),
                                 ["VARIABLE"] + [str(i) for i in range(1, mom.ar + 1)], names, z, lh, 8, 4))
        if mom.variance_decomposition is not None:
            vd = mom.variance_decomposition
            out.append(_dyntable(_filter_subtitle("VARIANCE DECOMPOSITION SIMULATING ONE SHOCK AT A TIME (in percent)", mom),
                                 [" "] + mom.exo_names + ["Tot. lin. contr."], names,
                                 np.column_stack([vd, vd.sum(axis=1)]), lh, 8, 2))
            out.append("Note: numbers do not add up to 100 due to non-zero correlation of simulated shocks in small "
                       "samples" if mom.order == 1 else
                       "Note: numbers do not add up to 100 due to i) non-zero correlation of simulated shocks in small "
                       "samples and ii) nonlinearity")
        return "\n".join(out) + "\n"
    if mom.kind == "pruned":
        title = "THEORETICAL MOMENTS BASED ON PRUNED STATE SPACE"
    else:
        title = _filter_subtitle(f"{approx}THEORETICAL MOMENTS", mom)
    out.append(_dyntable(title, ["VARIABLE", "MEAN", "STD. DEV.", "VARIANCE"], names,
                         np.column_stack([mom.mean, mom.std, mom.variance]), lh, 11, 4))
    st = np.nonzero(mom.stationary)[0]

    def decomposition_tables():
        if mom.variance_decomposition is not None and len(mom.exo_names) > 1:
            t = f"{approx}VARIANCE DECOMPOSITION (in percent)"
            if mom.kind == "pruned":
                t = "APPROXIMATED VARIANCE DECOMPOSITION (in percent)"
            t = _filter_subtitle(t, mom)
            out.append(_dyntable(t, [" "] + mom.exo_names, [names[i] for i in st],
                                 mom.variance_decomposition[st], lh, 8, 2))
            if mom.variance_decomposition_me is not None:
                out.append(_dyntable(t + " WITH MEASUREMENT ERROR", [" "] + mom.exo_names + ["ME"], mom.me_var_names,
                                     mom.variance_decomposition_me, lh, 8, 2))
        if mom.conditional_variance_decomposition is not None:
            ap = "APPROXIMATED " if mom.order == 2 else ""
            for cvd, addon, hdr in ((mom.conditional_variance_decomposition, "", mom.exo_names),
                                    (mom.conditional_variance_decomposition_me, " - WITH MEASUREMENT ERROR",
                                     mom.exo_names + ["ME"])):
                if cvd is None:
                    continue
                labs = names if addon == "" else mom.me_var_names
                out.append("\n" + f"{ap}CONDITIONAL VARIANCE DECOMPOSITION (in percent){addon}")
                for j, h in enumerate(mom.conditional_steps):
                    out.append(f"Period {h}:")
                    out.append(_dyntable("", [" "] + hdr, labs, 100 * cvd[:, j, :], max(len(x) for x in labs) + 2, 8, 2))

    def correlation_tables():
        if mom.corr is not None:
            i1 = np.nonzero(np.abs(mom.variance) > 1e-12)[0]
            lab = [names[i] for i in i1]
            t = "MATRIX OF CORRELATIONS" if mom.kind == "theoretical" else "MATRIX OF CORRELATIONS BASED ON PRUNED STATE SPACE"
            out.append(_dyntable(_filter_subtitle(f"{approx}{t}", mom) if mom.kind == "theoretical" else t,
                                 ["Variables"] + lab, lab, mom.corr[np.ix_(i1, i1)], max(len(x) for x in lab) + 2, 8, 4))
        if mom.autocorr:
            i1 = np.nonzero(np.abs(mom.variance) > 1e-12)[0]
            lab = [names[i] for i in i1]
            z = np.column_stack([np.diag(a)[i1] for a in mom.autocorr])
            t = "COEFFICIENTS OF AUTOCORRELATION" if mom.kind == "theoretical" else \
                "COEFFICIENTS OF AUTOCORRELATION BASED ON PRUNED STATE SPACE"
            out.append(_dyntable(_filter_subtitle(f"{approx}{t}", mom) if mom.kind == "theoretical" else t,
                                 ["Order "] + [str(i) for i in range(1, mom.ar + 1)], lab, z,
                                 max(len(x) for x in lab) + 2, 8, 4))

    # Dynare prints the decompositions before the correlations, except for the pruned state space
    if mom.kind == "pruned":
        correlation_tables()
        decomposition_tables()
    else:
        decomposition_tables()
        correlation_tables()
    return "\n".join(out) + "\n"
