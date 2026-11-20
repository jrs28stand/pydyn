"""Kalman filter, Kalman smoother and historical shock decomposition at the model's current
parameters: the counterpart of Dynare's ``calib_smoother``, ``shock_decomposition`` and
``initial_condition_decomposition`` commands.

This module is derived from Dynare 7.2 (GPL-3.0-or-later, Copyright Dynare Team):
``matlab/kalman/DsgeSmoother.m``, ``missing_DiffuseKalmanSmootherH1_Z.m``,
``missing_DiffuseKalmanSmootherH3_Z.m``, ``store_smoother_results.m``,
``matlab/estimation/compute_Pinf_Pstar.m``, ``matlab/shock_decomposition/shock_decomposition.m``,
``initial_condition_decomposition.m`` and ``make_the_groups`` in ``plot_shock_decomposition.m``.

It is self-contained on purpose: it does not use ``pydyn.kalman`` / ``pydyn.estimation``.

Conventions (all Dynare's, see ``SmootherResult``):

* the state vector holds every endogenous variable (Dynare's default, ``smoother_redux`` off);
  arrays are in declaration order;
* variables are returned as Dynare stores them: deviation plus steady state, or plus the log of
  the steady state with ``loglinear``; observed variables under ``prefilter=1`` get the sample
  mean of the data instead;
* ``FilteredVariables[name][t]`` is the forecast of period ``t+1`` made in period ``t``
  (Dynare's ``aK(1,:,2:end)``), so its last entry is an out-of-sample forecast.
"""
from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import linalg

__all__ = ["smooth", "shock_decomposition", "initial_condition_decomposition", "group_shocks",
           "check", "load_data", "SmootherResult", "ShockDecomposition", "SmootherError"]


class SmootherError(Exception):
    pass


# --------------------------------------------------------------------------------- options
_DEFAULTS = dict(
    datafile=None, first_obs=1, nobs=None, prefilter=0, loglinear=False, logdata=False,
    presample=0, diffuse_filter=False, lik_init=1, kalman_algo=0, kalman_tol=1e-10,
    diffuse_kalman_tol=1e-6, qz_criterium=None, filtered_vars=False, filter_step_ahead=None,
    filter_covariance=False, smoothed_state_uncertainty=False, parameter_set="calibration",
    varobs=None, replicate_dynare_bugs=False,
)
_STATEMENTS = ("estimation", "calib_smoother")
_NATIVE_OPTION = re.compile(r"^\s*options_\.(\w+)\s*=\s*(true|false|[-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*;")


def _statement_options(model, kinds):
    """Options of the given statements, merged in file order (as Dynare accumulates options_)."""
    out = {}
    for st in getattr(model, "statements", []):
        if st.get("statementName") in kinds:
            for k, v in (st.get("options") or {}).items():
                out[k.split(".")[-1] if k.startswith("shock_decomp.") else k] = v
        elif st.get("statementName") in ("native", "verbatim"):
            # plain MATLAB such as "options_.logdata = 1;" (calib_smoother has no logdata option)
            mt = _NATIVE_OPTION.match(st.get("string", ""))
            if mt and mt.group(1) in _DEFAULTS:
                val = mt.group(2)
                out[mt.group(1)] = {"true": True, "false": False}.get(val, None)
                if out[mt.group(1)] is None:
                    out[mt.group(1)] = float(val) if re.search(r"[.eE]", val) else int(val)
    return out


def _scalar(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) != 1:
            raise NotImplementedError(f"vector-valued option {v!r} (recursive estimation) is not supported")
        return v[0]
    return v


def _resolve_options(model, user, extra_statements=()):
    opts = dict(_DEFAULTS)
    from_file = _statement_options(model, _STATEMENTS + tuple(extra_statements))
    opts.update({k: v for k, v in from_file.items() if k in _DEFAULTS})
    unknown = set(user) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown smoother option(s): {sorted(unknown)}")
    opts.update({k: v for k, v in user.items() if v is not None or k in ("nobs", "filter_step_ahead")})
    opts["first_obs"] = int(_scalar(opts["first_obs"]))
    if opts["nobs"] is not None:
        opts["nobs"] = int(_scalar(opts["nobs"]))
    if opts["parameter_set"] not in (None, "calibration"):
        raise NotImplementedError("only parameter_set=calibration is available (no estimation results here)")
    if int(opts["prefilter"]) not in (0, 1):
        raise ValueError("prefilter must be 0 or 1")
    fsa = opts["filter_step_ahead"]
    if fsa is not None:
        fsa = sorted({int(k) for k in np.atleast_1d(fsa)})
        if fsa and fsa[0] < 1:
            raise ValueError("filter_step_ahead must hold positive integers")
    if not fsa:
        fsa = [1] if opts["filtered_vars"] else []
    opts["filter_step_ahead"] = fsa
    return opts


# --------------------------------------------------------------------------------- data
_M_BLOCK = re.compile(r"([A-Za-z_]\w*)\s*=\s*\[(.*?)\]\s*('?)\s*;?", re.S)


def _parse_m_file(path):
    """Read a Dynare-style data ``.m`` file made of ``name = [ numbers ];`` assignments."""
    text = "\n".join(line.split("%")[0] for line in Path(path).read_text().splitlines())
    out = {}
    for name, body, _ in _M_BLOCK.findall(text):
        toks = body.replace(";", " ").replace(",", " ").replace("...", " ").split()
        try:
            out[name] = np.array([float(t) for t in toks])     # float() understands NaN, Inf
        except ValueError as exc:
            raise SmootherError(f"{path}: cannot parse the values of {name!r}; only plain numeric "
                                "assignments are understood") from exc
    if not out:
        raise SmootherError(f"{path}: no 'name = [ ... ];' assignment found")
    return out


def _parse_csv(path):
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    names = [c.strip() for c in rows[0]]

    def num(s):
        s = s.strip()
        try:
            return float(s) if s else np.nan
        except ValueError:
            return np.nan
    cols = list(zip(*[[num(c) for c in r] + [np.nan] * (len(names) - len(r)) for r in rows[1:]]))
    return {n: np.array(c) for n, c in zip(names, cols) if n}


def load_data(source, varobs, base_dir=None) -> np.ndarray:
    """Return the observations as a ``(T, len(varobs))`` array, columns ordered like ``varobs``.

    ``source`` is an array (columns already in ``varobs`` order), a dict of series, or the path
    of a ``.m`` (plain ``name = [...]`` assignments), ``.csv`` (header row with names) or
    ``.mat`` file. A path without extension is completed as Dynare does. Missing values are NaN.
    """
    if isinstance(source, np.ndarray) or (isinstance(source, (list, tuple)) and not isinstance(source, str)):
        y = np.array(source, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] != len(varobs):
            raise SmootherError(f"data has {y.shape[1]} columns for {len(varobs)} observed variables")
        return y
    if isinstance(source, dict):
        series = source
    else:
        p = Path(source)
        cands = [p] if p.is_absolute() else [Path(base_dir) / p if base_dir else p, p]
        found = None
        for c in cands:
            for suffix in ("", ".m", ".mat", ".csv"):
                q = Path(str(c) + suffix)
                if q.is_file():
                    found = q
                    break
            if found:
                break
        if found is None:
            raise FileNotFoundError(f"data file {source!r} not found (looked in {[str(c) for c in cands]})")
        ext = found.suffix.lower()
        if ext == ".m":
            series = _parse_m_file(found)
        elif ext == ".csv":
            series = _parse_csv(found)
        elif ext == ".mat":
            from scipy.io import loadmat
            series = {k: np.asarray(v, float).ravel() for k, v in loadmat(found).items() if not k.startswith("__")}
        else:
            raise SmootherError(f"unsupported data file type {found.suffix!r} (use .m, .csv or .mat)")
    missing = [v for v in varobs if v not in series]
    if missing:
        raise SmootherError(f"observed variable(s) {missing} not in the data")
    cols = [np.asarray(series[v], float).ravel() for v in varobs]
    if len({len(c) for c in cols}) != 1:
        raise SmootherError("observed series have different lengths")
    return np.column_stack(cols)


# --------------------------------------------------------------------------------- results
@dataclass
class SmootherResult:
    """Smoother output, named after the fields of Dynare's ``oo_``.

    Dicts are keyed by variable / shock name and hold arrays of length ``T``. Higher-dimensional
    arrays keep Dynare's layout: ``FilteredVariablesKStepAhead`` is
    ``(len(filter_step_ahead), n_endo, T + max step)``, ``FilteredVariablesKStepAheadVariances``
    is ``(steps, n_endo, n_endo, T + max step)``, ``Variance`` is ``(n_endo, n_endo, T + 1)``
    and ``State_uncertainty`` is ``(n_endo, n_endo, T)``, all in declaration order.
    """
    endo_names: list
    exo_names: list
    varobs: list
    SmoothedVariables: dict
    UpdatedVariables: dict
    FilteredVariables: dict
    SmoothedShocks: dict
    SmoothedMeasurementErrors: dict
    Constant: dict
    Init_State: dict
    Init_State_Vector: np.ndarray
    SteadyState: np.ndarray
    loglinear: bool
    FilteredVariablesKStepAhead: np.ndarray | None = None
    FilteredVariablesKStepAheadVariances: np.ndarray | None = None
    filter_step_ahead: list = field(default_factory=list)
    Variance: np.ndarray | None = None
    State_uncertainty: np.ndarray | None = None
    Init_State_uncertainty: np.ndarray | None = None
    # --- working arrays (deviations from the constant, declaration order)
    smoothed: np.ndarray = None        # (T, n_endo) smoothed deviations
    updated: np.ndarray = None         # (T, n_endo)
    init_state: np.ndarray = None      # (n_endo,) smoothed deviation in period 0
    shocks: np.ndarray = None          # (T, n_exo)
    measurement_errors: np.ndarray = None   # (T, n_obs); NaN where the observation is missing
    data: np.ndarray = None            # (T, n_obs) data after log / prefilter, constant removed
    obs_constant: np.ndarray = None    # (n_obs,) constant removed from the data
    me_variance: np.ndarray = None     # (n_obs,) measurement-error variances
    G: np.ndarray = None               # transition used: dev_t = G dev_{t-1} + H u_t
    H: np.ndarray = None
    obs_index: np.ndarray = None       # declaration indices of the observed variables
    kalman_algo: int = 0               # filter actually used (1, 2, 3 or 4, Dynare's numbering)
    diffuse_periods: int = 0
    options: dict = field(default_factory=dict)

    @property
    def nobs(self):
        return self.smoothed.shape[0]


@dataclass
class ShockDecomposition:
    """``z`` has Dynare's layout ``(n_endo, n_columns, T)``: one column per shock (or per
    variable for the initial-condition decomposition, or per group), then the contribution of the
    initial conditions, then the smoothed deviation. ``columns`` labels the second axis."""
    z: np.ndarray
    endo_names: list
    columns: list
    kind: str = "shocks"               # "shocks", "initial_conditions" or "groups"
    init_state: int = 0

    def __getitem__(self, name):
        return dict(zip(self.columns, self.z[self.endo_names.index(name)]))


# --------------------------------------------------------------------------------- linear algebra
def _rank(a, tol):
    if a.size == 0:
        return 0
    return int(np.sum(np.linalg.svd(a, compute_uv=False) > tol))


def _rcond(a):
    """Reciprocal condition number in the 1-norm (MATLAB's rcond up to its estimation error)."""
    a = np.atleast_2d(a)
    if not np.all(np.isfinite(a)):
        return 0.0
    n1 = np.linalg.norm(a, 1)
    if n1 == 0.0:
        return 0.0
    try:
        with np.errstate(all="ignore"):
            inv = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return 0.0
    if not np.all(np.isfinite(inv)):
        return 0.0
    return 1.0 / (n1 * np.linalg.norm(inv, 1))


def compute_Pinf_Pstar(mf, T, R, Q, qz_criterium=1 + 1e-6):
    """Initial covariances of the diffuse filter (port of Dynare's ``compute_Pinf_Pstar``).

    ``mf``: indices of the observed variables in the state vector. Returns ``(Pstar, Pinf)``.
    """
    n_all = T.shape[0]
    if not np.any(Q):
        return np.zeros_like(T), np.zeros_like(T)
    colmax = np.max(np.abs(T), axis=0)
    indx = np.where(colmax >= 1e-10)[0]
    indx0 = np.where(colmax < 1e-10)[0]
    np0 = len(indx0)
    T0, R0 = T[np.ix_(indx0, indx)], R[indx0]
    Tr, Rr = T[np.ix_(indx, indx)], R[indx]
    n = Tr.shape[0]
    thr = 2 - qz_criterium
    ST, QT, nk = linalg.schur(Tr, output="real", sort=lambda re, im: np.hypot(re, im) > thr)
    R1 = QT.T @ Rr
    B = R1 @ Q @ R1.T
    Pstar = np.zeros((n, n))
    if n > nk:
        Pstar[nk:, nk:] = linalg.solve_discrete_lyapunov(ST[nk:, nk:], B[nk:, nk:])
        Pstar = (Pstar + Pstar.T) / 2
    if np0:
        Pstar = linalg.block_diag(np.zeros((np0, np0)), Pstar)
        ST = np.hstack([np.zeros((np0 + n, np0)), np.vstack([T0 @ QT, ST])])
        R1 = np.vstack([R0, R1])
        STinf = ST[np0:np0 + nk, np0:np0 + nk]
        iSTinf = np.linalg.inv(STinf) if nk else np.zeros((0, 0))
        ST0 = ST.copy()
        ST0[:, :np0 + nk] = 0
        ST00 = ST[:np0, np0:np0 + nk]
        ST0[:np0, np0 + nk:] = ST[:np0, np0 + nk:] - ST00 @ (iSTinf @ ST[np0:np0 + nk, np0 + nk:])
        R10 = R1.copy()
        R10[:np0] = R1[:np0] - ST00 @ (iSTinf @ R1[np0:np0 + nk])
        ST0[np0:np0 + nk] = 0
        R10[np0:np0 + nk] = 0
        Pstar = ST0 @ Pstar @ ST0.T + R10 @ Q @ R10.T
    Pinf = np.zeros((n, n))
    Pinf[:nk, :nk] = np.eye(nk)
    perm = np.concatenate([indx0, indx])
    if np0:
        STinf0 = ST00 @ (np.eye(nk) - iSTinf @ (STinf - np.eye(nk)))
        Pinf = linalg.block_diag(np.zeros((np0, np0)), Pinf)
        QT = linalg.block_diag(np.eye(np0), QT)
        QTinf = QT.copy()
        QTinf[:np0, np0:np0 + nk] = STinf0
        tmp = np.empty_like(QTinf)
        tmp[perm] = QTinf
        QTinf = tmp
        STinf1 = np.hstack([np.zeros((np0 + n, np0)),
                            np.vstack([STinf0, np.eye(nk), np.zeros((n - nk, nk))]),
                            np.zeros((np0 + n, n - nk))])
        mask = np.isin(perm, mf)                 # Dynare's indexing, kept as is
        for k in range(nk):
            if np.linalg.norm(QTinf[mask] @ ST[perm, k + np0]) < 1e-8:
                Pinf[k + np0, k + np0] = 0
        Pinf = STinf1 @ Pinf @ STinf1.T
        tmp = np.empty_like(QT)
        tmp[perm] = QT
        QT = tmp
    else:
        for k in range(nk):
            if np.linalg.norm(QT[np.asarray(mf)] @ ST[:, k]) < 1e-8:
                Pinf[k, k] = 0
    assert QT.shape[0] == n_all
    return QT @ Pstar @ QT.T, QT @ Pinf @ QT.T


def _stationary_variance(T, R, Q, qz_criterium):
    RQR = R @ Q @ R.T
    eig = np.abs(np.linalg.eigvals(T))
    if np.any(eig > min(qz_criterium, 2 - qz_criterium)):
        raise SmootherError("the model has a unit root: the stationary initialisation (lik_init=1) "
                            "is not available, use diffuse_filter=True")
    P = linalg.solve_discrete_lyapunov(T, RQR)
    return (P + P.T) / 2


# --------------------------------------------------------------------------------- multivariate smoother
def _smoother_multivariate(T, Z, R, Q, H, Pinf1, Pstar1, Y, di_list, nk, kalman_tol, dk_tol,
                           want_V, want_PK, compat):
    """Port of ``missing_DiffuseKalmanSmootherH1_Z``. ``Y`` is (n_obs, T). Returns ``None`` when
    Dynare's routine reports a singular forecast-error variance (the caller then switches to the
    univariate smoother, as ``DsgeSmoother`` does).

    ``compat=True`` reproduces three defects of Dynare 7.2 in the diffuse phase (see ``smooth``).
    """
    mm = T.shape[0]
    pp, smpl = Y.shape
    Tt = T.T
    a = np.zeros((smpl + 1, mm))
    atilde = np.zeros((smpl, mm))
    v = np.zeros((smpl, pp))
    aK = np.zeros((nk, mm, smpl + nk))
    PK = np.zeros((nk, mm, mm, smpl + nk)) if want_PK else None
    iF = np.zeros((smpl, pp, pp))
    L = np.zeros((smpl, mm, mm))
    P = np.zeros((smpl + 1, mm, mm))
    QQ = R @ Q @ R.T
    QRt = Q @ R.T
    a0, P0 = np.zeros(mm), Pstar1.copy()
    diffuse = Pinf1 is not None
    Pstar = {0: Pstar1.copy()}
    Pinf = {0: Pinf1.copy()} if diffuse else {}
    Linf, Lstar, Kstar, iFinf, iFstar, Fstar, kind = {}, {}, {}, {}, {}, {}, {}

    t = 0
    new_rank = _rank(Z @ Pinf[0] @ Z.T, dk_tol) if diffuse else 0
    if new_rank:
        Pinf_init, Pstar_init, a_init = Pinf[0].copy(), Pstar[0].copy(), a[0].copy()
        a[0] = T @ a[0]
        Pstar[0] = T @ Pstar[0] @ Tt + QQ
    while new_rank and t < smpl:
        di = di_list[t]
        store_forecast = True
        if len(di) == 0:
            kind[t] = "missing"
            atilde[t] = a[t]
            Linf[t] = T
            Pstar[t + 1] = T @ Pstar[t] @ Tt + QQ
            Pinf[t + 1] = T @ Pinf[t] @ Tt
            store_forecast = not compat           # Dynare leaves aK at zero here
        else:
            ZZ = Z[di]
            vt = Y[di, t] - ZZ @ a[t]
            v[t, di] = vt
            Finf = ZZ @ Pinf[t] @ ZZ.T
            # Dynare tests rcond(Finf) first; a Finf that is zero up to rounding (1e-30, say) but not
            # exactly zero then passes as regular and is inverted. Test for zero first instead.
            finf_zero = bool(np.all(np.abs(Finf) < dk_tol))
            if finf_zero or _rcond(Finf) < dk_tol:
                if not finf_zero:
                    return None                                          # error 420
                Fs = ZZ @ Pstar[t] @ ZZ.T + H[np.ix_(di, di)]
                Fstar[t] = Fs
                if _rcond(Fs) < kalman_tol:
                    if not np.all(np.abs(Fs) < kalman_tol):
                        return None                                      # error 421
                    kind[t] = "skip"
                    Pstar[t + 1] = T @ Pstar[t] @ Tt + QQ
                    Pinf[t + 1] = T @ Pinf[t] @ Tt
                    upd = a[t]
                    iFstar[t] = np.zeros_like(Fs)
                    Kstar[t] = np.zeros((mm, len(di)))
                    Lstar[t] = np.zeros((mm, mm)) if compat else T
                else:
                    kind[t] = "star"
                    iFstar[t] = np.linalg.inv(Fs)
                    Kstar[t] = Pstar[t] @ ZZ.T @ iFstar[t]
                    Pinf[t + 1] = T @ Pinf[t] @ Tt
                    Lstar[t] = T - T @ Kstar[t] @ ZZ
                    Pstar[t + 1] = T @ Pstar[t] @ Lstar[t].T + QQ
                    upd = a[t] + Kstar[t] @ vt
                # Dynare overwrites the state forecast with T*atilde where atilde was never set
                atilde[t] = 0.0 if compat else upd
            else:
                kind[t] = "inf"
                iFinf[t] = np.linalg.inv(Finf)
                Kinf = Pinf[t] @ ZZ.T @ iFinf[t]
                atilde[t] = a[t] + Kinf @ vt
                Linf[t] = T - T @ Kinf @ ZZ
                Fs = ZZ @ Pstar[t] @ ZZ.T + H[np.ix_(di, di)]
                Fstar[t] = Fs
                Kstar[t] = (Pstar[t] @ ZZ.T - Kinf @ Fs) @ iFinf[t]
                Pstar[t + 1] = T @ Pstar[t] @ Linf[t].T - T @ Kinf @ Finf @ Kstar[t].T @ Tt + QQ
                Pinf[t + 1] = T @ Pinf[t] @ Linf[t].T
        a[t + 1] = T @ atilde[t]
        if store_forecast:
            aK[0, :, t + 1] = a[t + 1]
            for j in range(1, nk):
                aK[j, :, t + 1 + j] = T @ aK[j - 1, :, t + j]
        t += 1
        new_rank = _rank(Z @ Pinf[t] @ Z.T, dk_tol)
    d = t
    P[d] = Pstar[d]

    while t < smpl:
        P[t] = np.tril(P[t]) + np.tril(P[t], -1).T
        di = di_list[t]
        if len(di) == 0:
            atilde[t] = a[t]
            L[t] = T
            P[t + 1] = T @ P[t] @ Tt + QQ
        else:
            ZZ = Z[di]
            vt = Y[di, t] - ZZ @ a[t]
            v[t, di] = vt
            F = ZZ @ P[t] @ ZZ.T + H[np.ix_(di, di)]
            dF = np.diag(F)
            if np.any(dF < kalman_tol):
                return None                                              # error 422
            sig = np.sqrt(dF)
            ss = np.outer(sig, sig)
            if _rcond(F / ss) < kalman_tol:
                return None
            iFt = np.linalg.inv(F / ss) / ss
            iF[t][np.ix_(di, di)] = iFt
            PZI = P[t] @ ZZ.T @ iFt
            atilde[t] = a[t] + PZI @ vt
            L[t] = T - T @ PZI @ ZZ
            P[t + 1] = T @ P[t] @ L[t].T + QQ
        a[t + 1] = T @ atilde[t]
        Pf = P[t + 1]
        aK[0, :, t + 1] = a[t + 1]
        for j in range(nk):
            if want_PK:
                if j > 0:
                    Pf = T @ Pf @ Tt + QQ
                PK[j, :, :, t + 1 + j] = Pf
            if j > 0:
                aK[j, :, t + 1 + j] = T @ aK[j - 1, :, t + j]
        t += 1

    # ---- backward pass
    alphahat = np.zeros((smpl, mm))
    etahat = np.zeros((smpl, Q.shape[0]))
    r = np.zeros((smpl + 1, mm))
    N = np.zeros((smpl + 1, mm, mm)) if want_V else None
    V = np.zeros((smpl, mm, mm)) if want_V else None
    V0 = None
    for t in range(smpl - 1, d - 1, -1):
        di = di_list[t]
        if len(di) == 0:
            r[t] = L[t].T @ r[t + 1]
            if want_V:
                N[t] = L[t].T @ N[t + 1] @ L[t]
        else:
            ZZ = Z[di]
            iFt = iF[t][np.ix_(di, di)]
            r[t] = ZZ.T @ iFt @ v[t, di] + L[t].T @ r[t + 1]
            if want_V:
                N[t] = ZZ.T @ iFt @ ZZ + L[t].T @ N[t + 1] @ L[t]
        alphahat[t] = a[t] + P[t] @ r[t]
        etahat[t] = QRt @ r[t]
        if want_V:
            V[t] = P[t] - P[t] @ N[t] @ P[t]
    if d == 0:
        alphahat0 = a0 + P0 @ (Tt @ r[0])
        if want_V:
            V0 = P0 - P0 @ (Tt @ N[0] @ T) @ P0
    else:
        r0 = np.zeros((d + 1, mm))
        r1 = np.zeros((d + 1, mm))
        r0[d] = r[d]
        if want_V:
            N1 = np.zeros((d + 1, mm, mm))
            N2 = np.zeros((d + 1, mm, mm))
        for t in range(d - 1, -1, -1):
            di = di_list[t]
            k = kind[t]
            if k == "missing" or (k == "skip" and not compat):
                r1[t] = Tt @ r1[t + 1]
                if not compat:
                    r0[t] = Tt @ r0[t + 1]
                    if want_V:
                        N[t] = Tt @ N[t + 1] @ T
                        N1[t] = Tt @ N1[t + 1] @ T
                        N2[t] = Tt @ N2[t + 1] @ T
            elif k == "inf":
                ZZ = Z[di]
                Li = Linf[t]
                r0[t] = Li.T @ r0[t + 1]
                r1[t] = ZZ.T @ (iFinf[t] @ v[t, di] - Kstar[t].T @ (Tt @ r0[t + 1])) + Li.T @ r1[t + 1]
                if want_V:
                    L_1 = -T @ Kstar[t] @ ZZ
                    N[t] = Li.T @ N[t + 1] @ Li
                    N1[t] = ZZ.T @ iFinf[t] @ ZZ + Li.T @ N1[t + 1] @ Li + L_1.T @ N[t + 1] @ Li
                    N2[t] = (ZZ.T @ (-iFinf[t] @ Fstar[t] @ iFinf[t]) @ ZZ + Li.T @ N2[t + 1] @ Li
                             + Li.T @ N1[t + 1] @ L_1 + L_1.T @ N1[t + 1].T @ Li + L_1.T @ N[t + 1] @ L_1)
            else:                                   # F_inf = 0 in this period
                ZZ = Z[di]
                Ls = Lstar[t]
                sign = -1.0 if compat else 1.0      # Dynare has "- Lstar' * r0"
                r0[t] = ZZ.T @ iFstar[t] @ v[t, di] + sign * (Ls.T @ r0[t + 1])
                r1[t] = Tt @ r1[t + 1]
                if want_V:
                    N[t] = ZZ.T @ iFstar[t] @ ZZ + Ls.T @ N[t + 1] @ Ls
                    N1[t] = Tt @ N1[t + 1] @ Ls
                    N2[t] = Tt @ N2[t + 1] @ (Tt if compat else T)     # Dynare has T' twice
            alphahat[t] = a[t] + Pstar[t] @ r0[t] + Pinf[t] @ r1[t]
            etahat[t] = QRt @ r0[t]
            if want_V:
                x = Pinf[t] @ N1[t] @ Pstar[t]
                V[t] = Pstar[t] - Pstar[t] @ N[t] @ Pstar[t] - x.T - x - Pinf[t] @ N2[t] @ Pinf[t]
        alphahat0 = a_init + Pstar_init @ (Tt @ r0[0]) + Pinf_init @ (Tt @ r1[0])
        if want_V:
            x = Pinf_init @ (Tt @ N1[0] @ T) @ Pstar_init
            V0 = (Pstar_init - Pstar_init @ (Tt @ N[0] @ T) @ Pstar_init - x.T - x
                  - Pinf_init @ (Tt @ N2[0] @ T) @ Pinf_init)
    return dict(alphahat=alphahat, etahat=etahat, updated=atilde, P=P, aK=aK, PK=PK, V=V, V0=V0,
                d=d, alphahat0=alphahat0)


# --------------------------------------------------------------------------------- univariate smoother
def _smoother_univariate(T, Z, R, Q, Hd, Pinf1, Pstar1, Y, di_list, nk, kalman_tol, dk_tol,
                         want_V, want_PK, compat=False):
    """Port of ``missing_DiffuseKalmanSmootherH3_Z`` (no OccBin). ``Hd`` is the diagonal of the
    measurement-error covariance. ``compat=True`` keeps a defect of Dynare 7.2 in the smoothed
    state uncertainty of the diffuse phase (see ``smooth``)."""
    mm = T.shape[0]
    pp, smpl = Y.shape
    Tt = T.T
    I = np.eye(mm)
    a = np.zeros((smpl, mm))
    a1 = np.zeros((smpl + 1, mm))
    v = np.zeros((smpl, pp))
    aK = np.zeros((nk, mm, smpl + nk))
    PK = np.zeros((nk, mm, mm, smpl + nk)) if want_PK else None
    Fstar = np.zeros((smpl, pp))
    Finf = np.zeros((smpl, pp))
    Fi = np.zeros((smpl, pp))
    Ki = np.zeros((smpl, mm, pp))
    Kstar, Kinf = {}, {}
    P = np.zeros((smpl + 1, mm, mm))
    P1 = np.zeros((smpl + 1, mm, mm))
    QQ = R @ Q @ R.T
    QRt = Q @ R.T
    ainit, Pinit = a1[0].copy(), Pstar1.copy()
    diffuse = Pinf1 is not None
    Pstar = {0: Pstar1.copy()}
    Pinf = {0: Pinf1.copy()} if diffuse else {}
    Pstar_pred, Pinf_pred = {}, {}

    t = 0
    new_rank = _rank(Z @ Pinf[0] @ Z.T, dk_tol) if diffuse else 0
    if new_rank:
        Pinf_init, Pstar_init = Pinf[0].copy(), Pstar[0].copy()
        Pstar[0] = T @ Pstar[0] @ Tt + QQ
    while new_rank and t < smpl:
        a[t] = a1[t]
        Pstar_pred[t], Pinf_pred[t] = Pstar[t].copy(), Pinf[t].copy()
        Kstar[t], Kinf[t] = np.zeros((mm, pp)), np.zeros((mm, pp))
        Ps, Pi = Pstar[t], Pinf[t]
        for i in di_list[t]:
            Zi = Z[i]
            v[t, i] = Y[i, t] - Zi @ a[t]
            Fstar[t, i] = Zi @ Ps @ Zi + Hd[i]
            Finf[t, i] = Zi @ Pi @ Zi
            Ks = Ps @ Zi
            Kstar[t][:, i] = Ks
            if Finf[t, i] > dk_tol:
                Kf = Pi @ Zi
                Kinf[t][:, i] = Kf
                KF = Kf / Finf[t, i]
                a[t] = a[t] + KF * v[t, i]
                Ps = Ps + np.outer(Kf, KF) * (Fstar[t, i] / Finf[t, i]) - np.outer(Ks, KF) - np.outer(KF, Ks)
                Pi = Pi - np.outer(Kf, Kf) / Finf[t, i]
            elif Fstar[t, i] > kalman_tol:
                a[t] = a[t] + Ks * v[t, i] / Fstar[t, i]
                Ps = Ps - np.outer(Ks, Ks) / Fstar[t, i]
        a1[t + 1] = T @ a[t]
        aK[0, :, t + 1] = a1[t + 1]
        for j in range(1, nk):
            aK[j, :, t + 1 + j] = T @ aK[j - 1, :, t + j]
        Pstar[t + 1] = T @ Ps @ Tt + QQ
        Pinf[t + 1] = T @ Pi @ Tt
        t += 1
        new_rank = _rank(Z @ Pinf[t] @ Z.T, dk_tol)
    d = t
    P[d] = Pstar[d]

    while t < smpl:
        a[t] = a1[t]
        P1[t] = P[t]
        Pt = P[t].copy()
        for i in di_list[t]:
            Zi = Z[i]
            v[t, i] = Y[i, t] - Zi @ a[t]
            Fi[t, i] = Zi @ Pt @ Zi + Hd[i]
            Ki[t, :, i] = Pt @ Zi
            if Fi[t, i] > kalman_tol:
                a[t] = a[t] + Ki[t, :, i] * v[t, i] / Fi[t, i]
                Pt = Pt - np.outer(Ki[t, :, i], Ki[t, :, i]) / Fi[t, i]
                Pt = (Pt + Pt.T) / 2
        a1[t + 1] = T @ a[t]
        P[t + 1] = T @ Pt @ Tt + QQ
        P[t + 1] = (P[t + 1] + P[t + 1].T) / 2
        Pf = P[t + 1]
        aK[0, :, t + 1] = a1[t + 1]
        for j in range(nk):
            if want_PK:
                if j > 0:
                    Pf = T @ Pf @ Tt + QQ
                PK[j, :, :, t + 1 + j] = Pf
            if j > 0:
                aK[j, :, t + 1 + j] = T @ aK[j - 1, :, t + j]
        t += 1
    P1[t] = P[t]

    # ---- backward pass
    alphahat = np.zeros((smpl, mm))
    etahat = np.zeros((smpl, Q.shape[0]))
    V = np.zeros((smpl, mm, mm)) if want_V else None
    V0 = None
    ri = np.zeros(mm)
    Ni = np.zeros((mm, mm))
    for t in range(smpl - 1, d - 1, -1):
        for i in di_list[t][::-1]:
            if Fi[t, i] > kalman_tol:
                Zi = Z[i]
                Li = I - np.outer(Ki[t, :, i], Zi) / Fi[t, i]
                ri = Zi / Fi[t, i] * v[t, i] + Li.T @ ri
                if want_V:
                    Ni = np.outer(Zi, Zi) / Fi[t, i] + Li.T @ Ni @ Li
        alphahat[t] = a1[t] + P1[t] @ ri
        etahat[t] = QRt @ ri
        ri = Tt @ ri
        if want_V:
            V[t] = P1[t] - P1[t] @ Ni @ P1[t]
            Ni = Tt @ Ni @ T
    if d == 0:
        alphahat0 = ainit + Pinit @ ri
        if want_V:
            V0 = Pinit - Pinit @ Ni @ Pinit
    else:
        r0, r1 = ri, np.zeros(mm)
        N0, N1, N2 = Ni, np.zeros((mm, mm)), np.zeros((mm, mm))
        for t in range(d - 1, -1, -1):
            for i in di_list[t][::-1]:
                Zi = Z[i]
                if Finf[t, i] > dk_tol:
                    Kf, Ks = Kinf[t][:, i], Kstar[t][:, i]
                    Li = I - np.outer(Kf, Zi) / Finf[t, i]
                    L0 = np.outer(Kf * (Fstar[t, i] / Finf[t, i]) - Ks, Zi) / Finf[t, i]
                    r1 = Zi * v[t, i] / Finf[t, i] + L0.T @ r0 + Li.T @ r1
                    r0 = Li.T @ r0
                    if want_V:
                        zz = np.outer(Zi, Zi)
                        N2 = (-zz / Finf[t, i] ** 2 * Fstar[t, i] + Li.T @ N2 @ Li + Li.T @ N1 @ L0
                              + L0.T @ N1.T @ Li + L0.T @ N0 @ L0)
                        N1 = zz / Finf[t, i] + Li.T @ N1 @ Li + L0.T @ N0 @ Li
                        N0 = Li.T @ N0 @ Li
                elif Fstar[t, i] > kalman_tol:
                    Ks = Kstar[t][:, i]
                    Li = I - np.outer(Ks, Zi) / Fstar[t, i]
                    r0 = Zi / Fstar[t, i] * v[t, i] + Li.T @ r0
                    if want_V:
                        N0 = np.outer(Zi, Zi) / Fstar[t, i] + Li.T @ N0 @ Li
                        if not compat:
                            N1 = N1 @ Li                # Dynare keeps N_1 fixed here
            Ps, Pi = Pstar_pred[t], Pinf_pred[t]
            alphahat[t] = a1[t] + Ps @ r0 + Pi @ r1
            etahat[t] = QRt @ r0
            if want_V:
                x = Pi @ N1 @ Ps
                V[t] = Ps - Ps @ N0 @ Ps - x.T - x - Pi @ N2 @ Pi
            r0, r1 = Tt @ r0, Tt @ r1
            if want_V:
                N0, N1, N2 = Tt @ N0 @ T, Tt @ N1 @ T, Tt @ N2 @ T
        alphahat0 = ainit + Pstar_init @ r0 + Pinf_init @ r1
        if want_V:
            x = Pinf_init @ N1 @ Pstar_init
            V0 = Pstar_init - Pstar_init @ N0 @ Pstar_init - x.T - x - Pinf_init @ N2 @ Pinf_init
    if d == smpl and diffuse:
        warnings.warn("the diffuse filter never left the diffuse phase: the data do not identify "
                      "the initial conditions of the non-stationary variables")
    return dict(alphahat=alphahat, etahat=etahat, updated=a, P=P1, aK=aK, PK=PK, V=V, V0=V0,
                d=d, alphahat0=alphahat0)


# --------------------------------------------------------------------------------- public: smoother
def _loglinear_rules(G, H, ys):
    if np.any(ys <= 0) or not np.all(np.isfinite(ys)):
        raise SmootherError("loglinear needs a strictly positive steady state for every variable "
                            "(models with auxiliary variables for lagged shocks are not handled)")
    return G * ys[None, :] / ys[:, None], H / ys[:, None]


def smooth(model, data=None, sol=None, **options) -> SmootherResult:
    """Kalman filter and smoother at the current parameters (Dynare's ``calib_smoother``).

    ``data``: array ``(T, n_obs)`` with columns in ``varobs`` order, dict of series, or a file
    path; when omitted the ``datafile`` option is used (relative to the ``.mod`` file's folder).
    ``sol``: a first-order solution (computed if omitted).

    Options are Dynare's and default to those of the ``estimation`` / ``calib_smoother``
    statements of the ``.mod`` file, in file order; keyword arguments take precedence:
    ``datafile, first_obs, nobs, prefilter, loglinear, logdata, diffuse_filter, lik_init (1 or 3),
    kalman_algo (0-4), filtered_vars, filter_step_ahead, filter_covariance,
    smoothed_state_uncertainty, kalman_tol, diffuse_kalman_tol, varobs``. ``presample`` is
    accepted and has no effect on the smoother (as in Dynare).

    ``replicate_dynare_bugs`` (default False) concerns the multivariate diffuse smoother
    (``kalman_algo=3``) only, when a period inside the diffuse phase has no observation at all or
    a zero ``F_inf``. Dynare 7.2 then drops terms of the backward recursion; its output breaks the
    transition equation and differs from its own univariate smoother. With False the textbook
    recursion is used (equal to Dynare's ``kalman_algo=4``); True reproduces Dynare's
    ``kalman_algo=3`` output.
    """
    from .perturbation import first_order

    opts = _resolve_options(model, options)
    for st in getattr(model, "statements", []):
        if st.get("statementName") in ("observation_trends", "filter_initial_state", "heteroskedastic_shocks"):
            raise NotImplementedError(f"{st['statementName']} blocks are not supported by the smoother")
    varobs = list(opts["varobs"] or model.varobs)
    if not varobs:
        raise SmootherError("no observed variables: declare varobs or pass varobs=[...]")
    if sol is None:
        sol = first_order(model)
    names = list(model.endo_names)
    n, nx, pp = len(names), model.n_exo, len(varobs)
    try:
        mf = np.array([names.index(v) for v in varobs])
    except ValueError as exc:
        raise SmootherError(f"observed variable not in the model: {exc}") from exc
    ys = np.asarray(sol.ys, float)
    loglinear = bool(opts["loglinear"])
    if loglinear:
        G, Hs = _loglinear_rules(sol.G, sol.H, ys)
        const_all = np.log(ys)
    else:
        G, Hs = np.array(sol.G, float), np.array(sol.H, float)
        const_all = ys.copy()

    # ---- data (Dynare's makedataset)
    src = data if data is not None else opts["datafile"]
    if src is None:
        raise SmootherError("no data: pass data=... or a datafile option")
    raw = load_data(src, varobs, base_dir=Path(model.json_dir).parents[2] if hasattr(model, "json_dir") else None)
    if loglinear and not opts["logdata"]:
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.log(raw)
    first = opts["first_obs"]
    nobs = opts["nobs"] if opts["nobs"] is not None else raw.shape[0] - first + 1
    if first < 1 or nobs < 1 or first - 1 + nobs > raw.shape[0]:
        raise SmootherError(f"first_obs={first}, nobs={nobs} do not fit the {raw.shape[0]} observations in the data")
    obs = raw[first - 1:first - 1 + nobs]
    obs_const = const_all[mf].copy()
    if int(opts["prefilter"]) == 1:
        if np.any(np.abs(obs_const) > 1e-9):
            raise SmootherError("prefilter=1 is inconsistent with observed variables that have a "
                                "non-zero steady state (same check as Dynare)")
        obs_const = np.nanmean(obs, axis=0)
    Yd = (obs - obs_const).T                          # (n_obs, T), deviations
    smpl = Yd.shape[1]
    di_list = [np.where(~np.isnan(Yd[:, t]))[0] for t in range(smpl)]

    # ---- state space
    Q = np.array(model.Sigma_e, float)
    Hme = np.diag([float(model.measurement_errors.get(v, 0.0)) for v in varobs])
    extra_me = set(model.measurement_errors) - set(varobs)
    if extra_me:
        raise SmootherError(f"measurement error declared on non-observed variable(s) {sorted(extra_me)}")
    Z = np.zeros((pp, n))
    Z[np.arange(pp), mf] = 1.0
    kalman_algo = int(opts["kalman_algo"])
    lik_init = int(opts["lik_init"])
    if opts["diffuse_filter"] or kalman_algo in (3, 4):
        lik_init = 3
    qz = opts["qz_criterium"]
    if lik_init == 1:
        qz = 1 - 1e-6 if qz is None else qz
        kalman_algo = 2 if kalman_algo == 2 else 1
        Pstar, Pinf = _stationary_variance(G, Hs, Q, qz), None
    elif lik_init == 3:
        qz = 1 + 1e-6 if qz is None else qz
        kalman_algo = 4 if kalman_algo == 4 else 3
        Pstar, Pinf = compute_Pinf_Pstar(mf, G, Hs, Q, qz)
    else:
        raise NotImplementedError(f"lik_init={lik_init}: only 1 (stationary) and 3 (diffuse) are implemented")
    nk = max(opts["filter_step_ahead"]) if opts["filter_step_ahead"] else 1
    want_V, want_PK = bool(opts["smoothed_state_uncertainty"]), bool(opts["filter_covariance"])
    ktol, dtol = float(opts["kalman_tol"]), float(opts["diffuse_kalman_tol"])

    out = None
    if kalman_algo in (1, 3):
        out = _smoother_multivariate(G, Z, Hs, Q, Hme, Pinf, Pstar, Yd, di_list, nk, ktol, dtol,
                                     want_V, want_PK, bool(opts["replicate_dynare_bugs"]))
        if out is None:
            kalman_algo += 1                        # Dynare: switch to the univariate filter
    if out is None:
        out = _smoother_univariate(G, Z, Hs, Q, np.diag(Hme).copy(), Pinf, Pstar, Yd, di_list, nk,
                                   ktol, dtol, want_V, want_PK, bool(opts["replicate_dynare_bugs"]))
    alphahat, updated, aK = out["alphahat"], out["updated"], out["aK"]
    eps = Yd.T - alphahat[:, mf]                      # NaN where the observation is missing

    # ---- Dynare's store_smoother_results
    is_obs = np.zeros(n, bool)
    is_obs[mf] = True
    const = np.tile(const_all, (smpl, 1))
    const[:, mf] = obs_const
    sv, uv = alphahat + const, updated + const
    fv = aK[0, :, 1:smpl + 1].T + const
    kstep = kvar = None
    fsa = opts["filter_step_ahead"]
    if fsa:
        idx = np.array(fsa) - 1
        kstep = aK[idx] + const_all[None, :, None]
        mx = max(fsa)
        for j, step in enumerate(fsa):                # observed variables: constant of the data
            hi = kstep.shape[2] - (mx - step)
            kstep[j, mf, step:hi] = aK[step - 1][mf, step:hi] + obs_const[:, None]
        if want_PK:
            kvar = out["PK"][idx]
    res = SmootherResult(
        endo_names=names, exo_names=list(model.exo_names), varobs=varobs,
        SmoothedVariables={v: sv[:, i] for i, v in enumerate(names)},
        UpdatedVariables={v: uv[:, i] for i, v in enumerate(names)},
        FilteredVariables={v: fv[:, i] for i, v in enumerate(names)},
        SmoothedShocks={v: out["etahat"][:, i] for i, v in enumerate(model.exo_names)},
        SmoothedMeasurementErrors=({v: eps[:, i] for i, v in enumerate(varobs)} if np.any(Hme) else {}),
        Constant={v: const[:, i] for i, v in enumerate(names)},
        Init_State={v: out["alphahat0"][i] + const_all[i] for i, v in enumerate(names)},
        Init_State_Vector=out["alphahat0"] + ys,      # Dynare adds the level steady state, also with loglinear
        SteadyState=ys, loglinear=loglinear,
        FilteredVariablesKStepAhead=kstep, FilteredVariablesKStepAheadVariances=kvar,
        filter_step_ahead=list(fsa),
        Variance=np.moveaxis(out["P"], 0, -1) if want_PK else None,
        State_uncertainty=np.moveaxis(out["V"], 0, -1) if want_V else None,
        Init_State_uncertainty=out["V0"] if want_V else None,
        smoothed=alphahat, updated=updated, init_state=out["alphahat0"], shocks=out["etahat"],
        measurement_errors=eps, data=Yd.T, obs_constant=obs_const, me_variance=np.diag(Hme).copy(), G=G, H=Hs, obs_index=mf,
        kalman_algo=kalman_algo, diffuse_periods=out["d"], options=opts,
    )
    return res


# --------------------------------------------------------------------------------- public: decompositions
def _as_result(model, res_or_data, options):
    if isinstance(res_or_data, SmootherResult):
        if options:
            raise TypeError("smoother options cannot be combined with an existing SmootherResult")
        return res_or_data
    return smooth(model, res_or_data, **options)


def shock_decomposition(model, res_or_data=None, init_state=None, **options) -> ShockDecomposition:
    """Historical shock decomposition (Dynare's ``shock_decomposition``, ``parameter_set=calibration``).

    ``res_or_data``: a ``SmootherResult``, or data / nothing, in which case the smoother is run
    with ``options`` (defaults also read from the file's ``shock_decomposition`` statement).
    ``init_state``: Dynare's option; shocks of periods ``1..init_state`` are attributed to the
    initial conditions (0: condition on the smoothed state of period 0).

    Returns ``z`` of shape ``(n_endo, n_exo + 2, T)``: shocks in declaration order, then initial
    conditions, then the smoothed deviation from the steady state (in logs with ``loglinear``).
    Observed variables under ``prefilter`` are measured from the model's steady state, not from
    the data mean, as in Dynare.
    """
    file_opts = _statement_options(model, ("shock_decomposition",))
    if init_state is None:
        init_state = int(file_opts.get("init_state", 0))
    if not isinstance(res_or_data, SmootherResult):
        merged = {k: v for k, v in file_opts.items() if k in _DEFAULTS}
        merged.update(options)
        options = merged
    res = _as_result(model, res_or_data, options)
    T, n = res.smoothed.shape
    nx = res.shocks.shape[1]
    z = np.zeros((n, nx + 2, T))
    z[:, nx + 1, :] = res.smoothed.T
    for t in range(T):
        if t > 0:
            z[:, :nx, t] = res.G @ z[:, :nx, t - 1]
        if t + 1 > init_state:
            z[:, :nx, t] += res.H * res.shocks[t][None, :]
        z[:, nx, t] = z[:, nx + 1, t] - z[:, :nx, t].sum(axis=1)
    return ShockDecomposition(z=z, endo_names=list(res.endo_names),
                              columns=list(res.exo_names) + ["initial_conditions", "smoothed"],
                              kind="shocks", init_state=int(init_state))


def initial_condition_decomposition(model, res_or_data=None, **options) -> ShockDecomposition:
    """Dynare's ``initial_condition_decomposition``: contribution of each variable's smoothed
    value in period 1 to the later path. ``z`` is ``(n_endo, n_endo + 2, T)``; column ``n_endo``
    (0-based) is the remainder, i.e. the contribution of all shocks from period 2 on, and the
    last column is the smoothed deviation."""
    res = _as_result(model, res_or_data, options)
    T, n = res.smoothed.shape
    z = np.zeros((n, n + 2, T))
    z[:, n + 1, :] = res.smoothed.T
    z[np.arange(n), np.arange(n), 0] = res.smoothed[0]
    for t in range(T):
        if t > 0:
            z[:, :n, t] = res.G @ z[:, :n, t - 1]
        z[:, n, t] = z[:, n + 1, t] - z[:, :n, t].sum(axis=1)
    return ShockDecomposition(z=z, endo_names=list(res.endo_names),
                              columns=list(res.endo_names) + ["shocks", "smoothed"],
                              kind="initial_conditions")


def group_shocks(decomp: ShockDecomposition, groups=None, model=None, name=None) -> ShockDecomposition:
    """Sum shock contributions by group, as Dynare does when plotting with ``use_shock_groups``.

    ``groups`` maps a label to a list of shock names; or pass ``model`` (and optionally the
    ``name`` of a ``shock_groups`` block) to use the groups of the ``.mod`` file. The result has
    one column per group, then ``"Others + Initial Values"`` (shocks in no group plus the initial
    conditions), then the smoothed deviation.
    """
    if groups is None:
        blocks = [st for st in getattr(model, "statements", []) if st.get("statementName") == "shock_groups"]
        if name is not None:
            blocks = [b for b in blocks if b.get("name") == name]
        if not blocks:
            raise SmootherError("no shock_groups block found")
        groups = {g["group_name"]: list(g["shocks"]) for g in blocks[-1]["groups"]}
    nshocks = len(decomp.columns) - 2
    z = decomp.z.copy()
    out = np.zeros((z.shape[0], len(groups) + 2, z.shape[2]))
    for i, (label, members) in enumerate(groups.items()):
        for s in members:
            if s not in decomp.columns[:nshocks]:
                raise SmootherError(f"shock {s!r} of group {label!r} is not in the decomposition")
            k = decomp.columns.index(s)
            out[:, i, :] += z[:, k, :]
            z[:, k, :] = 0                              # a shock listed twice counts once, as in Dynare
    out[:, -2, :] = z[:, :nshocks + 1, :].sum(axis=1)
    out[:, -1, :] = z[:, nshocks + 1, :]
    return ShockDecomposition(z=out, endo_names=list(decomp.endo_names),
                              columns=list(groups) + ["Others + Initial Values", "smoothed"],
                              kind="groups", init_state=decomp.init_state)


# --------------------------------------------------------------------------------- public: checks
def check(res: SmootherResult, decomp: ShockDecomposition | None = None, sol=None) -> dict:
    """Internal-consistency errors of a smoother result (all should be at rounding level):

    * ``transition``: smoothed shocks fed through ``dev_t = G dev_{t-1} + H u_t`` from the
      smoothed initial state, one step at a time, against the smoothed variables;
    * ``simulation``: the same, iterated from the initial state without re-anchoring;
    * ``measurement``: data minus smoothed observable, for observables declared without
      measurement error (periods with data only);
    * ``decomposition``: contributions summed against the smoothed deviation (``decomp`` given);
    * ``decomposition_initial``: for a shock decomposition with ``init_state=0``, the
      initial-condition column against the smoothed period-0 state carried forward by ``G``.

    ``sol`` (a first-order solution) replaces the stored transition matrices; it must not be
    passed for a ``loglinear`` result, whose rules are in logs.
    """
    G, H = (res.G, res.H) if sol is None else (np.asarray(sol.G), np.asarray(sol.H))
    if sol is not None and res.loglinear:
        raise SmootherError("loglinear result: use the stored (log) decision rules, not sol")
    prev = np.vstack([res.init_state, res.smoothed[:-1]])
    one_step = prev @ G.T + res.shocks @ H.T
    y, sim = res.init_state, np.empty_like(res.smoothed)
    for t in range(res.nobs):
        y = G @ y + H @ res.shocks[t]
        sim[t] = y
    out = {"transition": float(np.max(np.abs(one_step - res.smoothed))),
           "simulation": float(np.max(np.abs(sim - res.smoothed)))}
    eps = res.measurement_errors
    exact = res.me_variance == 0                      # observables without measurement error
    seen = ~np.isnan(res.data[:, exact])
    out["measurement"] = float(np.max(np.abs(eps[:, exact][seen]))) if seen.any() else 0.0
    if decomp is not None:
        z = decomp.z
        nc = z.shape[1] - 2
        out["decomposition"] = max(float(np.max(np.abs(z[:, :-1, :].sum(axis=1) - z[:, -1, :]))),
                                   float(np.max(np.abs(z[:, -1, :] - res.smoothed.T))))
        if decomp.kind == "shocks" and decomp.init_state == 0:
            # the initial-condition column must be the smoothed period-0 state carried forward
            y, err = res.init_state, 0.0
            for t in range(res.nobs):
                y = G @ y
                err = max(err, float(np.max(np.abs(z[:, nc, t] - y))))
            out["decomposition_initial"] = err
    return out
