"""Kalman filter, Kalman smoother and historical shock decompositions: the counterpart of Dynare's
``calib_smoother``, the smoother run by ``estimation`` / ``evaluate_smoother`` at a parameter set
(calibration, posterior mode / mean / median, ML mode, prior mean / mode, or any parameter vector),
``shock_decomposition``, ``initial_condition_decomposition``, ``realtime_shock_decomposition``,
``squeeze_shock_decomposition``, ``shock_groups`` / ``init2shocks`` and the numbers drawn by
``plot_shock_decomposition`` (the figures are in ``pydyn.plotting``).

This module is derived from Dynare 7.2 (GPL-3.0-or-later, Copyright Dynare Team):
``matlab/kalman/DsgeSmoother.m``, ``evaluate_smoother.m``, ``missing_DiffuseKalmanSmootherH1_Z.m``,
``missing_DiffuseKalmanSmootherH3_Z.m``, ``store_smoother_results.m``,
``matlab/estimation/compute_Pinf_Pstar.m``, ``matlab/compute_trend_coefficients.m``, ``set_prior.m`` (prior
modes), ``matlab/distributions/compute_prior_mode.m``, and in ``matlab/shock_decomposition/``:
``shock_decomposition.m``, ``initial_condition_decomposition.m``, ``realtime_shock_decomposition.m``,
``squeeze_shock_decomposition.m``, ``plot_shock_decomposition.m`` (with ``make_the_groups``),
``expand_group.m``, ``graph_decomp.m`` and ``graph_decomp_detail.m`` (screening and stacking rules).

The filter and smoother recursions are self-contained (they do not use ``pydyn.kalman``); a
``pydyn.estimation.Estimation`` object is only used, through its public interface, to map a vector
of estimated parameters into the model.

Conventions (all Dynare's, see ``SmootherResult``):

* the state vector holds every endogenous variable (Dynare's default, ``smoother_redux`` off);
  arrays are in declaration order;
* variables are returned as Dynare stores them: deviation plus steady state, or plus the log of
  the steady state with ``loglinear``; observed variables under ``prefilter=1`` get the sample
  mean of the data instead, and the linear trend of an ``observation_trends`` block is added;
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
           "check", "load_data", "SmootherResult", "ShockDecomposition", "SmootherError",
           "parameter_vector", "prior_mode", "shock_groups", "init2shocks", "apply_init2shocks",
           "squeeze_shock_decomposition", "realtime_shock_decomposition", "RealtimeDecomposition",
           "decomposition_plot_data", "DecompositionPlotData", "stack_bars", "PARAMETER_SETS"]


class SmootherError(Exception):
    pass


# --------------------------------------------------------------------------------- options
_DEFAULTS = dict(
    datafile=None, first_obs=1, nobs=None, prefilter=0, loglinear=False, logdata=False,
    presample=0, diffuse_filter=False, lik_init=1, kalman_algo=0, kalman_tol=1e-10,
    diffuse_kalman_tol=1e-6, qz_criterium=None, filtered_vars=False, filter_step_ahead=None,
    filter_covariance=False, smoothed_state_uncertainty=False, parameter_set="calibration",
    varobs=None, replicate_dynare_bugs=False, Harvey_scale_factor=10.0,
)
PARAMETER_SETS = ("calibration", "posterior_mode", "posterior_mean", "posterior_median", "mle_mode",
                  "prior_mode", "prior_mean")
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


#: options of the estimation command that concern the optimiser, the sampler or its output, not the smoother
_ESTIMATION_ONLY = {
    "mode_compute", "mode_file", "mode_check", "mh_replic", "mh_nblocks", "mh_jscale", "mh_drop", "mh_init_scale",
    "mh_tune_jscale", "optim", "cova_compute", "posterior_sampling_method", "bayesian_irf", "moments_varendo",
    "irf", "forecast", "conf_sig", "mh_conf_sig", "load_mh_file", "prior_trunc", "smoother", "order",
    "posterior_sampler_options", "additional_optimizer_steps", "nobs_vec", "lyapunov", "sub_draws",
    "mh_nblck", "mh_recover", "mcmc_jumping_covariance", "mode_check.status",
}


def _resolve_options(model, user, extra_statements=()):
    opts = dict(_DEFAULTS)
    from_file = _statement_options(model, _STATEMENTS + tuple(extra_statements))
    opts.update({k: v for k, v in from_file.items() if k in _DEFAULTS})
    from . import options as _opt
    dropped = sorted(set(from_file) - set(_DEFAULTS) - _opt.COSMETIC - _ESTIMATION_ONLY)
    if dropped:
        _opt.warn_once(model, "the smoother does not act on these options from the .mod file: " + ", ".join(dropped))
    unknown = set(user) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown smoother option(s): {sorted(unknown)}")
    opts.update({k: v for k, v in user.items() if v is not None or k in ("nobs", "filter_step_ahead")})
    opts["first_obs"] = int(_scalar(opts["first_obs"]))
    if opts["nobs"] is not None:
        opts["nobs"] = int(_scalar(opts["nobs"]))
    if opts["parameter_set"] is None:
        opts["parameter_set"] = "calibration"
    if opts["parameter_set"] not in PARAMETER_SETS:
        raise ValueError(f"unknown parameter_set {opts['parameter_set']!r}; use one of {PARAMETER_SETS}")
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
    # --- observation trends (Dynare's oo_.Smoother.Trend / TrendCoeffs); zero without the block
    Trend: dict = field(default_factory=dict)          # {name: (T,)}, zero for non-observed variables
    TrendCoeffs: np.ndarray = None                     # (n_endo,) slopes, declaration order
    # --- the calibration used
    parameter_set: str = "calibration"                 # or "theta" when a vector was passed
    theta: np.ndarray = None                           # estimated-parameter vector used (None: calibration)
    params: np.ndarray = None                          # structural parameters used (after the steady state)
    Sigma_e: np.ndarray = None                         # shock covariance used
    me_cov: np.ndarray = None                          # measurement-error covariance used (n_obs, n_obs)
    solution: object = None                            # first-order solution used (levels)
    raw: np.ndarray = None                             # data as loaded: all periods, before logs and prefilter
    first_obs: int = 1

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
    steady_state: np.ndarray = None    # level steady state of the variables at the parameters used
    parameter_set: str = "calibration"
    first_period: int = 1              # number of the first period (Dynare counts the sample from 1)

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
                           want_V, want_PK, compat, a_pred=None, a_zero=None):
    """Port of ``missing_DiffuseKalmanSmootherH1_Z``. ``Y`` is (n_obs, T). Returns ``None`` when
    Dynare's routine reports a singular forecast-error variance (the caller then switches to the
    univariate smoother, as ``DsgeSmoother`` does).

    ``compat=True`` reproduces three defects of Dynare 7.2 in the diffuse phase (see ``smooth``).
    ``a_pred`` is the state forecast for period 1 and ``a_zero`` the mean of the period-0 state
    (both zero by default; set by ``filter_initial_state``, stationary initialisations only).
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
    a0, P0 = (np.zeros(mm) if a_zero is None else np.asarray(a_zero, float)), Pstar1.copy()
    if a_pred is not None:
        a[0] = a_pred
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
                         want_V, want_PK, compat=False, a_pred=None, a_zero=None):
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
    if a_pred is not None:
        a1[0] = a_pred
    ainit, Pinit = (a1[0].copy() if a_zero is None else np.asarray(a_zero, float)), Pstar1.copy()
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


# --------------------------------------------------------------------------------- parameter sets
def prior_mode(prior) -> np.ndarray:
    """Mode of each prior of a ``pydyn.priors.Prior`` (Dynare's ``bayestopt_.p5``, port of
    ``compute_prior_mode.m`` as called by ``set_prior.m``). NaN for a uniform prior, as in Dynare."""
    out = np.full(len(prior), np.nan)
    eps = np.finfo(float).eps
    for i, sh in enumerate(prior.shape):
        a, b, lo, hi = prior.p6[i], prior.p7[i], prior.p3[i], prior.p4[i]
        if sh == 1:
            if a > 1 and b > 1:
                m = (a - 1) / (a + b - 2)
            elif a < 1 and b < 1:
                m = 0.0                                    # two modes, 0 and 1: Dynare keeps the first
            elif (a < 1 and b > 1 - eps) or (abs(a - 1) < 2 * eps and b > 1):
                m = 0.0
            elif (a > 1 and b < 1 + eps) or (abs(a - 1) < 2 * eps and b < 1):
                m = 1.0
            else:
                m = 0.5
            out[i] = m * (hi - lo) + lo
        elif sh == 2:
            out[i] = (0.0 if a < 1 else (a - 1) * b) + lo
        elif sh == 3:
            out[i] = prior.p1[i]
        elif sh == 4:
            out[i] = 1 / np.sqrt((b + 1) / a) + lo
        elif sh == 6:
            out[i] = a / (b + 2) + lo
        elif sh == 8:
            out[i] = (0.0 if a <= 1 else b * ((a - 1) / a) ** (1 / a)) + lo
    return out


def parameter_vector(estimation, parameter_set, mode=None, mcmc=None) -> np.ndarray:
    """The estimated-parameter vector Dynare's ``evaluate_smoother`` uses for ``parameter_set``.

    ``posterior_mode`` / ``mle_mode``: ``mode`` (a ``ModeResult`` or a vector), else the attribute
    ``estimation.posterior_mode`` / ``estimation.mle_mode`` if the caller stored one there;
    ``posterior_mean`` / ``posterior_median``: from ``mcmc`` (an ``MCMCResult``: mean / median of
    the draws kept after ``mh_drop``, or a vector), else ``estimation.posterior_mean`` /
    ``estimation.posterior_median``; ``prior_mean`` and ``prior_mode`` from the priors.
    """
    est = estimation
    if est is None:
        raise SmootherError(f"parameter_set={parameter_set!r} needs estimation=<pydyn.Estimation> "
                            "(or pass the vector with theta=...)")

    def as_vector(x, what):
        if x is None:
            x = getattr(est, parameter_set, None)
        if x is None:
            raise SmootherError(f"parameter_set={parameter_set!r}: pass {what}, or store the vector as "
                                f"estimation.{parameter_set}")
        return np.asarray(getattr(x, "x", x), float)

    bayes = bool(est.prior.is_bayesian)
    if parameter_set in ("posterior_mode", "mle_mode"):
        if (parameter_set == "mle_mode") == bayes:
            raise SmootherError(f"parameter_set={parameter_set!r} does not fit "
                                f"{'a Bayesian' if bayes else 'a maximum-likelihood'} estimation")
        return as_vector(mode, "mode=estimation.mode() (a ModeResult) or mode=<vector>")
    if parameter_set in ("posterior_mean", "posterior_median"):
        if not bayes:
            raise SmootherError(f"parameter_set={parameter_set!r} needs priors")
        if mcmc is not None and hasattr(mcmc, "kept"):
            kept = mcmc.kept()
            return kept.mean(axis=0) if parameter_set == "posterior_mean" else np.median(kept, axis=0)
        return as_vector(mcmc, "mcmc=estimation.sample(...) (an MCMCResult) or mcmc=<vector>")
    if parameter_set in ("prior_mean", "prior_mode"):
        if not bayes:
            raise SmootherError(f"parameter_set={parameter_set!r} needs priors")
        x = np.array(est.prior.p1, float) if parameter_set == "prior_mean" else prior_mode(est.prior)
        if not np.all(np.isfinite(x)):
            bad = [n for n, v in zip(est.names, x) if not np.isfinite(v)]
            raise SmootherError(f"parameter_set={parameter_set!r} is not defined for {bad} (Dynare's "
                                "mode of a uniform prior is NaN and its smoother then fails too)")
        return x
    raise ValueError(f"unknown parameter_set {parameter_set!r}")


def _calibration(estimation, theta):
    """Structural parameters, shock covariance and measurement-error covariance at ``theta``,
    through the public interface of ``Estimation``; nothing is written to the model."""
    theta = np.asarray(theta, float).ravel()
    if theta.shape != (len(estimation.names),):
        raise SmootherError(f"theta has {theta.size} entries for the {len(estimation.names)} estimated "
                            f"parameters {estimation.names}")
    if hasattr(estimation, "calibration"):
        c = estimation.calibration(theta)
        params, Q, H = c["params"], c["Sigma_e"], c["H"]
    else:
        params = estimation.model_params(theta)
        Q, H = estimation.covariances(theta)
    return np.array(params, float), np.array(Q, float), np.array(H, float), theta


def _statements(model, kind):
    return [st for st in getattr(model, "statements", []) if st.get("statementName") == kind]


def _trend_coefficients(model, varobs, params):
    """Slopes of the ``observation_trends`` block in ``varobs`` order (``compute_trend_coefficients.m``)."""
    coeff = np.zeros(len(varobs))
    blocks = _statements(model, "observation_trends")
    for st in blocks:
        tr = st.get("trends", st.get("vals", {}))
        table = ({d.get("var", d.get("name")): d.get("trend", d.get("value")) for d in tr}
                 if isinstance(tr, list) else dict(tr))
        unknown = [v for v in table if v not in varobs]
        if unknown:
            raise SmootherError(f"observation_trends on non-observed variable(s) {unknown}")
        for v, expr in table.items():
            coeff[varobs.index(v)] = float(model.eval_expr(expr, {"params": params}))
    return coeff, bool(blocks)


def _initial_state(model, names, state_var, ys, params, loglinear):
    """Deviation of the period-0 state set by a ``filter_initial_state`` block (zero elsewhere);
    ``None`` without a block. Only state variables are used, as in Dynare."""
    blocks = _statements(model, "filter_initial_state")
    if not blocks:
        return None
    a = np.zeros(len(names))
    states = set(int(k) for k in state_var)
    for st in blocks:
        for d in st.get("states", []):
            if int(d.get("lag", 0)) != 0:
                raise NotImplementedError("filter_initial_state on a lagged variable is not supported")
            i = names.index(d["var"])
            if i not in states:
                continue
            val = float(model.eval_expr(str(d["value"]).rstrip(";"), {"params": params}))
            a[i] = np.log(val) - np.log(ys[i]) if loglinear else val - ys[i]
    return a


_FROM_ESTIMATION = ("datafile", "first_obs", "nobs", "prefilter", "loglinear", "logdata", "presample", "lik_init",
                    "diffuse_filter", "kalman_algo", "kalman_tol", "diffuse_kalman_tol", "Harvey_scale_factor")


# --------------------------------------------------------------------------------- public: smoother
def _loglinear_rules(G, H, ys):
    if np.any(ys <= 0) or not np.all(np.isfinite(ys)):
        raise SmootherError("loglinear needs a strictly positive steady state for every variable "
                            "(models with auxiliary variables for lagged shocks are not handled)")
    return G * ys[None, :] / ys[:, None], H / ys[:, None]


def smooth(model, data=None, sol=None, estimation=None, theta=None, mode=None, mcmc=None, _reuse=None,
           **options) -> SmootherResult:
    """Kalman filter and smoother (Dynare's ``calib_smoother`` and ``evaluate_smoother``).

    ``data``: array ``(T, n_obs)`` with columns in ``varobs`` order, dict of series, or a file
    path; when omitted the ``datafile`` option is used (relative to the ``.mod`` file's folder).
    ``sol``: a first-order solution (computed if omitted; calibration only).

    Parameters used. By default the model's current calibration (``parameter_set="calibration"``).
    With ``estimation=`` (a ``pydyn.Estimation`` of the same model):

    * ``theta=vector``: the estimated-parameter vector, ordered as ``estimation.names`` (Dynare's
      ``xparam1``); it is mapped into structural parameters, shock standard deviations,
      measurement-error standard deviations and correlations as Dynare's ``set_all_parameters``;
    * ``parameter_set=`` ``"posterior_mode"`` / ``"mle_mode"`` (with ``mode=estimation.mode()`` or
      a vector), ``"posterior_mean"`` / ``"posterior_median"`` (with ``mcmc=estimation.sample()``
      or a vector), ``"prior_mean"``, ``"prior_mode"``: see ``parameter_vector``.

    The model object is never modified. With ``estimation`` the options that object was built with
    (sample, ``prefilter``, ``loglinear``, ``lik_init``, ...) are the defaults here; the data are
    still read from ``data`` / ``datafile``.

    Options are Dynare's and default to those of the ``estimation`` / ``calib_smoother``
    statements of the ``.mod`` file, in file order; keyword arguments take precedence:
    ``datafile, first_obs, nobs, prefilter, loglinear, logdata, diffuse_filter, lik_init (1, 2 or 3),
    Harvey_scale_factor, kalman_algo (0-4), filtered_vars, filter_step_ahead, filter_covariance,
    smoothed_state_uncertainty, kalman_tol, diffuse_kalman_tol, varobs``. ``presample`` is
    accepted and has no effect on the smoother (as in Dynare).

    Blocks of the ``.mod`` file. ``observation_trends``: the linear trends are removed from the data
    (demeaned over the sample under ``prefilter``) and returned in ``Trend`` / ``TrendCoeffs``;
    smoothed, updated and filtered *observed* variables include them (k-step-ahead forecasts with
    the trend moved k periods ahead), as in Dynare. ``filter_initial_state``: the filter starts
    from the given state (stationary initialisations only).

    ``replicate_dynare_bugs`` (default False):

    * multivariate diffuse smoother (``kalman_algo=3``) when a period inside the diffuse phase has
      no observation at all or a zero ``F_inf``: Dynare 7.2 then drops terms of the backward
      recursion; its output breaks the transition equation and differs from its own univariate
      smoother. With False the textbook recursion is used (equal to Dynare's ``kalman_algo=4``);
      True reproduces Dynare's ``kalman_algo=3`` output;
    * ``filter_initial_state``: Dynare centres the smoothed period-0 state on ``T a`` instead of
      the prescribed state ``a`` (``Init_State`` only; later periods are unaffected). True
      reproduces that.
    """
    from . import options as _opt
    _opt.refuse_occbin(model, "The smoother")
    from .perturbation import first_order

    user = dict(options)
    if estimation is not None:
        if list(estimation.model.endo_names) != list(model.endo_names):
            raise SmootherError("estimation= was built for another model")
        eo = estimation.options
        for k in _FROM_ESTIMATION:
            if k in eo and k not in user and eo[k] is not None:
                user[k] = eo[k]
    opts = _resolve_options(model, user)
    if _statements(model, "heteroskedastic_shocks"):
        raise NotImplementedError("heteroskedastic_shocks blocks are not supported by the smoother")
    varobs = list(opts["varobs"] or model.varobs)
    if not varobs:
        raise SmootherError("no observed variables: declare varobs or pass varobs=[...]")
    names = list(model.endo_names)
    n, nx, pp = len(names), model.n_exo, len(varobs)
    try:
        mf = np.array([names.index(v) for v in varobs])
    except ValueError as exc:
        raise SmootherError(f"observed variable not in the model: {exc}") from exc

    # ---- parameters (Dynare's set_all_parameters when a vector is used)
    pset = opts["parameter_set"]
    if _reuse is not None:
        sol, Q, Hme, theta, pset = _reuse.solution, _reuse.Sigma_e, _reuse.me_cov, _reuse.theta, _reuse.parameter_set
    else:
        if theta is None and pset != "calibration":
            theta = parameter_vector(estimation, pset, mode=mode, mcmc=mcmc)
        elif theta is not None:
            if estimation is None:
                raise SmootherError("theta= needs estimation=<pydyn.Estimation> to map the vector into the model")
            if "parameter_set" not in options:
                pset = "theta"
        if theta is not None:
            if sol is not None:
                raise TypeError("sol= cannot be combined with a parameter vector")
            if list(varobs) != list(estimation.varobs):
                raise SmootherError("varobs differs from the observed variables of the estimation")
            params, Q, Hme, theta = _calibration(estimation, theta)
            sol = first_order(model, params=params)
        else:
            Q = np.array(model.Sigma_e, float)
            Hme = np.diag([float(model.measurement_errors.get(v, 0.0)) for v in varobs])
            extra_me = set(model.measurement_errors) - set(varobs)
            if extra_me:
                raise SmootherError(f"measurement error declared on non-observed variable(s) {sorted(extra_me)}")
            if sol is None:
                sol = first_order(model)
    opts["parameter_set"] = pset
    params = np.array(getattr(sol, "params", model.params), float)
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
    loaded = load_data(src, varobs, base_dir=Path(model.json_dir).parents[2] if hasattr(model, "json_dir") else None)
    raw = loaded
    if loglinear and not opts["logdata"]:
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.log(raw)
    first = opts["first_obs"]
    nobs = opts["nobs"] if opts["nobs"] is not None else raw.shape[0] - first + 1
    if first < 1 or nobs < 1 or first - 1 + nobs > raw.shape[0]:
        raise SmootherError(f"first_obs={first}, nobs={nobs} do not fit the {raw.shape[0]} observations in the data")
    obs = raw[first - 1:first - 1 + nobs]
    obs_const = const_all[mf].copy()
    prefilter = int(opts["prefilter"]) == 1
    if prefilter:
        if np.any(np.abs(obs_const) > 1e-9):
            raise SmootherError("prefilter=1 is inconsistent with observed variables that have a "
                                "non-zero steady state (same check as Dynare)")
        obs_const = np.nanmean(obs, axis=0)
    trend_coeff, with_trend = _trend_coefficients(model, varobs, params)
    trend = np.outer(np.arange(first, first + nobs), trend_coeff)       # (T, n_obs)
    if prefilter:
        trend = trend - trend.mean(axis=0)
    Yd = (obs - obs_const - trend).T                  # (n_obs, T), deviations
    smpl = Yd.shape[1]
    di_list = [np.where(~np.isnan(Yd[:, t]))[0] for t in range(smpl)]

    # ---- state space
    Z = np.zeros((pp, n))
    Z[np.arange(pp), mf] = 1.0
    kalman_algo = int(opts["kalman_algo"])
    lik_init = int(opts["lik_init"])
    if opts["diffuse_filter"] or kalman_algo in (3, 4):
        if lik_init == 2:
            raise SmootherError("options diffuse_filter, lik_init and/or kalman_algo have contradictory settings")
        lik_init = 3
    qz = opts["qz_criterium"]
    if lik_init == 1:
        qz = 1 - 1e-6 if qz is None else qz
        kalman_algo = 2 if kalman_algo == 2 else 1
        Pstar, Pinf = _stationary_variance(G, Hs, Q, qz), None
    elif lik_init == 2:                               # Dynare's "old" diffuse filter: a large variance
        kalman_algo = 2 if kalman_algo == 2 else 1
        Pstar, Pinf = float(opts["Harvey_scale_factor"]) * np.eye(n), None
    elif lik_init == 3:
        qz = 1 + 1e-6 if qz is None else qz
        kalman_algo = 4 if kalman_algo == 4 else 3
        Pstar, Pinf = compute_Pinf_Pstar(mf, G, Hs, Q, qz)
    else:
        raise NotImplementedError(f"lik_init={lik_init}: only 1 (stationary), 2 (large variance) and 3 (diffuse) "
                                  "are implemented")
    compat = bool(opts["replicate_dynare_bugs"])
    a_init = _initial_state(model, names, sol.state_var, ys, params, loglinear)
    a_pred = a_zero = None
    if a_init is not None and np.any(a_init):
        if lik_init == 3:
            raise NotImplementedError("filter_initial_state with the diffuse filter is not supported (Dynare 7.2 "
                                      "applies the transition matrix twice to the prescribed state there)")
        a_pred = G @ a_init
        a_zero = a_pred if compat else a_init
    nk = max(opts["filter_step_ahead"]) if opts["filter_step_ahead"] else 1
    want_V, want_PK = bool(opts["smoothed_state_uncertainty"]), bool(opts["filter_covariance"])
    ktol, dtol = float(opts["kalman_tol"]), float(opts["diffuse_kalman_tol"])

    out = None
    if kalman_algo in (1, 3):
        out = _smoother_multivariate(G, Z, Hs, Q, Hme, Pinf, Pstar, Yd, di_list, nk, ktol, dtol,
                                     want_V, want_PK, compat, a_pred, a_zero)
        if out is None:
            kalman_algo += 1                        # Dynare: switch to the univariate filter
    if out is None:
        if not np.all(np.abs(Hme - np.diag(np.diag(Hme))) < 1e-14):
            raise NotImplementedError("the univariate smoother with correlated measurement errors is not "
                                      "implemented (Dynare augments the state vector)")
        out = _smoother_univariate(G, Z, Hs, Q, np.diag(Hme).copy(), Pinf, Pstar, Yd, di_list, nk,
                                   ktol, dtol, want_V, want_PK, compat, a_pred, a_zero)
    alphahat, updated, aK = out["alphahat"], out["updated"], out["aK"]
    eps = Yd.T - alphahat[:, mf]                      # NaN where the observation is missing

    # ---- Dynare's store_smoother_results
    const = np.tile(const_all, (smpl, 1))
    const[:, mf] = obs_const
    full_trend = np.zeros((smpl, n))
    full_trend[:, mf] = trend
    step = np.zeros(n)
    step[mf] = trend_coeff
    sv, uv = alphahat + const + full_trend, updated + const + full_trend
    fv = aK[0, :, 1:smpl + 1].T + const + full_trend + step
    kstep = kvar = None
    fsa = opts["filter_step_ahead"]
    if fsa:
        idx = np.array(fsa) - 1
        kstep = aK[idx] + const_all[None, :, None]
        mx = max(fsa)
        for j, k in enumerate(fsa):                   # observed variables: constant and trend of the data
            hi = kstep.shape[2] - (mx - k)
            kstep[j, mf, k:hi] = aK[k - 1][mf, k:hi] + obs_const[:, None] + (trend + k * trend_coeff).T
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
        Trend={v: full_trend[:, i] for i, v in enumerate(names)}, TrendCoeffs=step,
        parameter_set=pset, theta=None if theta is None else np.array(theta, float), params=params,
        Sigma_e=Q, me_cov=Hme, solution=sol, raw=loaded, first_obs=first,
    )
    return res


# --------------------------------------------------------------------------------- public: decompositions
#: options of the decomposition statements that only concern Dynare's figure files and menus
_DECOMP_COSMETIC = {"colormap", "fig_name", "graph_format", "nodisplay", "interactive", "nograph",
                    "shock_decomposition", "plot_shock_decomposition"}
_SMOOTH_ARGS = ("estimation", "theta", "mode", "mcmc", "sol")


def _short(key):
    return key.split(".")[-1]


def _decomp_statement_options(model, kind, which=None):
    """Options of the statements of one kind with the ``shock_decomp.`` / ``plot_shock_decomp.`` /
    ``initial_condition_decomp.`` prefixes removed. ``which=None`` merges them in file order;
    an integer picks one statement (Dynare resets the plot options before every plot statement).
    Returns ``(options, symbol_list)``."""
    sts = _statements(model, kind)
    if which is not None:
        sts = sts[which:][:1] if sts and -len(sts) <= which < len(sts) else []
    out, symbols = {}, []
    for st in sts:
        for k, v in (st.get("options") or {}).items():
            out[_short(k)] = v
        symbols = list(st.get("symbol_list", []) or [])
    return out, symbols


def _report_unused(model, kind, honoured, what, options=None):
    """Warn once (``pydyn.options``) about options of a statement that ``what`` does not act on."""
    from . import options as _opt
    if options is None:
        options, _ = _decomp_statement_options(model, kind)
    known = set(honoured) | _opt.COSMETIC | _DECOMP_COSMETIC
    unused = sorted(k for k in options if k not in known and f"no_graph.{k}" not in known)
    if unused and model is not None:
        _opt.warn_once(model, f"{what} does not act on these {kind} options from the .mod file: "
                              f"{', '.join(unused)}", stacklevel=4)
    return unused


def _as_result(model, res_or_data, options):
    if isinstance(res_or_data, SmootherResult):
        if options:
            raise TypeError("smoother options cannot be combined with an existing SmootherResult")
        return res_or_data
    return smooth(model, res_or_data, **options)


def _with_file_options(model, kind, options):
    """Smoother options of a decomposition statement as defaults under the keyword arguments."""
    file_opts, _ = _decomp_statement_options(model, kind)
    merged = {k: v for k, v in file_opts.items() if k in _DEFAULTS}
    merged.update(options)
    return file_opts, merged


def shock_decomposition(model, res_or_data=None, init_state=None, **options) -> ShockDecomposition:
    """Historical shock decomposition (Dynare's ``shock_decomposition``).

    ``res_or_data``: a ``SmootherResult``, or data / nothing, in which case the smoother is run
    with ``options`` (defaults also read from the file's ``shock_decomposition`` statement,
    including ``parameter_set``). The parameter set is chosen as in ``smooth``:
    ``estimation=est, theta=vector`` or ``estimation=est, parameter_set="posterior_mode", mode=...``.
    ``init_state``: Dynare's option; shocks of periods ``1..init_state`` are attributed to the
    initial conditions (0: condition on the smoothed state of period 0).

    Returns ``z`` of shape ``(n_endo, n_exo + 2, T)``: shocks in declaration order, then initial
    conditions, then the smoothed deviation from the steady state (in logs with ``loglinear``).
    Observed variables under ``prefilter`` are measured from the model's steady state, not from
    the data mean, and observation trends are not part of the decomposition, as in Dynare.
    Shocks are never grouped here: ``use_shock_groups`` acts when plotting (``group_shocks``,
    ``decomposition_plot_data``), as in Dynare where ``oo_.shock_decomposition`` is ungrouped.
    """
    file_opts, merged = _with_file_options(model, "shock_decomposition", options)
    _report_unused(model, "shock_decomposition", set(_DEFAULTS) | {"init_state", "use_shock_groups", "max_nrows"},
                   "shock_decomposition", file_opts)
    if init_state is None:
        init_state = int(file_opts.get("init_state", 0))
    if not isinstance(res_or_data, SmootherResult):
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
                              kind="shocks", init_state=int(init_state), steady_state=res.SteadyState.copy(),
                              parameter_set=res.parameter_set)


def initial_condition_decomposition(model, res_or_data=None, **options) -> ShockDecomposition:
    """Dynare's ``initial_condition_decomposition``: contribution of each variable's smoothed
    value in period 1 to the later path. ``z`` is ``(n_endo, n_endo + 2, T)``; column ``n_endo``
    (0-based) is the remainder, i.e. the contribution of all shocks from period 2 on, and the
    last column is the smoothed deviation. Parameter sets as in ``smooth``."""
    file_opts, _ = _decomp_statement_options(model, "initial_condition_decomposition")
    _report_unused(model, "initial_condition_decomposition",
                   {"type", "steadystate", "diff", "flip", "detail_plot", "max_nrows"},
                   "initial_condition_decomposition", file_opts)
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
                              kind="initial_conditions", steady_state=res.SteadyState.copy(),
                              parameter_set=res.parameter_set)


def shock_groups(model, name=None) -> dict:
    """Groups of a ``shock_groups`` block as ``{label: [shock, ...]}`` (the last block, or the one
    called ``name``; a block without a name is called ``default``)."""
    blocks = _statements(model, "shock_groups")
    if name is not None:
        blocks = [b for b in blocks if b.get("name", "default") == name]
    if not blocks:
        raise SmootherError("no shock_groups block found" + (f" with name {name!r}" if name else ""))
    return {g["group_name"]: list(g["shocks"]) for g in blocks[-1]["groups"]}


def group_shocks(decomp: ShockDecomposition, groups=None, model=None, name=None, expand=None) -> ShockDecomposition:
    """Sum shock contributions by group, as Dynare does when plotting with ``use_shock_groups``
    (``make_the_groups`` in ``plot_shock_decomposition.m``).

    ``groups`` maps a label to a list of shock names; or pass ``model`` (and optionally the
    ``name`` of a ``shock_groups`` block) to use the groups of the ``.mod`` file. The result has
    one column per group, then one column holding the shocks in no group plus the initial
    conditions, then the smoothed deviation. That column is labelled ``"Others + Initial Values"``
    when some shock outside the groups contributes and ``"Initial values"`` otherwise (Dynare's
    labels). A shock listed in two groups counts in the first only.

    ``expand="label"`` reproduces Dynare's "expand group" menu entry: the shocks of that group
    one by one, everything else in ``"Others + Initial Values"``.
    """
    if decomp.kind != "shocks":
        raise SmootherError("group_shocks needs an ungrouped shock decomposition")
    if groups is None:
        groups = shock_groups(model, name)
    if expand is not None:
        if expand not in groups:
            raise SmootherError(f"no group {expand!r} to expand; groups are {list(groups)}")
        groups = {s: [s] for s in groups[expand]}
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
    others = np.any(np.nan_to_num(z[:, :nshocks, :].sum(axis=1)) != 0)
    out[:, -2, :] = z[:, :nshocks + 1, :].sum(axis=1)
    out[:, -1, :] = z[:, nshocks + 1, :]
    return ShockDecomposition(z=out, endo_names=list(decomp.endo_names),
                              columns=list(groups) + ["Others + Initial Values" if others else "Initial values",
                                                      "smoothed"],
                              kind="groups", init_state=decomp.init_state, steady_state=decomp.steady_state,
                              parameter_set=decomp.parameter_set, first_period=decomp.first_period)


def init2shocks(model, name=None) -> list:
    """Pairs ``(endogenous, exogenous)`` of an ``init2shocks`` block (the last one, or ``name``)."""
    blocks = _statements(model, "init2shocks")
    if name is not None:
        blocks = [b for b in blocks if b.get("name", "default") == name]
    if not blocks:
        raise SmootherError("no init2shocks block found" + (f" with name {name!r}" if name else ""))
    return [(g["endogenous"], g.get("exogenous") or None) for g in blocks[-1]["groups"]]


def apply_init2shocks(decomp: ShockDecomposition, icd: ShockDecomposition, pairs, rename=True) -> ShockDecomposition:
    """Attribute the effect of the initial condition of some endogenous variables to shocks
    (Dynare's ``init2shocks`` option of ``plot_shock_decomposition``).

    ``pairs``: ``(endogenous, exogenous)`` tuples; ``icd`` the initial-condition decomposition of
    the same smoother run. For a shock decomposition the contribution of the variable's period-1
    value moves from the initial conditions to the shock, whose label becomes
    ``"shock + variable"`` (unless ``rename`` is false). For an initial-condition decomposition (pass it as ``decomp``; ``icd``
    is ignored) the listed variables are merged into the remainder (all shocks), as Dynare does
    in ``initial_condition_decomposition.m``.
    """
    z = decomp.z.copy()
    cols = list(decomp.columns)
    if decomp.kind == "initial_conditions":
        for endo, _ in pairs:
            j = decomp.endo_names.index(endo)
            z[:, -2, :] += z[:, j, :]
            z[:, j, :] = 0
    elif decomp.kind == "shocks":
        if icd is None or icd.kind != "initial_conditions":
            raise SmootherError("init2shocks needs the initial-condition decomposition (icd=...)")
        if icd.endo_names != decomp.endo_names:
            raise SmootherError("init2shocks: the initial-condition decomposition holds other variables "
                                "(apply init2shocks before squeezing)")
        T = z.shape[2]
        for endo, exo in pairs:
            j = icd.columns.index(endo)
            if exo is not None:
                jj = decomp.columns.index(exo)
                if rename:
                    cols[jj] = f"{cols[jj]} + {endo}"
                z[:, jj, :] += icd.z[:, j, :T]
            else:
                z[:, -1, :] -= icd.z[:, j, :T]
            z[:, -2, :] -= icd.z[:, j, :T]
    else:
        raise SmootherError("init2shocks applies before the shocks are grouped")
    return ShockDecomposition(z=z, endo_names=list(decomp.endo_names), columns=cols, kind=decomp.kind,
                              init_state=decomp.init_state, steady_state=decomp.steady_state,
                              parameter_set=decomp.parameter_set, first_period=decomp.first_period)


def squeeze_shock_decomposition(obj, var_list):
    """Dynare's ``squeeze_shock_decomposition``: keep the decomposition of the listed variables
    only (in the order given). Works on a ``ShockDecomposition`` or a ``RealtimeDecomposition``."""
    def pick(names):
        missing = [v for v in var_list if v not in names]
        if missing:
            raise SmootherError(f"variable(s) {missing} not in the decomposition")
        return [names.index(v) for v in var_list]

    if isinstance(obj, RealtimeDecomposition):
        k = pick(obj.endo_names)
        sq = lambda d: {key: v[k] for key, v in d.items()}   # noqa: E731
        return RealtimeDecomposition(
            endo_names=list(var_list), columns=list(obj.columns), nobs=obj.nobs, forecast=obj.forecast,
            presample=obj.presample, realtime=sq(obj.realtime), conditional=sq(obj.conditional),
            realtime_forecast=sq(obj.realtime_forecast), realtime_conditional=sq(obj.realtime_conditional),
            steady_state=None if obj.steady_state is None else obj.steady_state[k], parameter_set=obj.parameter_set)
    k = pick(obj.endo_names)
    return ShockDecomposition(z=obj.z[k], endo_names=list(var_list), columns=list(obj.columns), kind=obj.kind,
                              init_state=obj.init_state,
                              steady_state=None if obj.steady_state is None else np.asarray(obj.steady_state)[k],
                              parameter_set=obj.parameter_set, first_period=obj.first_period)


# --------------------------------------------------------------------------------- realtime decomposition
@dataclass
class RealtimeDecomposition:
    """Output of ``realtime_shock_decomposition``; the four dicts mirror Dynare's ``oo_`` fields
    and hold arrays ``(n_endo, n_exo + 2, periods)`` with the columns of a shock decomposition.

    * ``realtime`` (``oo_.realtime_shock_decomposition``): ``"pool"`` (``nobs + forecast`` periods:
      period ``t`` as decomposed with the data up to ``t``; the periods up to ``presample + 1`` come
      from the first vintage, the forecast periods from the last) and ``"time_j"`` for the saved
      vintages (data up to ``j``, ``j + forecast`` periods);
    * ``conditional`` (``oo_.conditional_shock_decomposition``): ``"pool"`` (one-step: period ``t``
      given the smoothed state of ``t - 1``, vintage ``t``) and ``"time_j"`` (``forecast + 1``
      periods starting at ``j``, conditional on the state at ``j``, from vintage ``j + forecast``);
    * ``realtime_forecast``: ``"pool"`` (``nobs + 1`` periods: the one-step forecast of ``t`` made
      with the data up to ``t - 1``) and ``"time_j"`` (period ``j`` and the ``forecast`` periods
      after it, as forecast in vintage ``j``);
    * ``realtime_conditional``: ``"time_j"``: realtime pool minus the forecast made at ``j``
      (the forecast itself is added to the initial-condition column).
    """
    endo_names: list
    columns: list
    nobs: int
    forecast: int
    presample: int
    realtime: dict
    conditional: dict
    realtime_forecast: dict
    realtime_conditional: dict
    steady_state: np.ndarray = None
    parameter_set: str = "calibration"

    def select(self, realtime=1, vintage=0) -> ShockDecomposition:
        """The decomposition Dynare's ``plot_shock_decomposition(realtime=..., vintage=...)`` draws:
        ``realtime`` 1 (realtime), 2 (conditional), 3 (forecast); ``vintage`` 0 for the pooled one."""
        if realtime not in (1, 2, 3):
            raise ValueError("realtime must be 1 (realtime), 2 (conditional) or 3 (forecast)")
        key = f"time_{int(vintage)}" if vintage else "pool"
        if realtime == 1:
            store, first = self.realtime, 1
        elif realtime == 2:
            store, first = (self.realtime_conditional, int(vintage)) if vintage else (self.conditional, 1)
        else:
            store, first = self.realtime_forecast, (int(vintage) if vintage else 1)
        if key not in store:
            raise SmootherError(f"vintage {vintage} was not stored (save_realtime); available: {sorted(store)}")
        return ShockDecomposition(z=store[key], endo_names=list(self.endo_names), columns=list(self.columns),
                                  kind="shocks", steady_state=self.steady_state, parameter_set=self.parameter_set,
                                  first_period=first)


def realtime_shock_decomposition(model, data=None, forecast=None, presample=None, save_realtime=None,
                                 fast_realtime=None, **options) -> RealtimeDecomposition:
    """Dynare's ``realtime_shock_decomposition``: the smoother is re-run on the samples
    ``1..j`` for ``j = presample + 1, ..., nobs`` and each period is decomposed with the data
    available then. ``forecast`` (periods ahead, shocks set to zero), ``presample``,
    ``save_realtime`` (vintages to keep) and ``fast_realtime`` (vintages at which the smoother is
    really re-run; other vintages reuse the next such run truncated to ``j`` periods) default to
    the statement in the ``.mod`` file. Other keyword arguments are those of ``smooth``
    (parameter sets included; the parameters are the same for all vintages).

    Conventions of Dynare 7.2 kept here:

    * ``presample`` is ``max(1, P - 1)`` with ``P`` the ``presample`` option of the *estimation*,
      and the option given here can only lower it (``min`` of the two), so with the default
      estimation ``presample=0`` the recursion always starts at ``j = 2`` whatever is asked;
    * the shocks of period 1 are attributed to the initial conditions (in
      ``shock_decomposition`` they are not), so the pooled result differs from the full-sample
      decomposition even in the last vintage.
    """
    file_opts, merged = _with_file_options(model, "realtime_shock_decomposition", options)
    _report_unused(model, "realtime_shock_decomposition",
                   set(_DEFAULTS) | {"forecast", "presample", "save_realtime", "fast_realtime"},
                   "realtime_shock_decomposition", file_opts)
    forecast = int(_scalar(file_opts.get("forecast", 0)) if forecast is None else forecast)
    my_presample = file_opts.get("presample") if presample is None else presample
    if "presample" in file_opts and "presample" not in options:
        merged.pop("presample", None)                 # the statement's presample is not the estimation's
    save = [int(v) for v in np.atleast_1d(file_opts.get("save_realtime", []) if save_realtime is None else save_realtime)]
    fast = [int(v) for v in np.atleast_1d(file_opts.get("fast_realtime", 0) if fast_realtime is None else fast_realtime)]
    fast = [v for v in fast if v > 0]

    full = smooth(model, data, **merged)
    nobs, n, nx = full.nobs, full.smoothed.shape[1], full.shocks.shape[1]
    pres = max(1, int(full.options["presample"]) - 1)
    if my_presample is not None:
        pres = min(pres, max(1, int(_scalar(my_presample))))
    G, H = full.G, full.H
    again = {k: v for k, v in merged.items() if k not in _SMOOTH_ARGS and k not in ("nobs", "datafile", "parameter_set")}

    def run(j):
        return full if j == nobs else smooth(model, full.raw, _reuse=full, estimation=merged.get("estimation"),
                                             nobs=j, **again)

    zreal = np.zeros((n, nx + 2, nobs + forecast))
    zcond = np.zeros((n, nx + 2, nobs))
    zfrcst = np.zeros((n, nx + 2, nobs + 1))
    rt, cond, fc, rtc, fc_all = {}, {}, {}, {}, {}
    init, gend0, res0 = True, 0, None
    for j in range(pres + 1, nobs + 1):
        if not fast:
            res = run(j)
        else:
            if j < min(fast) and gend0 < j:
                res0, gend0 = run(min(fast)), min(fast)
            if j in fast and gend0 < j:
                res = res0 = run(j)
                gend0 = j
            else:
                if j > gend0:
                    gend0 = nobs if j > max(fast) else min(f for f in fast if f > j)
                    res0 = run(gend0)
                res = res0
        gend = j
        sm = res.smoothed[:gend].T                                  # (n, gend)
        eps = np.hstack([res.shocks[:gend].T, np.zeros((nx, forecast))])
        z = np.zeros((n, nx + 2, gend + forecast))
        z[:, -1, :gend] = sm
        for i in range(gend + forecast):
            if i > 0:
                if i >= gend:
                    z[:, -1, i] = G @ z[:, -1, i - 1]
                z[:, :nx, i] = G @ z[:, :nx, i - 1] + H * eps[:, i][None, :]
            z[:, nx, i] = z[:, -1, i] - z[:, :nx, i].sum(axis=1)
        # conditional decomposition one step ahead
        z1 = np.zeros((n, nx + 2))
        z1[:, -1] = sm[:, gend - 1]
        z1[:, :nx] = H * eps[:, gend - 1][None, :]
        z1[:, nx] = z1[:, -1] - z1[:, :nx].sum(axis=1)
        # conditional decomposition k steps ahead
        if forecast and forecast < j:
            zn = np.zeros((n, nx + 2, forecast + 1))
            zn[:, -1, :] = sm[:, gend - forecast - 1:gend]
            for i in range(forecast + 1):
                if i > 0:
                    zn[:, :nx, i] = G @ zn[:, :nx, i - 1] + H * eps[:, i + gend - forecast - 1][None, :]
                zn[:, nx, i] = zn[:, -1, i] - zn[:, :nx, i].sum(axis=1)
            if j - forecast in save:
                cond[f"time_{j - forecast}"] = zn
        if init:
            zreal[:, :, :j] = z[:, :, :j]
        elif j < nobs:
            zreal[:, :, j - 1] = z[:, :, gend - 1]
        else:
            zreal[:, :, j - 1:] = z[:, :, gend - 1:]
        zcond[:, :, j - 1] = z1
        if j in save:
            rt[f"time_{j}"] = z
        if forecast:
            zfrcst[:, :, j] = z[:, :, gend]
            fc_all[j] = z[:, :, gend - 1:]
            if j in save:
                fc[f"time_{j}"] = fc_all[j]
            if j > forecast + pres:
                for back in ([forecast] + (list(range(forecast - 1, 0, -1)) if j == nobs else [])):
                    o = j - back
                    x = zreal[:, :, o - 1:j] - fc_all[o][:, :, :back + 1]
                    x[:, -2, :] += fc_all[o][:, -1, :back + 1]
                    x[:, -1, :] = zreal[:, -1, o - 1:j]
                    if o in save:
                        rtc[f"time_{o}"] = x
        init = False
    rt["pool"] = zreal
    cond["pool"] = zcond
    if forecast:
        fc["pool"] = zfrcst
    return RealtimeDecomposition(
        endo_names=list(full.endo_names), columns=list(full.exo_names) + ["initial_conditions", "smoothed"],
        nobs=nobs, forecast=forecast, presample=pres, realtime=rt, conditional=cond, realtime_forecast=fc,
        realtime_conditional=rtc, steady_state=full.SteadyState.copy(), parameter_set=full.parameter_set)


# --------------------------------------------------------------------------------- numbers behind the figures
@dataclass
class DecompositionPlotData:
    """What ``plot_shock_decomposition`` draws.

    ``z`` is ``(n_variables, n_components + 1, T)``: Dynare's ``z(i_var,:,:)`` after grouping,
    ``diff``, ``type`` and ``flip`` and before cropping; the rows are the components named in
    ``labels`` (the last one is the initial conditions, with the ungrouped shocks when groups are
    used) and then the line (the smoothed deviation). ``periods`` numbers the columns (sample
    periods from 1; years for ``type="aoa"``); ``steady_state`` is the value Dynare adds to the axis
    labels with the ``steadystate`` option (after the same transformations).

    ``panels`` holds one dict per figure, after cropping and screening: ``name``, ``labels``,
    ``contributions`` ``(n_components, T)``, ``total`` ``(T,)``, ``periods``, ``steady_state``.
    Variables whose bars span less than 1e-6 are skipped, as in Dynare.
    """
    var_list: list
    z: np.ndarray
    labels: list
    periods: np.ndarray
    steady_state: np.ndarray
    panels: list
    kind: str = "shocks"
    type: str = ""
    steadystate: bool = False
    title: str = ""
    parameter_set: str = "calibration"


def stack_bars(contributions):
    """Bottom and top of every bar of a stacked decomposition: positive contributions pile up
    from zero in component order, negative ones pile down (``graph_decomp.m``). Returns
    ``(bottom, top)`` with the shape of ``contributions`` (components, periods); NaN stays NaN."""
    c = np.asarray(contributions, float)
    pos = np.where(c > 0, c, 0.0)
    neg = np.where(c > 0, 0.0, np.nan_to_num(c))
    base_p = np.cumsum(pos, axis=0) - pos
    base_n = np.cumsum(neg, axis=0) - neg
    bottom = np.where(c > 0, base_p, base_n)
    return bottom, bottom + c


def decomposition_plot_data(decomp, var_list=None, model=None, use_shock_groups=None, groups=None, expand=None,
                            init2shocks=None, icd=None, type="", diff=False, flip=False, steadystate=False,
                            plot_init_date=None, plot_end_date=None, screen_shocks=False, first_quarter=1,
                            steady_state=None) -> DecompositionPlotData:
    """The numbers of Dynare's ``plot_shock_decomposition`` (and of the figure of
    ``initial_condition_decomposition`` when ``decomp`` is one).

    ``decomp``: a ``ShockDecomposition`` (ungrouped shocks, already grouped, or initial
    conditions); for realtime results pass ``rt.select(realtime, vintage)``.
    ``var_list``: variables to show (default: the model's original variables, else all).
    ``use_shock_groups`` (name of a ``shock_groups`` block of ``model``) or ``groups``
    (``{label: [shocks]}``); ``expand``: label of a group to show shock by shock.
    ``init2shocks``: name of an ``init2shocks`` block or a list of ``(endogenous, shock)`` pairs,
    with ``icd`` the initial-condition decomposition of the same smoother run.
    ``diff``: first difference (first period NaN, steady state 0). ``type``: ``""``/``"qoq"``
    (as is), ``"yoy"`` (sum of four consecutive periods, steady state times 4) or ``"aoa"``
    (every fourth period, for variables that are already annualised in the model; the sample is
    taken to start in quarter ``first_quarter``). ``flip``: change the sign.
    ``steadystate``: show the steady state on the axis. ``plot_init_date`` / ``plot_end_date``:
    first and last period shown, as integers in the numbering of ``periods``.
    ``screen_shocks``: with more than 18 components and no groups, keep the 16 shocks with the
    largest mean absolute contribution for each variable and sum the rest in ``"Others"``.
    ``steady_state``: overrides the steady state stored in ``decomp`` (see the note on Dynare in
    ``plotting.plot_shock_decomposition``).
    """
    if type not in ("", None, "qoq", "yoy", "aoa"):
        raise ValueError("plot_shock_decomposition: wrong type (use '', 'qoq', 'yoy' or 'aoa')")
    type = type or ""
    is_icd = decomp.kind == "initial_conditions"
    title = "Initial condition decomposition" if is_icd else "Historical shock decomposition"
    pairs = None
    if init2shocks:
        pairs = init2shocks if not isinstance(init2shocks, str) else globals()["init2shocks"](model, init2shocks)
        will_group = bool(use_shock_groups or groups or expand) and decomp.kind == "shocks"
        decomp = apply_init2shocks(decomp, icd, pairs, rename=not will_group)
    grouped = False
    if (use_shock_groups or groups or expand) and not is_icd:
        if decomp.kind != "shocks":
            raise SmootherError("the decomposition is grouped already")
        decomp = group_shocks(decomp, groups, model=model, name=use_shock_groups or None, expand=expand)
        grouped = True
        if use_shock_groups:
            title += f" group {use_shock_groups}"
    elif decomp.kind == "groups":
        grouped = True
    if is_icd:
        screen_shocks = True                          # forced by Dynare's initial_condition_decomposition
        labels = list(decomp.columns[:-2]) + ["Shocks"]
    elif grouped:
        labels = list(decomp.columns[:-1])
    else:
        labels = list(decomp.columns[:-2]) + ["Initial values"]

    names = list(decomp.endo_names)
    if var_list is None:
        orig = list(getattr(model, "orig_endo_names", [])) if model is not None else []
        var_list = [v for v in orig if v in names] or names
    var_list = list(dict.fromkeys(var_list))
    missing = [v for v in var_list if v not in names]
    if missing:
        raise SmootherError(f"variable(s) {missing} not in the decomposition")
    iv = [names.index(v) for v in var_list]
    z = decomp.z[iv].astype(float)
    ss = decomp.steady_state if steady_state is None else steady_state
    ss = np.zeros(len(names)) if ss is None else np.asarray(ss, float)
    ss = ss[iv].copy() if ss.shape[0] == len(names) else ss.copy()
    periods = decomp.first_period + np.arange(z.shape[2])

    if flip:
        title += " flip"
    if pairs:
        title += " init2shocks"
    if diff:
        title += " diff"
        z[:, :, 1:] = z[:, :, 1:] - z[:, :, :-1]
        z[:, :, 0] = np.nan
        ss = ss * 0
    if type == "yoy":
        if z.shape[2] < 4:
            raise SmootherError("type='yoy' needs at least four periods")
        z = z[:, :, :-3] + z[:, :, 1:-2] + z[:, :, 2:-1] + z[:, :, 3:]
        periods = periods[3:]
        ss = 4 * ss
    elif type == "aoa":
        if first_quarter not in (1, 2, 3, 4):
            raise ValueError("first_quarter must be 1, 2, 3 or 4")
        z = z[:, :, 4 - first_quarter::4]
        periods = 1 + np.arange(z.shape[2])           # years
    if flip:
        z = -z
        ss = -ss

    a, b = 0, z.shape[2]
    if plot_init_date is not None:
        hit = np.flatnonzero(periods == int(plot_init_date))
        if hit.size == 0:
            raise SmootherError(f"plot_init_date={plot_init_date} is outside the periods "
                                f"{periods[0]}..{periods[-1]} of the decomposition")
        a = int(hit[0])
    if plot_end_date is not None:
        if int(plot_end_date) <= periods[-1]:
            hit = np.flatnonzero(periods == int(plot_end_date))
            if hit.size == 0:
                raise SmootherError(f"plot_end_date={plot_end_date} is before the first period {periods[0]}")
            b = int(hit[0]) + 1
        else:
            warnings.warn(f"plot_end_date={plot_end_date} is after the last period {periods[-1]}: "
                          "the last period is used")
    zc, pc = z[:, :, a:b], periods[a:b]

    ncomp = z.shape[1] - 1
    screen = bool(screen_shocks) and not grouped and ncomp > 18
    panels = []
    for k, v in enumerate(var_list):
        z1 = zc[k]
        lab = list(labels)
        if screen:
            with np.errstate(invalid="ignore"):
                size = np.mean(np.abs(z1[:-2]), axis=1)
            order = np.argsort(-size, kind="stable")  # NaN (diff shown from its first period) keeps the order
            z1 = np.vstack([z1[order[:16]], z1[order[16:]].sum(axis=0), z1[ncomp - 1:]])
            lab = [labels[i] for i in order[:16]] + ["Others", labels[-1]]
        contrib = z1[:-1]
        clean = np.nan_to_num(contrib)
        ymax = np.max(np.sum(np.where(clean > 0, clean, 0), axis=0)) * 1.1
        ymin = np.min(np.sum(np.where(clean < 0, clean, 0), axis=0)) * 1.1
        if ymax - ymin < 1e-6:
            continue
        panels.append(dict(name=v, labels=lab, contributions=contrib, total=z1[-1], periods=pc,
                           steady_state=float(ss[k])))
    if screen:
        title += " SCREEN"
    return DecompositionPlotData(var_list=var_list, z=z, labels=labels, periods=periods, steady_state=ss,
                                 panels=panels, kind=decomp.kind, type=type, steadystate=bool(steadystate),
                                 title=title, parameter_set=decomp.parameter_set)


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
