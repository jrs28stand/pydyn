# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (GPL-3.0-or-later, Copyright (C) 2003-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: matlab/+forecasts/run.m, stochastic_classical.m and graph.m,
# matlab/+conditional_forecasts/run.m, get_shock_paths.m and plot.m,
# matlab/stochastic_solver/simult_varexo_det.m (confidence bands),
# matlab/perfect-foresight-models/make_ex_.m (the exo_det_simul matrix),
# matlab/estimation/prior_posterior_statistics.m, prior_posterior_statistics_core.m (forecast branches) and
# pm3.m (posterior moments of the forecasts).
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Forecasts: ``forecast``, ``estimation(forecast=...)`` and ``conditional_forecast``.

Three entry points, all with Dynare's conventions (see the docstrings):

* :func:`forecast` -- Dynare's ``forecast`` command after ``stoch_simul`` (``forecasts.run(...,'simul')``):
  the mean forecast from the steady state, from ``histval`` or from a given state, with confidence bands.
  Order 1 uses the analytical variance recursion; orders 2 and 3 simulate ``forecast_replic`` paths and take
  quantiles; ``varexo_det`` variables (order 1 and 2 without pruning) follow ``simult_varexo_det.m``.
* :func:`forecast_at` / :func:`posterior_forecast` -- ``estimation(forecast=N)``: the forecast starts from the
  smoothed state at the end of the sample (``forecasts.run(...,'smoother')``) at the posterior mode / ML mode
  / any parameter vector, or over posterior sub-draws (``oo_.MeanForecast``, ``oo_.PointForecast``,
  ``oo_.PointForecastME``).
* :func:`conditional_forecast` -- ``conditional_forecast`` with ``conditional_forecast_paths`` and
  ``controlled_varexo``; :func:`conditional_forecast_plot_data` returns what ``plot_conditional_forecast`` draws.

Random numbers: every function that draws takes ``rng`` with a ``randn(size)`` method filled in column-major
order, as Octave's (``pydyn.octave_rng.OctaveRandom`` reproduces Dynare-under-Octave's numbers bit for bit;
the default is a numpy generator).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import options as _opt
from .octave_rng import NumpyRandom
from .simulate import simulate as _simulate

__all__ = ["forecast", "Forecast", "forecast_at", "posterior_forecast", "PosteriorForecast", "conditional_forecast",
           "ConditionalForecast", "conditional_forecast_plot_data", "forecast_plot_data", "run_forecasts",
           "octave_quantile", "exo_det_paths", "plot_forecast", "plot_conditional_forecast"]

_DYNARE_FORECAST_REPLIC = 1000
_DYNARE_CONF_SIG = 0.9


# ====================================================================== small helpers
def octave_quantile(x, probs, axis=-1):
    """Octave's ``quantile`` (method 5, also MATLAB's): piecewise linear through the points
    ``(k - 0.5) / n``; returns the quantiles for ``probs`` on a new last axis."""
    x = np.sort(np.moveaxis(np.asarray(x, float), axis, -1), axis=-1)
    n = x.shape[-1]
    if n == 1:
        return np.repeat(x, len(probs), axis=-1)
    out = []
    for p in probs:
        pp = n * p + 0.5
        pi = int(max(min(np.floor(pp), n - 1), 1))
        pr = max(min(pp - pi, 1.0), 0.0)
        out.append((1 - pr) * x[..., pi - 1] + pr * x[..., pi])
    return np.stack(out, axis=-1)


def _rng(rng):
    if rng is None:
        return NumpyRandom()
    if isinstance(rng, (int, np.integer)):
        return NumpyRandom(np.random.default_rng(int(rng)))
    if isinstance(rng, np.random.Generator):
        return NumpyRandom(rng)
    return rng


def _randn(rng, shape):
    """Octave's ``randn(r, c)`` (column-major fill) on any generator with ``randn(size)``."""
    shape = tuple(int(s) for s in shape)
    if min(shape) == 0:
        return np.zeros(shape)
    return np.asarray(rng.randn(shape), float).reshape(shape)


def _last_option(model, key, kinds, default=None):
    """Value of ``key`` in the last statement among ``kinds`` that sets it (Dynare accumulates options_)."""
    val = default
    for st in getattr(model, "statements", []):
        if st.get("statementName") in kinds and key in (st.get("options") or {}):
            val = st["options"][key]
    return val


def _scalar(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        v = v[0]
    return v


def _maxlag(model) -> int:
    return int(model.info["maximum_lag"])


def _maxlead(model) -> int:
    return int(model.info.get("maximum_lead", 1))


def _native_option(model, name, default=None):
    """``options_.<name> = value;`` lines of the .mod file (pydyn stores them in ``matlab_options``)."""
    mo = getattr(model, "matlab_options", None)
    try:
        v = mo[name]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if v is None else v


def _chol_upper(S):
    """MATLAB's chol(): upper triangular R with R'R = S."""
    return np.linalg.cholesky(S).T


def _norminv(p):
    return float(stats.norm.ppf(p))


def _varobs(model, varobs=None):
    return list(model.varobs if varobs is None else varobs)


def _measurement_cov(model, varobs, H=None):
    """``M_.H`` in varobs order, or ``None`` when the .mod file declares no measurement error
    (Dynare then has ``M_.H = 0``)."""
    if H is not None:
        Hm = np.atleast_2d(np.asarray(H, float))
        return Hm if np.any(Hm != 0) else None
    if not getattr(model, "measurement_errors", None) and not getattr(model, "me_cov", None):
        return None
    try:
        Hm = model.me_H(varobs)
    except Exception as exc:                       # ModelError: a measurement error on a variable that is not observed
        raise ValueError(str(exc)) from None
    return Hm if np.any(Hm != 0) else None            # forecasts.run stores *_ME only if nnz(M_.H) > 0


def _settings(model, order=None, pruning=None, periods=None, conf_sig=None, replic=None):
    """Options of the .mod file that the forecast commands read (``options_.order``, ``pruning``, ``periods``,
    ``forecasts.conf_sig``, ``forecast_replic``), overridden by the arguments."""
    so = _opt.stoch_options(model)
    if order is None:
        order = int(_scalar(so.get("order", 2)))
    if model.linear:
        order = 1                                       # stoch_simul.m: options_.linear forces order = 1
    if pruning is None:
        pruning = bool(so.get("pruning", False))
    if conf_sig is None:
        conf_sig = _scalar(_last_option(model, "forecasts.conf_sig", ("forecast", "estimation", "stoch_simul"),
                                        _DYNARE_CONF_SIG))
    if replic is None:
        replic = int(_native_option(model, "forecast_replic", _DYNARE_FORECAST_REPLIC))
    if periods is None:
        periods = _scalar(_last_option(model, "periods", ("stoch_simul", "forecast"), 0))
    return int(order), bool(pruning), int(periods or 0), float(conf_sig), int(replic)


# ====================================================================== deterministic exogenous variables
def exo_det_paths(model, periods=0, horizon=None):
    """``oo_.exo_det_simul`` as ``make_ex_.m`` builds it for the forecast commands, with the rows that
    ``forecasts.run`` adds when the forecast is longer: ``maximum_lag + periods + maximum_lead`` rows of
    steady-state values with the ``shocks`` of the ``varexo_det`` variables written in (``periods`` is
    ``options_.periods``; a shock beyond the last row extends the matrix with zeros, as MATLAB does), then
    steady-state rows up to ``maximum_lag + horizon``. Returns ``(exo_det_simul, exo_det_length)``."""
    ml = _maxlag(model)
    ss = model.exo_det_steady()
    X = np.tile(ss, (ml + int(periods) + _maxlead(model), 1))
    for st in getattr(model, "statements", []):
        if st.get("statementName") == "mshocks":
            if any(d.get("var") in model._ixd for d in st.get("deterministic_shocks", [])):
                raise NotImplementedError("mshocks on varexo_det variables are not supported by the forecast")
    for i, p1, p2, val in model.det_shocks_exo_det:
        if not isinstance(p1, (int, np.integer)):
            raise NotImplementedError("forecast with dated shocks on varexo_det variables is not supported")
        k = np.arange(int(p1), int(p2) + 1) + ml                      # 1-based rows
        if np.any(k == 0):
            raise ValueError("make_ex_: The model does not have lags, so you cannot set values for period 0")
        if k.max() > X.shape[0]:
            X = np.vstack([X, np.zeros((k.max() - X.shape[0], X.shape[1]))])
        X[k - 1, i] = val
    length = X.shape[0] - ml
    if horizon is not None and horizon > length:
        X = np.vstack([X, np.tile(ss, (horizon - length, 1))])
        length = horizon
    return X, length


# ====================================================================== results
@dataclass
class Forecast:
    """Dynare's ``oo_.forecast``: dicts keyed by variable name with arrays of ``periods`` values
    (``Mean``, ``HPDinf``, ``HPDsup``, and for observed variables with measurement error ``HPDinf_ME`` and
    ``HPDsup_ME``); ``Exogenous`` holds the paths of the ``varexo_det`` variables."""
    var_list: list
    periods: int
    conf_sig: float
    order: int
    Mean: dict
    HPDinf: dict
    HPDsup: dict
    HPDinf_ME: dict = field(default_factory=dict)
    HPDsup_ME: dict = field(default_factory=dict)
    Exogenous: dict = field(default_factory=dict)
    initial_state: np.ndarray = None            # y0 in declaration order (levels; logs under loglinear)
    method: str = "analytical"                  # "analytical" (order 1) or "simulation"
    replic: int = 0
    task: str = "simul"

    def as_oo(self) -> dict:
        """``oo_.forecast`` as nested plain dicts."""
        d = {"Mean": self.Mean, "HPDinf": self.HPDinf, "HPDsup": self.HPDsup}
        if self.HPDinf_ME:
            d.update(HPDinf_ME=self.HPDinf_ME, HPDsup_ME=self.HPDsup_ME)
        if self.Exogenous:
            d["Exogenous"] = self.Exogenous
        return d

    def __getitem__(self, name):
        return {"Mean": self.Mean[name], "HPDinf": self.HPDinf[name], "HPDsup": self.HPDsup[name]}


def _ivar(model, var_list):
    names = list(model.endo_names)
    if not var_list:
        var_list = names[:model.raw.get("orig_endo_nbr", model.n_endo)]
    out = []
    for v in var_list:
        if v not in names:
            raise ValueError(f"{v} isn't an endogenous variable")
        out.append(names.index(v))
    return list(var_list), np.array(out, int)


# ====================================================================== the dr used
def _decision_rules(model, order, pruning, sol=None, dr=None):
    """``(dr, sol)``: decision-rule dict (``as_dr`` of the first-order solution, or ``k_order``) and the
    first-order solution ``sol`` (``G``, ``H`` in declaration order, ``ys`` in logs under loglinear)."""
    from .perturbation import first_order
    if dr is not None:
        dr = dr.as_dr() if hasattr(dr, "as_dr") else dict(dr)
        dr.setdefault("order", 1)
        if sol is None and hasattr(dr.get("_sol"), "G"):
            sol = dr["_sol"]
        return dr, sol
    if sol is None:
        sol = first_order(model)
    if order == 1:
        d = sol.as_dr()
        d["pruning"] = pruning
        return d, sol
    from . import korder
    d = korder.k_order(model, order=order, pruning=pruning)
    return d, sol


def _linear_rules(sol, dr):
    """G and H (declaration order, deviations from ``dr['ys']``) of the first-order part of ``dr``."""
    if sol is not None:
        return np.asarray(sol.G, float), np.asarray(sol.H, float)
    n, nx = len(dr["ys"]), dr["ghu"].shape[1]
    G, H = np.zeros((n, n)), np.zeros((n, nx))
    ov, sv = np.asarray(dr["order_var"], int), np.asarray(dr["state_var"], int)
    G[np.ix_(ov, sv)] = dr["ghx"]
    H[ov] = dr["ghu"]
    return G, H


# ====================================================================== the core of forecasts.run / stochastic_classical
def _forecast_core(model, dr, G, H, y0, horizon, ivar, Sigma_e, Hme, varobs, conf_sig, order, pruning, replic, rng,
                   ex_det=None, exo_det_dr=None):
    """Mean path, confidence bands and bands with measurement error.

    ``y0``: (n_endo,) initial state in the units of ``dr['ys']``. Returns ``(yf, ci, ci_me, method)`` with ``yf`` of
    shape (n_var, 1 + horizon') (first column the initial state), ``ci`` (n_var, 1 + horizon', 2) and ``ci_me`` the
    same or ``None`` when there is no measurement error to add; ``horizon'`` is ``horizon`` or, with varexo_det
    paths longer than the horizon, the length of those paths. The columns are Dynare's ``yf(:, maximum_lag + 1 : end)``
    preceded by the initial state."""
    n, nx = len(dr["ys"]), Sigma_e.shape[0]
    fact = _norminv((1 - conf_sig) / 2)
    obs = _varobs(model, varobs)
    obs_id = np.array([model._ie[v] for v in obs], int)
    # positions of the observed variables in the requested list, and which observed variables they are
    in_list = np.array([int(np.nonzero(ivar == i)[0][0]) if i in ivar else -1 for i in obs_id], int)
    me_ok = Hme is not None and np.any(in_list >= 0)

    use_xd = ex_det is not None
    if use_xd:
        from . import varexo_det as xd
        H_total = ex_det.shape[0] - 1
        ex = np.zeros((H_total, nx))
        horizon_sim = H_total
        yf_full = xd.simulate(model, exo_det_dr, ex, ex_det, y0=y0, order=order)
    else:
        horizon_sim = horizon
        yf_full = np.vstack([y0, _simulate(dr, np.zeros((horizon, nx)), order=order, pruning=pruning, y0=y0)])
    yf = yf_full[:, ivar].T                                           # (n_var, 1 + horizon_sim)

    if order == 1:
        V = np.zeros((n, n))
        var = np.zeros((len(ivar), 1 + horizon_sim))
        HSH = H @ Sigma_e @ H.T
        for t in range(1, horizon_sim + 1):
            V = G @ V @ G.T + HSH
            var[:, t] = np.diag(V)[ivar]
        width = -fact * np.sqrt(var)
        ci = np.stack([yf - width, yf + width], axis=-1)
        ci_me = None
        if me_ok:
            var_me = var.copy()
            for k, pos in enumerate(in_list):
                if pos >= 0:
                    var_me[pos] = var[pos] + Hme[k, k]
            width_me = -fact * np.sqrt(var_me)
            ci_me = np.stack([yf - width_me, yf + width_me], axis=-1)
        return yf, ci, ci_me, "analytical"

    # simulation: shocks of zero variance are not drawn
    nzero = np.nonzero(np.diag(Sigma_e) != 0)[0]
    chol_S = _chol_upper(Sigma_e[np.ix_(nzero, nzero)]) if nzero.size else np.zeros((0, 0))
    sims = np.empty((replic, n, 1 + horizon_sim))
    probs = [(1 - conf_sig) / 2, 1 - (1 - conf_sig) / 2]
    for it in range(replic):
        shocks = np.zeros((horizon_sim, nx))
        if nzero.size:
            shocks[:, nzero] = _randn(rng, (nzero.size, horizon_sim)).T @ chol_S
        if use_xd:
            sims[it] = xd.simulate(model, exo_det_dr, shocks, ex_det, y0=y0, order=order).T
        else:
            sims[it] = np.vstack([y0, _simulate(dr, shocks, order=order, pruning=pruning, y0=y0)]).T
    q = octave_quantile(np.moveaxis(sims[:, ivar, :], 0, -1), probs)       # (n_var, 1+h, 2)
    ci_me = None
    if me_ok:
        # measurement-error draws: one block of randn(nxs, horizon + 1) per replication, after all the shock draws
        iz = np.nonzero(np.diag(Hme) != 0)[0]
        chol_H = _chol_upper(Hme[np.ix_(iz, iz)]) if iz.size else np.zeros((0, 0))
        for it in range(replic):
            if iz.size:
                e = np.zeros((1 + horizon_sim, len(obs)))
                e[:, iz] = _randn(rng, (iz.size, 1 + horizon_sim)).T @ chol_H
                sims[it][obs_id, :] += e.T
        ci_me = octave_quantile(np.moveaxis(sims[:, ivar, :], 0, -1), probs)
    return yf, q, ci_me, "simulation"


def _initial_state_simul(model, dr, ys_used, raw=False):
    """``y0`` of the 'simul' task: the steady state, or ``histval`` (zero for variables the block leaves out; with
    ``loglinear`` the logs of the values it gives)."""
    st = [s for s in getattr(model, "statements", []) if s.get("statementName") == "histval"]
    if not st:
        return np.array(ys_used, float)
    from types import SimpleNamespace
    import importlib
    _pf = importlib.import_module("pydyn.perfect_foresight")      # the package attribute is the function of that name
    ws = SimpleNamespace(endo_histval=None, exo_histval=None)
    _pf._histval(model, ws, st[-1], model._stmt_values.get(id(st[-1])))
    h = ws.endo_histval
    if h is None:
        return np.array(ys_used, float)
    return _histval_state(model, h, raw)


def _histval_state(model, h, raw=False):
    """The initial state of ``forecasts.run`` from ``M_.endo_histval`` (``h``, one value per endogenous variable):
    ``h`` itself, or under ``loglinear`` ``log_variable(h)`` (zeros give -Inf).  ``raw``: ``h`` as given."""
    if raw:                                      # the values of the histval block, as given (zero where it is silent)
        return np.array(h, float)
    if _opt.stoch_options(model).get("loglinear", False):
        from .perturbation import log_mask
        mask = log_mask(model)
        out = np.zeros_like(h)
        out[mask & (h != 0)] = np.log(h[mask & (h != 0)])
        out[~mask] = h[~mask]
        # Dynare: log_variable(1:endo_nbr, M_.endo_histval, M_) takes the log of every value; zeros give -Inf
        out[mask & (h == 0)] = -np.inf
        return out
    return np.array(h, float)


# ====================================================================== forecast after stoch_simul
def forecast(model, var_list=None, *, periods=None, conf_sig=None, order=None, pruning=None, replic=None, rng=None,
             sol=None, dr=None, y0=None, Sigma_e=None, H=None, varobs=None, exo_det_periods=None) -> Forecast:
    """Dynare's ``forecast`` command after ``stoch_simul`` (``forecasts.run(var_list, M_, options_, oo_, 'simul')``).

    The path is the deterministic (zero-shock) path of the decision rules from ``y0`` -- the steady state, or
    ``histval`` when the .mod file has one -- ``Mean``; ``HPDinf``/``HPDsup`` are the ``conf_sig`` band of the
    forecast distribution ignoring measurement error and, for observed variables when the .mod file declares
    measurement errors, ``HPDinf_ME``/``HPDsup_ME`` the band including it.

    * order 1 (and linear models): analytical variance recursion, no random numbers;
    * orders 2 and 3: ``replic`` (``options_.forecast_replic``, 1000) simulations with shocks drawn as
      ``randn(nxs, horizon)' * chol(Sigma_e)`` and quantiles by Octave's method 5. At order 2 and 3 the mean path is
      ``simult_`` with zero shocks (so it includes the second-order constant and, with ``pruning``, the pruned
      recursion), not the mean of the simulations;
    * ``varexo_det``: ``simult_varexo_det.m`` with ``oo_.exo_det_simul`` built by :func:`exo_det_paths`
      (orders 1 and 2, no pruning at order 2); ``Exogenous`` returns the paths.

    Defaults (``order``, ``pruning``, ``periods``, ``conf_sig``, ``replic``) come from the .mod file as
    Dynare's ``options_`` would hold them; ``periods`` 0 means 5, as in Dynare. ``sol``/``dr``: an existing
    first-order solution or decision-rule dict. ``Sigma_e``, ``H`` (measurement-error covariance in ``varobs``
    order), ``varobs`` override the model's. ``y0``: initial state (declaration order, in the units of the
    solution: logs under loglinear). Refused: OccBin models, models without lags."""
    _opt.refuse_occbin(model, "forecast")
    order, pruning, periods_opt, conf_sig, replic = _settings(model, order, pruning, periods, conf_sig, replic)
    if _maxlag(model) == 0:
        raise NotImplementedError("forecast: models without lagged endogenous variables are not supported "
                                  "(Dynare's forecasts.run itself fails on them)")
    if model.n_exo_det and order > 2:
        raise ValueError("forecasts.run: forecasting with varexo_det does not support order>2.")
    if model.n_exo_det and order == 2 and pruning:
        raise ValueError("forecasts.run: forecasting with varexo_det does not support pruning.")
    if _opt.stoch_options(model).get("loglinear", False) and order > 1:
        raise NotImplementedError("loglinear is only available at order 1 (as in Dynare)")
    horizon = periods_opt if periods_opt else 5
    var_list, ivar = _ivar(model, var_list)
    dr, sol = _decision_rules(model, order, pruning, sol, dr)
    Sigma = np.array(model.Sigma_e if Sigma_e is None else Sigma_e, float)
    varobs = _varobs(model, varobs)
    Hme = _measurement_cov(model, varobs, H)
    ys = np.asarray(dr["ys"], float)
    y0 = _initial_state_simul(model, dr, ys) if y0 is None else np.asarray(y0, float)
    G, Hr = _linear_rules(sol, dr)
    rng = _rng(rng)
    ex_det = exo_dr = None
    exo_out = {}
    iorder = 1 if model.linear else order
    if model.n_exo_det:
        from . import varexo_det as xd
        X, length = exo_det_paths(model, exo_det_periods if exo_det_periods is not None else periods_opt, horizon)
        ex_det = X
        exo_dr = xd.decision_rules(model, dr)
        order_used = iorder
    else:
        order_used = order
    yf, ci, ci_me, method = _forecast_core(model, dr, G, Hr, y0, horizon, ivar, Sigma, Hme, varobs, conf_sig,
                                           order_used, pruning, replic, rng, ex_det, exo_dr)
    out = _pack(model, var_list, ivar, yf, ci, ci_me, horizon, conf_sig, order_used, varobs, Hme, y0, method, replic,
                "simul")
    if model.n_exo_det:
        ml = _maxlag(model)
        for i, nm in enumerate(model.exo_det_names):
            out.Exogenous[nm] = np.array(ex_det[ml:ml + horizon, i], float)
    return out


def _pack(model, var_list, ivar, yf, ci, ci_me, horizon, conf_sig, order, varobs, Hme, y0, method, replic, task):
    """Dynare's ``forecast.Mean`` etc.: periods 1 .. horizon (the initial-state column is dropped)."""
    out = Forecast(var_list=list(var_list), periods=int(horizon), conf_sig=float(conf_sig), order=int(order), Mean={},
                   HPDinf={}, HPDsup={}, initial_state=np.array(y0, float), method=method, replic=int(replic if method == "simulation" else 0), task=task)
    sl = slice(1, 1 + horizon)
    for i, v in enumerate(var_list):
        out.Mean[v] = np.array(yf[i, sl])
        out.HPDinf[v] = np.array(ci[i, sl, 0])
        out.HPDsup[v] = np.array(ci[i, sl, 1])
        if ci_me is not None and v in varobs:
            out.HPDinf_ME[v] = np.array(ci_me[i, sl, 0])
            out.HPDsup_ME[v] = np.array(ci_me[i, sl, 1])
    return out


# ====================================================================== forecast from the smoothed state (estimation)
def _options_forecast(model, name="forecast"):
    v = _last_option(model, name, ("estimation",), None)
    return None if v is None else int(_scalar(v))


def _smoothed_pieces(res):
    """What ``forecasts.run(..., 'smoother')`` and ``prior_posterior_statistics_core`` read from the smoother:
    constants, trends and the observed variables of a ``SmootherResult``."""
    names = list(res.endo_names)
    n = len(names)
    ys_c = np.log(np.asarray(res.SteadyState, float)) if res.loglinear else np.asarray(res.SteadyState, float)
    mf = np.asarray(res.obs_index, int)
    return names, n, ys_c, mf


def _dr_from_rules(sol, G, H, ys_c):
    ov, sv = np.asarray(sol.order_var, int), np.asarray(sol.state_var, int)
    return {"ys": np.asarray(ys_c, float), "order_var": ov, "state_var": sv, "ghx": G[np.ix_(ov, sv)], "ghu": H[ov],
            "order": 1, "pruning": False}


def forecast_at(model, estimation=None, theta=None, *, parameter_set=None, mode=None, mcmc=None, data=None,
                horizon=None, conf_sig=None, var_list=None, smoothed=None, shift_bands=False, **smoother_options):
    """``forecasts.run(var_list, M_, options_, oo_, 'smoother', dataset_info)``: the forecast of ``estimation(forecast=N)``
    at one parameter vector, started from the smoothed state of the last period of the sample.

    The vector is chosen as for :func:`pydyn.smoother.smooth` (``estimation=`` with ``theta=`` or
    ``parameter_set=`` ``"posterior_mode"``/``"mle_mode"``/``"posterior_mean"``/..., with ``mode=``/``mcmc=``;
    without ``estimation`` the calibration), or an existing ``SmootherResult`` is passed as ``smoothed=``; the
    other keyword arguments are smoother options (``datafile``, ``first_obs``, ``nobs``, ``prefilter``,
    ``loglinear``, ...). ``horizon`` defaults to the ``forecast`` option of the ``estimation`` statement,
    ``conf_sig`` to ``estimation(conf_sig=)`` / 0.9.

    Returns a :class:`Forecast` (order 1: ``Mean``, ``HPDinf``, ``HPDsup`` and, for observed variables with
    measurement error, ``HPDinf_ME``/``HPDsup_ME``). Observed variables get their ``observation_trends`` trend
    (moved to the forecast periods) and, with ``prefilter``, the data mean instead of the steady state in the mean
    forecast. As in Dynare, these are added to the mean only: the bands stay around the untrended forecast. Pass
    ``shift_bands=True`` to shift the bands by the same amount (what a user would expect; Dynare 7.2 does not).
    With ``varexo_det`` variables the path follows ``simult_varexo_det.m`` (not with ``loglinear``)."""
    from . import smoother as _sm
    _opt.refuse_occbin(model, "forecast_at")
    if smoothed is None:
        extra = {} if parameter_set is None else {"parameter_set": parameter_set}
        smoothed = _sm.smooth(model, data, estimation=estimation, theta=theta, mode=mode, mcmc=mcmc, **extra,
                              **smoother_options)
    res = smoothed
    if horizon is None:
        horizon = _options_forecast(model)
    if not horizon:
        raise ValueError("forecast_at: no horizon (pass horizon=, or use estimation(forecast=N) in the .mod file)")
    horizon = int(horizon)
    if conf_sig is None:
        conf_sig = float(_scalar(_last_option(model, "forecasts.conf_sig", ("estimation", "forecast"), _DYNARE_CONF_SIG)))
    names, n, ys_c, mf = _smoothed_pieces(res)
    ml = _maxlag(model)
    if ml == 0:
        raise NotImplementedError("forecast: models without lagged endogenous variables are not supported")
    var_list, ivar = _ivar(model, var_list)
    varobs = list(res.varobs)
    G, Hr = np.asarray(res.G, float), np.asarray(res.H, float)
    dr = _dr_from_rules(res.solution, G, Hr, ys_c)
    # initial state: smoothed value minus its constant (mean or steady state), plus the steady state, minus the trend
    y0 = np.empty(n)
    for i, v in enumerate(names):
        y0[i] = res.SmoothedVariables[v][-1] - res.Constant[v][-1] + ys_c[i] - res.Trend[v][-1]
    Hme = res.me_cov
    Hme = None if (Hme is None or not np.any(Hme != 0)) else np.asarray(Hme, float)
    Sigma = np.asarray(res.Sigma_e, float)
    gend, first_obs = res.nobs, int(res.first_obs)
    prefilter = int(res.options.get("prefilter", 0)) == 1
    obs_in_list = [(int(np.nonzero(ivar == mf[k])[0][0]), k) for k in range(len(varobs)) if mf[k] in ivar]
    ex_det = exo_dr = None
    if model.n_exo_det:
        if res.loglinear:
            raise NotImplementedError("forecast with varexo_det and loglinear is not supported")
        from . import varexo_det as xd
        X, _ = exo_det_paths(model, 0, horizon)
        ex_det = X
        saved = model.params
        try:
            model.params = np.asarray(res.params, float)       # varexo_det.decision_rules reads model.params
            exo_dr = xd.decision_rules(model, dict(dr, ys=np.asarray(res.SteadyState, float)))
        finally:
            model.params = saved
    yf, ci, ci_me, method = _forecast_core(model, dr, G, Hr, y0, horizon, ivar, Sigma, Hme, varobs, conf_sig, 1, False,
                                           0, None, ex_det, exo_dr)
    # trend and prefilter: added to the mean forecast of the observed variables (forecasts.run)
    cols = yf.shape[1]                                      # initial state + forecast periods
    adj = np.zeros_like(yf)
    tc = np.asarray(res.TrendCoeffs, float)[mf]
    t = first_obs + gend - 1 + np.arange(1 - ml, 1 - ml + cols)           # first_obs+gend-1+(1-maxlag:horizon)
    for pos, k in obs_in_list:
        tr = tc[k] * t
        if prefilter:
            tr = tr - np.mean(tc[k] * np.arange(first_obs, first_obs + gend))
            adj[pos] += tr - ys_c[mf[k]] + res.obs_constant[k]
        else:
            adj[pos] += tr
    if np.any(adj) and not shift_bands:
        from .options import warn_once
        warn_once(model, "forecast: the observation trend / prefilter mean is added to the mean forecast only, as in "
                         "Dynare 7.2; the HPD bands stay around the untrended forecast (pass shift_bands=True to shift them)",
                  stacklevel=3)
    yf = yf + adj
    if shift_bands:
        ci = ci + adj[..., None]
        if ci_me is not None:
            ci_me = ci_me + adj[..., None]
    out = _pack(model, var_list, ivar, yf, ci, ci_me, horizon, conf_sig, 1, varobs, Hme, y0, method, 0, "smoother")
    if model.n_exo_det:
        for i, nm in enumerate(model.exo_det_names):
            out.Exogenous[nm] = np.array(ex_det[ml:ml + horizon, i], float)
    return out


# ====================================================================== posterior forecasts (prior_posterior_statistics)
@dataclass
class PosteriorForecast:
    """Posterior distribution of the forecasts over parameter draws (Dynare's ``oo_.MeanForecast``,
    ``oo_.PointForecast`` and ``oo_.PointForecastME``).

    Each of ``MeanForecast``, ``PointForecast`` and ``PointForecastME`` is a dict with keys ``Mean``, ``Median``,
    ``Var``, ``HPDinf``, ``HPDsup`` (each ``{variable: array(horizon)}``) and ``deciles``
    (``{variable: array(9, horizon)}``), computed over the draws as ``posterior_moments.m`` does (``mh_conf_sig``
    HPD interval by Dynare's rule, deciles at ``round(q n)``). ``MeanForecast`` is the forecast without future
    shocks from the smoothed state at each draw, ``PointForecast`` adds one draw of the future shocks, and
    ``PointForecastME`` (observed variables, when measurement errors exist) one draw of the measurement errors."""
    var_list: list
    horizon: int
    mh_conf_sig: float
    draws: int
    MeanForecast: dict
    PointForecast: dict
    PointForecastME: dict = field(default_factory=dict)
    params: np.ndarray = field(default=None, repr=False)       # parameter vectors used (draws, n)
    mean_draws: np.ndarray = field(default=None, repr=False)   # (draws, horizon, n_endo) when keep_draws
    point_draws: np.ndarray = field(default=None, repr=False)
    me_draws: np.ndarray = field(default=None, repr=False)     # (draws, horizon, n_obs)


def _stats(draws, names_idx, names, mh_conf_sig):
    """pm3's moments for the series ``draws[:, t, i]`` (draws on the first axis)."""
    from .posterior import moments_along_draws
    out = {k: {} for k in ("Mean", "Median", "Var", "HPDinf", "HPDsup", "deciles")}
    for name, i in zip(names, names_idx):
        mom = moments_along_draws(draws[:, :, i], mh_conf_sig)          # one series per period
        out["Mean"][name] = mom["mean"]
        out["Median"][name] = mom["median"]
        out["Var"][name] = mom["variance"]
        out["HPDinf"][name] = mom["hpd_inf"]
        out["HPDsup"][name] = mom["hpd_sup"]
        out["deciles"][name] = mom["deciles"]                              # (9, horizon), as oo_ holds them
    return out


def posterior_forecast(estimation, params=None, mc=None, *, horizon=None, var_list=None, sub_draws=None,
                       mh_conf_sig=None, drop=None, rng=None, subsample_rng=None, keep_draws=False,
                       smoothers=None, **smoother_options) -> PosteriorForecast:
    """Forecasts over posterior draws: the forecast branch of ``prior_posterior_statistics_core.m`` with the moments
    of ``pm3.m`` (``estimation(forecast=N, mh_replic=...)``; with ``params`` also draws from the prior).

    ``params``: parameter vectors (rows, ordered as ``estimation.names``), or ``mc`` (the chains of
    ``estimation.sample()``) from which ``sub_draws`` vectors are drawn with replacement after the burn-in
    (:func:`pydyn.posterior.posterior_subsample`: Dynare's scheme, not its random numbers).
    For each vector the smoother is run (``pydyn.smoother.smooth``; ``smoother_options`` as there) and, from the
    smoothed state of the last period, a path without future shocks (``MeanForecast``), a path with one draw of
    the shocks ``randn(nxs, horizon)`` (``PointForecast``) and, if measurement errors exist, one draw of those
    (``PointForecastME``) are stored. ``rng`` (``randn(size)`` generator; ``OctaveRandom(seed)`` reproduces
    Dynare under Octave draw for draw) supplies the normals in Dynare's order: per draw the shocks, then the
    measurement errors. Dynare draws the measurement errors as ``chol(H) * randn`` (upper factor), which has
    covariance ``chol(H) chol(H)'``: this is reproduced, and equals ``H`` only for a diagonal ``H`` (pydyn
    warns otherwise). ``horizon`` defaults to ``estimation(forecast=N)``; ``var_list`` to the variables listed
    after ``estimation`` or all declared variables. ``smoothers``: a list of ready ``SmootherResult`` (one per draw)."""
    from . import smoother as _sm
    from . import posterior as _post
    model = estimation.model
    _opt.refuse_occbin(model, "posterior_forecast")
    if horizon is None:
        horizon = _options_forecast(model)
    if not horizon:
        raise ValueError("posterior_forecast: no horizon (pass horizon=, or estimation(forecast=N))")
    horizon = int(horizon)
    if mh_conf_sig is None:
        mh_conf_sig = float(estimation.options.get("mh_conf_sig", 0.9) or 0.9)
    if params is None:
        if mc is None:
            raise ValueError("posterior_forecast needs parameter draws (params=) or chains (mc=)")
        size = _post.number_of_subdraws(mc, estimation, sub_draws, drop)
        params, _ = _post.posterior_subsample(mc, size, drop, subsample_rng)
    params = np.atleast_2d(np.asarray(params, float))
    var_list = list(var_list or estimation.var_list or model.endo_names[:model.raw.get("orig_endo_nbr", model.n_endo)])
    _ivar(model, var_list)
    rng = _rng(rng)
    ml = _maxlag(model)
    if ml == 0:
        raise NotImplementedError("forecast: models without lagged endogenous variables are not supported")
    B, n = params.shape[0], model.n_endo
    obs_names = list(model.varobs)
    mean_d = np.full((B, horizon, n), np.nan)
    point_d = np.full((B, horizon, n), np.nan)
    me_d = np.full((B, horizon, len(obs_names)), np.nan)
    any_me = False
    for b in range(B):
        res = smoothers[b] if smoothers is not None else _sm.smooth(model, estimation=estimation, theta=params[b],
                                                                   **smoother_options)
        names, n_, ys_c, mf = _smoothed_pieces(res)
        G, Hr = np.asarray(res.G, float), np.asarray(res.H, float)
        Sigma = np.asarray(res.Sigma_e, float)
        Hme = np.asarray(res.me_cov, float) if res.me_cov is not None else np.zeros((len(obs_names),) * 2)
        dev0 = np.asarray(res.smoothed[-1], float)                      # alphahat(:, end): deviation of the last period
        gend, first_obs = res.nobs, int(res.first_obs)
        tc = np.asarray(res.TrendCoeffs, float)[mf]
        prefilter = int(res.options.get("prefilter", 0)) == 1
        nz = np.nonzero(np.diag(Sigma) != 0)[0]
        chol_S = _chol_upper(Sigma[np.ix_(nz, nz)]) if nz.size else np.zeros((0, 0))
        e = _randn(rng, (nz.size, horizon))
        B1 = Hr[:, nz] @ chol_S.T
        paths = (np.empty((horizon + 1, n)), np.empty((horizon + 1, n)))
        paths[0][0] = paths[1][0] = dev0
        for t in range(horizon):
            paths[0][t + 1] = G @ paths[0][t]
            paths[1][t + 1] = G @ paths[1][t] + B1 @ e[:, t]
        t_idx = (first_obs - 1) + gend + np.arange(1 - ml, horizon + 1)         # (1-maxlag:horizon)
        out = []
        for path in paths:
            if prefilter:
                path[:, mf] += res.obs_constant[None, :]
                path[:, mf] += np.outer(t_idx, tc) - np.mean(np.outer(np.arange(first_obs, first_obs + gend), tc),
                                                             axis=0)[None, :]
            else:
                path[:, mf] += np.outer(t_idx, tc)
            out.append((path + ys_c[None, :])[ml:])                             # drop the starting value
        mean_d[b], point_d[b] = out
        iz = np.nonzero(np.diag(Hme) != 0)[0]
        me_shocks = np.zeros((len(obs_names), horizon))
        if iz.size:
            any_me = True
            chol_H = _chol_upper(Hme[np.ix_(iz, iz)])
            if not np.allclose(chol_H, np.diag(np.diag(chol_H))):
                from .options import warn_once
                warn_once(model, "posterior_forecast: correlated measurement errors; Dynare draws them as chol(H)*randn, "
                                 "whose covariance is chol(H)*chol(H)', not H (reproduced here)", stacklevel=3)
            me_shocks[iz] = chol_H @ _randn(rng, (iz.size, horizon))
        me_d[b] = point_d[b][:, mf] + me_shocks.T
    idx = {v: model._ie[v] for v in var_list}
    res_mean = _stats(mean_d, list(idx.values()), list(idx), mh_conf_sig)
    res_point = _stats(point_d, list(idx.values()), list(idx), mh_conf_sig)
    res_me = {}
    if any_me:
        sel = [v for v in obs_names if v in var_list]
        res_me = _stats(me_d, [obs_names.index(v) for v in sel], sel, mh_conf_sig)
    return PosteriorForecast(var_list=var_list, horizon=horizon, mh_conf_sig=float(mh_conf_sig), draws=B,
                             MeanForecast=res_mean, PointForecast=res_point, PointForecastME=res_me, params=params,
                             mean_draws=mean_d if keep_draws else None, point_draws=point_d if keep_draws else None,
                             me_draws=me_d if (keep_draws and any_me) else None)


# ====================================================================== conditional forecasts (+conditional_forecasts/run.m)
def _matlab_round(x):
    """MATLAB's ``round`` (half away from zero) for positive numbers."""
    return int(np.floor(x + 0.5))


@dataclass
class ConditionalForecast:
    """What ``conditional_forecasts.run`` stores in ``oo_.conditional_forecast``.

    ``cond`` and ``uncond`` are ``{"Mean": {name: array(periods + 1)}, "ci": {name: array(2, periods + 1)}}`` for
    every endogenous variable (the first column is the last observation / initial state, as in Dynare; ``ci``
    rows are the lower and upper band). ``controlled_exo_variables`` has the same layout for the controlled shocks
    (the periods of the paths only). ``controlled_variables`` and ``instruments`` are the names of the constrained
    variables and of the controlled shocks (the i-th variable is controlled by the i-th shock). ``graph`` has
    Dynare's ``title`` and ``fname``. ``trend`` is the constant plus observation trend added back to the
    deviations (``(n_endo, periods + 1)``, declaration order), ``paths`` the constrained values."""
    cond: dict
    uncond: dict
    controlled_exo_variables: dict
    controlled_variables: list
    controlled_variables_index: np.ndarray        # position in decision-rule order (0-based), as forecasts.controlled_variables
    instruments: list
    graph: dict
    periods: int
    replic: int
    conf_sig: float
    paths: np.ndarray
    trend: np.ndarray
    initial_state: np.ndarray
    parameter_set: object
    endo_names: list

    def as_oo(self) -> dict:
        return {"cond": self.cond, "uncond": self.uncond, "controlled_exo_variables": self.controlled_exo_variables,
                "controlled_variables": self.controlled_variables_index + 1, "instruments": self.instruments,
                "graph": self.graph}

    def __getitem__(self, name):
        return getattr(self, name)


def _cf_statement(model):
    """Options of the last ``conditional_forecast`` statement."""
    last = None
    for st in getattr(model, "statements", []):
        if st.get("statementName") == "conditional_forecast":
            last = st
    return {} if last is None else (last.get("options") or {})


def _cf_paths_from_model(model):
    last = None
    for st in getattr(model, "statements", []):
        if st.get("statementName") == "conditional_forecast_paths":
            last = st
    if last is None:
        raise ValueError("conditional_forecast: no conditional_forecast_paths block (pass paths=)")
    vals = model._stmt_values.get(id(last), {})
    names, rows, L = [], [], 0
    for i, p in enumerate(last["paths"]):
        names.append(p["var"])
        row = []
        for k, v in enumerate(p["values"]):
            p1, p2 = int(v["period1"]), int(v["period2"])
            row.append((p1, p2, np.asarray(vals[(i, k)], float)))
            L = max(L, p2)
        rows.append(row)
    arr = np.full((len(names), L), np.nan)
    for i, row in enumerate(rows):
        for p1, p2, val in row:
            arr[i, p1 - 1:p2] = val
    return names, arr


def _cf_paths(model, paths):
    if paths is None:
        return _cf_paths_from_model(model)
    names = list(paths)
    L = max(len(np.atleast_1d(paths[k])) for k in names)
    arr = np.full((len(names), L), np.nan)
    for i, k in enumerate(names):
        v = np.atleast_1d(np.asarray(paths[k], float))
        arr[i, :len(v)] = v
    return names, arr


def _get_shock_paths(cL, H, mc, shocks, f0, T, R, mv, mu):
    """``conditional_forecasts.get_shock_paths`` for all draws at once. ``shocks`` is ``(draws, n_exo, H)``, ``f0``
    the initial state ``(n,)``; returns the states ``(draws, n, H + 1)`` and the controlled shocks
    ``(draws, n_controlled, cL)`` (zero where the path is NaN)."""
    nd, n = shocks.shape[0], T.shape[0]
    f = np.zeros((nd, n, H + 1))
    f[:, :, 0] = f0
    e = np.zeros((nd, mc.shape[0], cL))
    for t in range(cL):
        k = np.nonzero(np.isfinite(mc[:, t]))[0]
        s = shocks[:, :, t]
        ft = f[:, :, t]
        if k.size:
            mk = mv[k] @ R @ mu[:, k]                                    # (nk, nk)
            rhs = mc[k, t][None, :] - ft @ (mv[k] @ T).T - s @ (mv[k] @ R).T
            ek = np.linalg.solve(mk, rhs.T).T
            e[:, k, t] = ek
            f[:, :, t + 1] = ft @ T.T + (ek @ mu[:, k].T + s) @ R.T
        else:
            f[:, :, t + 1] = ft @ T.T + s @ R.T
    for t in range(cL, H):
        f[:, :, t + 1] = f[:, :, t] @ T.T + shocks[:, :, t] @ R.T
    return f, e


def conditional_forecast(model, paths=None, controlled_varexo=None, *, parameter_set=None, periods=None, replic=None,
                         conf_sig=None, rng=None, estimation=None, theta=None, mode=None, mcmc=None, data=None,
                         smoothed=None, y0=None, **smoother_options) -> ConditionalForecast:
    """Dynare's ``conditional_forecast`` (``conditional_forecasts.run``): forecasts conditional on paths for
    endogenous variables, which are reached by the shocks in ``controlled_varexo``.

    ``paths``: ``{variable: values}`` with NaN for unconstrained periods (the ``conditional_forecast_paths`` block
    when omitted; Dynare orders its variables by declaration, the i-th variable is controlled by the i-th shock of
    ``controlled_varexo``). ``parameter_set``: ``"calibration"`` (after ``stoch_simul``: starts from the steady
    state or ``histval``, or from ``y0``), or an estimated vector as for :func:`pydyn.smoother.smooth`
    (``"posterior_mode"``, ``"mle_mode"``, ``"posterior_mean"``, ``"posterior_median"``, ``"prior_mode"``,
    ``"prior_mean"`` with ``estimation=``, or ``theta=``): the forecast starts from the smoothed state of the last
    period of the data, and the observation trends are added back. Defaults from the ``conditional_forecast``
    statement of the .mod file, then Dynare's: ``periods`` 40, ``replic`` 5000, ``conf_sig`` 0.8.

    Two sets of ``replic`` simulations are drawn (shocks ``chol(Sigma_e, 'lower') * randn(n_exo, periods)``, the
    controlled shocks zeroed): the first controls the paths (``cond``, ``controlled_exo_variables``), the second
    is the unconditional forecast (``uncond``). Bands are the order statistics ``round(replic*(1-conf_sig)/2)``
    and ``round(replic*(1+conf_sig)/2)``. Refused: ``prefilter`` and OccBin models."""
    from . import smoother as _sm
    _opt.refuse_occbin(model, "conditional_forecast")
    so = _cf_statement(model)
    if parameter_set is None:
        parameter_set = so.get("parameter_set")
    if controlled_varexo is None:
        cv = so.get("controlled_varexo")
        controlled_varexo = list(cv.get("symbol_list", []) if isinstance(cv, dict) else (cv or []))
    controlled_varexo = [controlled_varexo] if isinstance(controlled_varexo, str) else list(controlled_varexo)
    periods = int(_scalar(so.get("periods", 40)) if periods is None else periods)
    replic = int(_scalar(so.get("replic", 5000)) if replic is None else replic)
    conf_sig = float(_scalar(so.get("conditional_forecast.conf_sig", 0.8)) if conf_sig is None else conf_sig)
    if parameter_set is None and theta is None and estimation is not None:
        parameter_set = "posterior_mode" if (mode is not None or getattr(estimation, "posterior_mode", None) is not None) \
            else "mle_mode"
    estimated = not (isinstance(parameter_set, str) and parameter_set == "calibration") and (
        theta is not None or parameter_set is not None)
    if parameter_set is None and theta is None:
        parameter_set = "calibration"
        estimated = False
    names, mcv = _cf_paths(model, paths)
    endo = list(model.endo_names)
    for v in names:
        if v not in endo:
            raise ValueError(f"conditional_forecast: {v} is not an endogenous variable")
    n1, n2 = len(names), len(controlled_varexo)
    if n1 != n2:
        raise ValueError("conditional_forecast: The number of constrained variables doesn't match the number of "
                         "controlled shocks")
    exo = list(model.exo_names)
    for v in controlled_varexo:
        if v not in exo:
            raise ValueError(f"conditional_forecast: Declared controlled shock {v} is not a shock in the model.")
    cL = mcv.shape[1]
    if cL > periods:
        raise ValueError(f"conditional_forecast: the paths have {cL} periods but periods={periods}")
    n = len(endo)
    if estimated:
        if smoothed is None:
            if int(_scalar(_last_option(model, "prefilter", ("estimation",), 0)) or 0) == 1 or \
                    int(smoother_options.get("prefilter", 0) or 0) == 1:
                raise ValueError("conditional_forecast: Conditional forecasting does not support the prefiltering option")
            extra = {} if theta is not None else {"parameter_set": parameter_set}
            smoothed = _sm.smooth(model, data, estimation=estimation, theta=theta, mode=mode, mcmc=mcmc, **extra,
                                  **smoother_options)
        res = smoothed
        if int(res.options.get("prefilter", 0)) == 1:
            raise ValueError("conditional_forecast: Conditional forecasting does not support the prefiltering option")
        sol = res.solution
        ys = np.asarray(res.SteadyState, float)
        G, Hr = np.asarray(res.G, float), np.asarray(res.H, float)
        Sigma = np.asarray(res.Sigma_e, float)
        gend, first_obs = res.nobs, int(res.first_obs)
        mf = np.asarray(res.obs_index, int)
        if res.options.get("noconstant", False):
            const = np.zeros((n, periods + 1))
        else:                                        # run.m: repmat(log(ys), 1, periods + 1) under loglinear
            with np.errstate(all="ignore"):
                const = np.repeat((np.log(ys) if res.loglinear else ys)[:, None], periods + 1, 1)
        t = first_obs + gend - 1 + np.arange(periods + 1)
        tc = np.asarray(res.TrendCoeffs, float)[mf]
        for k, v in enumerate(res.varobs):
            const[endo.index(v)] += tc[k] * t
        init = np.array([res.SmoothedVariables[v][-1] - res.Constant[v][-1] - res.Trend[v][-1] for v in endo])
        ov = np.asarray(sol.order_var, int)
        title = {"posterior_mode": "Posterior Mode", "posterior_mean": "Posterior Mean",
                 "posterior_median": "Posterior Median", "mle_mode": "ML Mode", "prior_mode": "Prior Mode",
                 "prior_mean": "Prior Mean"}.get(parameter_set if isinstance(parameter_set, str) else "", "Estimated")
        trend_full = const
    else:
        from .perturbation import first_order, log_mask
        sol = first_order(model)
        ys = np.asarray(sol.ys, float)                       # logs of the logged variables under loglinear
        G, Hr = np.asarray(sol.G, float), np.asarray(sol.H, float)
        Sigma = np.array(model.Sigma_e, float)
        ov = np.asarray(sol.order_var, int)
        if y0 is None and not _opt.stoch_options(model).get("loglinear", False):
            start = _initial_state_simul(model, sol.as_dr(), ys)
        elif y0 is None and not any(st_.get("statementName") == "histval" for st_ in getattr(model, "statements", [])):
            start = ys
        elif _opt.stoch_options(model).get("loglinear", False):
            if y0 is None:
                y0 = _initial_state_simul(model, sol.as_dr(), ys, raw=True)
            # run.m: y0 = zeros; y0(histval ~= 0) = log_variable(histval)(histval ~= 0): zeros stay zero
            h = np.asarray(y0, float)
            mask = log_mask(model)
            start = np.zeros_like(h)
            nz = h != 0
            start[nz & mask] = np.log(h[nz & mask])
            start[nz & ~mask] = h[nz & ~mask]
        else:
            start = np.asarray(y0, float)
        init = start - ys
        title = "Calibration"
        trend_full = np.repeat(ys[:, None], periods + 1, 1)
    T = G[np.ix_(ov, ov)]
    R = Hr[ov]
    state0 = init[ov]
    trend = trend_full[ov]
    nexo = len(exo)
    if np.allclose(Sigma, np.diag(np.diag(Sigma))):
        sQ = np.diag(np.sqrt(np.diag(Sigma)))
    else:
        from .options import warn_once
        warn_once(model, "conditional_forecast: the innovations are correlated, so the results depend on the ordering "
                         "of the innovations (a Cholesky factorization is used)", stacklevel=3)
        sQ = np.linalg.cholesky(Sigma)
    inv_ov = np.empty(n, int)
    inv_ov[ov] = np.arange(n)
    cv_dr = inv_ov[[endo.index(v) for v in names]]
    ctrl = [exo.index(v) for v in controlled_varexo]
    mv = np.zeros((n1, n))
    mu = np.zeros((nexo, n2))
    for i in range(n1):
        mv[i, cv_dr[i]] = 1.0
        mu[ctrl[i], i] = 1.0
    mc = mcv - trend[cv_dr][:, 1:1 + cL]
    rng = _rng(rng)

    def draw():
        s = np.empty((replic, nexo, periods))
        for b in range(replic):
            s[b] = sQ @ _randn(rng, (nexo, periods))
        s[:, ctrl, :] = 0.0
        return s

    f1, e1 = _get_shock_paths(cL, periods, mc, draw(), state0, T, R, mv, mu)
    f1 += trend[None]
    err = np.nanmax(np.abs(f1[:, cv_dr, 1:1 + cL] - (trend[cv_dr][:, 1:1 + cL] + mc)[None])) if np.isfinite(mc).any() else 0.0
    if err > 1e-4:
        import warnings
        warnings.warn("conditional_forecasts: controlling of variables was not successful. This can be due to numerical "
                      "imprecision (e.g. explosive simulations) or because the instrument(s) do not allow controlling "
                      "the variable(s).", stacklevel=2)
    tt = (1 - conf_sig) / 2
    t1 = max(1, _matlab_round(replic * tt))
    t2 = min(replic, _matlab_round(replic * (1 - tt)))

    def bands(f):
        srt = np.sort(f, axis=0)
        return np.stack([srt[t1 - 1], srt[t2 - 1]], axis=-2)         # (..., 2, T)

    cond = {"Mean": {}, "ci": {}}
    uncond = {"Mean": {}, "ci": {}}
    m1, b1 = f1.mean(axis=0), bands(f1)
    for i in range(n):
        nm = endo[ov[i]]
        cond["Mean"][nm] = m1[i]
        cond["ci"][nm] = b1[i]
    exo_out = {"Mean": {}, "ci": {}}
    me1, be1 = e1.mean(axis=0), bands(e1)
    for i, v in enumerate(controlled_varexo):
        exo_out["Mean"][v] = me1[i]
        exo_out["ci"][v] = be1[i]
    del f1, e1
    f2, _ = _get_shock_paths(0, periods, mc, draw(), state0, T, R, mv, mu)
    f2 += trend[None]
    m2, b2 = f2.mean(axis=0), bands(f2)
    for i in range(n):
        nm = endo[ov[i]]
        uncond["Mean"][nm] = m2[i]
        uncond["ci"][nm] = b2[i]
    return ConditionalForecast(cond=cond, uncond=uncond, controlled_exo_variables=exo_out, controlled_variables=names,
                               controlled_variables_index=cv_dr, instruments=controlled_varexo,
                               graph={"title": title, "fname": getattr(model, "fname", getattr(model, "name", ""))},
                               periods=periods, replic=replic, conf_sig=conf_sig, paths=mcv, trend=trend_full,
                               initial_state=init, parameter_set=parameter_set, endo_names=endo)


# ====================================================================== what the graphs draw
def forecast_plot_data(fc, var_list=None, *, panels_per_figure=12):
    """What ``forecasts.graph`` draws, as arrays: a list of figures ``{"name", "file", "panels"}`` (12 panels per
    figure, 3 rows by 4 columns, named ``Forecasts (I)``, ``Forecasts (2)``, ...; files ``forcst1``, ``forcst2``, ...),
    each panel ``{"variable", "x" (1..horizon), "Mean", "HPDinf", "HPDsup"}``. ``var_list`` defaults to the
    variables of the forecast."""
    names = list(fc.var_list if not var_list else var_list)
    for v in names:
        if v not in fc.Mean:
            raise ValueError(f"{v} isn't a variable of this forecast")
    figs = []
    for j, v in enumerate(names):
        if j % panels_per_figure == 0:
            k = len(figs) + 1
            figs.append({"name": "Forecasts (I)" if k == 1 else f"Forecasts ({k})", "file": f"forcst{k}", "panels": []})
        h = len(fc.Mean[v])
        figs[-1]["panels"].append({"variable": v, "x": np.arange(1, h + 1), "Mean": np.array(fc.Mean[v]),
                                   "HPDinf": np.array(fc.HPDinf[v]), "HPDsup": np.array(fc.HPDsup[v])})
    return figs


def conditional_forecast_plot_data(cf, var_list=None, periods=None):
    """What ``plot_conditional_forecast`` (``conditional_forecasts.plot``) draws, per variable: ``x`` (1..H),
    ``cond_ci`` and ``uncond_ci`` (``(2, H)``, lower and upper), ``cond_Mean`` (solid line), ``uncond_Mean`` (dashed
    line), ``base`` (the area baseline: the smallest lower bound of the two bands), the figure ``name`` and the
    ``file`` name. ``periods`` counts the forecast periods (the initial period is added, and the horizon is capped)."""
    names = list(var_list) if var_list else list(cf.cond["Mean"])
    for v in names:
        if v not in cf.cond["Mean"]:
            raise ValueError(f"{v} isn't an endogenous variable")
    total = len(cf.cond["Mean"][names[0]])
    if periods is None:
        use = total
    else:
        use = int(periods) + 1
        if use > total:
            import warnings
            warnings.warn(f"conditional_forecasts.plot:: Number of periods for plotting exceeds forecast horizon. "
                          f"Setting periods to {total - 1}.", stacklevel=2)
            use = total
    title = cf.graph["title"]
    out = []
    for v in names:
        c1, c2 = np.array(cf.cond["ci"][v])[:, :use], np.array(cf.uncond["ci"][v])[:, :use]
        out.append({"variable": v, "name": f"Conditional forecast ({title}): {v}.", "x": np.arange(1, use + 1),
                    "cond_Mean": np.array(cf.cond["Mean"][v])[:use], "uncond_Mean": np.array(cf.uncond["Mean"][v])[:use],
                    "cond_ci": c1, "uncond_ci": c2, "base": float(min(c1[0].min(), c2[0].min())),
                    "file": "Conditional_forecast_" + title.strip().replace(" ", "_") + "_" + v})
    return out


def plot_forecast(fc, var_list=None, *, save_dir=None, show=False):
    """Draws :func:`forecast_plot_data` with matplotlib (mean and band as blue lines, as ``forecasts.graph``);
    returns the figures, saving ``<file>.png`` into ``save_dir`` when given."""
    import matplotlib
    if not show:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    figs = []
    for fig_data in forecast_plot_data(fc, var_list):
        fig = plt.figure(fig_data["name"])
        for m, p in enumerate(fig_data["panels"]):
            ax = fig.add_subplot(3, 4, m + 1)
            for key in ("Mean", "HPDinf", "HPDsup"):
                ax.plot(p["x"], p[key], "b-")
            ax.set_title(p["variable"])
            ax.set_xlim(1, len(p["x"]))
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, fig_data["file"] + ".png"))
        figs.append(fig)
    return figs


def plot_conditional_forecast(cf, var_list=None, periods=None, *, save_dir=None, show=False):
    """Draws :func:`conditional_forecast_plot_data` with matplotlib (conditional band in grey, conditional mean solid,
    unconditional mean and band dashed, as ``conditional_forecasts.plot``); returns the figures."""
    import matplotlib
    if not show:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    figs = []
    for d in conditional_forecast_plot_data(cf, var_list, periods):
        fig = plt.figure(d["name"])
        ax = fig.add_subplot(1, 1, 1)
        x = d["x"]
        ax.fill_between(x, d["cond_ci"][0], d["cond_ci"][1], color=(.9, .9, .9))
        ax.plot(x, d["cond_Mean"], "-k", linewidth=3)
        ax.plot(x, d["uncond_Mean"], "--k", linewidth=3)
        ax.plot(x, d["uncond_ci"][0], "--k", linewidth=1)
        ax.plot(x, d["uncond_ci"][1], "--k", linewidth=1)
        ax.autoscale(tight=True)
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, d["file"] + ".png"))
        figs.append(fig)
    return figs


# ====================================================================== the statements of a .mod file
def run_forecasts(model, *, estimation=None, theta=None, mode=None, mcmc=None, data=None, rng=None, smoothed=None,
                  **smoother_options):
    """Runs the ``forecast`` and ``conditional_forecast`` statements of the .mod file in order, as Dynare would, and
    returns ``{"forecast": [...], "estimation_forecast": [...], "conditional_forecast": [...]}`` (lists of
    :class:`Forecast` / :class:`ConditionalForecast`, in statement order).

    ``options_`` accumulate through the file as in Dynare: each statement sees the options (``periods``,
    ``conf_sig``) set by the statements before it. ``forecast`` needs a previous ``stoch_simul``; an
    ``estimation(forecast=N)`` statement is run at ``estimation`` / ``theta`` (or ``mode``, ``mcmc``);
    ``conditional_forecast`` uses the last ``conditional_forecast_paths`` before it. ``rng`` is shared by the
    statements that draw numbers. Shocks / varobs blocks after a forecast statement do not change the forecast
    already computed (Dynare would use them for the next statement; pydyn evaluates the blocks at load time, so a
    warning is given)."""
    import copy
    rng = _rng(rng)
    stmts = list(getattr(model, "statements", []))
    out = {"forecast": [], "estimation_forecast": [], "conditional_forecast": []}
    seen_stoch = False
    last_paths = None
    for i, st in enumerate(stmts):
        kind = st.get("statementName")
        if kind == "stoch_simul":
            seen_stoch = True
        elif kind == "conditional_forecast_paths":
            last_paths = st
        elif kind in ("forecast", "estimation", "conditional_forecast"):
            later = [s.get("statementName") for s in stmts[i + 1:]]
            if kind == "estimation" and not (st.get("options") or {}).get("forecast"):
                continue
            if any(k in ("shocks", "mshocks", "varobs") for k in later):
                from .options import warn_once
                warn_once(model, f"run_forecasts: a shocks/mshocks/varobs block follows the {kind} statement; pydyn "
                                 "applies it at load time, so the forecast may differ from Dynare's", stacklevel=2)
            proxy = copy.copy(model)
            proxy.statements = stmts[:i + 1]
            if kind == "forecast":
                if not seen_stoch:
                    raise ValueError("forecast: you need to call stoch_simul before forecast")
                out["forecast"].append(forecast(proxy, st.get("symbol_list") or None, rng=rng))
            elif kind == "estimation":
                out["estimation_forecast"].append(forecast_at(proxy, estimation, theta, mode=mode, mcmc=mcmc, data=data,
                                                              smoothed=smoothed, **smoother_options))
            else:
                if last_paths is None:
                    raise ValueError("conditional_forecast: no conditional_forecast_paths block before the statement")
                out["conditional_forecast"].append(
                    conditional_forecast(proxy, estimation=estimation, theta=theta, mode=mode, mcmc=mcmc, data=data,
                                         smoothed=smoothed, rng=rng, **smoother_options))
    return out
