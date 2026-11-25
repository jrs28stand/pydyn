"""OccBin forecasts: Dynare 7.2's ``occbin.forecast`` (``+occbin/forecast.m``).

The forecast starts from ``M_.endo_histval(:, 1)`` (levels, declaration order), uses the base-case
shocks ``options_.occbin.forecast.SHOCKS0`` (zero by default) and, with ``replic > 0``, adds Gaussian
draws with the covariance matrix of the shocks (plain, or a Sobol sequence with ``qmc``); the
statistics of the simulated paths (mean, median, variance, HPD interval, deciles, min, max) are
Dynare's ``posterior_moments``. The random numbers come from ``rng`` (:class:`pydyn.octave_rng.OctaveRandom`
reproduces Octave's ``randn``; ``pydyn.dynare()`` passes its own stream).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from . import posterior as _post
from .occbin import Occbin, OccbinError
from .sobol import qmc_sequence

__all__ = ["OccbinForecast", "occbin_forecast", "STATISTICS"]

STATISTICS = ("Mean", "Median", "Var", "HPDinf", "HPDsup", "Deciles", "Min", "Max")


@dataclass
class OccbinForecast:
    """Dynare's ``forecast`` and ``yf``: ``linear`` and ``piecewise`` map a statistic (``Mean``, and with
    ``replic > 0`` also ``Median``, ``Var``, ``HPDinf``, ``HPDsup``, ``Deciles`` (horizon x 9), ``Min``,
    ``Max``) to a dict endogenous variable name -> array over the horizon."""
    linear: dict = field(default_factory=dict)
    piecewise: dict = field(default_factory=dict)
    regimes: object = None            # ``regime_history`` (replic = 0), or one list per successful replication
    error_flag: int = 0               # replic = 0: the solver's error flag; replic > 0: number of failed replications
    yf: np.ndarray = None             # (horizon + 1, n_endo) levels: the initial point, then the piecewise forecast
    replic: int = 0                   # replications requested (after the Sobol rounding)
    successful: np.ndarray = None     # replic > 0: indices (0-based) of the replications that converged
    paths: dict = None                # replic > 0: {"linear", "piecewise"} arrays (horizon, n_endo, successful)
    shocks: np.ndarray = None         # replic > 0: the shocks of every replication (horizon, n_shocks, replic)

    @property
    def ok(self) -> bool:
        return self.yf is not None


def _round_half_away(x: float) -> int:
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def _base_shocks(ob: Occbin, shocks, horizon: int) -> np.ndarray:
    base = np.zeros((horizon, ob.n_exo))
    if shocks is None or (hasattr(shocks, "__len__") and len(shocks) == 0):
        return base
    if isinstance(shocks, dict):
        shocks = list(shocks.items())
    if isinstance(shocks, (list, tuple)) and shocks and isinstance(shocks[0], (list, tuple)) \
            and len(shocks[0]) == 2 and isinstance(shocks[0][0], str):
        periods = len(shocks[0][1])
        for name, values in shocks:
            if name not in ob.model._ix and name not in ob.model.exo_names:
                raise ValueError(f"occbin.forecast: unknown exogenous shock {name}")
            if len(values) != periods:
                raise ValueError(f"occbin.forecast: unknown exogenous shock {name}")
            base[:, ob.model.exo_names.index(name)] = np.asarray(values, float)
        return base
    mat = np.atleast_2d(np.asarray(shocks, float))
    if mat.shape[0] > ob.n_exo:
        raise ValueError("occbin.forecast: SHOCKS0 has more rows than the model has shocks")
    base[:, :mat.shape[0]] = mat.T
    return base


def occbin_forecast(ob: Occbin, horizon: int, endo_histval=None, *, shocks=None, frcst_regimes=None,
                    replic: int = 0, qmc: bool = False, maxit: int = 30, check_ahead_periods: int = 30,
                    conf_sig: float = 0.9, sigma_e=None, rng=None, simul_options: dict | None = None) -> OccbinForecast:
    """``[forecast, error_flag, yf] = occbin.forecast(options_, M_, dr, ys, ..., forecast_horizon)``.

    ``endo_histval``: the initial state in levels, declaration order (``M_.endo_histval(:,1)``; default: the
    steady state). ``shocks`` (``options_.occbin.forecast.SHOCKS0``): a (n_shocks x horizon) matrix whose rows
    are the first shocks, or a dict / list of (name, values). ``frcst_regimes``: initial regime guess (``init_regime``
    format of :meth:`Occbin.simulate`). ``conf_sig`` is ``options_.forecasts.conf_sig``; ``sigma_e`` defaults to the
    model's covariance matrix of the shocks. ``simul_options``: the other ``options_.occbin.simul`` entries
    (``Occbin.mod_options()`` by default)."""
    horizon = int(horizon)
    m = ob.model
    ys = np.asarray(ob.ys, float)
    y0 = ys.copy() if endo_histval is None else np.asarray(endo_histval, float).ravel()
    sim = dict(ob.mod_options() if simul_options is None else simul_options)
    sim.update(maxit=int(maxit), check_ahead_periods=int(check_ahead_periods), periods=horizon)
    init = y0 - ys
    shocks_base = _base_shocks(ob, shocks, horizon)
    init_regime = None if frcst_regimes is None or len(frcst_regimes) == 0 else [list(r) for r in frcst_regimes]
    names = list(m.endo_names)
    res = OccbinForecast()
    if replic:
        S = np.asarray(m.Sigma_e if sigma_e is None else sigma_e, float)
        ishock = np.flatnonzero(np.sqrt(np.diag(S)))
        neff = len(ishock)
        U, s, _ = sla.svd(S[np.ix_(ishock, ishock)], lapack_driver="gesvd")
        replic = int(replic)
        if qmc:
            replic = 2 ** _round_half_away(np.log2(replic + 1)) - 1
            add, _ = qmc_sequence(horizon * neff, 1, 1, replic)
        else:
            if rng is None:
                raise ValueError("occbin.forecast: a random number generator (rng) is needed for replic > 0")
            add = np.asarray(rng.randn((horizon * neff, replic)), float)
        add = add.reshape((neff, horizon, replic), order="F")
        lin = np.full((horizon, ob.n, replic), np.nan)
        pw = np.full((horizon, ob.n, replic), np.nan)
        flags = np.ones(replic, bool)
        regs = [None] * replic
        sh_all = np.full((horizon, neff, replic), np.nan)
        L = U @ np.diag(np.sqrt(s))
        for it in range(replic):
            sh = shocks_base[:, ishock] + (L @ add[:, :, it]).T
            try:
                out = ob.simulate(sh, init=init, exo_pos=ishock, init_regime=init_regime, **sim)
            except OccbinError:
                continue
            flags[it] = False
            lin[:, :, it], pw[:, :, it], regs[it] = out.linear, out.piecewise, out.regime_history
            sh_all[:, :, it] = sh
        ok = np.flatnonzero(~flags)
        res.replic, res.successful = replic, ok
        res.error_flag = int(flags.sum())
        lin, pw = lin[:, :, ok], pw[:, :, ok]
        res.paths = {"linear": lin, "piecewise": pw}
        res.shocks = sh_all[:, :, ok]
        res.regimes = regs
        for fname, z in (("linear", lin), ("piecewise", pw)):
            d = {k: {} for k in STATISTICS}
            for i, nm in enumerate(names):
                mom = [_post.posterior_moments(z[j, i, :], conf_sig, density=False) for j in range(horizon)]
                d["Mean"][nm] = np.array([r["mean"] for r in mom])
                d["Median"][nm] = np.array([r["median"] for r in mom])
                d["Var"][nm] = np.array([r["variance"] for r in mom])
                d["HPDinf"][nm] = np.array([r["hpd"][0] for r in mom])
                d["HPDsup"][nm] = np.array([r["hpd"][1] for r in mom])
                d["Deciles"][nm] = np.array([r["deciles"] for r in mom])
                d["Min"][nm] = z[:, i, :].min(axis=1)
                d["Max"][nm] = z[:, i, :].max(axis=1)
            setattr(res, fname, d)
        # Dynare's last simulation (out) gives yf: with replications it is the last one run, whether or not it converged
        res.yf = np.vstack([y0[None, :], pw[:, :, -1] if len(ok) else np.full((horizon, ob.n), np.nan)])
        return res
    try:
        out = ob.simulate(shocks_base, init=init, init_regime=init_regime, **sim)
    except OccbinError as exc:
        res.error_flag = int(exc.error_flag) if exc.error_flag is not None else 1
        return res
    res.error_flag = int(out.error_flag)
    res.regimes = out.regime_history
    res.linear = {"Mean": {nm: out.linear[:, i] for i, nm in enumerate(names)}}
    res.piecewise = {"Mean": {nm: out.piecewise[:, i] for i, nm in enumerate(names)}}
    res.yf = np.vstack([y0[None, :], out.piecewise])
    return res
