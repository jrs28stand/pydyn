# This file is part of pydyn. It follows the OccBin branches of matlab/estimation/dsge_likelihood.m,
# matlab/+occbin/IVF_posterior.m, matlab/estimation/dynare_estimation_init.m and
# matlab/estimation/initial_estimation_checks.m from Dynare 7.2 (Copyright (C) Dynare Team,
# GPL-3.0-or-later), modified in 2026.
# Released under the GNU General Public License, version 3 or later; see LICENSE. No warranty.
"""Estimation of models with occasionally binding constraints (OccBin), after Dynare 7.2.

:class:`OccbinEstimation` is :class:`pydyn.estimation.Estimation` (data, priors, bounds, parameter
mapping, posterior mode, Hessian, Laplace approximation, Metropolis-Hastings) with the likelihood
replaced by the piecewise Kalman filter (Dynare's default for a model with ``occbin_constraints``)
or by the inversion filter (``occbin_setup(likelihood_inversion_filter)``).
"""
from __future__ import annotations

import numpy as np

from . import kalman
from . import occbin_filter as _of
from .estimation import Estimation, EstimationError, _Rejected
from .occbin import OBC_SELECTIONS, Occbin, OccbinError, canonical_obc
from .perturbation import BlanchardKahnError
from .steady import SteadyStateError


_LOGLINEAR_REFUSAL = (
    "loglinear is not supported with OccBin: Dynare 7.2 accepts it but gives incoherent results. +occbin builds the regime "
                    "matrices from the level-model Jacobians evaluated at endo_steady_state (get_deriv.m), which under loglinear is the "
                    "LOGGED steady state, while the base-regime rule oo_.dr is log-linearised. Octave 8.4 run, Dynare 7.2, borrcon model "
                    "with stoch_simul(loglinear) then occbin_solver: a surprise shock of 0.02 gives error_flag 313 where the level model "
                    "solves; with a shock of -0.5 the simulation 'solves' but returns a mixture of level and log quantities (the multiplier "
                    "lb jumps to 85.3 against 0.70 in levels, 0.0082 steady state in levels but -4.81 logged)")


class OccbinEstimation(Estimation):
    """Bayesian or maximum-likelihood estimation of an OccBin model.

    Parameters are those of :class:`pydyn.estimation.Estimation`, plus ``occbin``: a dict of
    OccBin options overriding the .mod file's ``occbin_setup`` (keys like ``'likelihood_maxit'`` or
    ``'likelihood.maxit'``; ``'likelihood_inversion_filter': True`` selects the inversion filter).
    The filter is the piecewise Kalman filter unless ``likelihood_inversion_filter`` is set, as in
    Dynare (``options_.occbin.likelihood.status`` is true for every model with ``occbin_constraints``).

    ``obc_method`` (``'occbin'``, default, or ``'news'``), ``obc_horizon`` (news horizon, default 40) and ``obc_selection``
    (``'occbin'`` default, ``'shortest'``, ``'longest'``, ``'min_wedge'``: which solution the news solver returns at a state
    that has several, one constraint only) are as in :meth:`pydyn.Occbin.simulate`; they are read from the
    ``options_.occbin.obc_*`` lines of the .mod file unless given, can be reassigned (``est.obc_method = 'news'``), and
    unknown values are an error.  ``'news'`` replaces OccBin's solver inside the inversion filter
    (:mod:`pydyn.obc_filter`); the piecewise Kalman filter has no news version and refuses it.
    ``obc_milp_time_limit`` (attribute) is the time the news solver may spend on an integer programme at a trial point that
    its cheap checks did not decide (default 0: none; the trial point has failed, as in OccBin).

    Not supported (raise): ``loglinear``, the diffuse filter (``lik_init = 3``), univariate filters
    (``kalman_algo = 2, 4``), particle filtering options.
    """

    handles_occbin = True

    # ``obc_method`` / ``obc_horizon`` / ``obc_selection`` are options of this class, as of ``Occbin.simulate`` and of
    # the native mod-file lines ``options_.occbin.obc_method = 'news';`` etc.  They can be given to the constructor,
    # read from those lines, and changed later by assignment (``est.obc_method = 'news'``); unknown values are an error.
    @property
    def obc_method(self):
        return self._obc_method

    @obc_method.setter
    def obc_method(self, value):
        value = _check_obc_method(value)
        if value == "minnorm":                       # JDyn's name for news with obc_selection='min_wedge'
            self._obc_method, self._obc_selection = canonical_obc(value, getattr(self, "_obc_selection", "occbin"))
        else:
            self._obc_method = value
        if self._obc_method == "news" and getattr(self, "inversion_filter", True) is False:
            raise EstimationError(_PKF_NEWS_MSG)

    @property
    def obc_horizon(self):
        return self._obc_horizon

    @obc_horizon.setter
    def obc_horizon(self, value):
        self._obc_horizon = _check_obc_horizon(value)

    @property
    def obc_selection(self):
        return self._obc_selection

    @obc_selection.setter
    def obc_selection(self, value):
        self._obc_selection = _check_obc_selection(value)

    #: seconds the news solver's integer programme may use at one trial point of the inversion filter (None: its default)
    obc_milp_time_limit = None

    def __init__(self, model, data=None, occbin=None, obc_method=None, obc_horizon=None, obc_selection=None, **options):
        if not getattr(model, "occbin_constraints", None):
            raise ValueError("the model has no occbin_constraints block: use pydyn.Estimation")
        native = _native_obc_options(model)          # options_.occbin.obc_* lines of the .mod file
        self._obc_method, self._obc_horizon, self._obc_selection = "occbin", 40, "occbin"
        method = native.get("obc_method", "occbin") if obc_method is None else obc_method
        selection = native.get("obc_selection", "occbin") if obc_selection is None else obc_selection
        _check_obc_method(method)
        _check_obc_selection(selection)
        method, selection = canonical_obc(method, selection)       # minnorm = news + min_wedge
        self.obc_method = method
        self.obc_horizon = native.get("obc_horizon", 40) if obc_horizon is None else obc_horizon
        self.obc_selection = selection
        # Estimation's data, prior and parameter handling is reused; the likelihood and smoother are
        # replaced below (handles_occbin tells Estimation not to refuse the model).
        super().__init__(model, data, **options)
        o = self.options
        if o["loglinear"]:
            raise NotImplementedError(_LOGLINEAR_REFUSAL)
        if o["lik_init"] == 3:
            raise NotImplementedError("the diffuse filter (lik_init = 3) is not available with OccBin")
        if o["kalman_algo"] in (2, 4) and not self._ivf_requested(occbin):
            raise EstimationError("the piecewise Kalman filter is incompatible with univariate filters "
                                  "(kalman_algo = 2, 4)")
        self.occbin_options = _of.occbin_options(model, occbin, "OccbinEstimation")
        ob = self.occbin_options
        for k in ("filter.particle.status", "filter.init_periods_using_particles"):
            if ob.get(k):
                raise NotImplementedError("particle filtering for OccBin models is not implemented")
        self.inversion_filter = bool(ob["likelihood.inversion_filter"])
        S = np.array(model.Sigma_e, float)
        if self.inversion_filter:
            mapping = ob["likelihood.IVF_shock_observable_mapping"]
            if mapping is None:
                mapping = np.flatnonzero(np.diag(S) != 0)
            else:
                mapping = np.array([model._ix[x] if isinstance(x, str) else int(x) for x in mapping])
                if np.any(np.diag(S)[mapping] == 0):
                    raise EstimationError("IVF-filter: an observable is mapped to a zero variance shock.")
            if np.any(self._H0 != 0) or self._groups["vn"]:
                raise EstimationError("IVF-filter: Measurement errors are not allowed with the inversion filter.")
            if len(mapping) != len(self.varobs):
                raise EstimationError("The IVF requires exactly as many shocks as observables.")
            self.ivf_mapping = np.asarray(mapping, int)
        self.last_filter = "inversion filter" if self.inversion_filter else "piecewise Kalman filter"
        self.last_result = None
        if self._obc_method == "news" and not self.inversion_filter:
            raise EstimationError(_PKF_NEWS_MSG)

    @staticmethod
    def _ivf_requested(occbin):
        return bool(occbin and (occbin.get("likelihood_inversion_filter") or occbin.get("likelihood.inversion_filter")))

    # ------------------------------------------------------------------ model at theta
    def occbin_at(self, theta) -> Occbin:
        """The piecewise-linear model at ``theta`` (raises ``_Rejected`` with Dynare's codes)."""
        try:
            ob = Occbin(self.model, self.model_params(np.asarray(theta, float)))
        except SteadyStateError as exc:
            raise _Rejected(f"steady state: {exc}", 20) from None
        except BlanchardKahnError as exc:
            code, pen = self._bk_penalty(theta)
            raise _Rejected(f"solution: {exc}", code, pen) from None
        if not (np.all(np.isfinite(ob.P)) and np.all(np.isfinite(ob.Q))):
            raise _Rejected("the decision rule contains NaN or Inf", 6)
        return ob

    def occbin_forecast(self, theta, horizon, endo_histval=None, **kw):
        """``occbin.forecast`` at ``theta``: the OccBin forecast (:func:`pydyn.occbin_forecast.occbin_forecast`)
        from the state ``endo_histval`` (levels, declaration order, e.g. the smoothed state of the last period
        plus the steady state; default: the steady state). The covariance matrix of the shocks is the one at
        ``theta``; ``replic``, ``qmc``, ``rng``, ``shocks``, ``frcst_regimes``, ``maxit``, ``check_ahead_periods``
        and ``conf_sig`` are passed on."""
        from .occbin_forecast import occbin_forecast
        theta = np.asarray(theta, float)
        ob = self.occbin_at(theta)
        kw.setdefault("sigma_e", self.covariances(theta)[0])
        return occbin_forecast(ob, horizon, endo_histval, **kw)

    def _context(self, theta):
        theta = np.asarray(theta, float)
        Q, Hm = self.covariances(theta)
        ob = self.occbin_at(theta)
        r = self._restrict
        mm = len(r)
        Z = np.zeros((len(self.varobs), mm))
        Z[np.arange(len(self.varobs)), self._mf] = 1.0
        ctx = _of._PKFContext(ob, r, Z, Hm, Q, dict(self.occbin_options,
                              rescale_prediction_error_covariance=self.options["rescale_prediction_error_covariance"]))
        return ctx, Q, Hm

    def _observations(self, ob):
        """Data minus the steady state (and trends) of the observed variables, (pp x T)."""
        Y = self.data.T - ob.ys[self._obs_decl][:, None]
        trend = self._trend(ob.params, self.nobs)
        if trend is not None:
            Y = Y - trend
        return Y

    def _initial_state(self, ctx):
        o = self.options
        mm = ctx.mm
        if o["lik_init"] == 1:
            P = kalman.lyapunov_symm(ctx.T0, ctx.R0 @ ctx.Q @ ctx.R0.T, o["qz_criterium"], o["lyapunov_complex_threshold"])
        elif o["lik_init"] == 2:
            P = o["Harvey_scale_factor"] * np.eye(mm)
            if not int(self.occbin_options["likelihood.first_period_occbin_update"]) == 1:
                if o["Harvey_scale_factor"] == 0:
                    P = ctx.T0 @ P @ ctx.T0.T + ctx.R0 @ ctx.Q @ ctx.R0.T
        else:
            raise NotImplementedError(f"lik_init={o['lik_init']} is not available with OccBin")
        return np.zeros(mm), P

    # ------------------------------------------------------------------ filters
    def run_filter(self, theta):
        """Run the OccBin filter at ``theta`` and return its result: an
        :class:`pydyn.occbin_filter.InversionFilterResult` or :class:`~pydyn.occbin_filter.PKFResult`
        (with ``.occbin`` the model at ``theta``). Raises ``_Rejected`` when the filter fails."""
        theta = np.asarray(theta, float)
        Q, Hm = self.covariances(theta)
        if self._groups["cx"] and not _ispd(Q[np.ix_(self._Q_check, self._Q_check)]):
            raise _Rejected("covariance matrix of the structural shocks is not positive definite (info 43)", 43)
        if self.inversion_filter:
            ob = self.occbin_at(theta)
            solver = self._news_solver(ob)
            ivf = _of.inversion_filter(ob, self.data, self._obs_decl, self.ivf_mapping, self.occbin_options,
                                       solver=solver)
            ivf.occbin = ob
            if ivf.error_code:
                raise _Rejected(f"inversion filter failed in period {ivf.error_period} (info {ivf.error_code})",
                                ivf.error_code, 1000.0)
            try:
                like, lik = _of.inversion_filter_loglik(ivf, self.data, self._obs_decl, Q, self.ivf_mapping,
                                                        int(self.options["presample"]))
            except ValueError as exc:
                raise _Rejected(f"inversion filter likelihood (info {exc.args[0]})", exc.args[0], 1000.0) from None
            ivf.LIK, ivf.lik = like, lik
            return ivf
        if self._obc_method != "occbin":
            raise EstimationError(_PKF_NEWS_MSG)
        ctx, Q, Hm = self._context(theta)
        a, P = self._initial_state(ctx)
        Y = self._observations(ctx.ob)
        res = _of.piecewise_kalman_filter(ctx, Y, a, P, self._data_index, int(self.options["presample"]),
                                          self.options["kalman_tol"])
        res.occbin, res.context = ctx.ob, ctx
        if res.error_code:
            raise _Rejected(f"piecewise Kalman filter failed in period {res.error_period} "
                            f"(occbin error {res.error_code}; info 320)", 320, 0.1)
        if not np.isfinite(res.LIK):
            raise _Rejected("the likelihood is NaN or Inf (info 320)", 320, 0.1)
        return res

    def _news_solver(self, ob):
        """The filter's solver at ``ob``: None (OccBin's own) or the news-shock solver."""
        if self._obc_method == "occbin":
            if self._obc_selection != "occbin":
                raise EstimationError("obc_selection other than 'occbin' needs obc_method='news'")
            return None
        from .obc_filter import NewsFilterSolver
        self.last_news_solver = NewsFilterSolver(ob, horizon=self._obc_horizon, selection=self._obc_selection,
                                                 milp_time_limit=self.obc_milp_time_limit)
        return self.last_news_solver

    def _minus_loglik(self, theta, return_lik=False):
        try:
            with np.errstate(all="ignore"):
                res = self.run_filter(theta)
        except OccbinError as exc:
            raise _Rejected(f"OccBin solver: {exc}", 320, 0.1) from None
        self.last_result = res
        if return_lik:
            return res.LIK, np.asarray(res.lik)[int(self.options["presample"]):]
        return res.LIK


_PKF_NEWS_MSG = ("obc_method='news' is implemented for the inversion filter only (occbin_setup(likelihood_inversion_filter)); "
                 "the piecewise Kalman filter needs OccBin's regime matrices. Use obc_method='occbin' or the inversion filter.")


def _check_obc_method(v):
    if not isinstance(v, str) or v not in ("occbin", "news", "minnorm"):
        raise ValueError(f"unknown obc_method {v!r}: use 'occbin', 'news' or 'minnorm'")
    return v


def _check_obc_selection(v):
    if not isinstance(v, str) or v not in OBC_SELECTIONS:
        raise ValueError(f"unknown obc_selection {v!r}: use one of {', '.join(map(repr, OBC_SELECTIONS))}")
    return v


def _check_obc_horizon(v):
    try:
        h = int(np.ravel(v)[0])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"obc_horizon must be a positive integer, not {v!r}") from None
    if h < 1 or h != np.ravel(v)[0]:
        raise ValueError(f"obc_horizon must be a positive integer, not {v!r}")
    return h


def _native_obc_options(model) -> dict:
    """``options_.occbin.obc_method / obc_horizon / obc_selection`` lines of the .mod file (as ``Occbin.mod_options``)."""
    mo = getattr(model, "matlab_options", None) or {}
    occ = mo.get("occbin", {}) if hasattr(mo, "get") else {}
    out = {}
    if hasattr(occ, "get"):
        for k in ("obc_method", "obc_horizon", "obc_selection"):
            if occ.get(k) is not None:
                out[k] = occ[k] if k == "obc_horizon" else str(occ[k])
    return out


def _ispd(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def _occbin_smoother_method(self, theta, inversion_filter=None):
    """OccBin smoother at ``theta`` (Dynare's ``calib_smoother``/``estimation(smoother)`` with
    OccBin): the piecewise Kalman smoother (``occbin.DSGE_smoother``), or the inversion filter when
    ``smoother_inversion_filter`` is set (or ``inversion_filter=True``). Returns
    :class:`pydyn.occbin_filter.OccbinSmootherResult` (deviations from the steady state in
    declaration order; ``.ys`` the steady state)."""
    theta = np.asarray(theta, float)
    o = self.options
    ivf = self.occbin_options["smoother.inversion_filter"] if inversion_filter is None else inversion_filter
    Q, Hm = self.covariances(theta)
    ob = self.occbin_at(theta)
    if ivf:
        mapping = getattr(self, "ivf_mapping", None)
        if mapping is None:
            mapping = np.flatnonzero(np.diag(np.array(self.model.Sigma_e, float)) != 0)
        res = _of.inversion_filter(ob, self.data, self._obs_decl, mapping, self.occbin_options,
                                   solver=self._news_solver(ob))
        return _of.inversion_filter_smoother(res, ob.ys, mapping, ob.n_exo)
    if self._obc_method != "occbin":
        raise EstimationError(_PKF_NEWS_MSG)
    Y = self._observations(ob)
    return _of.occbin_smoother(ob, Y, Q, Hm, self._obs_decl, self.occbin_options, self._data_index,
                               lik_init=int(o["lik_init"]), qz_criterium=o["qz_criterium"],
                               Harvey_scale_factor=o["Harvey_scale_factor"], kalman_tol=o["kalman_tol"])


def _occbin_shock_decomposition_method(self, theta, smoother_result=None, tinit=1):
    """OccBin shock decomposition at ``theta`` (occbin.shock_decomposition): see
    :func:`pydyn.occbin_filter.occbin_shock_decomposition`."""
    sm = self.occbin_smoother(theta) if smoother_result is None else smoother_result
    ob = self.occbin_at(theta)
    m = self.model
    return _of.occbin_shock_decomposition(sm, ob.P, ob.Q, m.endo_names, m.exo_names, tinit=tinit)


OccbinEstimation.occbin_smoother = _occbin_smoother_method
OccbinEstimation.smoother = _occbin_smoother_method
OccbinEstimation.occbin_shock_decomposition = _occbin_shock_decomposition_method
