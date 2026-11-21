# This file is part of pydyn.
#
# It is a Python port of, and follows, matlab/estimation/{prior_posterior_statistics, prior_posterior_statistics_core, pm3,
# posterior_analysis, prior_analysis, prior_sampler, dsge_simulated_theoretical_covariance,
# dsge_simulated_theoretical_correlation, dsge_simulated_theoretical_variance_decomposition,
# dsge_simulated_theoretical_conditional_variance_decomposition, covariance_mc_analysis, correlation_mc_analysis,
# variance_decomposition_mc_analysis, variance_decomposition_ME_mc_analysis, conditional_variance_decomposition_mc_analysis,
# conditional_variance_decomposition_ME_mc_analysis, check_list_of_variables, check_measurement_error_requested_vars}.m,
# matlab/moments/compute_moments_varendo.m and matlab/cli/prior.m of Dynare 7.2 (Copyright (C) 2003-2026 Dynare Team,
# https://www.dynare.org; GPL-3.0-or-later), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Posterior (and prior) distributions of statistics of the endogenous variables.

Two families of results, both over parameter draws (the sub-draws that Dynare takes from the
Metropolis-Hastings chains: :func:`pydyn.posterior.posterior_subsample`), with the statistics of
``posterior_moments.m`` for every series (mean, median, variance, shortest ``mh_conf_sig`` interval
(HPD), deciles and Gaussian kernel density on 512 points).

``moments_varendo`` (:func:`posterior_theoretical_moments`, Dynare's ``oo_.PosteriorTheoreticalMoments.dsge``)
    theoretical moments of the variables of a list at each draw: variances and covariances, contemporaneous
    correlations (``contemporaneous_correlation``), autocorrelations of order 1 to ``ar``, the unconditional
    variance decomposition (``nodecomposition`` skips it), the conditional variance decomposition
    (``conditional_variance_decomposition``), and, for observed variables when there are measurement
    errors, both decompositions with the measurement error as an additional shock (``VarianceDecompositionME``,
    ``ConditionalVarianceDecompositionME``). Variable list: the one after the ``estimation`` command, else
    the one selected by ``consider_all_endogenous`` (the original endogenous variables),
    ``consider_all_endogenous_and_auxiliary`` or ``consider_only_observed``; without any of them Dynare asks
    interactively and ``nointeractive`` makes it take all original endogenous variables: so does pydyn.

``smoother``, ``filtered_vars``, ``filter_step_ahead`` (:func:`posterior_smoother_statistics`, Dynare's
``oo_.SmoothedVariables``, ``UpdatedVariables``, ``SmoothedShocks``, ``SmoothedMeasurementErrors``, ``Constant``,
``Smoother.Trend`` / ``TrendCoeff`` / ``Init_State``, ``FilteredVariables``, ``Filtered_Variables_<k>_step_ahead``)
    the Kalman smoother (``pydyn.smoother.smooth``) at each draw and the statistics, period by period, of the
    smoothed, updated and filtered series.

Prior counterpart: :func:`prior_theoretical_moments` and :func:`prior_moments_distribution`; see their
docstrings for what Dynare 7.2 itself makes reachable.

Notes on Dynare 7.2 that this module reproduces on purpose (``replicate_dynare_bugs`` of the respective
functions switches them off where it matters):

* ``oo_.FilteredVariables`` is built by ``pm3`` from the *first listed* ``filter_step_ahead`` element shifted
  by one period (it is only the one-step-ahead forecast when that element is 1).
* ``oo_.FilteredVariablesKStepAheadVariances`` holds the *means* (the code copies ``Mean_filter_step_ahead``).
* with ``nodecomposition`` and ``conditional_variance_decomposition`` and a measurement error on a listed observed
  variable, Dynare 7.2 stops with an undefined variable (``observable_name_requested_vars``); pydyn computes the
  result the other branches define.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import moments as _mom
from . import posterior as _post
from .perturbation import first_order

__all__ = ["MomentFamily", "PosteriorTheoreticalMoments", "posterior_theoretical_moments", "theoretical_moments_at",
           "SeriesDistribution", "PosteriorSmootherStatistics", "posterior_smoother_statistics", "per_draw_series",
           "moment_variable_list", "prior_theoretical_moments", "prior_moments_distribution", "prior_draws"]

#: Dynare's options_.estimation.moments_posterior_density defaults
KERNEL = {"gridpoints": 2 ** 9, "bandwidth": 0, "kernel_function": "gaussian"}


# ====================================================================== statistics of series of draws
def _stats(x, mh_conf_sig, density=True, kernel_options=None) -> dict:
    """``posterior_moments.m`` for every series of ``x`` (draws on axis 0, any trailing shape).

    Returns ``mean``, ``median``, ``variance``, ``hpd_inf``, ``hpd_sup`` (trailing shape), ``deciles``
    (trailing shape + (9,)) and, if ``density``, ``density`` (trailing shape + (gridpoints, 2))."""
    x = np.asarray(x, float)
    tail = x.shape[1:]
    flat = x.reshape(x.shape[0], -1)
    ns = flat.shape[1]
    ng = int((kernel_options or KERNEL)["gridpoints"])
    out = {k: np.full(ns, np.nan) for k in ("mean", "median", "variance", "hpd_inf", "hpd_sup")}
    out["deciles"] = np.full((ns, 9), np.nan)
    if density:
        out["density"] = np.full((ns, ng, 2), np.nan)
    for j in range(ns):
        r = _post.posterior_moments(flat[:, j], mh_conf_sig, density=density, kernel_options=kernel_options)
        out["mean"][j], out["median"][j], out["variance"][j] = r["mean"], r["median"], r["variance"]
        out["hpd_inf"][j], out["hpd_sup"][j] = r["hpd"]
        out["deciles"][j] = r["deciles"]
        if density:
            out["density"][j] = r["density"]
    res = {k: v.reshape(tail + v.shape[1:]) for k, v in out.items()}
    return res


@dataclass
class MomentFamily:
    """Distribution of one kind of moment: arrays indexed like the moment (see the owner's docstring), with the
    statistics' names of Dynare (``Mean``, ``Median``, ``Variance``, ``HPDinf``, ``HPDsup``, ``deciles``, ``density``)."""
    mean: np.ndarray
    median: np.ndarray
    variance: np.ndarray
    hpd_inf: np.ndarray
    hpd_sup: np.ndarray
    deciles: np.ndarray                      # (..., 9)
    density: np.ndarray = field(default=None, repr=False)   # (..., gridpoints, 2) or None

    @classmethod
    def from_draws(cls, x, mh_conf_sig, density=True, kernel_options=None):
        s = _stats(x, mh_conf_sig, density, kernel_options)
        return cls(s["mean"], s["median"], s["variance"], s["hpd_inf"], s["hpd_sup"], s["deciles"], s.get("density"))

    def at(self, *idx) -> dict:
        """Dynare's field values for one moment: ``{"Mean": ..., "Median": ..., "Variance": ..., "HPDinf": ...,
        "HPDsup": ..., "deciles": ..., "density": ...}``."""
        d = {"Mean": self.mean[idx], "Median": self.median[idx], "Variance": self.variance[idx],
             "HPDinf": self.hpd_inf[idx], "HPDsup": self.hpd_sup[idx], "deciles": self.deciles[idx]}
        if self.density is not None:
            d["density"] = self.density[idx]
        return d


# ====================================================================== variable lists
def moment_variable_list(est, var_list=None, consider=None) -> list:
    """The variables whose moments are computed (``check_list_of_variables.m``): ``var_list`` (default: the list after
    the ``estimation`` command) made unique; or, without a list, ``consider`` / the ``consider_all_endogenous``,
    ``consider_all_endogenous_and_auxiliary``, ``consider_only_observed`` option of the .mod file; with neither,
    Dynare asks (interactively) and ``nointeractive`` selects all original endogenous variables, as pydyn does.
    A list together with a ``consider_*`` option is an error in Dynare and here."""
    m = est.model
    given = list(var_list) if var_list is not None else list(est.var_list)
    flag = consider if consider is not None else est.options.get("endo_vars_for_moment_computations_in_estimation")
    if isinstance(flag, bool) or flag in ("", None):
        flag = None
    if given and flag:
        raise ValueError("You cannot use the consider_all_endogenous, consider_all_endogenous_and_auxiliary or "
                         "consider_all_observed options when listing variables after the estimation command")
    if not given:
        if flag in (None, "all_endogenous_variables", "all_endogenous"):
            given = list(m.orig_endo_names)
        elif flag in ("all_endogenous_and_auxiliary_variables", "all_endogenous_and_auxiliary"):
            given = list(m.endo_names)
        elif flag in ("only_observed_variables", "only_observed"):
            given = list(est.varobs)
        else:
            raise ValueError(f"unknown endogenous variable selection {flag!r}")
    unknown = [v for v in given if v not in m.endo_names]
    if unknown:
        raise ValueError(f"The following symbols are not endogenous variables: {' '.join(unknown)}")
    seen, out = set(), []
    for v in given:
        if v in seen:
            warnings.warn(f"{v} is specified twice in the variable list and is considered only once")
            continue
        seen.add(v)
        out.append(v)
    return out


# ====================================================================== the draws
def _theta_draws(est, mc, params, sub_draws, drop, rng, what):
    if sub_draws is not None and int(sub_draws) < 1:
        raise ValueError(f"{what}: sub_draws must be a positive integer, got {sub_draws}")
    if params is None:
        if mc is None:
            raise ValueError(f"{what} needs the chains (mc) or parameter draws (params)")
        size = _post.number_of_subdraws(mc, est, sub_draws, drop)
        params, _ = _post.posterior_subsample(mc, size, drop, rng)
    return np.atleast_2d(np.asarray(params, float))


def _option(est, name, value, default=None):
    if value is not None:
        return value
    v = est.options.get(name)
    return default if v is None else v


def _moment_options(est, ar):
    """The ``options_`` that th_autocovariances reads, with the estimation command's solution tolerances."""
    m = est.model
    o = _mom._options(m, {"ar": ar, "qz_criterium": est.options["qz_criterium"],
                          "lyapunov_complex_threshold": est.options["lyapunov_complex_threshold"]}
                      if "lyapunov_complex_threshold" in _mom._DEFAULTS else {"ar": ar, "qz_criterium": est.options["qz_criterium"]})
    if o["hp_filter"] or o["one_sided_hp_filter"] or o["bandpass"]:
        raise NotImplementedError("a stoch_simul statement of the .mod file sets a filter (hp_filter, one_sided_hp_filter or "
                                  "bandpass_filter): Dynare would carry it into the posterior moments depending on the "
                                  "order of the statements; remove it from the file")
    o["order"] = 1
    return o


def _solve_at(est, theta):
    """First-order decision rules at ``theta`` (``compute_decision_rules`` after ``set_all_parameters``)."""
    cal = est.calibration(theta)
    return first_order(est.model, params=cal["params"], qz_criterium=est.options["qz_criterium"],
                       loglinear=bool(est.options["loglinear"])), cal


def theoretical_moments_at(est, theta, var_list=None, ar=None, steps=None, nodecomposition=False) -> dict:
    """The moments that enter the posterior distributions, at one parameter vector: ``variance`` (n, n),
    ``autocorr`` (n, n, ar), ``decomposition`` (n, nexo) in fractions (or None), ``decomposition_me`` (observed
    variables of the list, nexo + 1), ``conditional`` (n, steps, nexo) and ``conditional_me``. Unit-root variables
    give NaN."""
    var_list = moment_variable_list(est, var_list)
    ar = int(_option(est, "ar", ar, 5))
    steps = [int(s) for s in np.atleast_1d(_option(est, "conditional_variance_decomposition", steps, []))] \
        if np.size(_option(est, "conditional_variance_decomposition", steps, [])) else []
    o = _moment_options(est, ar)
    m = est.model
    ivar = _mom._var_indices(m, var_list)
    return _moments_one(est, np.asarray(theta, float), ivar, o, ar, steps, bool(nodecomposition))


def _moments_one(est, theta, ivar, o, ar, steps, nodecomposition):
    m = est.model
    n, nexo = len(ivar), len(m.exo_names)
    sol, cal = _solve_at(est, theta)
    R = _mom._Rules(m, sol.as_dr())
    Sigma = cal["Sigma_e"]
    Hd = np.diag(cal["H"]).copy()
    vid = est._obs_decl
    gamma, stat = _mom._th_autocovariances(R, Sigma, ivar, o, nodecomposition or nexo < 2)
    out = {"variance": gamma[0], "autocorr": np.full((n, n, ar), np.nan), "decomposition": None, "decomposition_me": None,
           "conditional": None, "conditional_me": None}
    for i in range(1, ar + 1):
        out["autocorr"][:, :, i - 1] = gamma[i]
    if nexo > 1:
        if not nodecomposition:
            vd = gamma[ar + 1] if len(stat) > 0 else np.full((n, nexo), np.nan)
            out["decomposition"] = vd
            obs_pos, isub, iobs = _mom._intersect_stable(ivar, vid)
            if np.any(Hd != 0) and len(obs_pos):
                v = np.diag(gamma[0])[isub]
                with np.errstate(all="ignore"):
                    me = vd[isub, :] * (v / (v + Hd[iobs]))[:, None]
                out["decomposition_me"] = np.column_stack([me, 1.0 - me.sum(axis=1)])
        if steps:
            cvd, cvd_me = _mom.conditional_variance_decomposition(m, R, steps, ivar, Sigma_e=Sigma, H=Hd, varobs_id=vid)
            out["conditional"], out["conditional_me"] = cvd, cvd_me
    return out


# ====================================================================== posterior distributions of theoretical moments
@dataclass
class PosteriorTheoreticalMoments:
    """Dynare's ``oo_.PosteriorTheoreticalMoments.dsge`` (or ``PriorTheoreticalMoments``) as arrays.

    Index conventions (variables in ``var_names`` order, shocks in declaration order):

    * ``covariance``, ``contemporaneous_correlation``: (n, n), the upper triangle as Dynare stores it, mirrored;
      Dynare spells the latter field ``contemporeaneous_correlation``;
    * ``correlation``: (n, n, ar), entry (a, b, k) the distribution of corr(y_a,t, y_b,t-k);
    * ``variance_decomposition``: (n, nexo) fractions; ``variance_decomposition_me``: (observed variables of the list,
      nexo + 1), the last column the measurement error;
    * ``conditional_variance_decomposition``: (n, nexo, nsteps) and ``..._me``: (observed, nexo + 1, nsteps); Dynare's
      ``deciles`` of these are (9, nsteps) per variable and shock: here ``deciles`` has a trailing axis of 9 after
      the steps, ``density`` (..., nsteps, gridpoints, 2).

    A family is ``None`` when Dynare does not compute it. ``oo()`` returns Dynare's nested structure."""
    var_names: list
    exo_names: list
    me_var_names: list
    steps: list
    ar: int
    mh_conf_sig: float
    n_draws: int
    covariance: MomentFamily = None
    contemporaneous_correlation: MomentFamily = None
    correlation: MomentFamily = None
    variance_decomposition: MomentFamily = None
    variance_decomposition_me: MomentFamily = None
    conditional_variance_decomposition: MomentFamily = None
    conditional_variance_decomposition_me: MomentFamily = None
    kind: str = "posterior"
    params: np.ndarray = field(default=None, repr=False)
    draws: dict = field(default=None, repr=False)        # per-draw moments (keep_draws=True)

    def oo(self) -> dict:
        """Dynare's structure: ``{"covariance": {"Mean": {var1: {var2: ...}}, ...}, "correlation": ..., ...}``; only the
        pairs (var1, var2) with var1 listed first exist for ``covariance`` and ``contemporeaneous_correlation``."""
        v, x = self.var_names, self.exo_names
        res = {}

        def fill(fam, rows, cols, pair_only=False, vector=False):
            d = {}
            for key in ("Mean", "Median", "Variance", "HPDinf", "HPDsup", "deciles", "density"):
                if key == "density" and fam.density is None:
                    continue
                d[key] = {}
                for i, a in enumerate(rows):
                    d[key][a] = {}
                    for j, b in enumerate(cols):
                        if pair_only and j < i:
                            continue
                        val = fam.at(i, j)[key]
                        if key in ("deciles", "density") and vector:
                            val = [np.take(val, k, axis=0) for k in range(val.shape[0])]
                        d[key][a][b] = val
            return d
        if self.covariance is not None:
            res["covariance"] = fill(self.covariance, v, v, pair_only=True)
        if self.contemporaneous_correlation is not None:
            res["contemporeaneous_correlation"] = fill(self.contemporaneous_correlation, v, v, pair_only=True)
        if self.correlation is not None:
            res["correlation"] = fill(self.correlation, v, v, vector=True)
        if self.variance_decomposition is not None:
            res["VarianceDecomposition"] = fill(self.variance_decomposition, v, x)
        if self.variance_decomposition_me is not None:
            res["VarianceDecompositionME"] = fill(self.variance_decomposition_me, self.me_var_names, list(x) + ["ME"])
        for name, fam, rows, cols in (
                ("ConditionalVarianceDecomposition", self.conditional_variance_decomposition, v, x),
                ("ConditionalVarianceDecompositionME", self.conditional_variance_decomposition_me, self.me_var_names,
                 list(x) + ["ME"])):
            if fam is not None:
                d = fill(fam, rows, cols)
                d["Steps"] = list(self.steps)
                res[name] = d
        return res


def _symmetrise(fam: MomentFamily) -> MomentFamily:
    """Fill the lower triangle from the upper one (Dynare stores covariances for var1 listed before var2 only)."""
    out = {}
    for k in ("mean", "median", "variance", "hpd_inf", "hpd_sup", "deciles", "density"):
        a = getattr(fam, k)
        if a is None:
            out[k] = None
            continue
        a = a.copy()
        n = a.shape[0]
        iu = np.triu_indices(n)
        a[(iu[1], iu[0])] = a[iu]
        out[k] = a
    return MomentFamily(**out)


def posterior_theoretical_moments(est, mc=None, params=None, var_list=None, sub_draws=None, ar=None, steps=None,
                                  nodecomposition=None, contemporaneous_correlation=None, mh_conf_sig=None, density=True,
                                  drop=None, rng=None, keep_draws=False, consider=None, kernel_options=None,
                                  kind="posterior") -> PosteriorTheoreticalMoments:
    """Posterior distribution of the theoretical moments of endogenous variables (``moments_varendo``;
    ``compute_moments_varendo('posterior', ...)``).

    The model is solved at ``sub_draws`` parameter vectors drawn from the kept draws of ``mc`` (or at the rows of
    ``params``), the first-order theoretical moments of ``var_list`` computed at each, and the distribution of every
    moment over draws summarised as ``posterior_moments.m`` does. Defaults come from the ``estimation`` command:
    ``ar`` (5), ``conditional_variance_decomposition``, ``nodecomposition``, ``contemporaneous_correlation``,
    ``mh_conf_sig`` (0.9), ``sub_draws``, the variable list or ``consider_*`` option (see :func:`moment_variable_list`).
    Decompositions need more than one shock (Dynare skips them otherwise) and are given for stationary variables only.
    Only first-order estimation exists in pydyn; Dynare itself skips the moments at order > 1 without pruning."""
    var_list = moment_variable_list(est, var_list, consider)
    ar = int(_option(est, "ar", ar, 5))
    st = _option(est, "conditional_variance_decomposition", steps, [])
    steps = [int(s) for s in np.atleast_1d(st)] if np.size(st) else []
    if any(s <= 0 for s in steps):
        raise ValueError("Conditional variance decomposition: All periods must be strictly positive")
    nodecomposition = bool(_option(est, "nodecomposition", nodecomposition, False))
    ccorr = bool(_option(est, "contemporaneous_correlation", contemporaneous_correlation, False))
    mh_conf_sig = float(_option(est, "mh_conf_sig", mh_conf_sig, 0.9))
    theta = _theta_draws(est, mc, params, sub_draws, drop, rng, "posterior_theoretical_moments")
    nd = theta.shape[0]
    if _post._iround((1 - mh_conf_sig) * nd) < 2:
        warnings.warn("posterior_theoretical_moments: mh_conf_sig times the number of draws is too small to generate "
                      "HPD intervals")
    m = est.model
    o = _moment_options(est, ar)
    ivar = _mom._var_indices(m, var_list)
    n, nexo = len(ivar), len(m.exo_names)
    obs_pos, isub, iobs = _mom._intersect_stable(ivar, est._obs_decl)
    # measurement errors present: Dynare looks at M_.H before the loop (the estimated values at the mode, never zero for
    # an estimated standard deviation): here the first draw's
    me_present = bool(np.any(np.diag(est.calibration(theta[0])["H"]) != 0) and len(obs_pos))
    me_names = [m.endo_names[i] for i in obs_pos] if me_present else []
    V = np.empty((nd, n, n))
    AC = np.empty((nd, n, n, ar))
    VD = np.empty((nd, n, nexo)) if (nexo > 1 and not nodecomposition) else None
    VDME = np.empty((nd, len(obs_pos), nexo + 1)) if (VD is not None and me_present) else None
    CV = np.empty((nd, n, nexo, len(steps))) if (nexo > 1 and steps) else None
    CVME = np.empty((nd, len(obs_pos), nexo + 1, len(steps))) if (CV is not None and me_present) else None
    for k, th in enumerate(theta):
        try:
            r = _moments_one(est, th, ivar, o, ar, steps, nodecomposition)
        except Exception as exc:
            raise RuntimeError(f"posterior moments: unable to compute the moments at draw {k} ({exc})") from exc
        V[k], AC[k] = r["variance"], r["autocorr"]
        if VD is not None:
            VD[k] = r["decomposition"]
            if VDME is not None:
                VDME[k] = r["decomposition_me"]
        if CV is not None:
            CV[k] = np.moveaxis(r["conditional"], 1, 2)                       # (n, nexo, steps)
            if CVME is not None:
                CVME[k] = np.moveaxis(r["conditional_me"], 1, 2)
    fam = lambda x: MomentFamily.from_draws(x, mh_conf_sig, density, kernel_options)       # noqa: E731
    res = PosteriorTheoreticalMoments(
        var_names=var_list, exo_names=list(m.exo_names), me_var_names=me_names, steps=steps, ar=ar,
        mh_conf_sig=mh_conf_sig, n_draws=nd, kind=kind, params=theta)
    res.covariance = _symmetrise(fam(V))
    if ccorr:
        sd = np.sqrt(np.einsum("kii->ki", V))
        with np.errstate(all="ignore"):
            cc = V / (sd[:, :, None] * sd[:, None, :])
        cc[V == 0] = 0.0                       # covariance_mc_analysis: 0 correlations where the covariance is 0
        res.contemporaneous_correlation = _symmetrise(fam(cc))
    if ar > 0:
        res.correlation = fam(AC)
    if VD is not None:
        res.variance_decomposition = fam(VD)
        if VDME is not None:
            res.variance_decomposition_me = fam(VDME)
    if CV is not None:
        res.conditional_variance_decomposition = fam(CV)
        if CVME is not None:
            res.conditional_variance_decomposition_me = fam(CVME)
    if keep_draws:
        res.draws = {"variance": V, "autocorr": AC, "decomposition": VD, "decomposition_me": VDME,
                     "conditional": CV, "conditional_me": CVME}
    return res


# ====================================================================== smoothed and filtered variables
@dataclass
class SeriesDistribution:
    """Distribution over draws of a set of time series (one call of ``pm3.m``): ``names`` and arrays in Dynare's layout,
    ``mean``, ``median``, ``var``, ``hpd_inf``, ``hpd_sup``: (periods, series); ``deciles``: (9, periods, series);
    ``density``: (gridpoints, 2, periods, series) or None."""
    names: list
    mean: np.ndarray
    median: np.ndarray
    var: np.ndarray
    hpd_inf: np.ndarray
    hpd_sup: np.ndarray
    deciles: np.ndarray
    density: np.ndarray = field(default=None, repr=False)

    @classmethod
    def from_draws(cls, names, x, mh_conf_sig, density=True, kernel_options=None):
        """``x``: (draws, periods, series)."""
        s = _stats(x, mh_conf_sig, density, kernel_options)
        return cls(list(names), s["mean"], s["median"], s["variance"], s["hpd_inf"], s["hpd_sup"],
                   np.moveaxis(s["deciles"], -1, 0),
                   None if not density else np.moveaxis(s["density"], (2, 3), (0, 1)))

    def __getitem__(self, name) -> dict:
        k = self.names.index(name)
        d = {"Mean": self.mean[:, k], "Median": self.median[:, k], "Var": self.var[:, k], "HPDinf": self.hpd_inf[:, k],
             "HPDsup": self.hpd_sup[:, k], "deciles": self.deciles[:, :, k]}
        if self.density is not None:
            d["density"] = self.density[..., k]
        return d


@dataclass
class PosteriorSmootherStatistics:
    """Distributions of Dynare's smoothed and filtered objects (``prior_posterior_statistics`` with ``smoother`` /
    ``filtered_vars`` / ``filter_step_ahead``). Each attribute is a :class:`SeriesDistribution` or None.

    ``smoothed_variables``, ``updated_variables``, ``constant``, ``filtered_variables`` and ``trend`` are over the
    variables of the list, ``smoothed_shocks`` over all shocks, ``smoothed_measurement_errors`` over all observed
    variables (named ``SE_EOBS_<variable>``; only when a measurement-error standard deviation is estimated, as in
    Dynare), ``trend_coeff`` and ``init_state`` are one-period series. ``filtered_k_step_ahead[k]`` is the distribution
    of the k-step-ahead forecast (``Filtered_Variables_<k>_step_ahead``), periods 1..T. ``filtered_k_step_ahead_mean``
    is Dynare's ``FilteredVariablesKStepAhead`` (steps, variables, T + max step) and ``..._variances`` its (mislabelled)
    twin (Dynare stores the means there too)."""
    var_names: list
    mh_conf_sig: float
    n_draws: int
    smoothed_variables: SeriesDistribution = None
    updated_variables: SeriesDistribution = None
    smoothed_shocks: SeriesDistribution = None
    smoothed_measurement_errors: SeriesDistribution = None
    constant: SeriesDistribution = None
    trend: SeriesDistribution = None
    trend_coeff: SeriesDistribution = None
    init_state: SeriesDistribution = None
    filtered_variables: SeriesDistribution = None
    filtered_k_step_ahead: dict = field(default_factory=dict)
    filtered_k_step_ahead_mean: np.ndarray = None
    filtered_k_step_ahead_variances: np.ndarray = None
    filter_step_ahead: list = field(default_factory=list)
    params: np.ndarray = field(default=None, repr=False)
    draws: dict = field(default=None, repr=False)

    def oo(self) -> dict:
        """Dynare's top-level fields: ``SmoothedVariables``, ``UpdatedVariables``, ``SmoothedShocks``,
        ``SmoothedMeasurementErrors``, ``Constant``, ``FilteredVariables``, ``Filtered_Variables_<k>_step_ahead`` and
        ``Smoother`` (with ``Trend``, ``TrendCoeff``, ``Init_State``); each ``{"Mean": {name: array}, "Median": ...,
        "Var": ..., "deciles": ..., "HPDinf": ..., "HPDsup": ..., "density": ...}``."""
        def nest(sd):
            out = {k: {} for k in ("Mean", "Median", "Var", "deciles", "HPDinf", "HPDsup")}
            if sd.density is not None:
                out["density"] = {}
            for nme in sd.names:
                for k, v in sd[nme].items():
                    out[k][nme] = v
            return out
        res = {}
        for key, sd in (("SmoothedVariables", self.smoothed_variables), ("UpdatedVariables", self.updated_variables),
                        ("SmoothedShocks", self.smoothed_shocks), ("SmoothedMeasurementErrors", self.smoothed_measurement_errors),
                        ("Constant", self.constant), ("FilteredVariables", self.filtered_variables)):
            if sd is not None:
                res[key] = nest(sd)
        for k, sd in self.filtered_k_step_ahead.items():
            res[f"Filtered_Variables_{k}_step_ahead"] = nest(sd)
        sm = {}
        for key, sd in (("Trend", self.trend), ("TrendCoeff", self.trend_coeff), ("Init_State", self.init_state)):
            if sd is not None:
                sm[key] = nest(sd)
        if sm:
            res["Smoother"] = sm
        if self.filtered_k_step_ahead_mean is not None:
            res["FilteredVariablesKStepAhead"] = self.filtered_k_step_ahead_mean
            res["FilteredVariablesKStepAheadVariances"] = self.filtered_k_step_ahead_variances
        return res


def per_draw_series(est, theta, data=None, filter_step_ahead=(), **smoother_options):
    """The series ``prior_posterior_statistics_core.m`` stores for one draw, all variables in declaration order, as
    a dict: ``smooth``, ``update``, ``innov`` (T, shocks), ``error`` (T, observed variables), ``constant``,
    ``trend`` (NaN for variables that are not observed, as Dynare stores it), ``trend_coeff`` (zero for them),
    ``init_state`` (n,), and ``filter_step_ahead`` (steps, n, T + max step). Smoothed and updated series and
    k-step forecasts are levels (logs under ``loglinear``). ``data``: the raw data array as loaded by the smoother
    (``SmootherResult.raw``) to avoid reading the file at every draw."""
    from . import smoother as _sm
    kw = dict(smoother_options)
    if filter_step_ahead:
        kw["filter_step_ahead"] = list(filter_step_ahead)
    res = _sm.smooth(est.model, data=data, estimation=est, theta=theta, **kw)
    return _store(est, res, filter_step_ahead), res


def _store(est, res, fsa):
    m = est.model
    n = len(m.endo_names)
    mf = np.asarray(res.obs_index)
    T = res.smoothed.shape[0]
    const = np.array([res.Constant[v] for v in m.endo_names]).T            # (T, n), observed columns: sample mean / SS
    sv = np.array([res.SmoothedVariables[v] for v in m.endo_names]).T
    uv = np.array([res.UpdatedVariables[v] for v in m.endo_names]).T
    trend = np.full((T, n), np.nan)
    trend[:, mf] = np.array([res.Trend[m.endo_names[i]] for i in mf]).T
    tcoef = np.zeros(n)
    tcoef[mf] = res.TrendCoeffs[mf]
    init = np.array([res.Init_State[v] for v in m.endo_names])
    level = init - res.init_state                   # steady state (logs under loglinear), the constant added to deviations
    # stock_init_smooth(IdObs) += trend_addition(:,1) (+ mean_correction(:,1) = sample mean - steady state under prefilter)
    init[mf] = res.init_state[mf] + res.obs_constant + trend[0, mf]
    fa = None
    if fsa:
        # prior_posterior_statistics_core adds the observed variables' constant and trend to every column of the k-step
        # forecasts; pydyn.smoother.smooth adjusts only the columns k:T (the others carry the steady state alone)
        fa = np.array(res.FilteredVariablesKStepAhead, dtype=float)
        mx = fa.shape[2] - T
        trend_ext = np.vstack([trend[:, mf], trend[-1, mf] + tcoef[mf] * np.arange(1, mx + 1)[:, None]])   # (T+mx, nobs)
        for j, k in enumerate(res.filter_step_ahead):
            hi = fa.shape[2] - (mx - k)
            cols = np.r_[0:k, hi:fa.shape[2]]
            for c in cols:
                fa[j, mf, c] = fa[j, mf, c] - level[mf] + res.obs_constant + trend_ext[c]
    del level
    out = {"smooth": sv, "update": uv, "innov": res.shocks,
           "error": res.measurement_errors if res.measurement_errors is not None else None,
           "constant": const, "trend": trend, "trend_coeff": tcoef, "init_state": init,
           "filter_step_ahead": fa}
    return out


def posterior_smoother_statistics(est, mc=None, params=None, var_list=None, sub_draws=None, smoother=None,
                                  filtered_vars=None, filter_step_ahead=None, mh_conf_sig=None, density=True, drop=None,
                                  rng=None, keep_draws=False, data=None, kernel_options=None, **smoother_options
                                  ) -> PosteriorSmootherStatistics:
    """Posterior distributions of smoothed, updated and filtered variables, shocks and measurement errors (the smoother
    branch of ``prior_posterior_statistics``, summarised by ``pm3``).

    ``smoother``, ``filtered_vars`` and ``filter_step_ahead`` default to the ``estimation`` command's options
    (``filtered_vars`` alone means ``filter_step_ahead=1`` and ``filter_step_ahead`` alone switches on
    ``filtered_vars``, as in ``dynare_estimation_init``). ``var_list`` defaults to the list after the command; without it
    Dynare would ask which variables (``nointeractive``: all original endogenous variables, sorted alphabetically for
    ``prior_posterior_statistics`` -- pydyn follows ``options_.varlist`` as given or all original endogenous variables in
    declaration order). ``smoother_options`` are passed to :func:`pydyn.smoother.smooth`.
    ``filter_covariance`` and ``smoothed_state_uncertainty`` distributions are not implemented."""
    o = est.options
    smoother = bool(_option(est, "smoother", smoother, False))
    fsa = _option(est, "filter_step_ahead", filter_step_ahead, [])
    fsa = [int(k) for k in np.atleast_1d(fsa)] if np.size(fsa) else []
    fv = bool(_option(est, "filtered_vars", filtered_vars, False))
    if fv and not fsa:
        fsa = [1]
    if fsa and not fv:
        fv = True
    if not (smoother or fsa):
        raise ValueError("neither smoother nor filtered_vars / filter_step_ahead requested")
    mo = est.model.options.get("estimation", {})
    for key in ("filter_covariance", "smoothed_state_uncertainty"):
        if smoother_options.get(key) or mo.get(key):
            raise NotImplementedError(f"posterior distributions of {key} are not implemented")
    mh_conf_sig = float(_option(est, "mh_conf_sig", mh_conf_sig, 0.9))
    theta = _theta_draws(est, mc, params, sub_draws, drop, rng, "posterior_smoother_statistics")
    nd = theta.shape[0]
    m = est.model
    names = list(var_list) if var_list is not None else (list(est.var_list) or list(m.orig_endo_names))
    unknown = [v for v in names if v not in m.endo_names]
    if unknown:
        raise ValueError(f"The following symbols are not endogenous variables: {' '.join(unknown)}")
    sel = np.array([m.endo_names.index(v) for v in dict.fromkeys(names)])
    names = [m.endo_names[i] for i in sel]
    if fv and fsa[0] != 1:
        warnings.warn(f"oo_.FilteredVariables of Dynare 7.2 is the forecast {fsa[0]} period(s) ahead shifted by one period "
                      "(pm3.m reads the first element of filter_step_ahead as if it were 1); reproduced here")
    if data is None and getattr(est, "first_date", None) is None and est.options.get("datafile"):
        # the smoother's own reader of .m data files only knows plain assignments; the estimation reader evaluates
        # expressions (Dynare runs the file)
        from . import estimation as _est
        base = est.model.json_dir.parents[2] if hasattr(est.model, "json_dir") else None
        data = _est.load_data(est.options["datafile"], est.varobs, base)
    srt = sorted(set(fsa))            # pydyn.smoother sorts the list, Dynare keeps the user's order
    stores, first = [], None
    for k, th in enumerate(theta):
        try:
            st, res = per_draw_series(est, th, data=data if data is not None else (first.raw if first is not None else None),
                                      filter_step_ahead=fsa, **smoother_options)
        except Exception as exc:
            raise RuntimeError(f"posterior smoother: unable to smooth at draw {k} ({exc})") from exc
        first = first or res
        stores.append(st)
    T = stores[0]["smooth"].shape[0]
    stack = lambda key: np.array([s[key] for s in stores])                                   # noqa: E731
    out = PosteriorSmootherStatistics(var_names=names, mh_conf_sig=mh_conf_sig, n_draws=nd, params=theta,
                                      filter_step_ahead=fsa)
    SD = lambda nm, x: SeriesDistribution.from_draws(nm, x, mh_conf_sig, density, kernel_options)       # noqa: E731
    if smoother:
        out.smoothed_variables = SD(names, stack("smooth")[:, :, sel])
        out.smoothed_shocks = SD(list(m.exo_names), stack("innov"))
        out.trend_coeff = SD(names, stack("trend_coeff")[:, None, :][:, :, sel])
        out.init_state = SD(names, stack("init_state")[:, None, :][:, :, sel])
        out.constant = SD(names, stack("constant")[:, :, sel])
        out.trend = SD(names, stack("trend")[:, :, sel])
        out.updated_variables = SD(names, stack("update")[:, :, sel])
        if est._groups["vn"]:                               # estim_params_.nvn: estimated measurement error std
            en = [f"SE_EOBS_{v}" for v in est.varobs]
            out.smoothed_measurement_errors = SD(en, stack("error"))
    fa = stack("filter_step_ahead")[:, [srt.index(k) for k in fsa]] if fv else None   # (draws, steps in the user's order, n, T + max)
    if fv:
        mx = max(fsa)
        out.filter_step_ahead = fsa
        out.filtered_variables = SD(names, np.moveaxis(fa[:, 0, :, 1:1 + T], 1, 2)[:, :, sel])
        means = np.zeros((len(fsa), len(names), T + mx))
        for j, k in enumerate(fsa):
            x = np.moveaxis(fa[:, j, :, k:T + k], 1, 2)[:, :, sel]            # (draws, T, vars)
            sd = SD(names, x)
            out.filtered_k_step_ahead[k] = sd
            means[j, :, k:T + k] = sd.mean.T
        out.filtered_k_step_ahead_mean = means
        out.filtered_k_step_ahead_variances = means.copy()                    # Dynare 7.2 stores the means here too
    if keep_draws:
        out.draws = {k: stack(k) for k in ("smooth", "update", "innov", "error", "constant", "trend", "trend_coeff", "init_state")
                     if stores[0][k] is not None}
        if fsa:
            out.draws["filter_step_ahead"] = fa
    return out


# ====================================================================== prior
def _prior_draw(pr, rng, trunc=0.0):
    """One draw from the independent marginal priors by inversion on [trunc, 1 - trunc] (``dprior.draw``)."""
    from scipy import stats
    x = np.empty(len(pr))
    for i, sh in enumerate(pr.shape):
        p3, p4, p6, p7 = pr.p3[i], pr.p4[i], pr.p6[i], pr.p7[i]
        for _ in range(10000):
            u = trunc + (1 - 2 * trunc) * rng.random()
            if sh == 1:
                v = p3 + (p4 - p3) * stats.beta.ppf(u, p6, p7)
            elif sh == 2:
                v = p3 + stats.gamma.ppf(u, p6, scale=p7)
            elif sh == 3:
                v = stats.norm.ppf(u, p6, p7)
            elif sh == 4:
                v = p3 + 1.0 / np.sqrt(stats.gamma.ppf(1 - u, p7 / 2, scale=2.0 / p6))
            elif sh == 5:
                v = p6 + (p7 - p6) * u
            elif sh == 6:
                v = p3 + 1.0 / stats.gamma.ppf(1 - u, p7 / 2, scale=2.0 / p6)
            elif sh == 8:
                v = p3 + stats.weibull_min.ppf(u, p6, scale=p7)
            else:
                raise ValueError("prior draws need a prior on every estimated parameter")
            if (np.isnan(p3) or v >= p3) and (np.isnan(p4) or v <= p4):
                break
        x[i] = v
    return x


def prior_draws(est, size, rng=None, max_tries=1000):
    """``size`` parameter vectors from the prior that give a determinate model (the draws of ``prior_sampler.m`` that
    solve; Dynare keeps the failures in its files, flagged by ``info``). Returns ``(draws, share_discarded)``."""
    rng = np.random.default_rng(0) if rng is None else rng
    pr = est.prior
    trunc = float(est.options.get("prior_trunc") or 0.0)
    out, bad = [], 0
    while len(out) < size:
        x = _prior_draw(pr, rng, trunc)
        try:
            est.solve(x)
        except Exception:                                     # noqa: BLE001
            bad += 1
            if bad > max_tries * size:
                raise RuntimeError("prior_draws: cannot find draws that solve")
            continue
        out.append(x)
    return np.array(out), bad / (bad + size)


def prior_theoretical_moments(est, params=None, size=None, rng=None, **kw) -> PosteriorTheoreticalMoments:
    """Prior distribution of the theoretical moments (``compute_moments_varendo('prior', ...)`` /
    ``prior_analysis.m``): :func:`posterior_theoretical_moments` over prior draws (``params``, or ``size`` draws made by
    :func:`prior_draws`, default ``prior_mc`` = 20000). Dynare 7.2 has no command that reaches this path (the
    ``prior`` command's ``moments(distribution)`` is :func:`prior_moments_distribution`); the statistics use the same
    code as the posterior ones."""
    if params is None:
        params, _ = prior_draws(est, int(size or est.options.get("prior_mc") or 20000), rng)
    out = posterior_theoretical_moments(est, params=params, kind="prior", **kw)
    return out


def prior_moments_distribution(est, params=None, size=None, rng=None, var_list=None) -> dict:
    """The ``prior moments(distribution)`` command (``matlab/cli/prior.m``): first and second order moments of all
    original endogenous variables over prior draws that solve the model. Returns ``FirstOrderMoments`` (n, draws),
    ``SecondOrderMoments`` (n, n, draws), and the prior expectation (``mean over draws``) and variance (``mean of the
    squared deviations``, divisor = number of draws) of both, ``PriorExpectationOfFirstOrderMoments``, etc. Draws for
    which a moment is NaN (unit roots) are removed as Dynare does."""
    m = est.model
    var_list = list(m.orig_endo_names) if var_list is None else list(var_list)
    if params is None:                     # as prior_sampler.m: every draw is kept, those that do not solve give NaN moments
        rng = np.random.default_rng(0) if rng is None else rng
        trunc = float(est.options.get("prior_trunc") or 0.0)
        params = np.array([_prior_draw(est.prior, rng, trunc) for _ in range(int(size or est.options.get("prior_mc") or 20000))])
    params = np.atleast_2d(params)
    ivar = _mom._var_indices(m, var_list)
    o = _moment_options(est, 0)
    n = len(ivar)
    F = np.full((n, len(params)), np.nan)
    S = np.full((n, n, len(params)), np.nan)
    for k, th in enumerate(params):
        try:
            sol, cal = _solve_at(est, th)
        except Exception:                                  # noqa: BLE001 - no steady state, Blanchard-Kahn: NaN, as in Dynare
            continue
        R = _mom._Rules(m, sol.as_dr())
        gamma, stat = _mom._th_autocovariances(R, cal["Sigma_e"], ivar, o, True)
        mean = R.ys[ivar].copy()
        mean[np.setdiff1d(np.arange(n), stat)] = np.nan
        F[:, k] = mean
        S[:, :, k] = gamma[0]
    keepF = ~np.any(np.isnan(F), axis=0)
    keepS = ~np.any(np.isnan(S), axis=(0, 1))
    F2, S2 = F[:, keepF], S[:, :, keepS]
    ef = F2.mean(axis=1)
    es = S2.mean(axis=2)
    return {"variables": var_list, "FirstOrderMoments": F, "SecondOrderMoments": S,
            "PriorExpectationOfFirstOrderMoments": ef,
            "PriorVarianceOfFirstOrderMoments": ((F2 - ef[:, None]) ** 2).mean(axis=1),
            "PriorExpectationOfSecondOrderMoments": es,
            "PriorVarianceOfSecondOrderMoments": ((S2 - es[:, :, None]) ** 2).mean(axis=2)}
