# This file is part of pydyn. It follows matlab/estimation/non_linear_dsge_likelihood.m,
# matlab/estimation/check_particle_filter_options.m, the particle branches of
# matlab/estimation/{dynare_estimation_1, dynare_estimation_init, select_qz_criterium_value}.m and
# matlab/kalman_transition_matrix.m of Dynare 7.2 (Copyright (C) Dynare Team, GPL-3.0-or-later),
# modified in 2026.
# Released under the GNU General Public License, version 3 or later; see LICENSE. No warranty.
"""Estimation of second-order approximated models with Dynare's nonlinear filters.

:class:`NonlinearEstimation` is :class:`pydyn.estimation.Estimation` (data, priors, bounds,
parameter mapping, posterior mode, Hessian, Metropolis-Hastings) with the likelihood of
``non_linear_dsge_likelihood.m``: the model is solved at order 2 (Dynare's k-order solver), reduced
to the state and observed variables, and one of the filters of :mod:`pydyn.nonlinear_filters` is run:

* ``sis`` (default): sequential importance (bootstrap) particle filter;
* ``apf``: auxiliary particle filter;
* ``cpf``: conditional particle filter (sigma-point proposal per particle);
* ``gf``: Gaussian filter (``proposal_approximation`` and ``distribution_approximation``:
  ``unscented`` (default), ``cubature`` or ``montecarlo``);
* ``nlkf``: nonlinear (sigma-point) Kalman filter.

Options are read from the ``estimation`` command (``filter_algorithm``, ``number_of_particles``,
``resampling`` = systematic/none/generic, ``resampling_threshold``, ``resampling_method`` =
kitagawa/stratified/smooth, ``proposal_approximation``, ``distribution_approximation``,
``cpf_weights``, ``nonlinear_filter_initialization`` 1 or 3, ``particle_filter_options`` with
``pruning``, ``unscented_alpha``/``beta``/``kappa``, ``initial_state_prior_std``) and can be given as
keyword arguments with the same names.

Random numbers: as Dynare, every likelihood evaluation starts the filter's random numbers afresh
(``set_dynare_seed_local_options([], false, 'default')``), so the particle likelihood is a
deterministic function of the parameters (common random numbers) and the optimisers and the
Metropolis-Hastings sampler of :class:`~pydyn.estimation.Estimation` work unchanged. ``rng='octave'``
(default) draws Octave's numbers for seed 0, so the likelihood equals Dynare's under Octave;
``rng='numpy'`` uses numpy's generator; ``particle_seed`` changes the seed of either.

Refused (``NotImplementedError``): ``order=3`` and above, OccBin constraints, ``loglinear`` (Dynare
refuses it too), ``dsge_var``, ``heteroskedastic_filter`` and ``endogenous_prior`` (which Dynare's
nonlinear likelihood ignores or bypasses), ``prefilter``, ``observation_trends`` and missing observations (Dynare's nonlinear
filters do not handle them), ``nonlinear_filter_initialization=2``, ``filter_algorithm=gmf``,
``resampling_method`` residual/multinomial (Dynare stops: "Unknown sampling method!"), and the
combinations of options with which Dynare's code stops with an error.
"""
from __future__ import annotations

import numpy as np

from . import kalman
from . import nonlinear_filters as nlf
from .estimation import Estimation, EstimationError, _Rejected, _ispd, _pd_penalty
from .korder import KOrderError, k_order
from .perturbation import dr_ordering
from .octave_rng import make_stream
from .steady import SteadyStateError, steady_state

__all__ = ["NonlinearEstimation", "particle_options"]

#: keyword names accepted for the options of options_.particle (preprocessor names and manual names)
_PARTICLE_KEYS = {
    "filter_algorithm", "number_of_particles", "resampling", "resampling_threshold", "resampling_method",
    "proposal_approximation", "distribution_approximation", "cpf_weights", "nonlinear_filter_initialization",
    "particle_filter_options", "initial_state_prior_std", "unscented_alpha", "unscented_beta",
    "unscented_kappa", "particle_pruning",
}


def _flag_group(opts, group):
    """The option set to true among the mutually exclusive flags ``particle.<group>.*``."""
    on = [k.split(".")[-1] for k, v in opts.items() if k.startswith(f"particle.{group}.") and v]
    return on[-1] if on else None


def _parse_key_values(s):
    """``read_key_value_string``: "'key', value, 'key', value" with numbers or true/false."""
    s = s.replace("''", "'")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) % 2:
        raise EstimationError("read_key_value_string: Wrong number of Key-Value pairs!")
    out = {}
    for k, v in zip(parts[0::2], parts[1::2]):
        k = k.strip("'\"")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v.strip("'\"")
    return out


def particle_options(raw: dict) -> tuple[nlf.ParticleOptions, dict]:
    """Split estimation options into :class:`~pydyn.nonlinear_filters.ParticleOptions` (as
    ``check_particle_filter_options.m``) and the remaining options."""
    po = nlf.ParticleOptions()
    rest = {}
    pfo = None
    # .mod-file names (particle.*) first, then their flag groups, then keyword names (which win)
    items = sorted(raw.items(), key=lambda kv: 0 if kv[0].startswith("particle.") else 1)
    for k, v in items:
        if isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        if k.startswith("particle."):
            sub = k[len("particle."):]
            if sub == "filter_algorithm":
                po.filter_algorithm = str(v)
            elif sub == "number_of_particles":
                po.number_of_particles = int(v)
            elif sub == "resampling.threshold":
                po.resampling_threshold = float(v)
            elif sub == "initialization":
                po.initialization = int(v)
            elif sub == "particle_filter_options":
                pfo = v
            elif sub == "status":
                pass
            elif sub.split(".")[0] in ("resampling", "proposal_approximation", "distribution_approximation",
                                       "cpf_weights_method"):
                pass                                   # flag groups, below
            else:
                raise NotImplementedError(f"particle option {k} is not supported")
        elif k in _PARTICLE_KEYS:
            if k == "particle_filter_options":
                pfo = v
            elif k == "nonlinear_filter_initialization":
                po.initialization = int(v)
            elif k == "number_of_particles":
                po.number_of_particles = int(v)
            elif k == "resampling_threshold":
                po.resampling_threshold = float(v)
            elif k == "particle_pruning":
                po.pruning = bool(v)
            elif k in ("unscented_alpha", "unscented_beta", "unscented_kappa", "initial_state_prior_std"):
                setattr(po, k, float(v))
            elif k == "cpf_weights":
                po.cpf_weights = str(v)
            else:
                setattr(po, k, str(v))
        else:
            rest[k] = v
    keyword_set = {k for k in raw if k in _PARTICLE_KEYS}
    for group, attr in (("resampling.status", "resampling"), ("resampling.method", "resampling_method"),
                        ("proposal_approximation", "proposal_approximation"),
                        ("distribution_approximation", "distribution_approximation"),
                        ("cpf_weights_method", "cpf_weights")):
        val = _flag_group(raw, group)
        kw = {"resampling": "resampling", "resampling_method": "resampling_method", "cpf_weights": "cpf_weights",
              "proposal_approximation": "proposal_approximation",
              "distribution_approximation": "distribution_approximation"}[attr]
        if val is not None and kw not in keyword_set:
            setattr(po, attr, val)
    if pfo:
        for k, v in (_parse_key_values(pfo) if isinstance(pfo, str) else dict(pfo)).items():
            if k == "pruning":
                if not isinstance(v, bool):
                    raise EstimationError("check_particle_filter_options:: the pruning options takes only true or false")
                po.pruning = v
            elif k in ("unscented_alpha", "unscented_beta", "unscented_kappa", "initial_state_prior_std"):
                if (k in ("unscented_alpha", "initial_state_prior_std") and v <= 0) or v < 0:
                    raise EstimationError(f"check_particle_filter_options:: the {k} option takes a positive argument")
                setattr(po, k, float(v))
            else:
                raise NotImplementedError(f"particle_filter_options: {k} is not supported")
    _check(po)
    return po, rest


def _check(po):
    if po.filter_algorithm not in nlf.FILTERS:
        if po.filter_algorithm == "gmf":
            raise NotImplementedError("filter_algorithm=gmf (Gaussian mixture filter) is not implemented")
        raise EstimationError(f"Estimation: Unknown filter {po.filter_algorithm}")
    if po.resampling not in ("systematic", "none", "generic"):
        raise EstimationError(f"unknown resampling option {po.resampling}")
    if po.resampling_method not in ("kitagawa", "stratified", "smooth"):
        raise NotImplementedError(f"resampling_method={po.resampling_method}: Dynare 7.2 stops with 'Unknown "
                                  "sampling method!' (resample.m only has kitagawa, stratified and smooth)")
    for name in ("proposal_approximation", "distribution_approximation"):
        if getattr(po, name) not in ("unscented", "cubature", "montecarlo"):
            raise EstimationError(f"unknown {name} {getattr(po, name)}")
    if po.cpf_weights not in ("amisanotristani", "murrayjonesparslow"):
        raise EstimationError(f"unknown cpf_weights {po.cpf_weights}")
    if po.initialization == 2:
        raise NotImplementedError("nonlinear_filter_initialization=2 (ergodic variance from a 5000-period "
                                  "simulation) is not implemented")
    if po.initialization not in (1, 3):
        raise EstimationError("Unknown initialization option!")
    if po.pruning and po.filter_algorithm in ("gf", "nlkf", "cpf"):
        name = {"gf": "gaussian_filter", "nlkf": "nonlinear_kalman_filter", "cpf": "conditional_particle_filter"}
        raise NotImplementedError(f"{name[po.filter_algorithm]}: pruning is not yet implemented (Dynare 7.2 stops too)")
    if po.filter_algorithm == "cpf" and po.cpf_weights == "murrayjonesparslow" and po.proposal_approximation != "unscented":
        raise NotImplementedError("cpf_weights=murrayjonesparslow with proposal_approximation=cubature or montecarlo: "
                                  "Dynare 7.2 uses a (points x points) matrix as the predicted state variance there")
    if po.filter_algorithm == "apf" and po.resampling_method == "smooth":
        raise NotImplementedError("auxiliary_particle_filter: resampling_method=smooth is not supported (Dynare 7.2)")


class NonlinearEstimation(Estimation):
    """Estimation of a second-order approximated model with a nonlinear filter (``estimation(order=2,
    ...)`` in Dynare). Parameters are those of :class:`pydyn.estimation.Estimation`, the particle
    options (see the module docstring), ``rng`` ('octave', 'numpy', a stream factory or stream) and
    ``particle_seed`` (default 0, Dynare's)."""

    def __init__(self, model, data=None, rng="octave", particle_seed=0, **options):
        if getattr(model, "occbin_constraints", None):
            raise NotImplementedError("OccBin constraints with a second-order approximation (nonlinear filters) "
                                      "are not a Dynare feature")
        raw = dict(model.options.get("estimation", {}))
        raw.update(options)
        order = raw.pop("order", 2)
        order = int(order[0] if isinstance(order, (list, tuple)) else order)
        if order == 1:
            raise EstimationError("order=1: use pydyn.Estimation (Kalman filter); nonlinear filters need order >= 2")
        if order != 2:
            raise NotImplementedError(f"estimation at order {order}: only order 2 is implemented")
        po, rest = particle_options(raw)
        self.model_pruning = bool(rest.pop("pruning", False))      # options_.pruning
        if po.filter_algorithm == "cpf" and self.model_pruning:
            raise NotImplementedError("filter_algorithm=cpf with the pruning option: Dynare 7.2 stops with an error "
                                      "(iterate_law_of_motion called with pruning but without the pruned states)")
        if rest.get("loglinear"):
            raise NotImplementedError("non_linear_dsge_likelihood: It is not possible to use a non linear filter "
                                      "with the option loglinear!")
        dv = rest.get("dsge_var")
        if dv is not None and dv is not False:
            raise NotImplementedError("dsge_var with order=2: Dynare 7.2 then evaluates the linear DSGE-VAR likelihood "
                                      "(dsge_var_likelihood takes precedence over the nonlinear filters); not supported")
        if rest.get("heteroskedastic_filter"):
            raise NotImplementedError("heteroskedastic_filter with a nonlinear filter: Dynare 7.2's "
                                      "non_linear_dsge_likelihood ignores the heteroskedastic shocks; not supported")
        if rest.get("endogenous_prior"):
            raise NotImplementedError("endogenous_prior with a nonlinear filter: Dynare 7.2's non_linear_dsge_likelihood "
                                      "does not add the endogenous prior; not supported")
        if rest.get("prefilter"):
            raise NotImplementedError("prefilter with a nonlinear filter: Dynare's nonlinear filters compare the "
                                      "demeaned data with the levels of the model; not supported")
        user_qz = rest.get("qz_criterium")
        saved = model.options.get("estimation")
        model.options["estimation"] = {}
        try:
            super().__init__(model, data, order=1, **rest)
        finally:
            if saved is None:
                model.options.pop("estimation", None)
            else:
                model.options["estimation"] = saved
        self.options["order"] = 2
        self.particle = po
        self.rng, self.particle_seed = rng, particle_seed
        if self._trend_expr is not None:
            raise NotImplementedError("observation_trends with a nonlinear filter: Dynare's "
                                      "non_linear_dsge_likelihood ignores them; not supported")
        if self.missing:
            raise NotImplementedError("missing observations with a nonlinear filter (Dynare's filters do not "
                                      "handle NaN in the data)")
        # select_qz_criterium_value.m (particle filters)
        if po.initialization == 3 and (user_qz is None or float(user_qz) <= 1):
            self.options["qz_criterium"] = 1 + 1e-6
        elif user_qz is None:
            self.options["qz_criterium"] = 1 - 1e-6
        else:
            self.options["qz_criterium"] = float(user_qz)
        m = model
        order_var, _, counts = dr_ordering(m.lead_lag_incidence)
        inv = np.empty(m.n_endo, int)
        inv[order_var] = np.arange(m.n_endo)
        obs_dr = inv[self._obs_decl]
        state_dr = np.arange(counts["nstatic"], counts["nstatic"] + counts["npred"] + counts["nboth"])
        self._k2 = np.union1d(obs_dr, state_dr)                     # dr.restrict_var_list (0-based)
        self._mf0 = np.searchsorted(self._k2, state_dr)              # bayestopt_.mf0
        self._mf1 = np.searchsorted(self._k2, obs_dr)                # bayestopt_.mf1
        self._order_var = order_var
        self.filter = nlf.FILTERS[po.filter_algorithm]
        if po.pruning:
            nlf.warn_mex_pruning(m.n_exo, len(state_dr))
        self.last_filter = f"nonlinear ({po.filter_algorithm})"
        self.last_result = None

    # ------------------------------------------------------------------ reduced form
    def reduced_form(self, theta) -> nlf.ReducedForm:
        """Second-order reduced form and initial state distribution at ``theta`` (the ReducedForm
        of ``non_linear_dsge_likelihood.m``); raises ``_Rejected`` with Dynare's info codes."""
        theta = np.asarray(theta, float)
        Q, Hm = self.covariances(theta)
        m = self.model
        try:
            ys, params = steady_state(m, self.model_params(theta))
        except SteadyStateError as exc:
            raise _Rejected(f"steady state: {exc}", 20) from None
        params = np.array(params, float)
        if not np.all(np.isfinite(params)):
            raise _Rejected("parameters are NaN or Inf after the steady-state computation", 20)
        try:
            dr = k_order(m, order=2, ys=np.asarray(ys, float), params=params, pruning=False, loglinear=False,
                         qz_criterium=self.options["qz_criterium"], Sigma_e=Q)
        except KOrderError as exc:
            code, pen = self._bk_penalty(theta)
            raise _Rejected(f"solution: {exc}", code, pen) from None
        k2, mf0 = self._k2, self._mf0
        g = {k: np.asarray(dr[k], float)[k2] for k in ("ghx", "ghu", "ghxx", "ghuu", "ghxu")}
        if not all(np.all(np.isfinite(v)) for v in g.values()):
            raise _Rejected("the decision rule contains NaN or Inf", 6)
        ghs2 = np.asarray(dr["ghs2"], float).ravel()[k2]
        yss = np.asarray(ys, float)[self._order_var[k2]]
        rf = nlf.ReducedForm(**g, ghs2=ghs2, steadystate=yss, constant=yss + 0.5 * ghs2,
                             state_variables_steady_state=yss[mf0], mf0=mf0, mf1=self._mf1, Q=Q, H=Hm)
        po = self.particle
        rf.StateVectorMean = rf.constant[mf0].copy()
        if po.initialization == 1:
            # kalman_transition_matrix(dr, restrict_var_list, restrict_columns); lyapunov_symm
            A = np.zeros((len(k2), len(k2)))
            A[:, mf0] = g["ghx"]
            B = g["ghu"]
            V = kalman.lyapunov_symm(A, B @ Q @ B.T, self.options["qz_criterium"],
                                     self.options["lyapunov_complex_threshold"])
            rf.StateVectorVariance = V[np.ix_(mf0, mf0)]
        else:
            rf.StateVectorVariance = po.initial_state_prior_std * np.eye(len(mf0))
        if not _ispd(rf.StateVectorVariance) or not np.all(np.isfinite(rf.StateVectorVariance)):
            raise _Rejected("the initial state covariance matrix is not positive definite (info 201)", 201)
        return rf

    # ------------------------------------------------------------------ likelihood
    def new_stream(self):
        """The random-number stream of one likelihood evaluation (reset at every evaluation)."""
        return make_stream(self.rng, self.particle_seed)

    def run_filter(self, theta, rng=None):
        """``(LIK, lik, reduced form)`` at ``theta``: minus the log likelihood from period presample+1 and
        the log density of every period. Raises ``_Rejected`` as ``non_linear_dsge_likelihood.m``."""
        theta = np.asarray(theta, float)
        Q, Hm = self.covariances(theta)
        if self._groups["cx"] and not _ispd(Q[np.ix_(self._Q_check, self._Q_check)]):
            raise _Rejected("covariance matrix of the structural shocks is not positive definite (info 43)",
                            43, _pd_penalty(Q[np.ix_(self._Q_check, self._Q_check)]))
        if self._groups["cn"] and not _ispd(Hm[np.ix_(self._H_check, self._H_check)]):
            raise _Rejected("covariance matrix of the measurement errors is not positive definite (info 44)",
                            44, _pd_penalty(Hm[np.ix_(self._H_check, self._H_check)]))
        rf = self.reduced_form(theta)
        Y = self.data.T
        start = int(self.options["presample"]) + 1
        stream = self.new_stream() if rng is None else rng
        kw = {"model_pruning": self.model_pruning} if self.particle.filter_algorithm == "cpf" else {}
        with np.errstate(all="ignore"):
            LIK, lik = self.filter(rf, Y, start, self.particle, stream, **kw)
        if np.iscomplexobj(LIK) and np.imag(LIK) != 0:
            raise _Rejected("the likelihood is complex (info 46)", 46)
        LIK = float(np.real(LIK))
        if np.isnan(LIK):
            raise _Rejected("the likelihood is NaN (info 45)", 45)
        if np.isinf(LIK):
            raise _Rejected("the likelihood is Inf (info 50)", 50)
        return LIK, lik, rf

    def _minus_loglik(self, theta, return_lik=False):
        LIK, lik, rf = self.run_filter(theta)
        self.last_result = (LIK, lik)
        if return_lik:
            return LIK, np.asarray(lik)[int(self.options["presample"]):]
        return LIK

    # ------------------------------------------------------------------ posterior mode
    def mode(self, x0=None, method=None, compute_hessian=None, mode_compute=None, optim=None,
             additional_optimizer_steps=None, **kw):
        """As :meth:`pydyn.estimation.Estimation.mode`, except that ``mode_compute=8`` (the optimiser
        Dynare's tests use with particle filters) runs a port of Dynare's own simplex routine
        (:func:`pydyn.simplex.simplex_optimization_routine`) instead of a scipy stand-in; ``optim``
        keys MaxIter, TolFun, TolX, MaxFunEvals, MaxFunEvalFactor, InitialSimplexSize apply to it."""
        from .estimation import ModeResult, _parse_optim
        from .simplex import simplex_optimization_routine, simplex_options
        o = self.options
        first = method if method is not None else (mode_compute if mode_compute is not None else o["mode_compute"])
        extra = additional_optimizer_steps if additional_optimizer_steps is not None else o["additional_optimizer_steps"]
        steps = [first] + (list(np.atleast_1d(extra)) if extra is not None and np.size(extra) else [])
        is8 = [isinstance(st, (int, float, np.integer, np.floating)) and int(st) == 8 for st in steps]
        if not any(is8):
            return super().mode(x0=x0, method=method, compute_hessian=compute_hessian, mode_compute=mode_compute,
                                optim=optim, additional_optimizer_steps=additional_optimizer_steps, **kw)
        if compute_hessian is None:
            compute_hessian = bool(o["cova_compute"])
        opt = _parse_optim(optim if optim is not None else o["optim"])
        x = np.array(self.x0 if x0 is None else x0, float)
        nfev, msgs, ok = 0, [], True

        def fun(v):
            val = -self.log_posterior(v)
            return val, bool(np.isfinite(val))
        for st, eight in zip(steps, is8):
            if eight:
                res = simplex_optimization_routine(fun, x, simplex_options(opt))
                x, nfev = res.x, nfev + res.func_count
                ok = ok and res.exitflag == 1
                msgs.append(f"simplex (mode_compute=8): {res.message}")
            else:
                r = super().mode(x0=x, mode_compute=st, compute_hessian=False, optim=optim,
                                 additional_optimizer_steps=(), **kw)
                x, nfev, ok = r.x, nfev + r.nfev, ok and r.success
                msgs.append(r.message)
        out = ModeResult(x=np.array(x), names=list(self.names), log_posterior=self.log_posterior(x),
                         success=ok, message="; ".join(msgs), nfev=nfev)
        if compute_hessian:
            self.add_hessian(out)
        return out

    def smoother(self, theta, *args, **kwargs):
        raise NotImplementedError("Dynare has no smoother for nonlinear filters (order > 1)")
