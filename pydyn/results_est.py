# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# The fields written follow Dynare 7.2 (matlab/kalman/store_smoother_results.m, matlab/estimation/
# dynare_estimation_1.m, matlab/calib_smoother.m; Copyright (C) Dynare Team, GPL-3.0-or-later).
"""``estimation`` (mode, smoother) and ``calib_smoother`` on the live workspace of :mod:`pydyn.results`."""
from __future__ import annotations

import warnings

import numpy as np

from .matlab import Cell, Struct, Tainted, Unsupported, farr, _F
from .results_post import PosteriorAnalysisMixin, ConvergenceMixin, DiscretionaryMixin, ModeCheckMixin, FileOutputMixin, EstimationChecksMixin


def _R():
    from . import results
    return results


class EstimationMixin(PosteriorAnalysisMixin, ConvergenceMixin, DiscretionaryMixin, ModeCheckMixin, FileOutputMixin, EstimationChecksMixin):
    # ------------------------------------------------------------------ smoother results
    def _store_smoother(self, res, nk_filtered):
        """oo_.SmoothedVariables, ... as matlab/kalman/store_smoother_results.m writes them (columns, T x 1)."""
        R = _R()
        m = self.model
        self._smooth_res = res                      # what forecasts.run(..., 'smoother') starts from
        sel = self._selected_variables()            # None, or (names, declaration-order columns of the k-step arrays)
        for f in ("Smoother", "SmoothedVariables", "UpdatedVariables", "FilteredVariables",
                  "FilteredVariablesKStepAhead", "FilteredVariablesShockDecomposition",
                  "FilteredVariablesKStepAheadVariances", "SmoothedShocks", "SmoothedMeasurementErrors"):
            self._drop_oo(f)
        sm = R.rstruct("oo_.Smoother")
        sm["occbin"] = np.bool_(False)
        sm["loglinear"] = np.bool_(bool(res.loglinear))
        sm["SteadyState"] = R.col(res.SteadyState)
        sm["TrendCoeffs"] = R.col(res.TrendCoeffs if res.TrendCoeffs is not None else np.zeros(m.n_endo))
        trend = R.rstruct("oo_.Smoother.Trend")
        for v in m.endo_names:
            trend[v] = R.col(res.Trend.get(v, np.zeros(res.nobs)))
        sm["Trend"] = trend
        const = R.rstruct("oo_.Smoother.Constant")
        keep = set(sel[0]) | set(m.varobs) if sel else None          # the observed variables' constants are always stored
        for v in m.endo_names:
            if v in res.Constant and (keep is None or v in keep):
                const[v] = R.col(res.Constant[v])
        sm["Constant"] = const
        init = R.rstruct("oo_.Smoother.Init_State")
        for v in m.endo_names:
            if v in res.Init_State and (sel is None or v in sel[0]):
                init[v] = _F(float(res.Init_State[v]))
        sm["Init_State"] = init
        sm["Init_State_Vector"] = R.col(res.Init_State_Vector)
        if res.Variance is not None:                   # options_.filter_covariance (n_endo x n_endo x T+1)
            sm["Variance"] = R.mat(res.Variance)
        if res.State_uncertainty is not None:          # options_.smoothed_state_uncertainty
            sm["State_uncertainty"] = R.mat(res.State_uncertainty)
            sm["Init_State_uncertainty"] = R.mat(res.Init_State_uncertainty)
        self.set_oo("Smoother", sm)
        for field, src in (("SmoothedVariables", res.SmoothedVariables), ("UpdatedVariables", res.UpdatedVariables)):
            s = R.rstruct("oo_." + field)
            for v in m.endo_names:
                if v in src and (sel is None or v in sel[0]):
                    s[v] = R.col(src[v])
            self.set_oo(field, s)
        if nk_filtered and res.FilteredVariables:
            s = R.rstruct("oo_.FilteredVariables")
            for v in m.endo_names:
                if v in res.FilteredVariables and (sel is None or v in sel[0]):
                    s[v] = R.col(res.FilteredVariables[v])
            self.set_oo("FilteredVariables", s)
            if res.FilteredVariablesKStepAhead is not None:     # (steps, n_endo, T + max step), declaration order
                fk, fv = res.FilteredVariablesKStepAhead, res.FilteredVariablesKStepAheadVariances
                if sel is not None:
                    # store_smoother_results.m: the saved variables in decision-rule order, then A(:,p,:) = A with
                    # p = sort permutation of their declaration positions (literally, as Dynare does)
                    dec = np.array(sel[1], int)                              # declaration indices, ascending dr position
                    p = np.argsort(dec, kind="stable")
                    sub = fk[:, dec, :]
                    fk = np.empty_like(sub)
                    fk[:, p, :] = sub
                self.set_oo("FilteredVariablesKStepAhead", R.mat(fk))
                if fv is not None:
                    self.set_oo("FilteredVariablesKStepAheadVariances", R.mat(fv))
            if self.opt_flag("filter_decomposition"):
                self.refuse("oo_", ["FilteredVariablesShockDecomposition"], "filter_decomposition is not computed by pydyn")
        s = R.rstruct("oo_.SmoothedShocks")
        for v in m.exo_names:
            s[v] = R.col(res.SmoothedShocks[v])
        self.set_oo("SmoothedShocks", s)
        if res.me_cov is not None and np.any(np.asarray(res.me_cov, float) != 0) and res.SmoothedMeasurementErrors:
            s = R.rstruct("oo_.SmoothedMeasurementErrors")
            for v in res.varobs:
                s[v] = R.col(res.SmoothedMeasurementErrors[v])
            self.set_oo("SmoothedMeasurementErrors", s)

    def _refuse_selected_variables(self, est, mh_replic):
        """selected_variables_only: the classical smoother (store_smoother_results.m) is reproduced; the posterior smoother's
        draws (pm3.m on the saved subset) and the covariance arrays of the restricted state space are not."""
        R = _R()
        self._mh_eff = mh_replic if est.prior.is_bayesian else 0         # dynare_estimation_init.m: no priors, no Metropolis
        sel = self._selected_variables()
        if sel is None:
            return
        bayes = bool(np.any(np.asarray(est.prior.shape) > 0))
        if bayes and (mh_replic > 0 or self.opt_flag("load_mh_file")) and (self.opt_flag("smoother") or self.opt_flag("filtered_vars")
                                                                 or self.opt("filter_step_ahead")):
            raise R.ResultsError("estimation: selected_variables_only with the posterior smoother or filtered variables "
                                 "(the saved-variable subset of the posterior draws) is not supported by pydyn.dynare()")
        if self.opt_flag("filter_covariance") or self.opt_flag("smoothed_state_uncertainty"):
            raise R.ResultsError("estimation: selected_variables_only with filter_covariance or smoothed_state_uncertainty "
                                 "is not supported by pydyn.dynare() (the covariance arrays of the restricted state space)")

    def _selected_variables(self):
        """options_.selected_variables_only (dynare_estimation_init.m): the variables of the estimation / calib_smoother
        list (names; declaration indices in decision-rule order) that the smoother results keep, or None for all.
        Classical forecasts without a posterior ignore the option."""
        if not self.opt_flag("selected_variables_only"):
            return None
        if int(self.opt("forecast", 0) or 0) > 0 and int(getattr(self, "_mh_eff", None) if getattr(self, "_mh_eff", None) is not None
                                                      else np.max(np.atleast_1d(self.opt("mh_replic", 0) or 0))) == 0 \
                and not self.opt_flag("load_mh_file"):
            return None
        m = self.model
        names = list(getattr(self, "_sel_vars", None) or [])
        from .perturbation import dr_ordering
        order_var = dr_ordering(m.lead_lag_incidence)[0]
        inv = np.empty(m.n_endo, int)
        inv[order_var] = np.arange(m.n_endo)
        dec = sorted({m._ie[v] for v in names}, key=lambda i: inv[i])
        return [m.endo_names[i] for i in dec], dec

    def _estimation_order(self, st, *kinds):
        """options_.order is a stoch_simul default (2) here; estimation and the smoothers work at the order the
        statement gives, else at 1 (dynare_estimation_init)."""
        if "order" not in (st.get("options") or {}):
            for k in kinds:
                self.model.options.get(k, {}).pop("order", None)
                self.model._options_parsed.get(k, {}).pop("order", None)

    def _drop_oo(self, name):
        oo = self.oo_
        if name in oo:
            d = {k: v for k, v in oo.items() if k != name}
            R = _R()
            self.set_global("oo_", R.tag_tree(Struct(d), "oo_"))

    def _smoother_options(self, kind):
        """Keywords of pydyn.smoother.smooth that the live options_ give (the model's option dicts follow options_)."""
        return {}

    def _init_filter_options(self):
        """dynare_estimation_init.m: filter_step_ahead requires filtered_vars and the reverse (filter_step_ahead = 1),
        and options_.nk = max(options_.filter_step_ahead)."""
        fsa = self.opt("filter_step_ahead", None)
        fsa = [] if fsa is None else (fsa if isinstance(fsa, list) else [fsa])
        if not self.opt_flag("filtered_vars") and fsa:
            self.set_options("filtered_vars", np.bool_(True))
        if self.opt_flag("filtered_vars") and (not fsa or fsa == [0]):
            self.set_options("filter_step_ahead", np.float64(1))
            fsa = [1]
        if fsa != [0] and fsa:
            self.set_options("nk", np.float64(max(fsa)))

    # ------------------------------------------------------------------ calib_smoother
    def st_calib_smoother(self, st):
        from .smoother import smooth, SmootherError
        R = _R()
        self._refresh_options(st)
        self._undo_logged_steady()                  # dynare_estimation.m: a steady state logged by stoch_simul(loglinear) is undone
        if "parameter_set" not in (st.get("options") or {}):
            self.set_options("parameter_set", "calibration")     # the preprocessor's CalibSmootherStatement
        fo = self.opt("first_obs", None)
        if fo is None or (np.ndim(fo) == 0 and fo != fo):
            self.set_options("first_obs", np.float64(1))        # dynare_estimation_init sets it; calib_smoother returns before
                                                                # dynare_estimation_1 resets it to NaN
        self._init_filter_options()
        self.pull()
        self._estimation_order(st, "calib_smoother", "estimation")
        self._sel_vars = list(self._estimation_var_list(st))
        self._mh_eff = None
        m = self.model
        self._refuse_unfaithful_smoother()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smooth(m)
        except (SmootherError, NotImplementedError) as exc:
            raise R.ResultsError(f"calib_smoother: {exc}") from None
        # with an estimated_params block Dynare smooths at its initial values (dynare_estimation_init)
        m.params[:] = np.asarray(res.params, float)
        if res.Sigma_e is not None:
            m.Sigma_e = np.asarray(res.Sigma_e, float)
        self.push()
        self._solve_dr_at_current()
        est_ = self._set_prior_if_any()
        self._store_smoother(res, bool(self.opt_flag("filtered_vars")))
        self.fig_classical_smoother(res, 0 if est_ is None else self._estimated_me_count(est_))
        self.set_oo("gui", R.struct_set(self.oo_["gui"], "ran_calib_smoother", np.bool_(True)))

    def _refuse_unfaithful_smoother(self):
        R = _R()
        for key in ("smoother_redux", "occbin.smoother.status"):
            if self.opt_flag(key):
                raise R.ResultsError(f"the smoother option {key} is not supported by pydyn.dynare()")

    def _solve_dr_at_current(self):
        """oo_.dr and oo_.steady_state at the live calibration (Dynare's resol before the smoother)."""
        from .perturbation import first_order
        R = _R()
        self.compute_steady(command=False)
        sol = first_order(self.model, qz_criterium=self._qz(), loglinear=self.opt_flag("loglinear"))
        self.set_oo("dr", self._make_dr(sol, sol.as_dr(), 1, self._qz()))
        self._set_restrict_fields()

    # ------------------------------------------------------------------ estimation
    def st_estimation(self, st):
        from .estimation import Estimation, EstimationError
        from .smoother import smooth, SmootherError
        R = _R()
        self._refresh_options(st)
        self._undo_logged_steady()                  # dynare_estimation.m: a steady state logged by stoch_simul(loglinear) is undone
        self._init_filter_options()
        self.pull()
        self._estimation_order(st, "estimation")
        m = self.model
        mh_replic = int(np.max(np.atleast_1d(self.opt("mh_replic", 0) or 0)))
        h_fc = int(self.opt("forecast", 0) or 0)
        st_vars = self._estimation_var_list(st)
        self._sel_vars = list(st_vars)
        self.vars["var_list_"] = R.cellstr([str(x) for x in (st.get("symbol_list") or [])])      # the driver's var_list_
        ord_ = (st.get("options") or {}).get("order")
        ord_ = int(float(ord_[0] if isinstance(ord_, (list, tuple)) else ord_)) if ord_ is not None else 1
        particle = ord_ >= 2                                  # the preprocessor then writes options_.particle.status = true
        if self.opt_flag("particle.status") and not particle:
            raise R.ResultsError("estimation: options_.particle.status with order=1 selects Dynare's nonlinear likelihood "
                                 "at a first-order approximation, which pydyn does not implement")
        for key in ("load_mh_file", "mh_recover",
                    "mh_posterior_mode_estimation", "estimate_initial_states_endogenous_prior"):
            if self.opt_flag(key):
                raise R.ResultsError(f"estimation option {key} is not supported by pydyn.dynare()")
        self._refuse_unfaithful_smoother()
        old_dr = self.oo_.get("dr")                  # what a preceding check / stoch_simul left
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if particle:
                    from .nonlinear_estimation import NonlinearEstimation
                    est = NonlinearEstimation(m)
                else:
                    est = Estimation(m)
                self._refuse_selected_variables(est, mh_replic)
                self._initial_estimation_checks(est)
                mc_ = est.options["mode_compute"]
                mode_file = est.options["mode_file"]
                theta = est.x0
                mode = None
                if mc_ is None:
                    mc_ = 5                                   # Dynare's default
                if int(mc_) != 0 or mode_file:
                    mode = est.mode(mode_compute=int(mc_))
                    theta = mode.x
        except (EstimationError, NotImplementedError) as exc:
            raise R.ResultsError(f"estimation: {exc}") from None
        if not self.opt_flag("dsge_var"):                # dynare_estimation_init: options_.nobs = dataset_.nobs
            self.set_options("nobs", _F(float(est.nobs)))
        self._set_likelihood_at_initial_parameters(est, mode_file)
        self._set_prior(est)
        if self.opt_flag("plot_priors") and np.any(np.asarray(est.prior.shape) > 0):
            self.fig_priors(est)
        mc = None
        if mode is None and self.opt_flag("mode_check.status"):
            raise R.ResultsError("estimation: mode_check without a computed mode (mode_compute=0 and no mode_file) is not "
                                 "supported by pydyn.dynare()")
        if mode is not None:
            self._set_mode(est, mode, mode.x, no_density=bool(mode_file))
            if self.opt_flag("mode_check.status"):
                self._mode_check(est, mode.x)
        # dynare_estimation_1.m returns right after the smoother when mode_compute=0 and there is no mode file: no MCMC
        early = mode is None
        if mh_replic > 0 and not early and est.prior.is_bayesian:       # dynare_estimation_1.m: any(pshape>0) && mh_replic
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mc = self._run_chains(est, mode)
                    theta = self._set_posterior(est, mc, mode)
            except (EstimationError, NotImplementedError, ValueError) as exc:
                raise R.ResultsError(f"estimation: {exc}") from None
        cal = est.calibration(theta, correlation_me=True)
        m.params[:] = np.asarray(cal["params"], float)
        m.Sigma_e = np.asarray(cal["Sigma_e"], float)
        if est._groups["cx"]:                          # set_all_parameters.m stores the estimated correlations
            C = np.array(est._corr0, float)
            o = len(est._groups["vx"]) + len(est._groups["vn"])
            for i, (_, (k1, k2), _) in enumerate(est._groups["cx"]):
                C[k1, k2] = C[k2, k1] = np.asarray(theta, float)[o + i]
            m.Correlation_matrix = C
        Hc = cal.get("H")
        if Hc is not None and (est._groups["vn"] or est._groups["cn"]) and m.varobs:
            # the estimated measurement-error covariances become the model's (calibrated covariances are kept)
            m.set_me_H(np.asarray(Hc, float), m.varobs, np.asarray(cal["Correlation_matrix_ME"], float))
        self.push()
        if Hc is not None and np.any(np.asarray(Hc, float) != 0):
            self.set_M("H", R.mat(np.atleast_2d(np.asarray(Hc, float))))
            if est._groups["vn"] or est._groups["cn"]:        # set_all_parameters stores M_.Correlation_matrix_ME then
                self.set_M("Correlation_matrix_ME", R.mat(np.atleast_2d(np.asarray(cal["Correlation_matrix_ME"], float))))
        self._solve_dr_at_current()
        self._steady_at_init(est, est.load_mode(mode_file)[0] if mode_file else None)
        if mc is not None:
            self._mcmc_diagnostics(est, mc)
            self._set_metropolis(est, mc)
            self.fig_prior_posterior(est)
        if mc is not None:
            self._posterior_analysis(est, mc, st_vars, st)
        if particle:
            # dynare_estimation_1.m: smoothed and filtered objects of a posterior sample need a particle smoother, which Dynare
            # does not have; after the mode and the sample the function returns (no classical smoother, no forecast)
            if mc is not None and (self.opt_flag("smoother") or self.opt("filter_step_ahead") or self.opt_flag("filtered_vars")
                                   or h_fc > 0):
                raise R.ResultsError("estimation: Particle Smoothers are not yet implemented.")
            self._partial_dr(old_dr)
            self.set_oo("gui", R.struct_set(self.oo_["gui"], "ran_estimation", np.bool_(True)))
            self._est_ctx = {"est": est, "mode": mode, "mc": mc, "theta": theta}
            return
        posterior_stats = mc is not None and (self.opt_flag("smoother") or bool(self.opt("filter_step_ahead"))
                                              or self.opt_flag("filtered_vars") or h_fc > 0)
        bayes_smoother = mc is not None and self.opt_flag("smoother")
        if posterior_stats:
            self._posterior_statistics(est, mc, st_vars, h_fc)
        if bayes_smoother or (early and not self.opt_flag("smoother")):
            # no classical smoother (dynare_estimation_1.m): nothing recomputes oo_.dr, which keeps what the commands
            # before estimation left, plus the state-space ordering of dynare_estimation_init
            self._partial_dr(old_dr)
        else:                         # options_.frequentist_smoother is true by default: classical smoother at the estimates
            if not self.opt_flag("frequentist_smoother"):
                if h_fc > 0 and mc is None:
                    raise R.ResultsError("estimation: forecast without the classical smoother (frequentist_smoother=0)")
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = smooth(m, estimation=est, theta=theta)
                except (SmootherError, NotImplementedError) as exc:
                    raise R.ResultsError(f"estimation: smoother: {exc}") from None
                self._store_smoother(res, bool(self.opt_flag("filtered_vars")) or bool(self.opt("filter_step_ahead")))
                self.fig_classical_smoother(res, self._estimated_me_count(est))
                if h_fc > 0 and mc is None:
                    self._forecast_at_mode(res, st_vars, h_fc)
        self.set_oo("gui", R.struct_set(self.oo_["gui"], "ran_estimation", np.bool_(True)))
        self._est_ctx = {"est": est, "mode": mode, "mc": mc, "theta": theta}      # what conditional_forecast's parameter sets read

    # ------------------------------------------------------------------ forecasts after estimation
    def _estimation_var_list(self, st):
        """The variables after ``estimation`` (dynare_estimation.m: check_list_of_variables.m): repeated names are
        dropped; without a list the endogenous variables chosen by consider_all_endogenous... or, for the outputs that
        need one, all original endogenous variables (what Dynare takes with nointeractive, or when its prompt reads
        nothing)."""
        R = _R()
        m = self.model
        names = [str(x) for x in (st.get("symbol_list") or [])]
        unknown = [v for v in names if v not in m.endo_names]
        if unknown:
            raise R.ResultsError("estimation: The following symbols are not endogenous variables: " + " ".join(unknown))
        out = list(dict.fromkeys(names))
        if out:
            return out
        orig = list(m.endo_names[:m.raw.get("orig_endo_nbr", m.n_endo)])
        kind = self.opt("endo_vars_for_moment_computations_in_estimation", None)
        if kind == "all_endogenous_variables":
            return orig
        if kind == "all_endogenous_and_auxiliary_variables":
            return list(m.endo_names)
        if kind == "only_observed_variables":
            return list(m.varobs)
        return orig

    def _forecast_at_mode(self, res, var_list, horizon):
        """oo_.forecast = forecasts.run(var_list_, M_, options_, oo_, 'smoother', dataset_info) after the classical smoother
        at the mode (or at the estimates): the forecast from the end-of-sample smoothed state."""
        from .forecast import forecast_at
        R = _R()
        cs = self.opt("forecasts.conf_sig", 0.9)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = forecast_at(self.model, smoothed=res, horizon=horizon, conf_sig=float(cs), var_list=list(var_list))
        except (NotImplementedError, ValueError, KeyError) as exc:
            raise R.ResultsError(f"estimation: forecast: {exc}") from None
        self._store_forecast(fc)

    def _forecast_struct(self, fc):
        R = _R()
        out = R.rstruct("oo_.forecast")
        for key in ("Mean", "HPDinf", "HPDsup", "HPDinf_ME", "HPDsup_ME", "Exogenous"):
            d = getattr(fc, key)
            if d:
                s = R.rstruct(f"oo_.forecast.{key}")
                for v, x in d.items():
                    s[v] = R.col(x)
                out[key] = s
        return out

    def _store_forecast(self, fc):
        self.set_oo("forecast", self._forecast_struct(fc))
        self.fig_forecast(fc)

    def _run_chains(self, est, mode):
        """The Metropolis chains with Dynare's random numbers (seeded from set_dynare_seed); afterwards the results'
        shared random streams are those of the last chain, as in Dynare (posterior_sampler.m loads each chain's own
        generator states into the global generators)."""
        R = _R()
        seed = float(self.seed)
        if not (seed == np.floor(seed) and 0 <= seed < 2 ** 32 - 1):
            raise R.ResultsError(f"estimation: the seed {seed} of the random generators is not an integer in [0, 2^32 - 1)")
        from . import posterior as _post
        kw = {}
        kind = est.options.get("mcmc_jumping_covariance")
        if isinstance(kind, str) and kind not in ("hessian", "prior_variance", "identity_matrix"):
            # set_mcmc_jumping_covariance.m: the matrix of the user's file is a Hessian (Dynare inverts it)
            kw["cov"] = _post.jumping_covariance(est, kind, as_dynare=True)
        mc = est.sample(mode=mode, rng="octave", seed=int(seed), **kw)
        fr = getattr(mc, "final_rng", None)
        if fr is not None:
            self.rng.u, self.rng.n = fr.u, fr.n
            self.I.rng_unknown = None
        return mc

    def _dynare_subsample(self, mc, B, npar, exact_random=False):
        """get_posterior_subsample.m, draw for draw: per draw ``ceil(rand*mh_nblck)`` then ``ceil(rand*draws per chain)``
        from the uniform stream, the sample ordered by (mh file, chain) with Dynare's file size
        ``ceil(MaxNumberOfBytes/(npar+2)/8)``; when B equals the kept draws per chain prior_posterior_statistics
        takes the first B pooled draws instead (GetAllPosteriorDraws)."""
        nb, total = mc.draws.shape[0], mc.draws.shape[1]
        drop = float(self.opt("mh_drop", 0.5))
        first = int(np.floor(drop * total))                         # 0-based index of FirstDraw
        per_chain = total - first
        if B == per_chain and not exact_random:
            pooled = mc.draws[:, first:, :].reshape(-1, mc.draws.shape[2])
            return np.array(pooled[:B], float)
        max_runs = int(np.ceil(float(self.opt("MaxNumberOfBytes", 1e8)) / (npar + 2) / 8))
        u = self.rng.u.uniforms(2 * B)
        chain = np.ceil(u[0::2] * nb).astype(int)                   # 1-based
        draw = np.ceil(u[1::2] * per_chain).astype(int)
        absolute = first + draw                                     # FirstDraw + DrawNumber - 1, 1-based
        mhfile = -(-absolute // max_runs)
        order = np.lexsort((np.arange(B), chain, mhfile))           # sortrows(SampleAddress, [3 2]) is stable
        chain, absolute = chain[order], absolute[order]
        return np.array(mc.draws[chain - 1, absolute - 1, :], float)

    def _posterior_statistics(self, est, mc, var_list, horizon):
        """prior_posterior_statistics('posterior', ...): the smoother (and the forecasts) at the sub-draws and the posterior
        moments of the results (pm3.m): oo_.SmoothedVariables, UpdatedVariables, SmoothedShocks, SmoothedMeasurementErrors,
        Constant, Smoother.{TrendCoeff, Init_State, Trend}, FilteredVariables and the k-step ahead ones, and
        oo_.MeanForecast, PointForecast and PointForecastME, with Dynare's draws and shocks."""
        from .forecast import posterior_forecast
        from .smoother import smooth, SmootherError
        R = _R()
        nb, total = mc.draws.shape[0], mc.draws.shape[1]
        drop = float(self.opt("mh_drop", 0.5))
        per_chain = total - int(np.floor(drop * total))
        sub = self.opt("sub_draws", None)
        B = int(min(float(self.opt("posterior_max_subsample_draws", 1200)), np.ceil(per_chain * nb))) if sub is None else int(sub)
        self.set_options("sub_draws", np.float64(B))                 # set_number_of_subdraws.m
        if B == 0:
            return
        if B > per_chain * nb * nb:
            raise R.ResultsError("estimation: I cannot compute the posterior statistics! (sub_draws is greater than "
                                 "the number of available draws in the MCMC)")
        for key in ("smoothed_state_uncertainty", "filter_covariance"):
            if self.opt_flag(key):
                raise R.ResultsError(f"estimation: the posterior distribution of the {key} is not implemented in "
                                     "pydyn.dynare() (smoothed_state_uncertainty draws random numbers; the covariances "
                                     "are not stored)")
        if getattr(mc, "final_rng", None) is None:
            raise R.ResultsError("estimation: the posterior statistics draw random numbers from the state the posterior "
                                 "sampler leaves; pydyn reproduces it only for the Metropolis-Hastings-type samplers")
        fsa = self.opt("filter_step_ahead", None)
        fsa = [] if fsa is None else (fsa if isinstance(fsa, list) else [fsa])
        fsa = [int(k) for k in fsa if k]
        m = self.model
        npar = mc.draws.shape[2]
        params = self._dynare_subsample(mc, B, npar)
        sm = R.rstruct("oo_.Smoother")
        sm["loglinear"] = np.bool_(self.opt_flag("loglinear"))
        self.set_oo("Smoother", sm)
        mcs = float(self.opt("mh_conf_sig", 0.9))
        smoothers = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for b in range(B):
                    smoothers.append(smooth(m, estimation=est, theta=params[b]))
        except (SmootherError, NotImplementedError, ValueError) as exc:
            raise R.ResultsError(f"estimation: posterior smoother: {exc}") from None
        endo, exo, obs = list(m.endo_names), list(m.exo_names), list(m.varobs)
        vl = list(var_list)
        cols = [endo.index(v) for v in vl]
        gend = smoothers[0].nobs

        # prior_posterior_statistics_core.m adds log(SteadyState) to every variable under loglinear, also to the auxiliary
        # variables of exogenous leads and lags (the smoother of store_smoother_results.m leaves those in levels: for a zero
        # steady state this is -Inf)
        pdelta = {id(r): self._posterior_log_delta(r) for r in smoothers}

        def stack(get, names, adjust=False):
            return np.stack([np.column_stack([np.asarray(get(r)[v], float) + (pdelta[id(r)][endo.index(v)] if adjust else 0.0)
                                              for v in names]) for r in smoothers])

        if self.opt_flag("smoother"):
            sm_draws = stack(lambda r: r.SmoothedVariables, endo, True)
            self._store_pm3("SmoothedVariables", sm_draws, vl, cols, mcs)
            self._store_pm3("SmoothedShocks", stack(lambda r: r.SmoothedShocks, exo), exo, list(range(len(exo))), mcs)
            sm = self.oo_["Smoother"]
            tc = np.stack([np.asarray(r.TrendCoeffs, float) for r in smoothers])[:, None, :]       # (B, 1, n_endo)
            self._store_pm3("TrendCoeff", tc, vl, cols, mcs, parent="Smoother")
            # a0T + the steady state, and for the observed variables the trend of the first period and the prefilter
            # correction (the classical Init_State has neither)
            def init_state(r):
                ys = np.log(r.SteadyState) if r.loglinear else np.asarray(r.SteadyState, float)
                return [float(r.Init_State[v]) + pdelta[id(r)][i] + (float(r.Trend[v][0]) + float(r.Constant[v][0]) - ys[i] if v in obs else 0.0)
                        for i, v in enumerate(endo)]
            init = np.stack([init_state(r) for r in smoothers])[:, None, :]
            self._store_pm3("Init_State", init, vl, cols, mcs, parent="Smoother")
            self._store_pm3("Constant", stack(lambda r: r.Constant, endo, True), vl, cols, mcs)
            trend = np.stack([np.column_stack([np.asarray(r.Trend[v], float) if v in obs else np.full(gend, np.nan)
                                               for v in endo]) for r in smoothers])    # NaN where the variable is not observed
            self._store_pm3("Trend", trend, vl, cols, mcs, parent="Smoother")
            self._store_pm3("UpdatedVariables", stack(lambda r: r.UpdatedVariables, endo, True), vl, cols, mcs)
            if int(self._estimated_me_count(est)):
                names = ["SE_EOBS_" + v for v in obs]
                me = np.stack([np.column_stack([np.asarray(r.SmoothedMeasurementErrors[v], float) for v in obs])
                               for r in smoothers])
                self._store_pm3("SmoothedMeasurementErrors", me, names, list(range(len(obs))), mcs)
        if self.opt_flag("filtered_vars") or fsa:
            self._store_filtered(smoothers, vl, cols, fsa, mcs, gend)
        if not horizon:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pf = posterior_forecast(est, params=params, rng=self.rng, horizon=horizon, var_list=list(var_list),
                                        mh_conf_sig=mcs, smoothers=smoothers, keep_draws=True)
        except (SmootherError, NotImplementedError, ValueError) as exc:
            raise R.ResultsError(f"estimation: posterior forecasts: {exc}") from None
        idx = {v: m._ie[v] for v in var_list}
        self._store_pm3("MeanForecast", pf.mean_draws, vl, [idx[v] for v in vl], mcs)
        self._store_pm3("PointForecast", pf.point_draws, vl, [idx[v] for v in vl], mcs)
        H = self.M_.get("H")
        me_declared = not (np.ndim(H) == 0 and float(H) == 0.0) if H is not None else False
        sel = sorted(v for v in set(obs) & set(vl))
        if me_declared and sel:
            if pf.me_draws is None:
                raise R.ResultsError("estimation: posterior forecasts with measurement errors that are all zero are "
                                     "not supported")
            self._store_pm3("PointForecastME", pf.me_draws, sel, [obs.index(v) for v in sel], mcs)

    def _posterior_analysis(self, est, mc, var_list, st):
        """After the chains (dynare_estimation_1.m): set_number_of_subdraws, then PosteriorIRF and compute_moments_varendo,
        each drawing its sub-sample from the shared stream before the smoother and forecasts do."""
        R = _R()
        B, too_many = self._number_of_subdraws(mc)
        if B == 0:
            return
        if too_many and (self.opt_flag("bayesian_irf") or self.opt_flag("moments_varendo")):
            raise R.ResultsError("estimation: I cannot compute the posterior IRFs or moments! (sub_draws is greater than "
                                 "the number of available draws in the MCMC)")
        if self.opt_flag("bayesian_irf"):
            self._bayesian_irf(est, mc, [str(x) for x in (st.get("symbol_list") or [])], B)
        if self.opt_flag("moments_varendo"):
            self._moments_varendo(est, mc, var_list, B)

    def _partial_dr(self, old_dr):
        """oo_.dr when no smoother recomputes it: the dr before ``estimation`` (a Struct or []) with order_var and
        inv_order_var (set_state_space) and restrict_var_list and restrict_columns."""
        R = _R()
        cur = self.oo_.get("dr")
        dr = R.rstruct("oo_.dr")
        if isinstance(old_dr, Struct):
            dr.update(old_dr)
        for k in ("order_var", "inv_order_var"):
            if isinstance(cur, Struct) and k in cur:
                dr[k] = cur[k]
        self.set_oo("dr", dr)
        self._set_restrict_fields()

    def _estimated_me_count(self, est):
        """estim_params_.nvn: the number of estimated standard deviations of measurement errors."""
        return sum(1 for n in est.names if n.startswith("stderr ") and n[7:] not in self.model.exo_names)

    def _posterior_log_delta(self, r):
        """log(ys) - ys for the variables loglinear leaves in levels (auxiliary variables of exogenous leads and lags), zero
        elsewhere: what the posterior statistics add on top of the smoother's constant."""
        from .perturbation import log_mask
        n = len(r.endo_names)
        if not r.loglinear:
            return np.zeros(n)
        keep = ~log_mask(self.model)
        out = np.zeros(n)
        if keep.any():
            ys = np.asarray(r.SteadyState, float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[keep] = np.log(ys[keep]) - ys[keep]
        return out

    def _store_filtered(self, smoothers, vl, cols, fsa, mcs, gend):
        """pm3.m on the _filter_step_ahead files: oo_.FilteredVariables (the first entry of filter_step_ahead, entries 2 to
        gend+1 of the stored array, as Dynare does), oo_.Filtered_Variables_<k>_step_ahead, and the Mean-based
        oo_.FilteredVariablesKStepAhead (and ...Variances, which Dynare fills with the same means)."""
        from . import posterior as _post
        R = _R()
        arr = np.stack([np.asarray(r.FilteredVariablesKStepAhead, float) + self._posterior_log_delta(r)[None, :, None]
                        for r in smoothers])   # (B, steps, n_endo, gend+max)
        if arr.ndim != 4:
            raise R.ResultsError("estimation: the smoother returned no k-step ahead filtered variables")
        self._store_pm3("FilteredVariables", np.transpose(arr[:, 0, :, 1:1 + gend], (0, 2, 1)), vl, cols, mcs)
        nvar = len(vl)
        kmax = max(fsa)
        means = np.zeros((len(fsa), nvar, gend + kmax))
        for ii, k in enumerate(fsa):
            d = np.transpose(arr[:, ii, :, k:gend + k], (0, 2, 1))                               # (B, gend, n_endo)
            s = self._pm3_struct(f"Filtered_Variables_{k}_step_ahead", d, vl, cols, mcs)
            self.set_oo(f"Filtered_Variables_{k}_step_ahead", s)
            for i, v in enumerate(vl):
                means[ii, i, k:gend + k] = np.asarray(s["Mean"][v], float).ravel()
        self.set_oo("FilteredVariablesKStepAhead", R.mat(means))
        self.set_oo("FilteredVariablesKStepAheadVariances", R.mat(means.copy()))

    def _drop_posterior_objects(self):
        for f in ("SmoothedVariables", "UpdatedVariables", "FilteredVariables", "SmoothedShocks",
                  "SmoothedMeasurementErrors", "Smoother", "Constant", "FilteredVariablesKStepAhead"):
            self._drop_oo(f)

    def _pm3_struct(self, tag, draws, names, columns, mh_conf_sig):
        """pm3.m's moments of ``draws`` (draws, periods, variables) as the struct oo_.<tag>: {Mean, Median, Var, deciles,
        HPDinf, HPDsup, density}.<variable>."""
        from . import posterior as _post
        R = _R()
        ko = {"gridpoints": int(self.opt("estimation.moments_posterior_density.gridpoints", 512)),
              "bandwidth": self.opt("estimation.moments_posterior_density.bandwidth", 0),
              "kernel_function": self.opt("estimation.moments_posterior_density.kernel_function", "gaussian")}
        dens = self.opt_flag("estimation.moments_posterior_density.indicator")
        T = draws.shape[1]
        out = {k: R.rstruct(f"oo_.{tag}.{k}") for k in ("Mean", "Median", "Var", "deciles", "HPDinf", "HPDsup")}
        if dens:
            out["density"] = R.rstruct(f"oo_.{tag}.density")
        for v, c in zip(names, columns):
            mom = [_post.posterior_moments(draws[:, t, c], mh_conf_sig, density=dens, kernel_options=ko) for t in range(T)]
            out["Mean"][v] = R.col([x["mean"] for x in mom])
            out["Median"][v] = R.col([x["median"] for x in mom])
            out["Var"][v] = R.col([x["variance"] for x in mom])
            out["deciles"][v] = R.mat(np.column_stack([x["deciles"] for x in mom]))
            out["HPDinf"][v] = R.col([x["hpd"][0] for x in mom])
            out["HPDsup"][v] = R.col([x["hpd"][1] for x in mom])
            if dens:
                d = np.stack([x["density"] for x in mom], axis=2)
                out["density"][v] = R.mat(d[:, :, 0] if T == 1 else d)
        s = R.rstruct(f"oo_.{tag}")
        for k, v in out.items():
            s[k] = v
        return s

    def _store_pm3(self, name, draws, names, columns, mh_conf_sig, parent=None):
        """pm3.m: oo_.<name> (or oo_.<parent>.<name>) from ``draws`` shaped (draws, periods, variables)."""
        R = _R()
        tag = name if parent is None else f"{parent}.{name}"
        s = self._pm3_struct(tag, draws, names, columns, mh_conf_sig)
        if parent is None:
            self.set_oo(name, s)
        else:
            self.set_oo(parent, R.struct_set(self.oo_[parent], name, s))
        if parent is None or name == "Trend":
            self.fig_pm3(name, s, names)

    def _set_likelihood_at_initial_parameters(self, est, mode_file):
        """initial_estimation_checks.m: with options_.debug, oo_.likelihood_at_initial_parameters is the objective
        (minus the log posterior kernel) at the initial values, or at the mode file's values."""
        if not self.opt_flag("debug"):
            return
        R = _R()
        try:
            theta0 = est.x0 if not mode_file else est.load_mode(mode_file)[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fval = est.objective(np.asarray(theta0, float))[0]
        except Exception as exc:                     # noqa: BLE001 - a by-product of the checks
            raise R.ResultsError(f"estimation: the likelihood at the initial parameters cannot be computed: {exc}") from None
        self.set_oo("likelihood_at_initial_parameters", _F(fval))

    def _steady_at_init(self, est, theta0=None):
        """oo_.steady_state as dynare_estimation_init leaves it: the steady state at the initial values of the
        estimated parameters, or at the vector of the mode file (M_.params and oo_.dr are at the calibration, or the
        mode)."""
        from .steady import steady_state
        R = _R()
        m = self.model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = est.calibration(est.x0 if theta0 is None else theta0, correlation_me=True)
            xs = np.array([m._oo.get(v, 0.0) for v in m.exo_names], float)
            ys, _ = steady_state(m, params=np.asarray(cal["params"], float).copy(), exo=xs)
        self.set_oo("steady_state", R.col(ys))

    def _set_prior_if_any(self):
        """oo_.prior and oo_.steady_state of calib_smoother with an estimated_params block."""
        from .estimation import Estimation
        m = self.model
        if not any(st.get("statementName") == "estimated_params" for st in m.statements):
            R = _R()                            # Dynare leaves empty prior fields (estimated_params is absent)
            s = R.rstruct("oo_.prior")
            for k in ("mean", "mode", "variance"):
                s[k] = np.zeros((0, 0))
            h = R.rstruct("oo_.prior.hyperparameters")
            h["first"] = np.zeros((0, 0))
            h["second"] = np.zeros((0, 0))
            s["hyperparameters"] = h
            self.set_oo("prior", s)
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = Estimation(m)
            self._steady_at_init(est)
        except Exception:                       # a by-product here; the smoother does not need it
            return None
        self._set_prior(est)
        if self.opt_flag("plot_priors") and np.any(np.asarray(est.prior.shape) > 0):
            self.fig_priors(est)
        return est

    def _set_prior(self, est):
        from .smoother import prior_mode
        R = _R()
        p = est.prior
        s = R.rstruct("oo_.prior")
        s["mean"] = R.col(p.p1)
        pm = np.asarray(prior_mode(p), float).copy()
        pm[np.asarray(p.shape) == 0] = 0.0                       # no prior: Dynare's mode is 0
        s["mode"] = R.col(pm)
        s["variance"] = R.mat(np.diag(np.asarray(p.p2, float) ** 2))
        h = R.rstruct("oo_.prior.hyperparameters")
        h["first"] = R.col(p.p6)
        h["second"] = R.col(p.p7)
        s["hyperparameters"] = h
        self.set_oo("prior", s)

    def _param_tree(self, est, values, tag):
        """posterior_mode-like struct: parameters, shocks_std, measurement_errors_std, shocks_corr, ..."""
        R = _R()
        groups = {}
        for name, v in zip(est.names, values):
            if name.startswith("stderr "):
                key = "shocks_std" if name[7:] in self.model.exo_names else "measurement_errors_std"
                groups.setdefault(key, {})[name[7:]] = float(v)
            elif name.startswith("corr "):
                a, b = [x.strip() for x in name[5:].split(",")]
                key = "shocks_corr" if a in self.model.exo_names else "measurement_errors_corr"
                groups.setdefault(key, {})[f"{a}_{b}"] = float(v)
            else:
                groups.setdefault("parameters", {})[name] = float(v)
        out = R.rstruct(tag)
        for key in ("parameters", "shocks_std", "measurement_errors_std", "shocks_corr", "measurement_errors_corr"):
            if key in groups:
                g = R.rstruct(f"{tag}.{key}")
                for k, v in groups[key].items():
                    g[k] = _F(v)
                out[key] = g
        return out

    def _get_param(self, tree, name):
        m = self.model
        if name.startswith("stderr "):
            return tree["shocks_std" if name[7:] in m.exo_names else "measurement_errors_std"][name[7:]]
        if name.startswith("corr "):
            a, b = [x.strip() for x in name[5:].split(",")]
            return tree["shocks_corr" if a in m.exo_names else "measurement_errors_corr"][f"{a}_{b}"]
        return tree["parameters"][name]

    def _put_param(self, tree, name, value):
        m = self.model
        if name.startswith("stderr "):
            g = "shocks_std" if name[7:] in m.exo_names else "measurement_errors_std"
            tree[g][name[7:]] = value
        elif name.startswith("corr "):
            a, b = [x.strip() for x in name[5:].split(",")]
            tree["shocks_corr" if a in m.exo_names else "measurement_errors_corr"][f"{a}_{b}"] = value
        else:
            tree["parameters"][name] = value

    def _set_mode(self, est, mode, theta, no_density=False):
        R = _R()
        ml = bool(np.all(np.asarray(est.prior.shape) == 0))
        post = R.rstruct("oo_.posterior")
        opt = R.rstruct("oo_.posterior.optimization")
        opt["mode"] = R.col(theta)
        opt["Variance"] = R.mat(mode.inv_hessian) if mode.inv_hessian is not None else np.zeros((0, 0))
        opt["log_density"] = np.zeros((0, 0)) if no_density else _F(float(mode.log_posterior))   # no fval from a mode file
        post["optimization"] = opt
        self.set_oo("posterior", post)
        self.refuse("oo_.posterior.optimization", ["optimization_info"], "optimizer run time and counters")
        which = "mle_mode" if ml else "posterior_mode"
        self.set_oo(which, self._param_tree(est, mode.x, "oo_." + which))
        if mode.inv_hessian is not None:
            std_name = "mle_std_at_mode" if ml else "posterior_std_at_mode"
            tree = self._param_tree(est, mode.std, "oo_." + std_name)
            var = np.diag(mode.inv_hessian)
            for name, v in zip(est.names, var):          # MATLAB's sqrt of a negative variance is complex
                if v < 0:
                    self._put_param(tree, name, Tainted("complex standard deviation: the inverse Hessian is not "
                                                        "positive definite at the mode"))
            self.set_oo(std_name, tree)
            if not ml and np.isfinite(mode.laplace):
                md = R.rstruct("oo_.MarginalDensity")
                md["LaplaceApproximation"] = _F(float(mode.laplace))
                self.set_oo("MarginalDensity", md)

    # ------------------------------------------------------------------ forecast (after stoch_simul)
    def st_forecast(self, st):
        from .forecast import forecast
        R = _R()
        self._refresh_options(st)
        self.pull()
        m = self.model
        dr = self.oo_.get("dr")
        if not isinstance(dr, Struct) or "ghx" not in dr:
            raise R.ResultsError("forecast: oo_.dr is empty: run stoch_simul first")
        order = int(self.opt("order", 2))
        if self.opt_flag("linear"):
            order = 1
        sopts = st.get("options") or {}
        # forecasts.run ('simul'): horizon = options_.periods (the statement's own, else that of an earlier stoch_simul), 5 if 0
        periods = int(self.opt("periods", 0) or 0) or 5
        var_list = list(st.get("symbol_list") or [])
        if not var_list and isinstance(self.oo_.get("var_list"), Cell):
            var_list = [str(x) for x in self.oo_["var_list"].items]
        kw = {}
        eh = self.M_.get("endo_histval")
        eh = None if eh is None else np.asarray(farr(eh), float)
        if eh is not None and eh.size and eh.ndim == 2 and eh.shape[1] == 1:
            from .forecast import _histval_state       # forecasts.run: y0 = M_.endo_histval (its logs under loglinear)
            kw["y0"] = _histval_state(m, eh[:, 0])
        cs = sopts.get("forecasts.conf_sig", sopts.get("conf_sig"))
        if cs is not None:
            kw["conf_sig"] = float(cs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = forecast(m, var_list or None, periods=periods, order=order,
                              pruning=self.opt_flag("pruning"), rng=self.rng, **kw)
        except (NotImplementedError, ValueError) as exc:
            raise R.ResultsError(f"forecast: {exc}") from None
        out = R.rstruct("oo_.forecast")
        for key in ("Mean", "HPDinf", "HPDsup", "HPDinf_ME", "HPDsup_ME", "Exogenous"):
            d = getattr(fc, key)
            if d:
                s = R.rstruct(f"oo_.forecast.{key}")
                for v, x in d.items():
                    s[v] = R.col(x)
                out[key] = s
        self.set_oo("forecast", out)
        self.fig_forecast(fc)

    # ------------------------------------------------------------------ shock decompositions
    def _decomposition_smoother(self, kind, run=True, allow_sets=False, st=None):
        """The smoother of Dynare's evaluate_smoother(parameter_set, ...) inside shock_decomposition.m and
        initial_condition_decomposition.m; parameter_set = 'calibration', and with ``allow_sets`` also the posterior mean,
        median and mode and the ML mode of the last estimation.  The parameter set used is left in ``self._sd_ps``."""
        from .smoother import smooth, SmootherError
        R = _R()
        ps = self._sd_resolve_ps(kind, allow_sets)
        kw = {}
        if not isinstance(self.oo_.get("SmoothedVariables"), Struct):
            df = self.opt("datafile", None)
            if not (allow_sets and isinstance(df, str) and df):
                raise R.ResultsError(f"{kind}: no smoothed variables: run estimation(smoother) or calib_smoother first")
            kw["datafile"] = df             # evaluate_smoother runs dynare_estimation_init on the live options
            for key in ("first_obs", "nobs"):
                v = self.opt(key, None)
                if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) and np.isfinite(v) and v > 0:
                    kw[key] = int(v)
        # the statement's own smoother options (shock_decomposition(lik_init=3, nobs=100, ...)): Dynare writes them into options_
        # before evaluate_smoother runs, so they govern this smoother run (found by tools/options_probe.py: they were ignored)
        from .smoother import _DEFAULTS as _SMOOTHER_OPTIONS
        for key, val in ((st or {}).get("options") or {}).items():
            if key in _SMOOTHER_OPTIONS and key not in ("datafile", "parameter_set", "varobs", "noprint"):
                kw[key] = R.py_value(R.option_value(val))
        sets = ("calibration", "posterior_mean", "posterior_median", "posterior_mode", "mle_mode") if allow_sets else ("calibration",)
        if ps not in sets:
            raise R.ResultsError(f"{kind}: parameter_set={ps} is not supported by pydyn.dynare() (only "
                                 + ", ".join(sets) + ")")
        self._sd_ps = ps
        self._refuse_unfaithful_smoother()
        self.pull()
        if ps != "calibration":
            ctx = getattr(self, "_est_ctx", None)
            if ctx is None:
                raise R.ResultsError(f"{kind}: parameter_set={ps} needs an estimation before the statement")
            if (ps in ("posterior_mean", "posterior_median") and ctx["mc"] is None) or \
                    (ps in ("posterior_mode", "mle_mode") and ctx["mode"] is None):
                raise R.ResultsError(f"{kind}: parameter_set={ps} is not available from the last estimation")
            kw.update(estimation=ctx["est"], parameter_set=ps, mode=ctx["mode"], mcmc=ctx["mc"])
        if not run:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return smooth(self.model, **kw)
        except (SmootherError, NotImplementedError) as exc:
            raise R.ResultsError(f"{kind}: {exc}") from None

    def _decomp_info(self, path, value):
        R = _R()
        info = self.oo_.get("shock_decomposition_info")
        info = info if isinstance(info, Struct) else R.rstruct("oo_.shock_decomposition_info")
        self.set_oo("shock_decomposition_info", R.struct_set(info, path, value))

    def st_shock_decomposition(self, st):
        """shock_decomposition.m: oo_.shock_decomposition (endo_nbr x exo_nbr + 2 x T) and
        oo_.shock_decomposition_info.parameter_set."""
        from .smoother import shock_decomposition
        R = _R()
        self._refresh_options(st)
        kind = "shock_decomposition"
        ft = self.opt("shock_decomp.forecast_type", None)
        ft = ft if isinstance(ft, str) and ft else None
        if self.opt_flag("shock_decomp.with_epilogue"):
            raise R.ResultsError(f"{kind}: with_epilogue is not supported by pydyn.dynare()")
        plot = not self.opt_flag("no_graph.shock_decomposition")
        if plot and (self.opt("plot_shock_decomp.init2shocks", None) or self.opt_flag("plot_shock_decomp.realtime")):
            raise R.ResultsError(f"{kind}: plot_shock_decomposition with init2shocks or realtime writes more to oo_ "
                                 "than pydyn.dynare() follows")
        if ft is not None:
            self._sd_forecast_inputs(ft)           # the checks that precede the smoother in shock_decomposition.m
        res = self._decomposition_smoother(kind, allow_sets=True, st=st)
        ps = self._sd_ps
        self._sd_initial_date = self._decomp_initial_date(res)
        if ft is not None:
            self._sd_forecast(st, res, ft, ps)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sd = shock_decomposition(self.model, res, init_state=int(self.opt("shock_decomp.init_state", 0) or 0))
            self.set_oo("shock_decomposition", R.mat(sd.z))
            self.unrefuse("oo_", ["shock_decomposition"])
            self._decomp_info("parameter_set", ps)
        if plot:
            # plot_shock_decomposition.m (called unless nograph) records the plotted variables in oo_
            from .matlab import BUILTINS
            m = self.model
            names = list(st.get("symbol_list") or []) or list(m.endo_names[:int(self.M_["orig_endo_nbr"])])
            i_var = BUILTINS["varlist_indices"](self.I, [Cell(names, (len(names), 1)), self.M_["endo_names"]], 1)
            i_var = np.asarray(farr(i_var), float).reshape(-1, 1)
            info = self.oo_.get("plot_shock_decomposition_info")
            info = info if isinstance(info, Struct) else R.rstruct("oo_.plot_shock_decomposition_info")
            if "i_var" in info:
                i_var = np.unique(np.concatenate([i_var.ravel(), np.asarray(farr(info["i_var"]), float).ravel()])).reshape(-1, 1)
            self.set_oo("plot_shock_decomposition_info", R.struct_set(info, "i_var", R.mat(i_var)))
            self.fig_shock_decomposition(i_var.ravel(), forecast_type=ft)
        self.set_oo("gui", R.struct_set(self.oo_["gui"], "ran_shock_decomposition", np.bool_(True)))

    def st_plot_shock_decomposition(self, st):
        """``plot_shock_decomposition.m``: the figures (plain decompositions) and the variables recorded in oo_."""
        from .matlab import BUILTINS
        R = _R()
        # set_default_plot_shock_decomposition_options
        empty = np.zeros((0, 0))
        for key, val in (("use_shock_groups", ""), ("colormap", ""), ("graph_format", "eps"), ("detail_plot", np.bool_(False)),
                         ("init2shocks", empty), ("interactive", np.bool_(False)), ("screen_shocks", np.bool_(False)),
                         ("steadystate", np.bool_(False)), ("type", ""), ("fig_name", ""), ("write_xls", np.bool_(False)),
                         ("realtime", np.float64(0)), ("vintage", np.float64(0)), ("plot_init_date", empty),
                         ("plot_end_date", empty), ("diff", np.bool_(False)), ("flip", np.bool_(False)),
                         ("max_nrows", np.float64(6)), ("forecast_length", empty)):
            self.set_options("plot_shock_decomp." + key, val)
        self.set_options("shock_decomp.forecast_type", empty)
        self._refresh_options(st)
        if not isinstance(self.oo_.get("shock_decomposition"), (np.ndarray, Struct)) and "shock_decomposition" not in self.oo_:
            raise R.ResultsError("plot_shock_decomposition: oo_.shock_decomposition is missing: run shock_decomposition first")
        names = list(st.get("symbol_list") or [])
        m = self.model
        allnames = names or list(m.endo_names[:int(self.M_["orig_endo_nbr"])])
        i_var = BUILTINS["varlist_indices"](self.I, [Cell(allnames, (len(allnames), 1)), self.M_["endo_names"]], 1)
        i_var = np.asarray(farr(i_var), float).reshape(-1, 1)
        info0 = self.oo_.get("shock_decomposition_info")
        if names and not (isinstance(info0, Struct) and "i_var" in info0):
            info = self.oo_.get("plot_shock_decomposition_info")
            info = info if isinstance(info, Struct) else R.rstruct("oo_.plot_shock_decomposition_info")
            iv = i_var
            if "i_var" in info:
                iv = np.unique(np.concatenate([i_var.ravel(), np.asarray(farr(info["i_var"]), float).ravel()])).reshape(-1, 1)
            self.set_oo("plot_shock_decomposition_info", R.struct_set(info, "i_var", R.mat(iv)))
        if not self.opt_flag("no_graph.plot_shock_decomposition"):
            self.fig_shock_decomposition(i_var.ravel())

    def st_initial_condition_decomposition(self, st):
        """initial_condition_decomposition.m: oo_.initval_decomposition (endo_nbr x endo_nbr + 2 x T)."""
        from .smoother import initial_condition_decomposition
        R = _R()
        empty = np.zeros((0, 0))
        for key, val in (("colormap", ""), ("graph_format", "eps"), ("fig_name", ""), ("detail_plot", np.bool_(False)),
                         ("init2shocks", empty), ("steadystate", np.bool_(False)), ("write_xls", np.bool_(False)),
                         ("type", ""), ("plot_init_date", empty), ("plot_end_date", empty), ("diff", np.bool_(False)),
                         ("flip", np.bool_(False)), ("max_nrows", np.float64(6))):
            self.set_options("initial_condition_decomp." + key, val)         # set_default_initial_condition_decomposition_options
        self._refresh_options(st)
        kind = "initial_condition_decomposition"
        if self.opt_flag("initial_condition_decomp.with_epilogue"):
            raise R.ResultsError(f"{kind}: with_epilogue is not supported by pydyn.dynare()")
        if "initval_decomposition" in self.oo_:
            self._plot_initval_decomposition(st)
            return                  # Dynare keeps the decomposition it has (and only plots it again)
        res = self._decomposition_smoother(kind, st=st)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            icd = initial_condition_decomposition(self.model, res)
        self.set_oo("initval_decomposition", R.mat(icd.z))
        self.unrefuse("oo_", ["initval_decomposition"])
        self._decomp_info("initval_decomposition.parameter_set", "calibration")
        self._plot_initval_decomposition(st)

    def _plot_initval_decomposition(self, st):
        """The plot of initial_condition_decomposition.m: plot_shock_decomposition on the initial-condition decomposition
        with the endogenous variables as the components."""
        if not self.fig_on():
            return
        from .matlab import BUILTINS
        for key in ("colormap", "nodisplay", "graph_format", "fig_name", "detail_plot", "init2shocks", "steadystate", "write_xls",
                    "type", "plot_init_date", "plot_end_date", "diff", "flip", "max_nrows"):
            v = self.opt("initial_condition_decomp." + key, None)
            if v is not None:
                self.set_options("plot_shock_decomp." + key, v)
        self.set_options("plot_shock_decomp.realtime", np.float64(0))
        self.set_options("plot_shock_decomp.use_shock_groups", "")
        names = list(st.get("symbol_list") or [])
        allnames = names or list(self.model.endo_names[:int(self.M_["orig_endo_nbr"])])
        i_var = BUILTINS["varlist_indices"](self.I, [Cell(allnames, (len(allnames), 1)), self.M_["endo_names"]], 1)
        self.fig_shock_decomposition(np.asarray(farr(i_var), float).ravel(), init_cond=True)

    def st_realtime_shock_decomposition(self, st):
        """realtime_shock_decomposition.m: oo_.realtime_shock_decomposition, .conditional_shock_decomposition,
        .realtime_forecast_shock_decomposition (with forecast) and .realtime_conditional_shock_decomposition
        (structs of time_<j> and pool fields, each endo_nbr x exo_nbr + 2 x periods)."""
        from .smoother import realtime_shock_decomposition, SmootherError
        R = _R()
        self._refresh_options(st)
        kind = "realtime_shock_decomposition"
        if self.opt_flag("shock_decomp.with_epilogue"):
            raise R.ResultsError(f"{kind}: with_epilogue is not supported by pydyn.dynare()")
        self._decomposition_smoother(kind, run=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rt = realtime_shock_decomposition(self.model)
        except (SmootherError, NotImplementedError, ValueError) as exc:
            raise R.ResultsError(f"{kind}: {exc}") from None

        def put(field, d, info):
            if not d:
                return
            cur = self.oo_.get(field)
            s = cur.clone() if isinstance(cur, Struct) else R.rstruct("oo_." + field)
            for k in sorted((k for k in d if k != "pool"), key=lambda k: int(k.split("_")[1])) + \
                    (["pool"] if "pool" in d else []):
                s[k] = R.mat(d[k])
            self.set_oo(field, s)
            self.unrefuse("oo_", [field])
            if info:
                self._decomp_info(f"{field}.parameter_set", "calibration")
        put("realtime_conditional_shock_decomposition", rt.realtime_conditional, True)
        put("realtime_shock_decomposition", rt.realtime, True)
        put("conditional_shock_decomposition", rt.conditional, True)
        put("realtime_forecast_shock_decomposition", rt.realtime_forecast, True)
        self.set_oo("gui", R.struct_set(self.oo_["gui"], "ran_realtime_shock_decomposition", np.bool_(True)))


    # ------------------------------------------------------------------ OccBin
    def st_occbin_setup(self, st):
        m = self.model
        m.options["occbin_setup"] = {**m.options.get("occbin_setup", {}), **(st.get("options") or {})}

    def st_occbin_solver(self, st):
        from .occbin import Occbin, OccbinError
        R = _R()
        m = self.model
        m._exec_statement(st)                       # records the statement's options (read by Occbin.mod_options)
        self.pull()
        if self.opt_flag("loglinear"):
            from .occbin_estimation import _LOGLINEAR_REFUSAL
            raise R.ResultsError(f"occbin_solver: {_LOGLINEAR_REFUSAL}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ob = Occbin(m)
                res = ob.run()
        except OccbinError as exc:
            raise R.ResultsError(f"occbin_solver: {exc}") from None
        except (ValueError, NotImplementedError) as exc:
            raise R.ResultsError(f"occbin_solver: {exc}") from None
        pos = np.asarray(res.exo_pos, int)
        if res.shocks.ndim == 2 and res.shocks.shape[1] == len(m.exo_names):
            nz = np.flatnonzero(np.any(res.shocks != 0, axis=0))      # occbin.setup: exo_pos = find(~all(temp == 0, 1))
            pos = nz if nz.size else pos
        sim = R.rstruct("oo_.occbin.simul")
        sim["shocks_sequence"] = R.mat(np.asarray(res.shocks, float)[:, pos] if res.shocks.ndim == 2 and pos.size
                                       else np.asarray(res.shocks, float))
        sim["exo_pos"] = R.row(pos + 1) if pos.size != 1 else _F(float(pos[0] + 1))
        sim["ys"] = R.col(res.ys)
        sim["piecewise"] = R.mat(res.piecewise)
        sim["linear"] = R.mat(res.linear)
        sim["error_flag"] = _F(float(res.error_flag))
        occ = R.rstruct("oo_.occbin")
        occ["simul"] = sim
        self.set_oo("occbin", occ)
        self.set_oo("dr", self._make_dr(ob.ref, ob.ref.as_dr(), 1, self._qz()))    # the reference regime's rules
        self.refuse("oo_.occbin.simul", ["regime_history"], "a struct array")
        self._occbin_shocks = np.asarray(res.shocks, float)

    def st_rplot(self, st):
        """``rplot(var_list_)``."""
        self.fig_rplot([str(x) for x in (st.get("symbol_list") or [])])

    def st_occbin_graph(self, st):
        """``occbin.graph(M_, options_, options_occbin_, oo_, var_list_)``."""
        opts = st.get("options") or {}
        ss = opts.get("graph.steady_state", True)             # noconstant sets it to false (deviations from the steady state)
        if isinstance(ss, str):
            ss = ss.strip().lower() not in ("false", "0")
        self.fig_occbin([str(x) for x in (st.get("symbol_list") or [])], bool(ss))

    def fn_occbin_forecast(self, args, nargout):
        """``[forecast, error_flag, yf] = occbin.forecast(options_, M_, dr, ys, exo_ys, exo_det_ys, horizon)`` for the
        model of the .mod file at its current parameters (``forecast.regimes`` is not provided)."""
        from .occbin import Occbin, OccbinError
        from .occbin_forecast import occbin_forecast, STATISTICS
        R = _R()
        m = self.model
        if len(args) != 7:
            raise R.ResultsError("occbin.forecast needs seven arguments")
        horizon = int(np.asarray(farr(args[6])).ravel()[0])
        self.pull()
        opts = args[0]
        g = lambda k, d=None: (lambda v: d if v is None or (isinstance(v, np.ndarray) and v.size == 0) else R.py_value(v))(
            R.struct_get(opts, "occbin.forecast." + k))
        sh = R.struct_get(opts, "occbin.forecast.SHOCKS0")
        shocks = None
        if isinstance(sh, np.ndarray) and sh.size:
            shocks = np.asarray(sh, float)
        elif sh is not None and not isinstance(sh, np.ndarray) and hasattr(sh, "items") is False and len(sh):
            shocks = [(str(c[0]), np.asarray(farr(c[1]), float).ravel()) for c in sh]
        eh = np.asarray(farr(args[1]["endo_histval"]), float)
        if eh.size == 0:
            raise R.ResultsError("occbin.forecast: M_.endo_histval is empty (set the initial state first)")
        occ = self.oo_.get("occbin")
        if g("replic", 0) == 0 and isinstance(occ, Struct) and "simul" in occ and "exo_pos" in occ["simul"]:
            npos = np.asarray(farr(occ["simul"]["exo_pos"])).size
            if npos != len(m.exo_names):
                raise R.ResultsError(
                    f"occbin.forecast: with replic = 0 Dynare's solver uses options_.occbin.simul.exo_pos left by occbin_solver "
                    f"({npos} of {len(m.exo_names)} shocks) and stops with a size mismatch (a defect of +occbin/forecast.m)")
        fr = g("frcst_regimes")
        if fr not in (None, [], 0):
            raise R.ResultsError("occbin.forecast: options_.occbin.forecast.frcst_regimes is not supported by pydyn.dynare()")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ob = Occbin(m)
            fc = occbin_forecast(ob, horizon, eh.reshape(eh.shape[0], -1, order="F")[:, 0], shocks=shocks,
                                 replic=int(g("replic", 0)), qmc=bool(g("qmc", 0)), maxit=int(g("maxit", 30)),
                                 check_ahead_periods=int(g("check_ahead_periods", 30)),
                                 conf_sig=float(R.py_value(R.struct_get(opts, "forecasts.conf_sig")) or 0.9),
                                 sigma_e=np.asarray(farr(args[1]["Sigma_e"]), float), rng=self.rng,
                                 simul_options=ob.mod_options())
        out = R.rstruct("forecast")
        for fld in ("linear", "piecewise"):
            d = R.rstruct(f"forecast.{fld}")
            for stat, per in getattr(fc, fld).items():
                sv = R.rstruct(f"forecast.{fld}.{stat}")
                for nm, v in per.items():
                    sv[nm] = R.mat(v) if np.ndim(v) == 2 else R.col(v)
                d[stat] = sv
            out[fld] = d
        res = [out, _F(float(fc.error_flag if fc.replic == 0 else 0)) if fc.replic == 0 else _F(0.0)]
        res.append(R.mat(fc.yf) if fc.yf is not None else np.zeros((0, 0)))
        return res[:max(nargout, 1)]

    def fn_unpack_simulations(self, args, nargout):
        R = _R()
        m = self.model
        occ = self.oo_.get("occbin")
        if not isinstance(occ, Struct) or "simul" not in occ:
            raise R.ResultsError("occbin.unpack_simulations: oo_.occbin.simul is empty: run occbin_solver first")
        sim = occ["simul"]
        lin, pw, ys = (np.asarray(farr(sim[k]), float) for k in ("linear", "piecewise", "ys"))
        ys = ys.ravel(order="F")
        out = {"endo_linear": R.rstruct("oo_.occbin.endo_linear"), "endo_piecewise": R.rstruct("oo_.occbin.endo_piecewise"),
               "endo_ss": R.rstruct("oo_.occbin.endo_ss")}
        for i, v in enumerate(m.endo_names):
            out["endo_linear"][v] = R.col(lin[:, i])
            out["endo_piecewise"][v] = R.col(pw[:, i])
            out["endo_ss"][v] = _F(float(ys[i]))
        ex = R.rstruct("oo_.occbin.exo")
        sh = self._occbin_shocks
        pos = np.asarray(farr(sim["exo_pos"]), float).ravel(order="F").astype(int) - 1
        for k, j in enumerate(pos):
            ex[m.exo_names[k]] = R.col(sh[:, j])           # unpack_simulations indexes the names by position (as Dynare)
        out["exo"] = ex
        new = R.rstruct("oo_.occbin")
        for k, v in occ.items():
            new[k] = v
        new.update(out)
        self.set_oo("occbin", new)
        return [self.oo_]

    # ------------------------------------------------------------------ optimal policy
    def st_ramsey_model(self, st):
        """ramsey_model: the preprocessor already wrote the planner's first-order conditions; pydyn.steady_state
        routes such models to the Ramsey steady state."""
        self.model._exec_statement(st)

    def st_evaluate_planner_objective(self, st):
        from .optimal_policy import evaluate_planner_objective, OptimalPolicyError
        R = _R()
        self.pull()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u0 = None
                es = self.oo_.get("exo_simul")
                if not self.model.det_shocks and isinstance(es, np.ndarray) and es.size:
                    # get_initial_state: u = oo_.exo_simul(1,:)' (the first simulated shock of stoch_simul with periods > 0)
                    u0 = np.asarray(farr(es), float)[0]
                pv = evaluate_planner_objective(self.model, order=int(self.opt("order", 2)), u=u0)
        except (OptimalPolicyError, NotImplementedError, ValueError) as exc:
            raise R.ResultsError(f"evaluate_planner_objective: {exc}") from None
        out = R.rstruct("oo_.planner_objective_value")
        if pv.zero_initial_multiplier is not None:
            c = R.rstruct("oo_.planner_objective_value.conditional")
            c["zero_initial_multiplier"] = _F(float(pv.zero_initial_multiplier))
            c["steady_initial_multiplier"] = _F(float(pv.steady_initial_multiplier))
            out["conditional"] = c
        else:
            out["conditional"] = _F(float(pv.conditional))
        out["unconditional"] = _F(float(pv.unconditional))
        self.set_oo("planner_objective_value", out)

    def st_ramsey_policy(self, st):
        """ramsey_policy = ramsey_model + stoch_simul + evaluate_planner_objective (the options are stoch_simul's)."""
        self.model._exec_statement(st)
        self.st_stoch_simul(st)
        self.st_evaluate_planner_objective(st)

    def st_osr(self, st):
        """osr: optimal simple rule parameters (pydyn.optimal_policy.osr), then stoch_simul at the optimum."""
        from .optimal_policy import osr, OptimalPolicyError
        R = _R()
        m = self.model
        m._exec_statement(st)
        self._refresh_options(st)
        self.pull()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = osr(m, order=int(self.opt("order", 1)) if st.get("options", {}).get("order") is not None else None)
        except (OptimalPolicyError, NotImplementedError, ValueError) as exc:
            raise R.ResultsError(f"osr: {exc}") from None
        self.push()
        o = R.rstruct("oo_.osr")
        o["error_indicator"] = _F(float(res.error_indicator))
        o["objective_function"] = _F(float(res.objective_function))
        p = R.rstruct("oo_.osr.optim_params")
        for k, v in res.optim_params.items():
            p[k] = _F(float(v))
        o["optim_params"] = p
        self.set_oo("osr", o)
        self.refuse("oo_.osr", ["optimization_info"], "optimizer run time and counters")
        self.st_stoch_simul(st)

    # ------------------------------------------------------------------ posterior statistics of the draws
    def _set_posterior(self, est, mc, mode):
        """Posterior mean (the parameters Dynare sets after the draws); fills oo_.posterior_* and the
        marginal density. Returns the posterior mean vector."""
        from .posterior import posterior_statistics, modified_harmonic_mean
        R = _R()
        stats = posterior_statistics(mc, est, drop=est.options.get("mh_drop"))
        self._posterior_stats = stats
        for field, vals in (("posterior_mean", stats.mean), ("posterior_hpdinf", stats.hpd_inf),
                            ("posterior_hpdsup", stats.hpd_sup), ("posterior_median", stats.median),
                            ("posterior_variance", stats.variance), ("posterior_std", stats.std)):
            self.set_oo(field, self._param_tree(est, vals, "oo_." + field))
        dec = self._param_tree(est, [0.0] * len(est.names), "oo_.posterior_deciles")
        den = self._param_tree(est, [0.0] * len(est.names), "oo_.posterior_density")
        for j, name in enumerate(est.names):
            self._put_param(dec, name, R.col(stats.deciles[j]))
            self._put_param(den, name, R.mat(np.asarray(stats.density[j], float)))
        self.set_oo("posterior_deciles", dec)
        self.set_oo("posterior_density", den)
        md = self.oo_.get("MarginalDensity")
        new = R.rstruct("oo_.MarginalDensity")
        if isinstance(md, Struct):
            new.update(md)
        md = new
        try:
            mhm = float(modified_harmonic_mean(mc, drop=est.options.get("mh_drop")).value)
        except np.linalg.LinAlgError:
            mhm = np.inf                      # a singular covariance of the draws (a very short chain): Octave's inv gives Inf
        md["ModifiedHarmonicMean"] = _F(mhm)
        self.set_oo("MarginalDensity", md)
        return np.asarray(stats.mean, float)

    def _set_metropolis(self, est, mc):
        """oo_.posterior.metropolis.{mean, Variance} as GetPosteriorMeanVariance.m computes them: chain after chain and
        file after file (``ceil(MaxNumberOfBytes/(npar+2)/8)`` draws per file), recursively, the demeaned products
        divided by the number of kept draws (not N - 1) and the mean of the first file's kept draws fixed by
        ``ceil(mh_drop*draws - draws already seen + 1)``."""
        R = _R()
        d = np.asarray(mc.draws, float)
        nb, T, npar = d.shape
        drop = float(self.opt("mh_drop", 0.5))
        max_runs = int(np.ceil(float(self.opt("MaxNumberOfBytes", 1e8)) / (npar + 2) / 8))
        mean, var, kept_n = np.zeros(npar), np.zeros((npar, npar)), 0
        thr = drop * T
        for i in range(nb):
            seen = 0
            for j in range(int(np.ceil(T / max_runs))):
                x2 = d[i, j * max_runs:min((j + 1) * max_runs, T)]
                nf = x2.shape[0]
                if seen + nf <= thr:
                    seen += nf
                    continue
                if seen < thr:
                    x2 = x2[int(np.ceil(thr - seen + 1)) - 1:]
                k = x2.shape[0]
                mean = (kept_n * mean + x2.sum(axis=0)) / (kept_n + k)
                dm = x2 - mean
                var = (kept_n * var + dm.T @ dm) / (kept_n + k)
                seen += nf
                kept_n += k
        post = self.oo_.get("posterior")
        met = R.rstruct("oo_.posterior.metropolis")
        met["mean"] = R.col(mean)
        met["Variance"] = R.mat(var)
        new = R.rstruct("oo_.posterior")
        if isinstance(post, Struct):
            new.update(post)
        post = new
        post["metropolis"] = met
        self.set_oo("posterior", post)

    # ------------------------------------------------------------------ functional forms
    def fn_resol(self, args, nargout):
        """[dr, info, params] = resol(check_flag, M_, options_, dr, ys, exo_steady_state, exo_det_steady_state):
        first-order rules at the parameters of the M_ passed (higher orders are refused)."""
        from .perturbation import first_order, BlanchardKahnError
        from .steady import SteadyStateError
        R = _R()
        if len(args) < 6 or not isinstance(args[1], Struct):
            raise Unsupported("resol: unsupported argument list", "results")
        Mst, ost = args[1], args[2]
        order = int(R.py_value(R.struct_get(ost, "order")) or 1)
        if order != 1:
            raise Unsupported(f"resol at order {order} is not supported (only order 1)", "results")
        m = self.model
        saved = m.params.copy()
        m.params[:] = np.asarray(farr(Mst["params"]), float).ravel(order="F")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ys0 = np.asarray(farr(args[4]), float).ravel(order="F")
                if ys0.size != m.n_endo:
                    ys0 = self._ss_start()
                xs0 = np.asarray(farr(args[5]), float).ravel(order="F")
                ys_, p_ = self._evaluate_ss(ys0, xs0 if xs0.size == m.n_exo else np.zeros(m.n_exo), True)   # resol.m line 75
                sol = first_order(m, ys=ys_, params=p_, qz_criterium=self._qz(),
                                  loglinear=bool(R.py_value(R.struct_get(ost, "loglinear"))))
            dr = self._make_dr(sol, sol.as_dr(), 1, self._qz(), ys=ys_)
            info = 0.0
        except BlanchardKahnError:
            dr, info = args[3], 3.0
        except SteadyStateError as exc:
            m.params[:] = saved
            raise R.ResultsError(f"resol: {exc}") from None
        params = np.asarray(farr(Mst["params"]), float)
        m.params[:] = saved
        return [dr, _F(info), R.col(params.ravel(order="F"))][:max(nargout, 1)]
