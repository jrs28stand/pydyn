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


def _R():
    from . import results
    return results


class EstimationMixin:
    # ------------------------------------------------------------------ smoother results
    def _store_smoother(self, res, nk_filtered):
        """oo_.SmoothedVariables, ... as matlab/kalman/store_smoother_results.m writes them (columns, T x 1)."""
        R = _R()
        m = self.model
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
        for v in m.endo_names:
            if v in res.Constant:
                const[v] = R.col(res.Constant[v])
        sm["Constant"] = const
        init = R.rstruct("oo_.Smoother.Init_State")
        for v in m.endo_names:
            if v in res.Init_State:
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
                if v in src:
                    s[v] = R.col(src[v])
            self.set_oo(field, s)
        if nk_filtered and res.FilteredVariables:
            s = R.rstruct("oo_.FilteredVariables")
            for v in m.endo_names:
                if v in res.FilteredVariables:
                    s[v] = R.col(res.FilteredVariables[v])
            self.set_oo("FilteredVariables", s)
            if res.FilteredVariablesKStepAhead is not None:     # (steps, n_endo, T + max step), declaration order
                self.set_oo("FilteredVariablesKStepAhead", R.mat(res.FilteredVariablesKStepAhead))
                if res.FilteredVariablesKStepAheadVariances is not None:
                    self.set_oo("FilteredVariablesKStepAheadVariances", R.mat(res.FilteredVariablesKStepAheadVariances))
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

    # ------------------------------------------------------------------ calib_smoother
    def st_calib_smoother(self, st):
        from .smoother import smooth, SmootherError
        R = _R()
        self._refresh_options(st)
        if "parameter_set" not in (st.get("options") or {}):
            self.set_options("parameter_set", "calibration")     # the preprocessor's CalibSmootherStatement
        self.pull()
        self._estimation_order(st, "calib_smoother", "estimation")
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
        self._set_prior_if_any()
        self._store_smoother(res, bool(self.opt_flag("filtered_vars")))
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
        self.refuse("oo_.dr", ["restrict_var_list", "restrict_columns"], "state-space restriction of dr1.m")

    # ------------------------------------------------------------------ estimation
    def st_estimation(self, st):
        from .estimation import Estimation, EstimationError
        from .smoother import smooth, SmootherError
        R = _R()
        self._refresh_options(st)
        self.pull()
        self._estimation_order(st, "estimation")
        m = self.model
        mh_replic = int(np.max(np.atleast_1d(self.opt("mh_replic", 0) or 0)))
        for key in ("load_mh_file", "mh_recover", "bayesian_irf", "moments_varendo", "forecast", "particle.status",
                    "mode_check.status", "mh_posterior_mode_estimation"):
            if self.opt_flag(key):
                raise R.ResultsError(f"estimation option {key} is not supported by pydyn.dynare()")
        self._refuse_unfaithful_smoother()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = Estimation(m)
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
        self._set_prior(est)
        mc = None
        if mode is not None:
            self._set_mode(est, mode, mode.x)
        if mh_replic > 0:
            if mode is None:
                raise R.ResultsError("estimation with mh_replic>0 needs a mode (mode_compute or mode_file)")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mc = est.sample(mode=mode, rng="octave")
                    theta = self._set_posterior(est, mc, mode)
            except (EstimationError, NotImplementedError, ValueError) as exc:
                raise R.ResultsError(f"estimation: {exc}") from None
        cal = est.calibration(theta)
        m.params[:] = np.asarray(cal["params"], float)
        m.Sigma_e = np.asarray(cal["Sigma_e"], float)
        self.push()
        Hc = cal.get("H")
        if Hc is not None and np.any(np.asarray(Hc, float) != 0):
            self.set_M("H", R.mat(np.atleast_2d(np.asarray(Hc, float))))
        self._solve_dr_at_current()
        self._steady_at_init(est)
        if mc is not None:
            self._set_metropolis(est, mc)
        bayes_smoother = mc is not None and (self.opt_flag("smoother") or self.opt_flag("filtered_vars")
                                              or bool(self.opt("filter_step_ahead")) or bool(self.opt("forecast")))
        if bayes_smoother:
            for f in ("SmoothedVariables", "UpdatedVariables", "FilteredVariables", "SmoothedShocks",
                      "SmoothedMeasurementErrors", "Smoother", "Constant", "forecast", "PointForecast", "MeanForecast"):
                self._drop_oo(f)
            self.refuse("oo_", ["SmoothedVariables", "UpdatedVariables", "FilteredVariables", "SmoothedShocks",
                                "SmoothedMeasurementErrors", "Smoother", "Constant", "forecast", "PointForecast",
                                "MeanForecast"],
                        "the posterior distributions of smoothed objects and forecasts use random sub-draws from "
                        "Octave's rand stream, which pydyn does not reproduce (pydyn.posterior_moments computes them "
                        "with numpy's generator)")
        else:                         # options_.frequentist_smoother is true by default: classical smoother at the estimates
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = smooth(m, estimation=est, theta=theta)
            except (SmootherError, NotImplementedError) as exc:
                raise R.ResultsError(f"estimation: smoother: {exc}") from None
            self._store_smoother(res, bool(self.opt_flag("filtered_vars")) or bool(self.opt("filter_step_ahead")))
        self.set_oo("gui", R.struct_set(self.oo_["gui"], "ran_estimation", np.bool_(True)))

    def _steady_at_init(self, est):
        """oo_.steady_state as dynare_estimation_init leaves it: the steady state at the initial values of the
        estimated parameters (M_.params and oo_.dr are at the calibration, or the mode)."""
        from .steady import steady_state
        R = _R()
        m = self.model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = est.calibration(est.x0)
            xs = np.array([m._oo.get(v, 0.0) for v in m.exo_names], float)
            ys, _ = steady_state(m, params=np.asarray(cal["params"], float).copy(), exo=xs)
        self.set_oo("steady_state", R.col(ys))

    def _set_prior_if_any(self):
        """oo_.prior and oo_.steady_state of calib_smoother with an estimated_params block."""
        from .estimation import Estimation
        m = self.model
        if not any(st.get("statementName") == "estimated_params" for st in m.statements):
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = Estimation(m)
            self._steady_at_init(est)
        except Exception:                       # a by-product here; the smoother does not need it
            return
        self._set_prior(est)

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

    def _set_mode(self, est, mode, theta):
        R = _R()
        ml = bool(np.all(np.asarray(est.prior.shape) == 0))
        post = R.rstruct("oo_.posterior")
        opt = R.rstruct("oo_.posterior.optimization")
        opt["mode"] = R.col(theta)
        opt["Variance"] = R.mat(mode.inv_hessian) if mode.inv_hessian is not None else np.zeros((0, 0))
        opt["log_density"] = _F(float(mode.log_posterior))
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
        if order > 1:
            raise R.ResultsError("forecast at order > 1 draws random numbers, which pydyn.dynare() does not "
                                 "reproduce from Dynare's stream")
        sopts = st.get("options") or {}
        periods = int(sopts.get("periods", 0) or 0) or 5          # the statement's own periods (options_.forecast is reset)
        var_list = list(st.get("symbol_list") or [])
        if not var_list and isinstance(self.oo_.get("var_list"), Cell):
            var_list = [str(x) for x in self.oo_["var_list"].items]
        kw = {}
        cs = sopts.get("forecasts.conf_sig", sopts.get("conf_sig"))
        if cs is not None:
            kw["conf_sig"] = float(cs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = forecast(m, var_list or None, periods=periods, order=order,
                              pruning=self.opt_flag("pruning"), **kw)
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

    # ------------------------------------------------------------------ shock decompositions
    def _decomposition_smoother(self, kind, run=True):
        """The smoother of Dynare's evaluate_smoother(parameter_set, ...) inside shock_decomposition.m and
        initial_condition_decomposition.m; only parameter_set = 'calibration' is followed."""
        from .smoother import smooth, SmootherError
        R = _R()
        if not isinstance(self.oo_.get("SmoothedVariables"), Struct):
            raise R.ResultsError(f"{kind}: no smoothed variables: run estimation(smoother) or calib_smoother first")
        ps = self.opt("parameter_set", None)
        if ps is None:
            if any(k in self.oo_ for k in ("posterior_mean", "mle_mode", "posterior")):
                raise R.ResultsError(f"{kind}: the default parameter set (posterior mean or mode) is not supported by "
                                     "pydyn.dynare(): give parameter_set=calibration")
            msg = "shock_decomposition: option parameter_set is not specified and posterior mode is not available"
            raise R.ResultsError(f"{kind}: {msg}", matlab_message=msg)
        if ps != "calibration":
            raise R.ResultsError(f"{kind}: parameter_set={ps} is not supported by pydyn.dynare() (only calibration)")
        self._refuse_unfaithful_smoother()
        self.pull()
        if not run:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return smooth(self.model)
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
        if self.opt("shock_decomp.forecast_type", None):
            raise R.ResultsError(f"{kind}: forecast_type is not supported by pydyn.dynare()")
        if self.opt_flag("shock_decomp.with_epilogue"):
            raise R.ResultsError(f"{kind}: with_epilogue is not supported by pydyn.dynare()")
        plot = not self.opt_flag("no_graph.shock_decomposition")
        if plot and (self.opt("plot_shock_decomp.init2shocks", None) or self.opt_flag("plot_shock_decomp.realtime")):
            raise R.ResultsError(f"{kind}: plot_shock_decomposition with init2shocks or realtime writes more to oo_ "
                                 "than pydyn.dynare() follows")
        res = self._decomposition_smoother(kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sd = shock_decomposition(self.model, res, init_state=int(self.opt("shock_decomp.init_state", 0) or 0))
        self.set_oo("shock_decomposition", R.mat(sd.z))
        self.unrefuse("oo_", ["shock_decomposition"])
        self._decomp_info("parameter_set", "calibration")
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
        self.set_oo("gui", R.struct_set(self.oo_["gui"], "ran_shock_decomposition", np.bool_(True)))

    def st_initial_condition_decomposition(self, st):
        """initial_condition_decomposition.m: oo_.initval_decomposition (endo_nbr x endo_nbr + 2 x T)."""
        from .smoother import initial_condition_decomposition
        R = _R()
        self._refresh_options(st)
        kind = "initial_condition_decomposition"
        if self.opt_flag("initial_condition_decomp.with_epilogue"):
            raise R.ResultsError(f"{kind}: with_epilogue is not supported by pydyn.dynare()")
        if "initval_decomposition" in self.oo_:
            return                  # Dynare keeps the decomposition it has (and only plots it again)
        res = self._decomposition_smoother(kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            icd = initial_condition_decomposition(self.model, res)
        self.set_oo("initval_decomposition", R.mat(icd.z))
        self.unrefuse("oo_", ["initval_decomposition"])
        self._decomp_info("initval_decomposition.parameter_set", "calibration")

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

    def fn_unpack_simulations(self, args, nargout):
        R = _R()
        m = self.model
        occ = self.oo_.get("occbin")
        if not isinstance(occ, Struct) or "simul" not in occ:
            raise R.ResultsError("occbin.unpack_simulations: oo_.occbin.simul is empty: run occbin_solver first")
        sim = occ["simul"]
        lin, pw, ys = (np.asarray(R.farr(sim[k]), float) for k in ("linear", "piecewise", "ys"))
        ys = ys.ravel(order="F")
        out = {"endo_linear": R.rstruct("oo_.occbin.endo_linear"), "endo_piecewise": R.rstruct("oo_.occbin.endo_piecewise"),
               "endo_ss": R.rstruct("oo_.occbin.endo_ss")}
        for i, v in enumerate(m.endo_names):
            out["endo_linear"][v] = R.col(lin[:, i])
            out["endo_piecewise"][v] = R.col(pw[:, i])
            out["endo_ss"][v] = _F(float(ys[i]))
        ex = R.rstruct("oo_.occbin.exo")
        sh = self._occbin_shocks
        pos = np.asarray(R.farr(sim["exo_pos"]), float).ravel(order="F").astype(int) - 1
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
                pv = evaluate_planner_objective(self.model, order=int(self.opt("order", 2)))
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
        md["ModifiedHarmonicMean"] = _F(float(modified_harmonic_mean(mc, drop=est.options.get("mh_drop")).value))
        self.set_oo("MarginalDensity", md)
        return np.asarray(stats.mean, float)

    def _set_metropolis(self, est, mc):
        R = _R()
        kept = mc.kept(est.options.get("mh_drop"))
        post = self.oo_.get("posterior")
        met = R.rstruct("oo_.posterior.metropolis")
        met["mean"] = R.col(kept.mean(axis=0))
        met["Variance"] = R.mat(np.cov(kept.T, bias=False))
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
                sol = first_order(m, qz_criterium=self._qz(), loglinear=bool(R.py_value(R.struct_get(ost, "loglinear"))))
            dr = self._make_dr(sol, sol.as_dr(), 1, self._qz())
            info = 0.0
        except BlanchardKahnError:
            dr, info = args[3], 3.0
        params = np.asarray(farr(Mst["params"]), float)
        m.params[:] = saved
        return [dr, _F(info), R.col(params.ravel(order="F"))][:max(nargout, 1)]
