# This file is part of pydyn.
#
# It follows matlab/estimation/{PosteriorIRF, PosteriorIRF_core1, posterior_analysis, get_posterior_subsample,
# set_number_of_subdraws}.m, matlab/moments/compute_moments_varendo.m and matlab/convergence_diagnostics/mcmc_diagnostics.m
# of Dynare 7.2 (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org; GPL-3.0-or-later), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Posterior analyses of ``estimation`` in :func:`pydyn.dynare`: ``bayesian_irf`` (``oo_.PosteriorIRF.dsge``),
``moments_varendo`` (``oo_.PosteriorTheoreticalMoments.dsge``) and the MCMC convergence diagnostics
(``oo_.convergence``), with Dynare's order of random draws.

Dynare draws its sub-samples from the global uniform stream: ``PosteriorIRF`` takes one with ``get_posterior_subsample``
(always at random, even when the sub-sample is as large as the chains), ``posterior_analysis`` (``moments_varendo``) takes
one at its first call and re-uses it for the rest, and ``prior_posterior_statistics`` (smoother, forecasts) one more,
in this order.  pydyn replays them from the shared stream of the results (``Results.rng``)."""
from __future__ import annotations

import warnings

import numpy as np

from .matlab import Cell, Struct, Tainted, _F, farr


def _R():
    from . import results
    return results


class PosteriorAnalysisMixin:
    # ------------------------------------------------------------------ sub-draws
    def _number_of_subdraws(self, mc):
        """set_number_of_subdraws.m: options_.sub_draws (set in options_), ``B`` and the error flag."""
        nb, total = mc.draws.shape[0], mc.draws.shape[1]
        drop = float(self.opt("mh_drop", 0.5))
        per_chain = total - int(np.floor(drop * total))
        sub = self.opt("sub_draws", None)
        B = int(min(float(self.opt("posterior_max_subsample_draws", 1200)), np.ceil(per_chain * nb))) if sub is None else int(sub)
        self.set_options("sub_draws", np.float64(B))
        return B, B > per_chain * nb * nb

    def _post_sampler_ok(self, mc, what):
        R = _R()
        if getattr(mc, "final_rng", None) is None:
            raise R.ResultsError(f"estimation: {what} draws random numbers from the state the posterior sampler leaves; "
                                 "pydyn reproduces it only for the Metropolis-Hastings-type samplers")

    # ------------------------------------------------------------------ bayesian_irf
    def _bayesian_irf(self, est, mc, st_vars_given, B):
        """PosteriorIRF('posterior', ...): oo_.PosteriorIRF.dsge.{Mean, Median, Var, deciles, HPDinf, HPDsup}.<variable>_<shock>."""
        from .posterior import irf_at, moments_along_draws, irf_shocks as _irf_shocks, _iround
        R = _R()
        if self.opt_flag("dsge_var"):
            raise R.ResultsError("estimation: bayesian_irf with dsge_var (the DSGE-VAR IRFs draw inverse-Wishart numbers) "
                                 "is not supported by pydyn.dynare()")
        if int(self.opt("order", 1) or 1) != 1:
            raise R.ResultsError("estimation: bayesian_irf at order > 1 is not supported by pydyn.dynare()")
        self._post_sampler_ok(mc, "bayesian_irf")
        m = self.model
        periods = int(self.opt("irf", 0) or 0) or 40
        relative = self.opt_flag("relative_irf")
        mcs = float(self.opt("mh_conf_sig", 0.9))
        variables = list(st_vars_given) if st_vars_given else list(m.varobs)
        unknown = [v for v in variables if v not in m._ie]
        if unknown:
            raise R.ResultsError(f"estimation: PosteriorIRF :: {unknown[0]} is not a declared endogenous variable!")
        names = self.opt("irf_shocks", None)
        if isinstance(names, str):
            names = [names]
        try:
            shocks = _irf_shocks(est, list(names) if names else [])
        except ValueError as exc:
            raise R.ResultsError(f"estimation: {exc}") from None
        params = self._dynare_subsample(mc, B, mc.draws.shape[2], exact_random=True)
        stock = np.zeros((params.shape[0], periods, len(variables), len(m.exo_names)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for k, theta in enumerate(params):
                    stock[k] = irf_at(est, theta, periods, relative, shocks, variables)
        except Exception as exc:
            raise R.ResultsError(f"estimation: PosteriorIRF :: Dynare is unable to solve the model ({exc}) with sample posterior") from None
        mom = moments_along_draws(stock[:, :, :, shocks], mcs)
        stats = {k: R.rstruct(f"oo_.PosteriorIRF.dsge.{k}") for k in ("Mean", "Median", "Var", "deciles", "HPDinf", "HPDsup")}
        for a, i in enumerate(shocks):
            for j, v in enumerate(variables):
                name = f"{v}_{m.exo_names[i]}"
                stats["Mean"][name] = R.col(mom["mean"][:, j, a])
                stats["Median"][name] = R.col(mom["median"][:, j, a])
                stats["Var"][name] = R.col(mom["variance"][:, j, a])
                stats["deciles"][name] = R.mat(mom["deciles"][:, :, j, a].T)
                stats["HPDinf"][name] = R.col(mom["hpd_inf"][:, j, a])
                stats["HPDsup"][name] = R.col(mom["hpd_sup"][:, j, a])
        dsge = R.rstruct("oo_.PosteriorIRF.dsge")
        dsge.update(stats)
        top = R.rstruct("oo_.PosteriorIRF")
        top["dsge"] = dsge
        self.set_oo("PosteriorIRF", top)
        self.fig_posterior_irf(mom, shocks, variables, periods)

    # ------------------------------------------------------------------ moments_varendo
    def _moments_varendo(self, est, mc, var_list, B):
        """compute_moments_varendo('posterior', ...): oo_.PosteriorTheoreticalMoments.dsge (draws taken once, at the first
        posterior_analysis call, and shared by all the moments)."""
        from .posterior_moments import posterior_theoretical_moments
        R = _R()
        if int(self.opt("order", 1) or 1) != 1:
            raise R.ResultsError("estimation: moments_varendo at order > 1 is not supported by pydyn.dynare()")
        self._post_sampler_ok(mc, "moments_varendo")
        if self.opt_flag("one_sided_hp_filter"):
            return                                     # compute_moments_varendo skips the computations
        steps = self.opt("conditional_variance_decomposition", None)
        steps = [] if steps is None else [int(s) for s in np.atleast_1d(steps)]
        params = self._dynare_subsample(mc, B, mc.draws.shape[2], exact_random=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = posterior_theoretical_moments(
                    est, params=params, var_list=list(var_list), ar=int(self.opt("ar", 5)), steps=steps,
                    nodecomposition=self.opt_flag("nodecomposition"),
                    contemporaneous_correlation=self.opt_flag("contemporaneous_correlation"),
                    mh_conf_sig=float(self.opt("mh_conf_sig", 0.9)),
                    density=self.opt_flag("estimation.moments_posterior_density.indicator"))     # the *_mc_analysis files use the default kernel
        except (NotImplementedError, ValueError, RuntimeError) as exc:
            raise R.ResultsError(f"estimation: moments_varendo: {exc}") from None
        self.set_oo("PosteriorTheoreticalMoments", R.struct_set(R.rstruct("oo_.PosteriorTheoreticalMoments"), "dsge",
                                                                 self._moments_struct(res)))

    def _moments_struct(self, res):
        R = _R()
        dsge = R.rstruct("oo_.PosteriorTheoreticalMoments.dsge")
        for fam, tree in res.oo().items():
            tag = f"oo_.PosteriorTheoreticalMoments.dsge.{fam}"
            s = R.rstruct(tag)
            for stat, byvar in tree.items():
                if stat == "Steps":
                    s["Steps"] = R.mat(np.asarray(byvar, float)) if np.size(byvar) != 1 else _F(float(np.ravel(byvar)[0]))
                    continue
                t1 = R.rstruct(f"{tag}.{stat}")
                for a, d2 in byvar.items():
                    t2 = R.rstruct(f"{tag}.{stat}.{a}")
                    for b, val in d2.items():
                        t2[b] = self._moment_value(stat, val, fam.startswith("Conditional"))
                    t1[a] = t2
                s[stat] = t1
            dsge[fam] = s
        return dsge

    @staticmethod
    def _moment_value(stat, val, conditional=False):
        R = _R()
        if isinstance(val, list):                      # correlation: one entry per lag
            return Cell([_PostValue.one(stat, np.asarray(x), False) for x in val], (len(val), 1))
        return _PostValue.one(stat, np.asarray(val), conditional)


class _PostValue:
    @staticmethod
    def one(stat, a, row=False):
        R = _R()
        if a.ndim == 0:
            return _F(float(a))
        if stat == "deciles" and a.ndim == 2:          # (steps, 9) -> Dynare's (9, steps)
            return R.mat(a.T)
        if stat == "density" and a.ndim == 3:          # (steps, grid, 2) -> (grid, 2, steps)
            return R.mat(np.moveaxis(a, 0, 2))
        if a.ndim == 1 and row:                        # conditional decompositions: one value per horizon, as a row
            return R.row(a)
        return R.mat(a) if a.ndim >= 2 else R.col(a)


class StructArray(Tainted):
    """A MATLAB struct array held as a list of structs (``items``).  The interpreter does not index struct arrays: MATLAB
    code that reads one is refused (this is a ``Tainted`` value); Python code reads ``.items``."""
    __slots__ = ("items",)

    def __init__(self, items, why):
        super().__init__(why)
        self.items = list(items)


def _dynare_parameter_name(est, name):
    """get_the_name.m: the name a parameter has in the convergence diagnostics."""
    m = est.model
    if name.startswith("stderr "):
        v = name[7:]
        return f"SE_{v}" if v in m.exo_names else f"SE_EOBS_{v}"
    if name.startswith("corr "):
        a, b = [x.strip() for x in name[5:].split(",")]
        return f"CC_{a}_{b}" if a in m.exo_names else f"CC_EOBS_{a}_{b}"
    return name


class ConvergenceMixin:
    def _mcmc_diagnostics(self, est, mc):
        """mcmc_diagnostics.m: oo_.convergence.geweke(chain).<parameter> and, with raftery_lewis_diagnostics,
        oo_.convergence.raftery_lewis(chain) (stored only for chains of more than 2000 draws, as in Dynare)."""
        from . import posterior as _post
        R = _R()
        method = self.opt("posterior_sampler_options.posterior_sampling_method", "random_walk_metropolis_hastings")
        if self.opt_flag("nodiagnostic") or method in ("hssmc", "dsmh", "online", "dime_mcmc", "dime"):
            return
        nb, T, npar = mc.draws.shape
        if method != "slice" and T <= 2000:
            return                                   # "MCMC convergence diagnostics are not computed ..."
        names = [_dynare_parameter_name(est, n) for n in est.names]
        gi = self.opt("convergence.geweke.geweke_interval", [0.2, 0.5])
        gi = np.atleast_1d(np.asarray(gi, float))
        ts = [int(t) for t in np.atleast_1d(self.opt("convergence.geweke.taper_steps", [4, 8, 15]))]
        drop = float(self.opt("mh_drop", 0.5))
        gw = _post.geweke(mc, tuple(gi), tuple(ts), drop)
        items = []
        for b in range(nb):
            st = R.rstruct("oo_.convergence.geweke")
            for j, nm in enumerate(names):
                e = R.rstruct(f"oo_.convergence.geweke.{nm}")
                for k, arr in gw["moments"].items():
                    e[k] = _F(float(arr[b, j]))
                e["pooled_mean"] = R.row(gw["pooled_mean"][b, j])
                e["pooled_nse"] = R.row(gw["pooled_nse"][b, j])
                e["prob_chi2_test"] = R.row(gw["prob_chi2_test"][b, j])
                st[nm] = e
            items.append(st)
        conv = R.rstruct("oo_.convergence")
        conv["geweke"] = StructArray(items, "oo_.convergence.geweke is a struct array")
        if self.opt_flag("convergence.rafterylewis.indicator"):
            qrs = self.opt("convergence.rafterylewis.qrs", [0.025, 0.005, 0.95])
            rl = _post.raftery_lewis_diagnostics(mc, qrs)
            ritems = []
            for b in range(nb):
                r = rl[b]
                e = R.rstruct("oo_.convergence.raftery_lewis")
                for k in ("M_burn", "N_prec", "k_thin", "k_ind", "I_stat", "N_total"):
                    e[k] = R.col(r[k])
                e["N_min"] = _F(float(r["N_min"]))
                e["parameter_names"] = R.cellstr(names)
                ritems.append(e)
            conv["raftery_lewis"] = StructArray(ritems, "oo_.convergence.raftery_lewis is a struct array")
        self.set_oo("convergence", conv)
        if nb > 1:
            self.fig_brooks_gelman(est, mc, names)


class DiscretionaryMixin:
    def st_discretionary_policy(self, st):
        """discretionary_policy.m: M_.instr_id, stoch_simul at order 1 with the rules of discretionary_policy_1
        (oo_.dr has order_var, inv_order_var, ys, ghu, ghx), oo_.steady_state = oo_.dr.ys and
        oo_.planner_objective_value (evaluate_planner_objective)."""
        R = _R()
        m = self.model
        m._exec_statement(st)
        self._refresh_options(st)
        self.set_options("discretionary_policy", np.bool_(True))        # written by the preprocessor, never reset
        self.pull()
        if self.opt_flag("loglinear"):
            raise R.ResultsError("discretionary_policy is not compatible with `loglinear` option set to 1")
        from .optimal_policy import _instrument_names
        try:
            names = _instrument_names(m)
        except Exception as exc:
            raise R.ResultsError(f"discretionary_policy: {exc}") from None
        instr_nbr = m.n_endo - m.jac_dims[0]
        if instr_nbr == 0:
            raise R.ResultsError("discretionary_policy:: There are no available instruments, because the model has as many "
                                 "equations as variables.")
        if len(names) != instr_nbr:
            raise R.ResultsError("discretionary_policy:: There are " + ("fewer" if len(names) < instr_nbr else "more")
                                 + " declared instruments than omitted equations.")
        if any(v not in m._ie for v in names):
            raise R.ResultsError("discretionary_policy:: instrument " + next(v for v in names if v not in m._ie) + " not found")
        self.set_M("instr_id", R.col([m._ie[v] + 1 for v in names]))
        order0 = self.opt_raw("order")
        self.set_options("order", np.float64(1))
        try:
            self.stoch_simul_run(list(st.get("symbol_list") or []))
            self.set_oo("steady_state", R.col(farr(self.oo_["dr"]["ys"])))
            self.st_evaluate_planner_objective({**st, "options": {}})
        finally:
            if order0 is not None:
                self.set_options("order", order0)

    def _discretionary_solution(self, qz):
        """discretionary_policy_1.m at the live calibration: (sol, dr) for stoch_simul_run."""
        from .optimal_policy import discretionary_policy, OptimalPolicyError
        R = _R()
        try:
            res = discretionary_policy(self.model, qz_criterium=qz, tol=self.opt("discretionary_tol", None),
                                       maxit=self.opt("dp.maxit", None), welfare=False)
        except (OptimalPolicyError, NotImplementedError, ValueError) as exc:
            raise R.ResultsError(f"stoch_simul: {exc}") from None
        return res

    def _discretionary_dr(self, res):
        """oo_.dr as discretionary_policy_1 leaves it."""
        R = _R()
        dr = res.dr
        n = self.model.n_endo
        order_var = np.asarray(dr["order_var"], int)
        inv = np.empty(n, int)
        inv[order_var] = np.arange(n)
        D = R.rstruct("oo_.dr")
        D["order_var"] = R.col(order_var + 1)
        D["inv_order_var"] = R.row(inv + 1)
        D["ys"] = R.col(np.asarray(dr["ys"], float))
        D["ghu"] = R.mat(np.asarray(dr["ghu"], float))
        D["ghx"] = R.mat(np.asarray(dr["ghx"], float))
        return D


class ModeCheckMixin:
    def _mode_check(self, est, theta):
        """``mode_check.m`` (called by dynare_estimation_1.m when ``mode_check`` is set): writes
        ``<dname>/graphs/<fname>_check_plot_data.mat`` (``mcheck.cross.<name> = [z, log posterior, log-lik kernel]``,
        ``mcheck.emode.<name>`` = the mode, whatever ``nograph``) and draws the check plots."""
        from . import posterior as _post
        o = est.options
        theta = np.asarray(theta, float)
        nolik = self.opt_flag("mode_check.nolik")
        ll = float(o["mode_check_neighbourhood_size"])
        sym = bool(o["mode_check_symmetric_plots"]) and not np.isinf(ll)
        npts = o["mode_check_number_of_points"]
        data = _post.mode_check(est, theta, ll, sym, npts, as_dynare=True)
        cross, emode, items = {}, {}, []
        for k, name in enumerate(est.names):
            nam = _dynare_parameter_name(est, name)
            y = data[name]
            cross[nam] = y[:, :2] if nolik else y
            emode[nam] = float(theta[k])
            _, bl, bu = _post.mode_check_grid(theta[k], est.lb[k], est.ub[k], est.prior.p2[k], ll, sym, npts,
                                              o["huge_number"], True)
            bound = est.lb[k] if bl else (est.ub[k] if bu else None)
            items.append((nam, y[:, 0], y[:, 1:2] if nolik else y[:, 1:3], bl, bu, bound))
        self.gfx.write_mat(f"{self.name}/graphs/{self.name}_check_plot_data.mat",
                           {"mcheck": {"cross": cross, "emode": emode}})
        self.fig_mode_check(items, nolik, emode)


def _g16(x):
    """fprintf's ``%.16g`` as Octave prints it (NaN, Inf, -Inf for the non-finite values)."""
    x = float(x)
    if x != x:
        return "NaN"
    if x in (float("inf"), float("-inf")):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.16g}"


class FileOutputMixin:
    def fn_save_params_and_steady_state(self, args, nargout):
        """``save_params_and_steady_state(filename)``: the file of ``name value`` lines (parameters, endogenous variables,
        exogenous variables and deterministic exogenous variables, ``%.16g``), written in the .mod file's folder."""
        import re
        from .matlab import MatlabError, Unsupported, farr
        if nargout > 0:
            raise MatlabError("Invalid call to save_params_and_steady_state")
        if len(args) != 1 or not isinstance(args[0], str):
            raise Unsupported("save_params_and_steady_state with a computed (non-char) file name", "file")
        self._save_pss(args[0])
        return []

    def st_save_params_and_steady_state(self, st):
        name = st.get("filename")
        if not isinstance(name, str):
            raise _R().ResultsError("save_params_and_steady_state: the statement carries no file name")
        self._save_pss(name)

    def _save_pss(self, name):
        import re
        from .matlab import Unsupported, farr
        R = _R()
        if not re.fullmatch(r"[A-Za-z0-9_][\w.-]*", name) or ".." in name:
            raise Unsupported(f"save_params_and_steady_state: {name!r} is not a plain file name in the .mod file's folder", "file")
        self.pull()
        M, oo = self.M_, self.oo_
        lines = []

        def block(names, values, what):
            names = [str(s) for s in names.items] if hasattr(names, "items") else []
            vals = np.atleast_1d(np.asarray(farr(values), float)).ravel(order="F") if values is not None else np.zeros(0)
            if len(vals) < len(names):
                raise R.ResultsError(f"save_params_and_steady_state: oo_.{what} has {len(vals)} values for {len(names)} names")
            lines.extend(f"{n} {_g16(v)}" for n, v in zip(names, vals))
        block(M["param_names"], M["params"], "params")
        block(M["endo_names"], oo.get("steady_state"), "steady_state")
        block(M["exo_names"], oo.get("exo_steady_state"), "exo_steady_state")
        det = M.get("exo_det_names")
        if det is not None and hasattr(det, "items") and len(det.items):
            block(det, oo.get("exo_det_steady_state"), "exo_det_steady_state")
        if self.gfx.write_text(name, "\n".join(lines) + "\n") is None:
            raise Unsupported(f"save_params_and_steady_state: {name} exists and was not written by pydyn.dynare()", "file")


class EstimationChecksMixin:
    def _initial_estimation_checks(self, est):
        """The errors of ``initial_estimation_checks.m`` that depend on the data, the shocks and the options: the numbers of
        observed variables against shocks (and measurement errors), the stochastic-singularity checks on the first period,
        ``mh_drop``, ``mh_nblocks`` and a prior that is -Inf at the initial values.  Dynare's messages are the errors'
        MATLAB messages."""
        R = _R()

        def fail(msg):
            raise R.ResultsError(f"estimation: {msg}", matlab_message=msg)
        o = est.options
        x0 = np.asarray(est.x0, float)
        if np.any(np.asarray(est.prior.shape) > 0):      # check_prior_bounds.m (dynare_estimation_init.m)
            out = [i for i in range(len(x0)) if x0[i] < est.lb[i] or x0[i] > est.ub[i]]
            if out:
                nms = [_dynare_parameter_name(est, est.names[i]) for i in out]
                msg = f"Initial value(s) of {', '.join(nms)} are outside parameter bounds.\nPotentially, you should set"
                cc, sk = any("CC_" in n for n in nms), any("SKEW_" in n for n in nms)
                if cc:
                    msg += "\n- correlations to be within theoretical bound of [-1;1]"
                if sk:
                    msg += ("\n- skewness coefficients to be within theoretical bound of [-0.9953;0.9953] of the skew normal "
                            "distribution")
                msg += "\n- prior_trunc = 0\n" if (cc or sk) else " prior_trunc = 0. "
                fail(msg + "If you used the mode_file-option, check whether your mode-file is consistent with the priors.")
            zero_ig = [i for i in range(len(x0)) if est.prior.shape[i] == 4 and x0[i] == 0]
            if zero_ig:
                nms = ", ".join(_dynare_parameter_name(est, est.names[i]) for i in zero_ig)
                fail(f"Initial value(s) of {nms} is zero. This is not allowed when using an inverse gamma prior.\n")
        if int(o.get("order", 1) or 1) > 1:
            return                                    # the particle-filter checks are not followed here
        data = np.asarray(est.data, float)
        valid = ~np.isnan(data)
        per_period = valid.sum(axis=1)
        max_non_missing = int(per_period[1:].max()) if per_period.size > 1 else 0
        init_non_missing = int(per_period[0]) if per_period.size else 0
        cal = est.calibration(est.x0)
        Sigma = np.atleast_2d(np.asarray(cal["Sigma_e"], float))
        H = cal.get("H")
        H = np.zeros((0, 0)) if H is None else np.atleast_2d(np.asarray(H, float))
        nvarobs = len(self.model.varobs)
        rows = set(np.flatnonzero(np.any(H != 0, axis=1)).tolist()) if H.size else set()
        for n in est.names:
            if n.startswith("stderr ") and n[7:] not in self.model.exo_names and n[7:] in self.model.varobs:
                rows.add(self.model.varobs.index(n[7:]))
        non_zero_me = len(rows)
        n_exo = len(self.model.exo_names)
        n_shocks = int(np.count_nonzero(np.diag(Sigma)))
        first_ok = bool(self.opt_flag("no_init_estimation_check_first_obs"))
        if max_non_missing > n_exo + non_zero_me:
            fail("initial_estimation_checks:: Estimation can't take place because there are less declared shocks than "
                 "observed variables!")
        if init_non_missing > n_exo + non_zero_me and not first_ok:
            fail("initial_estimation_checks:: Estimation can't take place because there are less declared shocks than "
                 "observed variables in first period!")
        if not self.opt_flag("heteroskedastic_filter"):
            if max_non_missing > n_shocks + non_zero_me:
                fail("initial_estimation_checks:: Estimation can't take place because too many shocks have been calibrated "
                     "with a zero variance!")
        if init_non_missing > n_shocks + non_zero_me and not first_ok:
            fail("initial_estimation_checks:: Estimation can't take place because too many shocks have been calibrated "
                 "with a zero variance in first period!")
        bayesian = bool(np.any(np.asarray(est.prior.shape) > 0))
        if bayesian and float(self.opt("mh_replic", 0) or 0) > 0 and int(self.opt("mh_nblck", 2) or 0) < 1:
            fail("initial_estimation_checks:: Bayesian estimation cannot be conducted with mh_nblocks=0.")
        drop = float(self.opt("mh_drop", 0.5))
        if drop < 0 or drop >= 1:
            fail("initial_estimation_checks:: mh_drop must be in [0,1).")
        if bayesian and not np.isfinite(est.log_prior(np.asarray(est.x0, float))):
            fail("The initial value of the prior is -Inf")
