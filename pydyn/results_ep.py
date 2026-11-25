# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2 (matlab/ep/extended_path.m, extended_path_initialization.m, extended_path_shocks.m and the
# ExtendedPathStatement of the preprocessor; Copyright (C) Dynare Team, GPL-3.0-or-later).
"""``extended_path`` (statement and function) on the live workspace of :mod:`pydyn.results`."""
from __future__ import annotations

import importlib

import numpy as np

from . import dseries as _ds
from .dates import Date as _Date
from . import extended_path as _ep
from . import extended_path_stoch as _sep
from . import matlab as _ml
from .matlab import MatlabError, Struct, Unsupported, farr, _F


def _R():
    from . import results
    return results


def _pfmod():
    return importlib.import_module(__package__ + ".perfect_foresight")


def _empty(v):
    return isinstance(v, np.ndarray) and v.size == 0


class ExtendedPathMixin:
    # ------------------------------------------------------------------------------ the statement
    def st_extended_path(self, st):
        """``extended_path(periods=N, ...)``: the options are written into ``options_`` (``periods`` is the sample size), then
        ``[Simulated_time_series, oo_] = extended_path([], N, [], options_, M_, oo_)``."""
        opts = dict(st.get("options", {}) or {})
        periods = opts.pop("periods", None)
        if periods is None:
            raise _R().ResultsError("the 'periods' option of 'extended_path' is mandatory")
        self._refresh_options({"options": opts})
        hyb = opts.get("ep.stochastic.hybrid_order")
        if hyb is not None and int(float(hyb)) > 2:
            self.set_options("k_order_solver", np.bool_(True))
        self.pull()
        ts, oo = self._extended_path(np.zeros((0, 0)), float(periods), np.zeros((0, 0)))
        self.set_global("oo_", oo)
        self.vars["Simulated_time_series"] = ts

    # ------------------------------------------------------------------------------ the function
    def fn_extended_path(self, args, nargout):
        """``[ts, oo_] = extended_path(initialconditions, samplesize, exogenousvariables, options_, M_, oo_)``."""
        if len(args) != 6:
            raise MatlabError("Invalid call to extended_path")
        if not all(isinstance(a, Struct) for a in args[3:6]):
            raise MatlabError("extended_path: the last three arguments must be options_, M_ and oo_")
        saved = {k: self.I.globals[k] for k in ("M_", "options_", "oo_")}
        try:
            for k, v in zip(("options_", "M_", "oo_"), args[3:6]):
                self.set_global(k, v)
            self.pull()
            ts, oo = self._extended_path(args[0], args[1], args[2])
        finally:
            for k, v in saved.items():
                self.I.globals[k] = v
        return [ts, oo][:max(nargout, 1)]

    # ------------------------------------------------------------------------------ the work
    def _extended_path(self, initial, samplesize, exogenous):
        R = _R()
        pf = _pfmod()
        _ds.HOST = self
        m = self.model
        n, nx = m.n_endo, m.n_exo
        info = m.info
        if m.n_exo_det:
            raise MatlabError("Extended path does not support varexo_det.")
        sample = _ml.scalar(samplesize, "extended_path: samplesize")
        if sample != int(sample) or sample < 1:
            raise MatlabError("extended_path: samplesize must be a positive integer")
        sample = int(sample)
        if info["maximum_lag"] == 0:
            raise MatlabError("(Stochastic) Extended path is not supported for purely forward-looking models.")
        o_ = self.opt
        order = int(o_("ep.stochastic.order", 0))
        ep_periods = int(o_("ep.periods", 200))
        if info["maximum_lead"] == 0:
            if order > 0:
                raise MatlabError("Stochastic extended path does not support purely backward-looking models.")
            ep_periods = 1
        if int(info.get("maximum_lag", 1)) != 1:
            raise Unsupported("extended_path for a model with maximum_lag > 1", "function")
        if self.opt_flag("ep.use_first_order_solution_as_initial_guess"):
            raise Unsupported("options_.ep.use_first_order_solution_as_initial_guess", "function")
        if int(o_("ep.stochastic.hybrid_order", 0)):
            raise Unsupported("options_.ep.stochastic.hybrid_order (hybrid stochastic extended path)", "function")
        if order > 0:
            if int(o_("ep.stochastic.algo", 0)) != 0:
                raise Unsupported("the sparse tree of the stochastic extended path (options_.ep.stochastic.algo = 1; only the "
                                  "full tree, algo = 0, is implemented)", "function")
            if self.opt_flag("lmmcp.status") or int(o_("ep.solve_algo", 9)) != 9 or int(o_("ep.stack_solve_algo", 7)) != 7:
                raise Unsupported("the stochastic extended path with ep.solve_algo other than 9 (trust region)", "function")
        cpb = self.M_.get("perfect_foresight_controlled_paths")
        if cpb is not None and not _empty(cpb):
            raise MatlabError("extended_path: command is not compatible with perfect_foresight_controlled_paths block.")
        live = self._sim_live()
        ss, xss = live["ss"], live["xss"]
        # --- initial conditions
        if _empty(initial):
            eh = np.asarray(farr(self.M_["endo_histval"]), float)
            if eh.size == 0:
                y0 = np.asarray(ss, float).copy()
            else:
                if eh.size != n:
                    raise Unsupported("extended_path: M_.endo_histval with several columns", "function")
                y0 = eh.reshape(-1, order="F")
        else:
            y0 = np.asarray(farr(initial), float).reshape(-1, order="F")
            if y0.size != n:
                raise MatlabError("extended_path: the initial conditions must be a vector with one value per endogenous variable")
        # --- options of the perfect-foresight solver
        self._pf_options()
        o = pf._resolve_options(m, False, {})
        o["dynatol.f"] = float(o_("dynatol.f", 1e-5))              # Dynare's defaults (default_option_values.m)
        o["dynatol.x"] = float(o_("dynatol.x", 1e-5))
        o["trust_region_initial_step_bound_factor"] = float(o_("trust_region_initial_step_bound_factor", 1.0))
        o["simul.maxit"] = int(o_("ep.maxit", 500))
        o["solve_algo"] = int(o_("ep.solve_algo", 9))
        o["stack_solve_algo"] = int(o_("ep.stack_solve_algo", 7))
        o["periods"] = ep_periods
        o["linear"] = pf._model_is_linear(m)
        if o["solve_algo"] == 11:
            raise Unsupported("extended_path with ep.solve_algo = 11 (the PATH solver)", "function")
        if o["solve_algo"] == 10:
            o["lmmcp.status"] = True
        # --- the shocks
        dist = o_("ep.innovation_distribution", "gaussian")
        if self.opt("ep.set_dynare_seed_to_default", 1):
            self.rng.u, self.rng.n = _ml._mt_stream(_F(0)), _ml._mt_stream(_F(0))
        if _empty(exogenous):
            if dist != "gaussian":
                raise Unsupported(f"extended_path: options_.ep.innovation_distribution = '{dist}'", "function")
            try:
                shocks = _ep.draw_shocks(self.rng, m.Sigma_e, sample)
            except np.linalg.LinAlgError:
                raise MatlabError("chol: the covariance matrix of the structural innovations is not positive definite") from None
        else:
            if dist == "gaussian" and np.any(np.asarray(xss) != 0):
                raise MatlabError("extended_path: a Gaussian innovation_distribution is incompatible with non-mean 0 exogenous variables.")
            shocks = np.asarray(farr(exogenous), float)
            if shocks.shape[1] != nx or shocks.shape[0] < sample:
                raise MatlabError("extended_path: the exogenous variables must have one column per exogenous variable and at least "
                                  "as many rows as the sample size")
        ip = o_("initial_period", None)
        if ip is None or (isinstance(ip, float) and np.isnan(ip)):
            first = _Date(1, 1)
        elif isinstance(ip, str):
            first = _Date.parse(ip)
        else:
            raise MatlabError("extended_path: option initial_period be a date.")
        sim = pf._Sim(m, m.params.copy(), np.zeros((0, n)), np.zeros((0, nx)), ss, xss, None, None, o, False, None, None)
        try:
            with np.errstate(all="ignore"):
                if order > 0:
                    res = _sep.run(m, m.params.copy(), ss, xss, m.Sigma_e, {
                        "periods": ep_periods, "order": order, "nodes": int(o_("ep.stochastic.quadrature.nodes", 5)),
                        "IntegrationAlgorithm": o_("ep.stochastic.IntegrationAlgorithm", "Tensor-Gaussian-Quadrature")},
                        sample, y0, shocks, o, log=self.warnings.append)
                else:
                    res = _ep.run(m, sim, {"periods": ep_periods}, sample, y0, shocks, ss, xss,
                                  no_homotopy=bool(self.opt_flag("no_homotopy")), log=self.warnings.append)
        except (_ep.EPError, _sep.SEPError, NotImplementedError, ValueError) as exc:
            raise Unsupported(f"extended_path: {exc}", "function") from None
        lag = int(info["maximum_lag"])
        oo = self.oo_.clone()
        ep = oo.get("extended_path")
        ep = ep.clone() if isinstance(ep, Struct) else Struct()
        ep["status"] = np.bool_(bool(res.status))
        oo["extended_path"] = ep
        oo["endo_simul"] = R.mat(res.endo.T)
        oo["exo_simul"] = R.mat(res.exo)
        ts = _ds.DSeries(res.endo, list(m.endo_names), first)
        return ts, oo
