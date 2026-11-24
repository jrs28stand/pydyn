# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour and fields follow Dynare 7.2 (matlab/+conditional_forecasts/{run, plot}.m and the driver lines that
# preprocessor/src/ComputingTasks.cc writes for conditional_forecast, conditional_forecast_paths and
# plot_conditional_forecast; Copyright (C) Dynare Team, GPL-3.0-or-later).
"""``conditional_forecast`` and ``plot_conditional_forecast`` on the live workspace of :mod:`pydyn.results`.

The numerics are :func:`pydyn.forecast.conditional_forecast`; this module is the statement-level glue: the paths of
the latest ``conditional_forecast_paths`` block before the statement, ``options_cond_fcst_`` (which Dynare's driver
never resets: an option given to an earlier ``conditional_forecast`` stays for the later ones), the parameter set
(calibration, or a vector of the last ``estimation``), the shared random stream (two sets of ``replic`` draws of
``randn(n_exo, periods)``, in Dynare's order), and ``oo_.conditional_forecast``.  Nothing is drawn."""
from __future__ import annotations

import warnings

import numpy as np

from .matlab import Cell, Struct, Unsupported, farr, _F


def _R():
    from . import results
    return results


def _num(x):
    return float(np.asarray(farr(x), float).ravel()[0])


def _option_value(v):
    """A value of the preprocessor's option JSON as the driver assigns it to ``options_cond_fcst_``."""
    R = _R()
    if isinstance(v, dict) and "symbol_list" in v:
        return R.cellstr(v["symbol_list"])
    if isinstance(v, bool):
        return np.bool_(v)
    if isinstance(v, (int, float)):
        return _F(float(v))
    if isinstance(v, str):
        return v
    raise Unsupported(f"conditional_forecast: the option value {v!r} is not supported", "function")


class ConditionalForecastMixin:
    # ------------------------------------------------------------------ options_cond_fcst_
    def _cf_update_options(self, st):
        """``options_list.writeOutput(output, "options_cond_fcst_")``: assignments that accumulate over the statements."""
        R = _R()
        cur = self.vars.get("options_cond_fcst_")
        if not isinstance(cur, Struct):
            cur = R.rstruct("options_cond_fcst_")
        for key, v in (st.get("options") or {}).items():
            cur = R.struct_set(cur, key, _option_value(v))
        self.vars["options_cond_fcst_"] = cur
        return cur

    def _cf_paths(self):
        """The latest ``conditional_forecast_paths`` block before the current statement: ``{variable: values}`` with NaN
        where a period is not constrained (``constrained_paths_ = NaN(n, L)``, then the assignments in order)."""
        R = _R()
        m = self.model
        k0 = self.last_k if self.last_k is not None else len(m.statements)
        for k in range(k0 - 1, -1, -1):
            st = m.statements[k]
            if st["statementName"] == "conditional_forecast_paths":
                break
        else:
            raise R.ResultsError("conditional_forecast: no conditional_forecast_paths block before the statement "
                                 "(Dynare: constrained_paths_ is undefined)")
        vals = m._stmt_values.get(id(st), {})
        paths = st["paths"]
        L = max(int(v["period2"]) for p in paths for v in p["values"])
        out = {}
        for i, p in enumerate(paths):
            row = np.full(L, np.nan)
            for j, v in enumerate(p["values"]):
                row[int(v["period1"]) - 1:int(v["period2"])] = np.asarray(vals[(i, j)], float)
            out[p["var"]] = row
        return out

    # ------------------------------------------------------------------ the statement
    def st_conditional_forecast(self, st):
        from .forecast import conditional_forecast
        R = _R()
        opts = self._cf_update_options(st)
        self.pull()
        m = self.model

        def get(path, default=None):
            cur = opts
            for part in path.split("."):
                if not isinstance(cur, Struct) or part not in cur:
                    return default
                cur = cur[part]
            return default if (isinstance(cur, np.ndarray) and cur.size == 0) else cur

        pset = get("parameter_set")
        if pset is None:
            for name, key in (("posterior_mode", "posterior_mode"), ("mle_mode", "mle_mode")):
                if name in self.oo_:
                    pset = key
                    break
            else:
                raise R.ResultsError("conditional_forecast: No valid parameter set found")
        if not isinstance(pset, str):
            raise Unsupported("conditional_forecast: a numerical parameter_set", "function")
        controlled = get("controlled_varexo")
        controlled = [str(x) for x in controlled.items] if isinstance(controlled, Cell) else \
            ([controlled] if isinstance(controlled, str) else [])
        periods = int(_num(get("periods", 40)))
        replic = int(_num(get("replic", 5000)))
        conf_sig = _num(get("conditional_forecast.conf_sig", 0.8))
        paths = self._cf_paths()
        kw = {}
        if pset != "calibration":
            ctx = getattr(self, "_est_ctx", None)
            if ctx is None:
                raise R.ResultsError(f"conditional_forecast: parameter_set={pset} needs an estimation before the statement")
            if self.opt_flag("prefilter"):
                raise R.ResultsError("MATLAB error in the .mod file: conditional_forecast: Conditional forecasting does "
                                     "not support the prefiltering option")
            kw.update(estimation=ctx["est"], mode=ctx["mode"], mcmc=ctx["mc"])
        else:
            eh = self.M_.get("endo_histval")
            eh = None if eh is None else np.asarray(farr(eh), float)
            if eh is not None and eh.size:
                kw["y0"] = eh.reshape(-1, order="F")[:m.n_endo]        # y0 = M_.endo_histval (indexed by order_var)
            self._cf_calibration_ready()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cf = conditional_forecast(m, paths, controlled, parameter_set=pset, periods=periods, replic=replic,
                                          conf_sig=conf_sig, rng=self.rng, **kw)
        except (NotImplementedError, ValueError, KeyError) as exc:
            raise R.ResultsError(f"conditional_forecast: {exc}") from None
        except Exception as exc:                               # SmootherError and the like
            if type(exc).__name__ in ("SmootherError", "EstimationError"):
                raise R.ResultsError(f"conditional_forecast: {exc}") from None
            raise
        self._store_cf(cf)

    def _cf_calibration_ready(self):
        R = _R()
        dr = self.oo_.get("dr")
        if not isinstance(dr, Struct) or "ghx" not in dr:
            raise R.ResultsError("MATLAB error in the .mod file: You need to call stoch_simul before conditional_forecast")

    def _store_cf(self, cf):
        R = _R()
        out = R.rstruct("oo_.conditional_forecast")
        # constrained_vars = oo_.dr.inv_order_var(constrained_vars): indexing a row vector gives a row
        out["controlled_variables"] = R.row(np.asarray(cf.controlled_variables_index, float) + 1)
        out["instruments"] = R.cellstr(cf.instruments)
        cond = R.rstruct("oo_.conditional_forecast.cond")
        cond["Mean"] = self._cf_fields("oo_.conditional_forecast.cond.Mean", cf.cond["Mean"], R.col)
        cond["ci"] = self._cf_fields("oo_.conditional_forecast.cond.ci", cf.cond["ci"], R.mat)
        out["cond"] = cond
        ex = R.rstruct("oo_.conditional_forecast.controlled_exo_variables")
        ex["Mean"] = self._cf_fields("oo_.conditional_forecast.controlled_exo_variables.Mean",
                                     cf.controlled_exo_variables["Mean"], R.col)
        ex["ci"] = self._cf_fields("oo_.conditional_forecast.controlled_exo_variables.ci",
                                   cf.controlled_exo_variables["ci"], R.mat)
        out["controlled_exo_variables"] = ex
        unc = R.rstruct("oo_.conditional_forecast.uncond")
        unc["Mean"] = self._cf_fields("oo_.conditional_forecast.uncond.Mean", cf.uncond["Mean"], R.col)
        unc["ci"] = self._cf_fields("oo_.conditional_forecast.uncond.ci", cf.uncond["ci"], R.mat)
        out["uncond"] = unc
        g = R.rstruct("oo_.conditional_forecast.graph")
        g["title"] = cf.graph["title"]
        g["fname"] = self.M_["fname"]
        out["graph"] = g
        self.set_oo("conditional_forecast", out)

    @staticmethod
    def _cf_fields(tag, d, conv):
        R = _R()
        s = R.rstruct(tag)
        for name, x in d.items():
            s[name] = conv(x)
        return s

    # ------------------------------------------------------------------ plot_conditional_forecast
    def st_plot_conditional_forecast(self, st):
        """``conditional_forecasts.plot(var_list_, periods, options_, oo_)``: the checks of the function; nothing is drawn."""
        R = _R()
        self.pull()
        cf = self.oo_.get("conditional_forecast")
        if not isinstance(cf, Struct):
            raise R.ResultsError("MATLAB error in the .mod file: Can't find conditional forecasts")
        names = [str(x) for x in (st.get("symbol_list") or [])]
        self.vars["var_list_"] = R.cellstr(names)
        if not names:
            raise R.ResultsError("MATLAB error in the .mod file: index (_,1): out of bound; value 1 out of bound 0")
        for v in names:
            for part in ("Mean", "ci"):
                for grp in ("cond", "uncond"):
                    if v not in cf[grp][part]:
                        raise R.ResultsError(f"MATLAB error in the .mod file: invalid use of undefined value: {v}")

    # ------------------------------------------------------------------ simult_
    def fn_simult_(self, args, nargout):
        """``y_ = simult_(M_, options_, y0, dr, ex_, iorder)`` (matlab/stochastic_solver/simult_.m): first order only."""
        R = _R()
        if len(args) != 6 or not all(isinstance(a, Struct) for a in args[:2]) or not isinstance(args[3], Struct):
            raise Unsupported("simult_(M_, options_, y0, dr, ex_, iorder) needs the structs M_, options_ and dr", "function")
        M, o, dr = args[0], args[1], args[3]
        iorder = int(_num(args[5]))
        if iorder != 1:
            raise Unsupported("simult_ at an order other than 1", "function")
        loglin = bool(_num(o.get("loglinear", 0))) and not bool(_num(o.get("logged_steady_state", 0)))
        ex = np.atleast_2d(np.asarray(farr(args[4]), float))
        y0 = np.asarray(farr(args[2]), float)
        y0 = y0.reshape(-1, 1) if y0.ndim < 2 else y0
        ys = np.asarray(farr(dr["ys"]), float).reshape(-1, 1)
        if loglin:                       # dr.ys(k) = log(dr.ys(k)) for the variables but the lagged/leaded exogenous ones
            from .perturbation import log_mask
            k = log_mask(self.model)
            ys = ys.copy()
            ys[k] = np.log(ys[k])
        ov = np.asarray(farr(dr["order_var"]), float).ravel().astype(int) - 1
        ghx, ghu = np.asarray(farr(dr["ghx"]), float), np.asarray(farr(dr["ghu"]), float)
        nst, nsp, ml = int(_num(M["nstatic"])), int(_num(M["nspred"])), int(_num(M["maximum_lag"]))
        iters = ex.shape[0] if ex.size else 0
        y = np.zeros((y0.shape[0], iters + ml))
        y[:, 0] = y0[:, 0] - ys[:, 0]
        k2 = ov[nst:nst + nsp]
        eps = ghu @ ex.T if ghu.size else None
        for i in range(1, iters + ml):
            if ghu.size == 0:
                y[ov, i] = ghx @ y[k2, i - 1]
            elif ghx.size == 0:
                y[ov, i] = eps[:, i - 1]
            else:
                y[ov, i] = ghx @ y[k2, i - 1] + eps[:, i - 1]
        if ghx.size == 0 and ghu.size:
            y[ov, :] = eps
        return [R.mat(y + ys)]
