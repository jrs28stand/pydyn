# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2 (matlab/shock_decomposition/shock_decomposition.m; Copyright (C) Dynare Team,
# GPL-3.0-or-later).
"""``shock_decomposition(forecast_type=...)`` in the results layer: oo_.forecast_shock_decomposition.<type> and
oo_.shock_decomposition_info.forecast.<type>.parameter_set."""
from __future__ import annotations

import numpy as np

from .matlab import Cell, Struct, farr


def _R():
    from . import results
    return results


def _matlab_error(msg, prefix="shock_decomposition"):
    R = _R()
    return R.ResultsError(f"{prefix}: {msg}", matlab_message=msg)


class ForecastDecompMixin:
    def _sd_resolve_ps(self, kind, allow_sets):
        """options_.parameter_set, with the default of shock_decomposition.m (posterior mean, ML mode, posterior mode)."""
        R = _R()
        ps = self.opt("parameter_set", None)
        if ps is not None and not (isinstance(ps, np.ndarray) and ps.size == 0):
            return ps
        has = [k for k in ("posterior_mean", "mle_mode", "posterior") if k in self.oo_]
        if has and allow_sets:
            return ("posterior_mean" if "posterior_mean" in self.oo_ else
                    "mle_mode" if "mle_mode" in self.oo_ else "posterior_mode")
        if has:
            raise R.ResultsError(f"{kind}: the default parameter set (posterior mean or mode) is not supported by "
                                 "pydyn.dynare(): give parameter_set=calibration")
        raise _matlab_error("shock_decomposition: option parameter_set is not specified and posterior mode is not available", kind)

    def _sd_forecast_inputs(self, ft):
        """The forecasts shock_decomposition.m looks up before the smoother (errors as in Dynare)."""
        R = _R()
        if ft not in ("unconditional", "conditional"):
            raise R.ResultsError(f"shock_decomposition: forecast_type={ft} is not supported by pydyn.dynare()")
        ps = self._sd_resolve_ps("shock_decomposition", True)
        if ft == "unconditional":
            if ps in ("calibration", "mle_mode"):
                if "forecast" not in self.oo_:
                    raise _matlab_error("Can't find unconditional forecasts in oo_.forecasts. Did you run the "
                                        + ("forecast command?" if ps == "calibration" else "ML estimation with the forecast option?"))
                uf = self.oo_["forecast"]
            else:
                if "MeanForecast" not in self.oo_:
                    raise _matlab_error("Can't find unconditional forecasts in oo_.MeanForecast. Did you run Bayesian "
                                        "estimation with the forecast option?")
                uf = self.oo_["MeanForecast"]
            return uf
        if "conditional_forecast" not in self.oo_:
            raise _matlab_error("Can't find conditional forecasts")
        return self.oo_["conditional_forecast"]

    def _sd_forecast(self, st, res, ft, ps):
        """The decomposition of the forecast of the chosen type (the smoother result ``res`` is the sample's)."""
        from .forecast_decomp import forecast_decomposition, ForecastDecompositionError
        from .matlab import BUILTINS
        R = _R()
        src = self._sd_forecast_inputs(ft)
        m = self.model
        names = [str(x) for x in (st.get("symbol_list") or [])] or list(m.endo_names[:int(self.M_["orig_endo_nbr"])])
        i_var = BUILTINS["varlist_indices"](self.I, [Cell(names, (len(names), 1)), self.M_["endo_names"]], 1)
        i_var = [int(x) - 1 for x in np.asarray(farr(i_var), float).ravel()]
        endo = list(m.endo_names)
        exo = list(m.exo_names)

        def field(s, path):
            cur = s
            for p in path:
                if not isinstance(cur, Struct) or p not in cur:
                    raise _matlab_error(f"reference to non-existent field '{p}'")
                cur = cur[p]
            return np.asarray(farr(cur), float).ravel()

        try:
            if ft == "unconditional":
                means = {i: field(src, ("Mean", endo[i])) for i in i_var}
                cexo = None
            else:
                means = {i: field(src, ("cond", "Mean", endo[i])) for i in i_var}
                cexo = [field(src, ("controlled_exo_variables", "Mean", e)) for e in exo]
            z = forecast_decomposition(res.smoothed, res.shocks, res.G, res.H, ft, i_var, means,
                                       np.asarray(res.SteadyState, float).ravel(),
                                       init_state=int(self.opt("shock_decomp.init_state", 0) or 0), controlled_exo=cexo)
        except ForecastDecompositionError as exc:
            raise _matlab_error(str(exc)) from None
        self._sd_forecast_length = len(means[i_var[0]])
        cur = self.oo_.get("forecast_shock_decomposition")
        self.set_oo("forecast_shock_decomposition", R.struct_set(cur if isinstance(cur, Struct) else Struct(), ft, R.mat(z)))
        self.unrefuse("oo_", ["forecast_shock_decomposition"])
        self._decomp_info(f"forecast.{ft}.parameter_set", ps)
