# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Dynare's internal routines that ``.mod`` files call from MATLAB lines (``results`` mode): the model's own epilogue
function and the refusals that name what is missing."""
from __future__ import annotations

import numpy as np

from .matlab import MatlabError, Unsupported, farr


def _R():
    from . import results
    return results


class DynareObject:
    """A Dynare global that pydyn does not build in Dynare's layout (``estim_params_``, the estimation outputs ``dataset_``,
    ``bayestopt_`` ...): it can be passed on to the Dynare routines that pydyn provides, any other use is refused."""
    obj_kind = "dynare-object"

    def __init__(self, name, why):
        self.name, self.why = name, why

    def _refuse(self, what):
        from .matlab import Unsupported
        raise Unsupported(f"{self.name}{what}: pydyn does not build Dynare's {self.name} ({self.why})", "results")

    def get_field(self, name):
        self._refuse(f".{name}")

    def ml_index(self, args, brace):
        self._refuse("(...)")

    def snapshot(self):
        return self


_WHY_ESTIM = ("it can only be passed to dynare_estimation_init and initial_estimation_checks")


#: Dynare routines and globals that pydyn does not have, with the missing piece (sizes: Dynare 7.2's matlab/ files)
_MISSING = {
    "cherrypick": "Dynare's cherrypick.m (525 lines: it cuts equations out of the model's JSON, rewrites them and writes a "
                  "new .mod file) is not ported",
    "dyn_ols": "Dynare's dyn_ols (matlab/ols, about 1,800 lines with the equation parser of the JSON model and the "
               "dseries transformations, writing M_.params) is not ported",
    "sur": "Dynare's sur (matlab/ols/sur.m, 258 lines, on top of the equation parser of dyn_ols, about 1,800 lines) is not ported",
    "pooled_ols": "Dynare's pooled_ols (matlab/ols/pooled_ols.m, 236 lines, on top of the equation parser of dyn_ols, "
                  "about 1,800 lines) is not ported",
    "pooled_fgls": "Dynare's pooled_fgls (matlab/ols/pooled_fgls.m, 131 lines, on top of the equation parser of dyn_ols, "
                   "about 1,800 lines) is not ported",
    "surgibbs": "Dynare's surgibbs (235 lines, on top of the equation parser of dyn_ols) is not ported",
    "olsgibbs": "Dynare's olsgibbs (278 lines, on top of the equation parser of dyn_ols) is not ported",
    "var_list_": "var_list_ is a char matrix (one padded row per variable) in Dynare, and pydyn's interpreter has no char "
                 "matrices",
}


class BayesTopt(DynareObject):
    """``bayestopt_`` for the fields that describe the restricted state space (set by dynare_estimation_init): the others are
    refused."""
    FIELDS = ("mf0", "mf1", "mf2", "mfys", "smoother_var_list", "smoother_mf", "smoother_restrict_columns",
              "smoother_saved_var_list")

    def __init__(self, res):
        super().__init__("bayestopt_", "pydyn builds only the fields " + ", ".join(self.FIELDS))
        self.res = res

    def get_field(self, name):
        if name not in self.FIELDS:
            self._refuse(f".{name}")
        return self.res._mi_bayestopt(name)

    def has_field(self, name):
        return name in self.FIELDS


class MiMixin:
    def _mi_install(self):
        """The Dynare globals estim_params_ and bayestopt_ that the estimation drivers declare."""
        self.I.globals["estim_params_"] = DynareObject("estim_params_", _WHY_ESTIM)
        self.I.globals["bayestopt_"] = BayesTopt(self)
        self.I.base.global_names |= {"estim_params_", "bayestopt_"}

    def _mi_layout(self):
        """The restricted state space of dynare_estimation_init (decision-rule order, 0-based): order_var, k2 (the union of the
        observed and the predetermined variables), the observed and the predetermined ones."""
        from .perturbation import dr_ordering
        m = self.model
        if not m.varobs:
            raise Unsupported("bayestopt_: the model has no varobs", "results")
        order_var, _sv, cnt = dr_ordering(m.lead_lag_incidence)
        inv = np.empty(m.n_endo, int)
        inv[order_var] = np.arange(m.n_endo)
        obs_decl = np.array([m._ie[v] for v in m.varobs], int)
        obs_dr = inv[obs_decl]
        nst = cnt["nstatic"]
        state_dr = np.arange(nst, nst + cnt["npred"] + cnt["nboth"])
        return order_var, inv, np.union1d(obs_dr, state_dr), obs_decl, obs_dr, state_dr

    def _mi_bayestopt(self, name):
        R = _R()
        _ov, inv, k2, obs_decl, obs_dr, state_dr = self._mi_layout()
        n = self.model.n_endo
        sel = self._selected_variables()
        if sel is None:
            k3 = np.arange(n)
        else:
            k3 = np.array(sorted(int(inv[i]) for i in sel[1]), int)       # dynare_estimation_init.m: k3 in dr order
        svl = np.union1d(k2, k3)                                           # smoother_var_list
        vals = {"mf0": np.searchsorted(k2, state_dr) + 1, "mf1": np.searchsorted(k2, obs_dr) + 1, "mf2": obs_dr + 1,
                "mfys": obs_decl + 1, "smoother_var_list": svl + 1, "smoother_mf": np.searchsorted(svl, obs_dr) + 1,
                "smoother_restrict_columns": np.searchsorted(svl, state_dr) + 1,
                "smoother_saved_var_list": np.searchsorted(svl, k3) + 1}
        return R.col(np.asarray(vals[name], float))

    def _missing(self, name, why):
        def refuse(args, nargout):
            raise Unsupported(f"{name}: {why}", "function")
        return refuse

    def _functions_mi(self):
        return {**{n: self._missing(n, w) for n, w in _MISSING.items()}, **self._functions_mi_()}

    def _functions_mi_(self):
        return {f"{self.name}.epilogue_dynamic": self.fn_epilogue_dynamic,
                "dynare_estimation_init": self.fn_dynare_estimation_init,
                "initial_estimation_checks": self.fn_initial_estimation_checks}

    def fn_dynare_estimation_init(self, args, nargout):
        """``[dataset_, dataset_info, xparam1, hh, M_, options_, oo_, estim_params_, bayestopt_, bounds] =
        dynare_estimation_init(var_list, fname, [], M_, options_, oo_, estim_params_, bayestopt_)``: runs what the
        ``estimation`` statement runs first (the options, the data, the estimated parameters); the structs M_, options_ and
        oo_ are the live ones, the data and prior objects are :class:`DynareObject` values."""
        import warnings
        from .estimation import Estimation, EstimationError
        R = _R()
        if len(args) != 8 or not all(isinstance(args[i], type(self.M_)) for i in (3, 4, 5)) \
                or args[3] is not self.M_ or args[4] is not self.options_ or args[5] is not self.oo_:
            raise Unsupported("dynare_estimation_init(var_list, fname, [], M_, options_, oo_, estim_params_, bayestopt_) with "
                              "the live M_, options_ and oo_")
        if nargout > 10:
            raise MatlabError("dynare_estimation_init: called with too many outputs")
        st = {"statementName": "estimation", "options": {}, "symbol_list": []}
        self._refresh_options(st)
        self._undo_logged_steady()
        self._init_filter_options()
        self.pull()
        self._estimation_order(st, "estimation")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = Estimation(self.model)
        except (EstimationError, NotImplementedError) as exc:
            raise R.ResultsError(f"dynare_estimation_init: {exc}") from None
        self._mi_est = est
        if not self.opt_flag("dsge_var"):
            self.set_options("nobs", np.float64(float(est.nobs)))
        self._mi_state_space(est)
        self._set_prior(est)
        self.push()
        why = "it is an internal object of dynare_estimation_init"
        outs = [DynareObject("dataset_", why), DynareObject("dataset_info", why), DynareObject("xparam1", why),
                DynareObject("hh", why), self.M_, self.options_, self.oo_, DynareObject("estim_params_", _WHY_ESTIM),
                DynareObject("bayestopt_", why), DynareObject("bounds", why)]
        return outs[:max(nargout, 1)]

    def fn_initial_estimation_checks(self, args, nargout):
        """``oo_ = initial_estimation_checks(objective_function, xparam1, dataset_, dataset_info, M_, estim_params_, options_,
        bayestopt_, bounds, oo_)`` after :meth:`fn_dynare_estimation_init`: the checks of initial_estimation_checks.m that
        pydyn has (:meth:`_initial_estimation_checks`), and the steady state at the initial parameters."""
        import warnings
        R = _R()
        est = getattr(self, "_mi_est", None)
        if est is None:
            raise Unsupported("initial_estimation_checks without a preceding dynare_estimation_init", "results")
        if len(args) != 10 or args[4] is not self.M_ or args[6] is not self.options_ or args[9] is not self.oo_:
            raise Unsupported("initial_estimation_checks with arguments other than the live M_, options_ and oo_", "results")
        self.pull()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._initial_estimation_checks(est)
        # check_steady_state_changes_parameters: the steady state at the current M_.params (not at the initial values)
        R = _R()
        exo = np.concatenate([np.asarray(farr(self.oo_[k]), float).ravel(order="F") if k in self.oo_ else np.zeros(0)
                              for k in ("exo_steady_state", "exo_det_steady_state")])
        ys, _ = self.evaluate_steady_state(farr(self.oo_["steady_state"]) if "steady_state" in self.oo_ else
                                           np.zeros(self.model.n_endo), exo[:self.model.n_exo],
                                           check=not self.opt_flag("diffuse_filter"))
        self.set_oo("steady_state", R.col(ys))
        self.push()
        return [self.oo_]

    def _set_restrict_fields(self):
        """oo_.dr.restrict_var_list and restrict_columns as dynare_estimation_init.m leaves them (k2 = union of the observed
        variables and the predetermined ones in decision-rule order; restrict_columns the places of the states in k2)."""
        from .perturbation import dr_ordering
        from .matlab import Struct
        R = _R()
        m = self.model
        if not m.varobs:
            return
        order_var, _sv, cnt = dr_ordering(m.lead_lag_incidence)
        inv = np.empty(m.n_endo, int)
        inv[order_var] = np.arange(m.n_endo)
        obs_dr = inv[np.array([m._ie[v] for v in m.varobs], int)]
        nst, npred, nboth = cnt["nstatic"], cnt["npred"], cnt["nboth"]
        state_dr = np.arange(nst, nst + npred + nboth)
        k2 = np.union1d(obs_dr, state_dr)
        nspred = int(float(self.M_["nspred"]))
        ic = np.searchsorted(k2, state_dr) + 1
        dr = self.oo_.get("dr")
        dr = R.rstruct("oo_.dr") if not isinstance(dr, Struct) else dr
        dr = R.struct_set(dr, "restrict_var_list", R.col(k2 + 1))
        dr = R.struct_set(dr, "restrict_columns", R.col(np.concatenate([ic, len(k2) + np.arange(1, nspred - (npred + nboth) + 1)])))
        self.set_oo("dr", dr)

    def _mi_state_space(self, est):
        """oo_.dr after dynare_estimation_init: set_state_space (order_var, inv_order_var) and the restricted state space
        (restrict_var_list: the union of observed and predetermined variables in decision-rule order; restrict_columns)."""
        from .perturbation import dr_ordering
        from .matlab import Struct
        R = _R()
        m = self.model
        order_var, _sv, cnt = dr_ordering(m.lead_lag_incidence)
        inv = np.empty(m.n_endo, int)
        inv[order_var] = np.arange(m.n_endo)
        obs_dr = inv[np.asarray(est._obs_decl, int)]
        nst, npred, nboth = cnt["nstatic"], cnt["npred"], cnt["nboth"]
        state_dr = np.arange(nst, nst + npred + nboth)
        k2 = np.union1d(obs_dr, state_dr)
        nspred = int(float(self.M_["nspred"]))
        ic = np.searchsorted(k2, state_dr) + 1
        dr = R.rstruct("oo_.dr")
        old = self.oo_.get("dr")
        if isinstance(old, Struct):
            dr.update(old)
        dr["order_var"] = R.col(order_var + 1)
        dr["inv_order_var"] = R.row(inv + 1)
        dr["restrict_var_list"] = R.col(k2 + 1)
        dr["restrict_columns"] = R.col(np.concatenate([ic, len(k2) + np.arange(1, nspred - (npred + nboth) + 1)]))
        self.set_oo("dr", dr)

    def fn_epilogue_dynamic(self, args, nargout):
        """``ds = <model>.epilogue_dynamic(params, ds)``: the epilogue variables added to (or updated in) the dseries by the
        definitions of the ``epilogue`` block (the preprocessor's file uses ``from d1 to d2 do ...`` of the dseries class; the
        evaluation is :class:`pydyn.epilogue.Epilogue`, which follows ``from.m``)."""
        from .dseries import DSeries
        from .epilogue import Epilogue
        if len(args) != 2 or not isinstance(args[1], DSeries):
            raise MatlabError(f"{self.name}.epilogue_dynamic: called with (params, ds) where ds is a dseries")
        ds = args[1]
        try:
            epi = Epilogue(self.model, mod_file=self.mod_file)
        except Exception as exc:                                    # an unreadable epilogue block
            raise Unsupported(f"{self.name}.epilogue_dynamic: cannot read the epilogue block ({exc})", "function") from None
        series = {nm: ds.data[:, i] for i, nm in enumerate(ds.names)}
        out = epi.dynamic(series, farr(args[0]).ravel())
        names = list(ds.names) + [n for n in out if n not in ds.names]
        data = np.column_stack([out[n] for n in names])
        return [DSeries(data, names, ds.init)]
