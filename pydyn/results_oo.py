# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# The order of operations and the fields written follow Dynare 7.2 (matlab/steady.m, matlab/check.m,
# matlab/stochastic_solver/stoch_simul.m, irf.m, simult.m, matlab/moments/*.m; Copyright (C) Dynare Team,
# GPL-3.0-or-later).
"""Dynare's commands on the live workspace of :mod:`pydyn.results`: each calls the existing pydyn function and
writes what Dynare writes into ``M_``, ``options_`` and ``oo_`` (names, shapes and orientation included)."""
from __future__ import annotations

import warnings

import numpy as np

from . import matlab as _ml
from .matlab import Cell, MatlabError, Struct, Unsupported, farr, _F
from .model import ModelError
from .perturbation import BlanchardKahnError, dr_ordering


def cp_used(cp):
    return bool(cp)


def Tainted_cp():
    return _ml.Tainted("controlled_paths_by_period is a struct array, which pydyn does not represent")


def _R():
    from . import results
    return results


def _pfmod():
    import importlib
    return importlib.import_module(__package__ + ".perfect_foresight")      # the package attribute is the function


class CommandsMixin:
    """Methods of :class:`pydyn.results.Results` that run Dynare's commands."""

    # ------------------------------------------------------------------ option access
    def opt(self, name, default=None):
        """options_.<name> as a Python value (``default`` when absent or empty)."""
        R = _R()
        v = R.struct_get(self.options_, name)
        if v is None or (isinstance(v, np.ndarray) and v.size == 0):
            return default
        return R.py_value(v)

    def opt_raw(self, name):
        return _R().struct_get(self.options_, name)

    def opt_flag(self, name):
        v = self.opt(name, False)
        return bool(v[0]) if isinstance(v, list) and v else bool(v)

    def _refresh_options(self, st):
        """Write the statement's options into options_ and let the model's option dicts follow."""
        self.write_options(st.get("options"))
        self._pull_options()

    # ------------------------------------------------------------------ steady
    def st_steady(self, st):
        """``steady``: oo_.steady_state (and M_.params, where a steady-state file or block sets parameters)."""
        R = _R()
        m = self.model
        if st.get("options"):
            m._exec_statement(st)                # homotopy (parameters / exogenous variables to final values)
        self._refresh_options(st)
        if st.get("options", {}).get("homotopy_mode"):
            self.push()
            self.push_steady()
        self.pull()
        self.compute_steady(command=True)

    def compute_steady(self, command=True):
        """Dynare's evaluate_steady_state at the live calibration; writes oo_.steady_state and M_.params."""
        from .steady import SteadyStateError, steady_state
        from .perfect_foresight import steady_command_algo
        R = _R()
        m = self.model
        algo = self.opt("solve_algo", 4)
        xs = np.array([m._oo.get(v, 0.0) for v in m.exo_names], float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ys, p = steady_state(m, params=m.params.copy(), exo=xs, solve_algo=int(algo) if command else None)
        except SteadyStateError as exc:
            raise R.ResultsError(f"steady: {exc}") from None
        m.params[:] = p
        for name, v in zip(m.endo_names, ys):
            m._oo[name] = float(v)
        self.push()
        self.push_steady()
        return ys

    # ------------------------------------------------------------------ check
    def st_check(self, st):
        self._refresh_options(st)
        self.pull()
        self.compute_check()

    def compute_check(self):
        R = _R()
        from .perturbation import first_order
        m = self.model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = first_order(m, qz_criterium=self._qz())
        except BlanchardKahnError as exc:
            eig = getattr(exc, "eigenvalues", None)
            if eig is None:
                raise R.ResultsError(f"check: {exc}") from None
            return self._eigval(eig), False
        eig = self._eigval(sol.eigenvalues)
        self.set_oo("dr", R.struct_set(self.oo_.get("dr") if isinstance(self.oo_.get("dr"), Struct) else Struct(),
                                       "eigval", R.col(eig)))
        return eig, True

    def _qz(self):
        q = self.opt("qz_criterium")
        return float(q) if q is not None and not isinstance(q, list) else 1.0 + 1e-6

    @staticmethod
    def _eigval(eig):
        e = np.asarray(eig)
        big = np.abs(e) > 1e15
        if np.any(big):
            e = e.astype(float).copy()
            e[big] = np.where(e[big] > 0, np.inf, -np.inf)
        return e

    # ------------------------------------------------------------------ stoch_simul
    def st_stoch_simul(self, st):
        self._refresh_options(st)
        self.pull()
        self.stoch_simul_run(list(st.get("symbol_list") or []))

    def stoch_simul_run(self, var_list):
        R = _R()
        from .korder import KOrderError, k_order
        from .moments import theoretical_moments, stoch_simul_moments, solve as _solve
        from .perturbation import first_order
        from .simulate import irf as _irf, simulate
        m = self.model
        info = m.info
        order = int(self.opt("order", 2))
        if self.opt_flag("linear"):
            order = 1
        if order == 0:
            raise R.ResultsError("stoch_simul: The order of the Taylor approximation cannot be 0!")
        if m.n_exo == 0:
            raise R.ResultsError("stoch_simul: does not support having no varexo in the model. As a workaround you "
                                 "could define a dummy exogenous variable.")
        replic = 1 if order == 1 else int(self.opt("replic", 50))
        if order != 1 and m.max_deriv_order < 2:
            warnings.warn("stoch_simul: using order = 1 because Hessian is equal to zero")
            order = 1
            replic = 1
        pruning = self.opt_flag("pruning")
        loglinear = self.opt_flag("loglinear")
        for key in ("partial_information", "discretionary_policy"):
            if self.opt_flag(key):
                raise R.ResultsError(f"stoch_simul with {key} is not supported")
        qz = self._qz()
        names = list(var_list) if var_list else list(m.orig_endo_names)
        # varlist_indices: unique names, in the order given
        seen, uniq = set(), []
        for v in names:
            if v not in m._ie:
                raise R.ResultsError(f"stoch_simul: {v} isn't an endogenous variable")
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        names = uniq
        ivar = np.array([m._ie[v] for v in names], int)
        self.set_oo("var_list", R.cellstr(names))
        periods = int(self.opt("periods", 0))
        lag, lead = int(info["maximum_lag"]), int(info["maximum_lead"])
        xs = np.asarray(farr(self.oo_["exo_steady_state"]), float).ravel(order="F")
        self.set_oo("exo_simul", R.mat(np.ones((max(periods, 1) + lag + lead, 1)) @ xs.reshape(1, -1)))
        # --- the decision rules
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                sol = first_order(m, qz_criterium=qz, loglinear=loglinear)
                dr = sol.as_dr()
                if order >= 2:
                    dr = k_order(m, order=order, pruning=pruning)
                    dr["order"] = order
            except BlanchardKahnError as exc:
                raise R.ResultsError(f"stoch_simul: {exc}") from None
            except KOrderError as exc:
                raise R.ResultsError(f"stoch_simul: {exc}") from None
        if loglinear and order > 1:
            raise R.ResultsError("stoch_simul: loglinear is only available at order 1")
        m.params[:] = sol.params
        self.push()
        self.set_oo("dr", self._make_dr(sol, dr, order, qz))
        # --- simulation
        Sigma = np.array(m.Sigma_e, float)
        ys = np.asarray(dr["ys"], float)
        shocks_used = None
        if periods > 0:
            drop = int(self.opt("drop", 100))
            if periods <= drop:
                warnings.warn("STOCH_SIMUL error: The horizon of simulation is shorter than the number of observations "
                              "to be dropped. Either increase periods or decrease drop.")
                return
            shocks_used = self._draw_shocks(periods)
            y0 = ys
            hv = self.M_.get("endo_histval")
            if isinstance(hv, np.ndarray) and hv.size:
                y0 = np.asarray(hv, float).ravel(order="F")
            y = simulate(dr, shocks_used, order=order, pruning=pruning, y0=y0)
            self.set_oo("endo_simul", R.mat(y.T))
            self.set_oo("exo_simul", R.mat(shocks_used))
        # --- moments
        if not self.opt_flag("nomoments"):
            self._moments(dr, names, ivar, order, pruning, shocks_used)
        # --- impulse responses
        irf = int(self.opt("irf", 40))
        if irf > 0:
            self._irfs(dr, names, ivar, irf, order, pruning, replic, Sigma)
        self.set_oo("gui.ran_stoch_simul", np.bool_(True))

    def _draw_shocks(self, periods):
        R = _R()
        m = self.model
        self._n_simul += 1
        given = self.shocks.get(self._n_simul)
        if given is not None:
            g = np.asarray(given, float)
            if g.shape != (periods, m.n_exo):
                raise R.ResultsError(f"the shocks given for stoch_simul number {self._n_simul} must be {periods}x{m.n_exo}")
            return g
        Sigma = np.array(m.Sigma_e, float)
        active = np.nonzero(np.diag(Sigma) != 0)[0]
        shocks = np.zeros((periods, m.n_exo))
        if len(active):
            chol = np.linalg.cholesky(Sigma[np.ix_(active, active)]).T            # chol(): upper factor
            z = self.rng.randn((len(active), periods))                           # randn(nxs, periods)'
            shocks[:, active] = z.T @ chol
        return shocks

    def _make_dr(self, sol, dr, order, qz):
        R = _R()
        m = self.model
        n = m.n_endo
        order_var = np.asarray(sol.order_var, int)
        inv = np.empty(n, int)
        inv[order_var] = np.arange(n)
        _, _, cnt = dr_ordering(m.lead_lag_incidence)
        eig = self._eigval(sol.eigenvalues)
        ghx = np.asarray(dr["ghx"], float)
        nst, nsp = cnt["nstatic"], cnt["npred"] + cnt["nboth"]
        D = R.rstruct("oo_.dr")
        D["inv_order_var"] = R.row(inv + 1)
        D["order_var"] = R.col(order_var + 1)
        ys_dr = np.asarray(dr["ys"], float)
        loglin = self.opt_flag("loglinear")
        if loglin:                  # dr1.m: ys stays in levels and gx, Gy are the derivatives before the log transformation
            ys_dr = np.asarray(farr(self.oo_["steady_state"]), float).ravel(order="F")
        D["ys"] = R.col(ys_dr)
        D["ghx"] = R.mat(ghx)
        D["ghu"] = R.mat(np.asarray(dr["ghu"], float))
        if order < 3:                       # Dynare's third-order dr (k_order_perturbation) has none of these
            D["state_var"] = R.row(np.asarray(sol.state_var, int) + 1)
            D["eigval"] = R.col(eig)
            sdim = int(np.sum(np.abs(eig) < qz))
            D["sdim"] = _F(sdim)
            D["edim"] = _F(len(eig) - sdim)
            gx, Gy = ghx[nst + cnt["npred"]:, :], ghx[nst:nst + nsp, :]
            if loglin:
                yo = ys_dr[order_var]
                gx = gx * yo[nst + cnt["npred"]:, None] / yo[None, nst:nst + nsp]
                Gy = Gy * yo[nst:nst + nsp, None] / yo[None, nst:nst + nsp]
            D["gx"] = R.mat(gx)
            D["Gy"] = R.mat(Gy)
        if order >= 2:
            for k in ("ghxx", "ghuu", "ghxu", "ghs2"):
                D[k] = R.mat(np.asarray(dr[k], float))
        if order >= 3:
            for k in ("ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss"):
                D[k] = R.mat(np.asarray(dr[k], float))
        self.refused = {k: v for k, v in self.refused.items() if k[0] != "oo_.dr"}
        if order == 2:
            self.refuse("oo_.dr", ["fuu"], "second-order auxiliary matrix of dr2.m")
        if order >= 3:
            self.refuse("oo_.dr", ["g_0", "g_1", "g_2", "g_3", "pruning"],
                        "k_order_perturbation's derivative tensors and pruned state space")
        return D

    # ------------------------------------------------------------------ moments
    def _moments(self, dr, names, ivar, order, pruning, shocks_used):
        R = _R()
        from .moments import simulated_moments, theoretical_moments
        m = self.model
        periods = int(self.opt("periods", 0))
        kw = dict(order=order, pruning=pruning, ar=int(self.opt("ar", 5)), nodecomposition=self.opt_flag("nodecomposition"),
                  nocorr=self.opt_flag("nocorr"), qz_criterium=self._qz())
        hp = self.opt("hp_filter", 0)
        if hp:
            kw["hp_filter"] = float(hp)
        ohp = self.opt("one_sided_hp_filter", 0)
        if ohp:
            kw["one_sided_hp_filter"] = float(ohp)
        if self.opt_flag("bandpass.indicator"):
            kw["bandpass_filter"] = tuple(self.opt("bandpass.passband", [6, 32]))
        cvd = self.opt("conditional_variance_decomposition")
        if cvd:
            kw["conditional_variance_decomposition"] = cvd if isinstance(cvd, list) else [cvd]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if periods == 0:
                if order == 3 and not pruning:
                    return                         # Dynare computes no theoretical moments here
                mom = theoretical_moments(m, dr, names, **kw)
            else:
                y = np.asarray(farr(self.oo_["endo_simul"]), float).T
                mom = simulated_moments(y, ivar=ivar, shocks=shocks_used, dr=dr, model=m, y0=None, **kw)
        oo = self.oo_
        theoretical = periods == 0
        self.set_oo("mean", R.col(mom.mean))
        self.set_oo("var", R.mat(mom.var))
        if theoretical:
            nar = int(self.opt("ar", 5))
            items = [R.mat(g) for g in mom.gamma_y]
            single_shock_vd = m.n_exo == 1 and order >= 2 and not self.opt_flag("nodecomposition") and len(items) > nar + 1
            if single_shock_vd:      # observed under order >= 2 with one shock: Dynare's decomposition has a row per endogenous variable
                items[nar + 1] = R.mat(np.ones((m.n_endo, 1)))
            # th_autocovariances gives a column; the pruned state-space moments (order >= 2, pruning) a row
            shape = (1, len(items)) if (pruning and order >= 2) else (len(items), 1)
            self.set_oo("gamma_y", Cell(items, shape))
        if mom.autocorr and (not theoretical or mom.stationary.any()):
            self.set_oo("autocorr", Cell([R.mat(a) for a in mom.autocorr], (1, len(mom.autocorr))))
        if mom.corr is not None and self.opt_flag("contemporaneous_correlation"):
            self.set_oo("contemporaneous_correlation", R.mat(mom.corr))
        if mom.skewness is not None:
            self.set_oo("skewness", R.col(mom.skewness))
            self.set_oo("kurtosis", R.col(mom.kurtosis))
        if theoretical and m.n_exo == 1 and order >= 2 and not self.opt_flag("nodecomposition"):
            self.set_oo("variance_decomposition", R.mat(100.0 * np.ones((m.n_endo, 1))))
        elif mom.variance_decomposition is not None and (m.n_exo > 1 or not theoretical):
            self.set_oo("variance_decomposition", R.mat(mom.variance_decomposition))
        if mom.variance_decomposition_me is not None:
            self.set_oo("variance_decomposition_ME", R.mat(mom.variance_decomposition_me))
        if mom.conditional_variance_decomposition is not None:
            if theoretical:
                # disp_th_moments.m: (nvar, nsteps, nexo) fractions; _ME is (nobs, nsteps, nexo + 1) or []
                self.set_oo("conditional_variance_decomposition", R.mat(mom.conditional_variance_decomposition))
                me = mom.conditional_variance_decomposition_me
                self.set_oo("conditional_variance_decomposition_ME", R.mat(me) if me is not None else np.zeros((0, 0)))
            else:
                self.refuse("oo_", ["conditional_variance_decomposition", "conditional_variance_decomposition_ME"],
                            "Dynare computes it with theoretical moments only (periods = 0)")

    # ------------------------------------------------------------------ irfs
    def _irfs(self, dr, names, ivar, irf, order, pruning, replic, Sigma):
        R = _R()
        from .moments import get_lower_cholesky_covariance
        from .simulate import irf as _irf
        m = self.model
        cs = get_lower_cholesky_covariance(Sigma, float(self.opt("add_tiny_number_to_cholesky", 1e-14)))
        shocks = self.opt("irf_shocks")
        if shocks:
            sel = [m._ix[s] for s in (shocks if isinstance(shocks, list) else [shocks])]
        else:
            sel = list(range(m.n_exo))
        relative = self.opt_flag("relative_irf")
        drop = int(self.opt("drop", 100))
        irfs = Struct()
        old = self.oo_.get("irfs")
        if isinstance(old, Struct):
            irfs.update(old)
        nx = m.n_exo
        active = np.nonzero(np.diag(Sigma) != 0)[0]
        for i in sel:
            if not cs[i, i] > 5e-7:
                continue
            impulse = cs[:, i] / cs[i, i] / 100 if (order > 1 and relative) else cs[:, i]
            draws = None
            if order > 1 and not m.max_deriv_order < 2:
                draws = np.stack([self.rng.randn((irf + drop, len(active))) for _ in range(replic)], axis=2)
            y = _irf(dr, impulse, Sigma_e=Sigma, periods=irf, order=order, pruning=pruning, drop=drop,
                     replic=replic, draws=draws).T                      # n x irf
            if relative and order == 1:
                y = 100 * y / cs[i, i]
            for j, name in zip(ivar, names):
                irfs[f"{name}_{m.exo_names[i]}"] = R.row(y[j, :])
        self.set_oo("irfs", irfs)

    # ------------------------------------------------------------------ perfect foresight
    #: option keys of the perfect-foresight commands that pydyn reads (see pydyn.perfect_foresight)
    def _pf_options(self):
        """Tell perfect_foresight's option reader to use the options of the live workspace (Dynare's defaults
        included: its tolerance on the residuals, ``dynatol.f``, is 1e-5)."""
        R = _R()
        pf = _pfmod()
        keys = set(pf._SOLVER_KEYS.values()) | set(pf._SETUP_KEYS.values())
        legacy = {}
        for k in keys:
            v = self.opt_raw(k)
            if v is None or isinstance(v, Struct) or (isinstance(v, np.ndarray) and v.size == 0):
                continue
            pv = R.py_value(v)
            if k == "periods" and not pv:
                continue
            if k in ("stack_solve_algo", "solve_algo", "simul.steady_solve_algo", "simul.steady_maxit") and \
                    k not in self.touched:
                continue                                  # defaults: nothing to report
            legacy[k] = pv
        for k, v in self.model.options["stoch_simul"].items():
            if k in self._stmt_keys.get("perfect_foresight", ()) and k not in legacy:
                legacy[k] = v
        names = ("perfect_foresight_setup", "perfect_foresight_solver", "perfect_foresight_with_expectation_errors_setup",
                 "perfect_foresight_with_expectation_errors_solver")
        self.model._live_statement_options = {n: dict(legacy) for n in names}

    def _sim_live(self):
        R = _R()
        m, oo = self.model, self.oo_
        ss = np.asarray(farr(oo["steady_state"]), float).ravel(order="F")
        xss = np.asarray(farr(oo["exo_steady_state"]), float).ravel(order="F") if m.n_exo else np.zeros(0)
        ss0, xss0 = oo.get("initial_steady_state"), oo.get("initial_exo_steady_state")
        ss0 = None if isinstance(ss0, np.ndarray) and ss0.size == 0 else np.asarray(farr(ss0), float).ravel(order="F")
        xss0 = None if isinstance(xss0, np.ndarray) and xss0.size == 0 else np.asarray(farr(xss0), float).ravel(order="F")
        return dict(ss=ss, xss=xss, ss0=ss0, xss0=xss0)

    def st_perfect_foresight_setup(self, st):
        self._stmt_keys.setdefault("perfect_foresight", set()).update(st.get("options", {}) or {})
        self._refresh_options(st)
        self._pf_options()
        self.pull()
        self.pf_setup()

    def pf_setup(self, kind="perfect_foresight_setup"):
        R = _R()
        pf = _pfmod()
        m = self.model
        self._pf_options()
        live = self._sim_live()
        ws = pf._replay(m, params=m.params.copy(), steady=False, upto=self.last_k, live=live)
        o = pf._resolve_options(m, False, {})
        if o["periods"] is None and ws.initval_periods is not None:
            o["periods"] = ws.initval_periods
        sim_periods, first_date, last_date = pf.simulation_periods(o)
        if o["periods"] is None and sim_periods is not None:
            o["periods"] = sim_periods
        pf._undate(ws, first_date, last_date)
        if o["periods"] is None:
            raise R.ResultsError("perfect_foresight_setup: the number of periods is not given (options_.periods)")
        T = int(o["periods"])
        try:
            endo, x = pf._make_paths(m, ws, T)
            self._pf_cp = pf.controlled_paths_by_period(m, ws, T)
        except pf.PerfectForesightError as exc:
            raise R.ResultsError(f"perfect_foresight_setup: {exc}") from None
        self.set_oo("endo_simul", R.mat(endo.T))
        self.set_oo("exo_simul", R.mat(x))
        ds = self.oo_.get("deterministic_simulation")
        ds = ds.clone() if isinstance(ds, Struct) else Struct()
        ds["controlled_paths_by_period"] = R.empty() if not cp_used(self._pf_cp) else Tainted_cp()
        self.set_oo("deterministic_simulation", ds)
        self._pf_ws = ws

    def st_perfect_foresight_solver(self, st):
        self._stmt_keys.setdefault("perfect_foresight", set()).update(st.get("options", {}) or {})
        self._refresh_options(st)
        self._pf_options()
        self.pull()
        self.pf_solve()

    def pf_solve(self):
        R = _R()
        pf = _pfmod()
        m = self.model
        self._pf_options()
        live = self._sim_live()
        o = pf._resolve_options(m, False, {})
        ws = getattr(self, "_pf_ws", None)
        if o["periods"] is None and ws is not None and ws.initval_periods is not None:
            o["periods"] = ws.initval_periods
        sim_periods, first_date, last_date = pf.simulation_periods(o)
        if o["periods"] is None and sim_periods is not None:
            o["periods"] = sim_periods
        if o["periods"] is None:
            raise R.ResultsError("perfect_foresight_solver: the number of periods is not given (options_.periods)")
        T = int(o["periods"])
        lag, lead = m.info["maximum_lag"], m.info["maximum_lead"]
        rows = lag + T + lead
        E = self.oo_["endo_simul"]
        X = self.oo_["exo_simul"]
        E = np.asarray(farr(E), float) if isinstance(E, (np.ndarray, np.floating)) else None
        if E is None or E.shape != (m.n_endo, rows):
            raise R.ResultsError("perfect_foresight_solver: 'oo_.endo_simul' has wrong size. Did you run "
                                 "'perfect_foresight_setup'?")
        X = np.asarray(farr(X), float)
        if m.n_exo and X.shape != (rows, m.n_exo):
            raise R.ResultsError("perfect_foresight_solver: 'oo_.exo_simul' has wrong size. Did you run "
                                 "'perfect_foresight_setup'?")
        o["linear"] = pf._model_is_linear(m)
        if pf.has_complementarity(m) and not o["lmmcp.status"]:
            warnings.warn("the model has complementarity conditions but perfect_foresight_solver is used without "
                          "the lmmcp option: the conditions are ignored, as in Dynare")
        lm = R.struct_get(self.options_, "lmmcp.TolFun")
        if lm is not None and not (isinstance(lm, np.ndarray) and lm.size == 0):
            o["lmmcp.tolfun"] = float(lm)
        cp = getattr(self, "_pf_cp", None)
        if cp is None and ws is not None:
            cp = pf.controlled_paths_by_period(m, ws, T)
        sim = pf._Sim(m, m.params.copy(), E.T.copy(), X.copy(), live["ss"], live["xss"], live["ss0"], live["xss0"], o,
                      False, None, cp)
        try:
            out = sim.solve(T)
        except pf.PerfectForesightError as exc:
            raise R.ResultsError(f"perfect_foresight_solver: {exc}") from None
        self.set_oo("endo_simul", R.mat(out["endo"].T))
        if m.n_exo:
            self.set_oo("exo_simul", R.mat(out["exo"]))
        if not out["status"] or self._pf_recomputed(o, live):
            self.set_oo("steady_state", R.col(out["ss"]))
            if m.n_exo:
                self.set_oo("exo_steady_state", R.col(out["xss"]))
            self._last_ss = (np.asarray(out["ss"], float).copy(), np.asarray(out["xss"], float).copy())
            for v, x in zip(m.endo_names, out["ss"]):
                m._oo[v] = float(x)
        ds = self.oo_.get("deterministic_simulation")
        ds = ds.clone() if isinstance(ds, Struct) else Struct()
        ds["status"] = np.bool_(bool(out["status"]))
        ds["error"] = _F(out["maxerror"])
        ds["homotopy_completion_share"] = _F(out["share"])
        ds["homotopy_iterations"] = _F(out["homotopy_iterations"])
        if out.get("iterations") is not None:
            ds["iterations"] = _F(out["iterations"])
        self.set_oo("deterministic_simulation", ds)
        self.set_oo("gui.ran_perfect_foresight", np.bool_(bool(out["status"])))
        return out

    def _pf_recomputed(self, o, live):
        return bool(o["simul.endval_steady"]) or live["ss0"] is not None
