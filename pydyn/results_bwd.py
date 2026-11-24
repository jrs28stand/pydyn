# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2 (matlab/backward/simul_backward_model.m, simul_backward_nonlinear_model.m,
# simul_backward_linear_model.m, matlab/+backward_model/initialize.m; Copyright (C) 2012-2025 Dynare Team,
# GPL-3.0-or-later).
"""``simul_backward_model`` and friends on the live workspace of :mod:`pydyn.results`."""
from __future__ import annotations

import numpy as np

from . import backward as _bw
from . import dseries as _ds
from . import matlab as _ml
from .matlab import Cell, MatlabError, Struct, Unsupported, farr, _F


def _R():
    from . import results
    return results


def _pfmod():
    import importlib
    return importlib.import_module(__package__ + ".perfect_foresight")


def _lower_cholesky(Sigma, tiny):
    """get_lower_cholesky_covariance.m."""
    pos = np.flatnonzero(np.sqrt(np.diag(Sigma)) != 0)
    C = np.zeros_like(Sigma)
    sub = Sigma[np.ix_(pos, pos)]
    try:
        C[np.ix_(pos, pos)] = np.linalg.cholesky(sub)
    except np.linalg.LinAlgError:
        C[np.ix_(pos, pos)] = np.linalg.cholesky(sub + tiny * np.eye(len(pos)))
    return C


def _empty(v):
    return isinstance(v, np.ndarray) and v.size == 0


class BackwardMixin:
    """Methods of :class:`pydyn.results.Results` for the simulation of backward-looking models."""

    # ------------------------------------------------------------------------------ the functions
    def fn_simul_backward_model(self, args, nargout):
        """``[simulation, errorflag] = simul_backward_model(initialconditions, samplesize, innovations)``."""
        if len(args) < 2 or len(args) > 3:
            raise MatlabError("Invalid call to simul_backward_model")
        m = self.model
        if m.info["maximum_lead"]:
            raise MatlabError(f"Model defined in {self.name}.mod is not backward or static.")
        if not m.info["maximum_lag"]:
            raise Unsupported("simul_backward_model of a static model (simul_static_model)", "function")
        sim, oo, flag = self._simul_backward(args[0], args[1], args[2] if len(args) > 2 else None, name="simul_backward_model",
                                             need_dseries_innovations=True)
        self.set_global("oo_", oo)
        return [sim, np.bool_(flag)][:max(nargout, 1)]

    def fn_simul_backward_nonlinear_model(self, args, nargout):
        return self._explicit(args, nargout, "simul_backward_nonlinear_model", linear=False)

    def fn_simul_backward_linear_model(self, args, nargout):
        return self._explicit(args, nargout, "simul_backward_linear_model", linear=True)

    def _explicit(self, args, nargout, name, linear):
        """``[simulations, oo_, errorflag] = f(initialconditions, samplesize, options_, M_, oo_, innovations)``: the structs
        are values; the globals are left as they are."""
        if len(args) < 5 or len(args) > 6:
            raise MatlabError(f"Invalid call to {name}")
        if not all(isinstance(a, Struct) for a in args[2:5]):
            raise MatlabError(f"{name}: the third to fifth arguments must be options_, M_ and oo_")
        m = self.model
        if m.info["maximum_lead"] or not m.info["maximum_lag"]:
            raise MatlabError(f"Model defined in {self.name}.mod is not backward.")
        saved = {k: self.I.globals[k] for k in ("M_", "options_", "oo_")}
        try:
            for k, v in zip(("options_", "M_", "oo_"), args[2:5]):
                self.set_global(k, v)
            innov = args[5] if len(args) > 5 else None
            sim, oo, flag = self._simul_backward(args[0], args[1], innov, name=name, force_linear=linear,
                                                 need_dseries_innovations=False)
        finally:
            for k, v in saved.items():
                self.I.globals[k] = v
        return [sim, oo, np.bool_(flag)][:max(nargout, 1)]

    # ------------------------------------------------------------------------------ dseries files
    def dseries_method(self, ds, name, args):
        """``ts.save(basename[, format])``: Dynare's default format is ``'mat'`` (INIT__, FREQ__, NAMES__, TEX__, OPS__, TAGS__,
        DATA__).  The file goes through ``save`` (same sandbox: the folder of the .mod file, plain names)."""
        if name != "save":
            raise Unsupported(f"dseries method {name}", "dseries")
        if len(args) > 2 or not all(isinstance(a, str) for a in args):
            raise MatlabError("dseries::save: invalid arguments")
        base = args[0] if args else "dynare_series"
        fmt = args[1] if len(args) > 1 else "mat"
        if fmt != "mat":
            raise Unsupported(f"dseries save in the format '{fmt}' (only 'mat' is written)", "file")
        from .matlab import Cell
        tmp = {"INIT__": str(ds.init), "FREQ__": _F(ds.init.freq), "NAMES__": Cell(list(ds.names), (len(ds.names), 1)),
               "TEX__": Cell([nm.replace("_", "\\_") for nm in ds.names], (len(ds.names), 1)),
               "OPS__": Cell([np.zeros((0, 0)) for _ in ds.names], (len(ds.names), 1)), "TAGS__": Struct(),
               "DATA__": ds.data.copy()}
        sc = self.I.scope_stack[-1]
        old = {k: sc.vars.get(k) for k in tmp}
        try:
            sc.vars.update(tmp)
            self.fn_save([base + ".mat"] + list(tmp), 0)
        finally:
            for k, v in old.items():
                if v is None:
                    sc.vars.pop(k, None)
                else:
                    sc.vars[k] = v
        return np.zeros((0, 0))

    def fn_save_or_dseries(self, args, nargout):
        """``save(ts, basename[, format])`` is the dseries method; every other call is the workspace ``save``."""
        if args and isinstance(args[0], _ds.DSeries):
            _ds.HOST = self
            self.dseries_method(args[0], "save", list(args[1:]))
            return []
        return self.fn_save(args, nargout)

    def fn_delete(self, args, nargout):
        """``delete file.mat``: only a file ``save`` (or ``ts.save``) of this run wrote and nobody changed since (the same record
        as the overwrite rule of ``save``); anything else is refused."""
        import hashlib
        import json
        if nargout or len(args) != 1 or not isinstance(args[0], str):
            raise Unsupported("delete with computed arguments or outputs", "file")
        fname = args[0]
        if "/" in fname or ".." in fname or any(c in fname for c in "*?[]"):
            raise Unsupported(f"delete: {fname!r} is not a plain file name in the .mod file's folder", "file")
        base = self.model.json_dir.parents[2]
        path = base / fname
        if not path.exists():
            self.warnings.append(f"delete: no such file or directory: {fname}")
            return []
        if not (path.is_file() and not path.is_symlink()):
            raise Unsupported(f"delete: {fname} is not a plain file", "file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # the record of this run, then those of the other models of the same folder (the files of one test series)
        records = [self.model.json_dir.parents[1] / self._WRITTEN] + sorted(base.glob(f"*/{self._WRITTEN}"))
        for rec_path in records:
            try:
                rec = json.loads(rec_path.read_text()) if rec_path.is_file() else {}
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get(fname) == digest:
                path.unlink()
                rec.pop(fname)
                rec_path.write_text(json.dumps(rec, indent=0))
                return []
        raise Unsupported(f"delete: {fname} was not written by pydyn.dynare(), or was changed since; it is not deleted", "file")

    def dseries_read(self, fname):
        """``dseries('file.mat')``: a MAT-file of the .mod file's folder written by ``save`` (INIT__, NAMES__, DATA__)."""
        from scipy.io import loadmat
        from .matlab import Cell
        if ".." in fname or fname.startswith("/") or "/" in fname:
            raise Unsupported("dseries('file') with a path", "file")
        if not fname.endswith(".mat"):
            raise Unsupported(f"dseries('{fname}'): only MAT-files ('.mat') can be read", "file")
        path = self.model.json_dir.parents[2] / fname
        if not path.is_file():
            raise MatlabError(f"dseries: file {fname} does not exist!")
        try:
            d = loadmat(str(path), squeeze_me=False, chars_as_strings=True, struct_as_record=True)
            init = str(d["INIT__"].ravel()[0]) if d["INIT__"].dtype.kind in "US" else str(d["INIT__"])
            names = [str(x.ravel()[0]) if isinstance(x, np.ndarray) else str(x) for x in d["NAMES__"].ravel()]
            data = np.asarray(d["DATA__"], float)
        except Exception as exc:  # noqa: BLE001
            raise Unsupported(f"dseries('{fname}'): the file is not a MAT-file written by dseries.save ({exc})", "file") from None
        from .dates import Date
        return _ds.DSeries(data, names, Date.parse(init))

    def _functions_backward(self):
        _ds.HOST = self
        return {"dseries": lambda args, nargout: _ds.make(args),
                "isdseries": lambda args, nargout: np.bool_(len(args) == 1 and isinstance(args[0], _ds.DSeries)),
                "copy": lambda args, nargout: (args[0].copy() if len(args) == 1 and isinstance(args[0], _ds.DSeries)
                                               else _ds._fail("copy: only a dseries can be copied")),
                "simul_backward_model": self.fn_simul_backward_model,
                "simul_backward_nonlinear_model": self.fn_simul_backward_nonlinear_model,
                "simul_backward_linear_model": self.fn_simul_backward_linear_model,
                "simul_static_model": self.fn_simul_static_model,
                "model_inversion": self.fn_model_inversion,
                "extended_path": self.fn_extended_path,
                "backward_model_irf": self.fn_backward_model_irf,
                "backward_model_forecast": self.fn_backward_model_forecast,
                "set_historical_values": self.fn_set_historical_values,
                "delete": self.fn_delete,
                "save": self.fn_save_or_dseries}

    # ------------------------------------------------------------------------------ the work
    def _simul_backward(self, initial, samplesize, innov, name, force_linear=None, need_dseries_innovations=True):
        _ds.HOST = self
        R = _R()
        pf = _pfmod()
        m = self.model
        info = m.info
        if m.n_exo_det:
            raise Unsupported(f"{name}: models with varexo_det", "function")
        T = _ml.scalar(samplesize, f"{name}: samplesize")
        if T != int(T) or T < 0:
            raise MatlabError(f"{name}: samplesize must be a non-negative integer")
        T = int(T)
        self.pull()
        self._pf_options()
        o = pf._resolve_options(m, False, {})
        linear = bool(self.opt_flag("linear")) if force_linear is None else force_linear
        if need_dseries_innovations and force_linear is None:
            linear = bool(self.opt_flag("linear"))
        params = m.params.copy()
        live = self._sim_live()
        ss = live["ss"]
        n, nx = m.n_endo, m.n_exo
        n_orig = int(m.raw.get("orig_endo_nbr", n))
        # --- initial conditions (backward_model.initialize)
        have_init_given = isinstance(initial, _ds.DSeries)
        if not (isinstance(initial, _ds.DSeries) or _empty(initial)):
            raise MatlabError("First input argument must be a dseries object or an empty array!")
        nlag_orig = int(info.get("orig_maximum_lag", 1))
        if _empty(initial):
            initial = self._initial_from_histval(name)
        have_init = True
        endo0, exo0, first, n0 = self._dseries_initial(initial, name)
        # --- innovations
        innovations = None
        if innov is not None and not _empty(innov):
            if not isinstance(innov, _ds.DSeries):
                raise MatlabError("Third argument must be a dseries object!" if need_dseries_innovations else
                                  f"{name}: innovations must be a matrix")
            if have_init_given and innov.init - 1 != first + (n0 - 1) and need_dseries_innovations:
                raise MatlabError("Time spans in first and third arguments should be contiguous!")
            if innov.nobs < T:
                raise MatlabError(f"Time span in third argument is too short (should not be less than {T}, the value of the second argument)")
            innovations = np.zeros((T, nx))
            for i, nm in enumerate(m.exo_names):
                if nm in innov.names:
                    innovations[:, i] = innov.data[:T, innov.names.index(nm)]
        elif innov is not None and not need_dseries_innovations:
            innovations = None
        drawn = False
        if innovations is None:
            # Dynare's draw: randn(T, k)*chol(Sigma_e restricted to the shocks with a positive variance)
            if self.opt("bnlms.set_dynare_seed_to_default", True):
                self.rng.u, self.rng.n = _ml._mt_stream(_F(0)), _ml._mt_stream(_F(0))
            dist = self.opt("bnlms.innovation_distribution", "gaussian")
            if dist != "gaussian":
                raise MatlabError(f"simul_backward_nonlinear_model:: {dist} distribution for the structural innovations is not (yet) implemented!")
            try:
                innovations = _bw.draw_innovations(self.rng, m.Sigma_e, T)
            except _bw.BackwardError as exc:
                raise MatlabError(str(exc)) from None
            drawn = True
        try:
            res = _bw.simulate(m, endo0, exo0, T, innovations, params, ss, o, linear=linear, rng=self.rng,
                               log=self.warnings.append)
        except (_bw.BackwardError, NotImplementedError) as exc:
            raise Unsupported(f"{name}: {exc}", "function") from None
        oo = self.oo_.clone()
        if drawn:
            bn = oo.get("bnlms") if isinstance(oo.get("bnlms"), Struct) else Struct()
            bn = bn.clone()
            pos = np.flatnonzero(np.diag(np.asarray(m.Sigma_e, float)) > 0)
            bn["shocks"] = R.mat(innovations[:, pos]) if pos.size else R.empty()
            oo["bnlms"] = bn
        oo["endo_simul"] = R.mat(res.endo)
        oo["exo_simul"] = R.mat(res.exo)
        # --- the dseries of the original endogenous variables and the exogenous variables
        nobs = n0 + T
        data = np.full((nobs, n_orig + nx), np.nan)
        data[:res.endo.shape[1], :n_orig] = res.endo[:n_orig].T
        data[:, n_orig:] = res.exo
        names = list(m.endo_names[:n_orig]) + list(m.exo_names)
        sim = _ds.DSeries(data, names, first)
        return sim, oo, res.errorflag

    def _dseries_initial(self, ds, name):
        """Initial conditions from a dseries (checkdatabase + the initialisation of backward_model.initialize)."""
        m = self.model
        info = m.info
        n, nx = m.n_endo, m.n_exo
        n_orig = int(m.raw.get("orig_endo_nbr", n))
        aux = m.raw.get("aux_vars", [])
        lli = m.lead_lag_incidence
        # get_lags_on_endogenous_variables / get_lags_on_exogenous_variables of backward_model.initialize
        lag_of = {int(i): 1 for i in (np.flatnonzero(lli[:n_orig, 0]) if lli.shape[1] and info["maximum_endo_lag"] else [])}
        for a in aux:
            if a.get("type") == 1:
                lag_of[a["orig_index"] - 1] = abs(a["orig_lead_lag"])
        exo_lag_of = {a["orig_index"] - 1: abs(a["orig_lead_lag"] - 1) for a in aux if a.get("type") == 3}
        aux_names = {m.endo_names[x["endo_index"] - 1] for x in aux if x.get("type") != 6}
        for i in sorted(lag_of):
            if m.endo_names[i] not in ds.names and m.endo_names[i] not in aux_names:     # checkdatabase adds the auxiliary variables
                raise MatlabError(f"You have to initialize the following endogenous variables: {m.endo_names[i]}\n"
                                  "Please fix the dseries object used for setting the initial conditions!")
        for i in sorted(exo_lag_of):
            if m.exo_names[i] not in ds.names:
                raise MatlabError(f"You have to initialize the following exogenous variables: {m.exo_names[i]}\n"
                                  "Please fix the dseries object used for setting the initial conditions!")
        maxlag = max(list(lag_of.values()) + list(exo_lag_of.values()), default=0)
        if maxlag > ds.nobs:
            raise MatlabError(f"The dseries object provided as first input argument should at least have {maxlag} periods!")
        endo0 = np.full((n, ds.nobs), np.nan)
        exo0 = np.full((ds.nobs, nx), np.nan)
        for i, nm in enumerate(m.endo_names):
            if nm in ds.names:
                endo0[i] = ds.data[:, ds.names.index(nm)]
        for i, nm in enumerate(m.exo_names):
            if nm in ds.names:
                exo0[:, i] = ds.data[:, ds.names.index(nm)]
        # the auxiliary variables the data do not hold (dynamic_set_auxiliary_series)
        # (checkdatabase calls dynamic_set_auxiliary_series, which sets every auxiliary variable that has a definition, also
        # those the database holds)
        def _set_aux(a):
            return a.get("type") != 6
        if any(_set_aux(a) for a in aux):
            live = self._sim_live()
            with np.errstate(all="ignore"):
                filled = _bw.aux_series(m, endo0, np.where(np.isnan(exo0), 0.0, exo0), m.params.copy(), live["ss"])
            for a in aux:
                if _set_aux(a):
                    endo0[a["endo_index"] - 1] = filled[a["endo_index"] - 1]
        for names_, lags_, rows in ((m.endo_names, lag_of, endo0), (m.exo_names, exo_lag_of, exo0.T)):
            for i, lg in lags_.items():
                if np.any(np.isnan(rows[i][ds.nobs - lg:])):
                    raise MatlabError(f"Variable {names_[i]} should not have a NaN value in period "
                                      f"{ds.init + (ds.nobs - 1)}.\nPlease fix the dseries object used for setting the initial conditions!")
        return endo0, exo0, ds.init, ds.nobs

    def _initial_from_histval(self, name):
        """The dseries ``initialize.m`` builds from ``M_.endo_histval`` / ``M_.exo_histval`` when the initial conditions are
        empty (it reads them as the driver wrote them: one column per period up to 0, the columns of the first ``orig_endo_nbr``
        entries as ``yinitdata(:,1)``).  Dynare's code is followed to the letter, which is only meaningful for
        ``M_.maximum_lag == 1``; other models are refused."""
        m = self.model
        info = m.info
        n, nx = m.n_endo, m.n_exo
        n_orig = int(m.raw.get("orig_endo_nbr", n))
        eh = np.asarray(farr(self.M_["endo_histval"]), float)
        xh = np.asarray(farr(self.M_["exo_histval"]), float)
        if not eh.size:
            raise MatlabError(f"{name}: the initial conditions are not given and M_.endo_histval is empty: a histval block is needed")
        if int(info["maximum_lag"]) != 1:
            raise Unsupported(f"{name}: initial conditions from histval for a model with M_.maximum_lag > 1 (Dynare's initialize.m "
                              "reads only the first column of M_.endo_histval)", "function")
        L = int(info.get("orig_maximum_lag", 1))
        yinit = np.zeros((n_orig, L))
        yinit[:, 0] = eh.reshape(-1, order="F")[:n_orig]
        xinit = np.zeros((nx, L))
        aux = m.raw.get("aux_vars", [])
        if int(info.get("orig_maximum_endo_lag", 1)) > 1:
            for i, a in enumerate(aux, start=1):
                if a.get("type") == 1:
                    yinit[a["orig_index"] - 1, abs(a["orig_lead_lag"])] = eh.reshape(-1, order="F")[n_orig + i - 1]
            yinit = yinit[:, ::-1]
        if int(info.get("orig_maximum_exo_lag", 0)) > 0:
            flat = xh.reshape(-1, order="F")
            for i, a in enumerate(aux, start=1):
                if a.get("type") == 3:
                    k = n_orig + i - 1
                    if k >= flat.size:
                        raise MatlabError(f"index ({k + 1}): out of bound; value {k + 1} out of bound {flat.size}")
                    xinit[a["orig_index"] - 1, abs(a["orig_lead_lag"])] = flat[k]
            xinit = xinit[:, ::-1]
        return _ds.DSeries(np.hstack([yinit.T, xinit.T]), list(m.endo_names[:n_orig]) + list(m.exo_names), _ds.Date(1, 1))

    # ------------------------------------------------------------------------------ impulse response functions
    def fn_backward_model_irf(self, args, nargout):
        """``[deviations, baseline, irfs] = backward_model_irf(initialcondition, innovationbaseline, listofshocks,
        listofvariables[, periods[, transform]])`` (matlab/+backward_model/irf.m)."""
        R = _R()
        _ds.HOST = self
        if not 4 <= len(args) <= 6:
            raise MatlabError("Invalid call to backward_model_irf")
        initial, base, shocks, variables = args[:4]
        if isinstance(base, _ds.DSeries) and base.nobs == 0:
            base = np.zeros((0, 0))                             # dseries(): isempty
        m = self.model
        n, nx = m.n_endo, m.n_exo
        n_orig = int(m.raw.get("orig_endo_nbr", n))
        if m.info["maximum_lead"]:
            raise MatlabError("backward_model_irf:: The specified model is not backward looking!")
        if m.n_exo_det:
            raise Unsupported("backward_model_irf: models with varexo_det", "function")
        periods = 40 if len(args) < 5 else _ml.scalar(args[4], "backward_model_irf: periods")
        if periods != int(periods) or periods < 1:
            raise MatlabError("backward_model_irf: periods must be a positive integer")
        periods = int(periods)
        if len(args) > 5 and not _empty(args[5]):
            raise Unsupported("backward_model_irf with a transformation function", "function")
        self.pull()
        self._pf_options()
        o = pf_resolve = _pfmod()._resolve_options(m, False, {})
        linear = bool(self.opt_flag("linear"))
        if _empty(initial):
            eh = np.asarray(farr(self.M_["endo_histval"]), float)
            if not eh.size:
                raise MatlabError("Use histval block to set the initial condition.")
            initial = _ds.DSeries(eh.reshape(n, -1, order="F").T, list(m.endo_names), _ds.Date(1, 1))
        if not isinstance(shocks, Cell):
            raise MatlabError("Third input argument has to be a cell of string or dseries objects!")
        items = list(shocks.items)
        if all(isinstance(x, str) for x in items):
            deterministic = False
        elif all(isinstance(x, _ds.DSeries) for x in items):
            deterministic = True
        else:
            raise MatlabError("Elements of third input argument must all be char arrays or dseries objects!")
        if not isinstance(initial, _ds.DSeries):
            raise MatlabError("First input argument must be a dseries object or an empty array!")
        last0 = initial.init + (initial.nobs - 1)
        if deterministic:
            for i, sh in enumerate(items, start=1):
                if any(nm not in m.exo_names for nm in sh.names):
                    raise MatlabError(f"In experiment n\u00b0{i}, some of the declared shocks are unknown!")
                if not (last0 < sh.init):
                    raise MatlabError(f"In experiment n\u00b0{i}, the shock period must follow {last0}!")
        if _empty(variables) or (isinstance(variables, Cell) and not variables.items):
            variables = Cell(list(m.endo_names[:n_orig]), (n_orig, 1))
        if not isinstance(variables, Cell):
            raise MatlabError("Fourth input argument has to be a cell of row char arrays or an empty object.")
        varnames = [str(v) for v in variables.items]
        # default initial conditions for the innovations
        cols, names = [initial.data], list(initial.names)
        for nm in m.exo_names:
            if nm not in initial.names:
                cols.append(np.zeros((initial.nobs, 1)))
                names.append(nm)
        initial = _ds.DSeries(np.hstack(cols), names, initial.init)
        Innov = np.zeros((periods, nx))
        if not _empty(base):
            if not isinstance(base, _ds.DSeries):
                raise MatlabError("If not empty, the second argument has to be a dseries object!")
            if base.init - last0 != 1:
                raise MatlabError("The first date of the second input argument must follow the last date of the first input argument!")
            if base.nobs < periods:
                raise MatlabError(f"The second input argument must at least have {periods} observations or lower the number of periods.")
            for i, nm in enumerate(m.exo_names):
                if nm in base.names:
                    Innov[:, i] = base.data[:periods, base.names.index(nm)]
        endo0, exo0, first, n0 = self._dseries_initial(initial, "backward_model_irf")
        if not deterministic:
            Sigma = np.asarray(m.Sigma_e, float)
            if not np.any(Sigma):
                raise MatlabError("You did not specify the size of the shocks!")
            C = _lower_cholesky(Sigma, float(self.opt("add_tiny_number_to_cholesky", 1e-14)))
        params, live = m.params.copy(), self._sim_live()

        def simulate(innov):
            try:
                return _bw.simulate(m, endo0, exo0, periods, innov, params, live["ss"], o, linear=linear, rng=self.rng,
                                    log=self.warnings.append)
            except (_bw.BackwardError, NotImplementedError) as exc:
                raise Unsupported(f"backward_model_irf: {exc}", "function") from None

        def as_ds(y):
            return _ds.DSeries(y[:n_orig].T, list(m.endo_names[:n_orig]), first)

        r0 = simulate(Innov)
        if r0.errorflag:
            raise MatlabError("Simulation failed. Cannot simulate baseline.")
        y0 = r0.endo
        deviations, irfs = Struct(), Struct()
        for i, sh in enumerate(items, start=1):
            innov = Innov.copy()
            if deterministic:
                for j, nm in enumerate(sh.names):
                    k = m.exo_names.index(nm)
                    for l in range(sh.nobs):
                        t = (sh.init + l) - last0                      # position in the simulation, 1-based
                        innov[t - 1, k] += sh.data[l, j]
                name = f"experiment_{i}"
            else:
                if sh not in m.exo_names:
                    raise MatlabError(f"backward_model_irf: Exogenous variable {sh} is unknown!")
                innov[0, :] += C[:, m.exo_names.index(sh)]
                name = sh
            r1 = simulate(innov)
            if r1.errorflag:
                self.warnings.append(f"Simulation failed. Cannot compute IRF for {name}.")
                continue
            innov_ds = _ds.DSeries(innov, list(m.exo_names), last0 + 1)
            deviations[name] = _ds.hcat([as_ds(r1.endo - y0).extract(varnames), innov_ds])
            if nargout > 2:
                irfs[name] = _ds.hcat([as_ds(r1.endo).extract(varnames), innov_ds])
        out = [deviations]
        if nargout > 1:
            baseline = as_ds(y0)
            out.append(_ds.hcat([baseline, base]) if isinstance(base, _ds.DSeries) and base.nobs else baseline)
        if nargout > 2:
            out.append(irfs)
        return out

    def fn_backward_model_forecast(self, args, nargout):
        """``forecasts = backward_model_forecast(initialcondition[, listofvariables[, periods[, withuncertainty]]])``
        (matlab/+backward_model/forecast.m): the point forecast and, with uncertainty, the mean, median, standard deviation and the
        95% bounds of 1000 simulations with innovations drawn from the covariance matrix of the shocks."""
        _ds.HOST = self
        if not 1 <= len(args) <= 4:
            raise MatlabError("Invalid call to backward_model_forecast")
        m = self.model
        n, nx = m.n_endo, m.n_exo
        n_orig = int(m.raw.get("orig_endo_nbr", n))
        if m.info["maximum_lead"]:
            raise MatlabError("backward_model_forecast:: The specified model is not backward looking!")
        if m.n_exo_det:
            raise Unsupported("backward_model_forecast: models with varexo_det", "function")
        initial = args[0]
        if not isinstance(initial, _ds.DSeries):
            raise MatlabError("First input argument must be a dseries object or an empty array!")
        variables = list(m.endo_names) if len(args) < 2 else args[1]
        if isinstance(variables, Cell):
            variables = [str(v) for v in variables.items]
        periods = 8 if len(args) < 3 else _ml.scalar(args[2], "backward_model_forecast: periods")
        uncertainty = False if len(args) < 4 else bool(np.asarray(args[3]).reshape(-1)[0])
        if periods != int(periods) or periods < 1:
            raise MatlabError("backward_model_forecast: the number of periods must be a positive integer")
        periods = int(periods)
        self.pull()
        self._pf_options()
        o = _pfmod()._resolve_options(m, False, {})
        linear = bool(self.opt_flag("linear"))
        start = initial.init + initial.nobs
        cols, names = [initial.data], list(initial.names)
        for nm in m.exo_names:
            if nm not in initial.names:
                cols.append(np.zeros((initial.nobs, 1)))
                names.append(nm)
        initial = _ds.DSeries(np.hstack(cols), names, initial.init)
        endo0, exo0, first, n0 = self._dseries_initial(initial, "backward_model_forecast")
        idy = []
        for v in variables:
            if v not in m.endo_names:
                raise MatlabError(f"backward_model_forecast:: Variable {v} is unknown!")
            idy.append(list(m.endo_names).index(v))
        if any(j >= n_orig for j in idy):
            raise MatlabError("index out of bound; value exceeds the number of original endogenous variables")
        B = 1000
        if uncertainty:
            C = _lower_cholesky(np.asarray(m.Sigma_e, float), float(self.opt("add_tiny_number_to_cholesky", 1e-14)))
        params, live = m.params.copy(), self._sim_live()

        def simulate(innov):
            try:
                return _bw.simulate(m, endo0, exo0, periods, innov, params, live["ss"], o, linear=linear, rng=self.rng,
                                    log=self.warnings.append)
            except (_bw.BackwardError, NotImplementedError) as exc:
                raise Unsupported(f"backward_model_forecast: {exc}", "function") from None

        r0 = simulate(np.zeros((periods, nx)))
        if r0.errorflag:
            raise MatlabError("Simulation failed.")
        out = Struct()
        out["pointforecast"] = _ds.DSeries(r0.endo[idy].T, variables, first)
        out["start"] = start
        if uncertainty:
            arr = np.zeros((len(idy), periods + n0, B))
            for b in range(B):
                innov = (C @ self.rng.randn((nx, periods))).T
                r = simulate(innov)
                if r.errorflag:
                    raise MatlabError("Simulation failed.")
                arr[:, :, b] = r.endo[idy]
            out["meanforecast"] = _ds.DSeries(arr.mean(axis=2).T, variables, first)
            out["medianforecast"] = _ds.DSeries(np.median(arr, axis=2).T, variables, first)
            out["stdforecast"] = _ds.DSeries(arr.std(axis=2).T, variables, first)
            arr.sort(axis=2)
            out["lb"] = _ds.DSeries(arr[:, :, int(round(0.025 * B)) - 1].T, variables, first)
            out["ub"] = _ds.DSeries(arr[:, :, int(round(0.975 * B)) - 1].T, variables, first)
        return [out]

    def fn_set_historical_values(self, args, nargout):
        """``set_historical_values ds 1991Q1`` / ``set_historical_values(ds, initialperiod)``: M_.endo_histval and M_.exo_histval
        from a dseries (matlab/set_historical_values.m)."""
        R = _R()
        if len(args) != 2:
            raise MatlabError("Invalid call to set_historical_values")
        ds, per = args
        if isinstance(ds, str):
            sc = self.I.scope_stack[-1]
            if ds not in sc.vars:
                raise MatlabError(f"'{ds}' undefined")
            ds = sc.vars[ds]
        if isinstance(per, str):
            per = _ds.as_date(per)
        if not isinstance(ds, _ds.DSeries) or not isinstance(per, _ds.Date):
            raise Unsupported("set_historical_values with arguments that are not a dseries and a date", "function")
        m = self.model
        n, nx = m.n_endo, m.n_exo
        n_orig = int(m.raw.get("orig_endo_nbr", n))
        maxlag = int(m.info["maximum_lag"])
        aux = m.raw.get("aux_vars", [])
        lli = m.lead_lag_incidence
        eh = np.zeros((n, maxlag))

        def value(nm, date):
            k = date - ds.init
            if k < 0 or k >= ds.nobs:
                raise MatlabError("dseries::subsref: Wrong date (out of bounds)")
            return ds.data[k, ds.names.index(nm)]

        k = 0
        for i in range(n):
            if i < n_orig:
                if lli[i, 0] > 0:
                    if m.endo_names[i] not in ds.names:
                        raise MatlabError(f"Can't find {m.endo_names[i]} in dseries")
                    eh[i, maxlag - 1] = value(m.endo_names[i], per)
            else:
                a = aux[k]
                if a.get("type") == 1:
                    nm = m.endo_names[a["orig_index"] - 1]
                    if nm not in ds.names:
                        raise MatlabError(f"Can't find {nm} in dseries")
                    eh[i, maxlag - 1] = value(nm, per + int(a["orig_lead_lag"]))
                k += 1
        xh = np.zeros((nx, maxlag))
        avail = [nm in ds.names for nm in m.exo_names]
        for t in range(1, int(m.info.get("maximum_exo_lag", m.info.get("orig_maximum_exo_lag", 0))) + 1):
            for i in range(nx):
                if avail[i]:
                    xh[i, maxlag - t] = value(m.exo_names[i], per - t)
        self.set_M("endo_histval", R.mat(eh))
        self.set_M("exo_histval", R.mat(xh))
        return []

    # ------------------------------------------------------------------------------ static models
    def fn_simul_static_model(self, args, nargout):
        """``[simulation, oo_, errorflag] = simul_static_model(samplesize, innovations)`` (matlab/backward/simul_static_model.m
        calls backward_model.simul_static_model(M_, options_, oo_, samplesize, innovations))."""
        R = _R()
        if len(args) not in (1, 2):
            raise MatlabError("Invalid call to simul_static_model")
        m = self.model
        if m.info["maximum_lag"]:
            raise MatlabError(f"{self.name}.mod has lagged variables, but it should be a static model.")
        if m.info["maximum_lead"]:
            raise MatlabError(f"{self.name}.mod has leaded variables, but it should be a static model.")
        if m.n_exo_det:
            raise Unsupported("simul_static_model: models with varexo_det", "function")
        T = int(_ml.scalar(args[0], "samplesize"))
        innov = args[1] if len(args) > 1 else None
        self.pull()
        self._pf_options()
        pf = _pfmod()
        o = pf._resolve_options(m, False, {})
        n, nx = m.n_endo, m.n_exo
        n_orig = int(m.raw.get("orig_endo_nbr", n))
        oo = self.oo_.clone()
        init = _ds.Date(1, 1)
        if innov is None or _empty(innov):
            if self.opt("bnlms.set_dynare_seed_to_default", True):
                self.rng.u, self.rng.n = _ml._mt_stream(_F(0)), _ml._mt_stream(_F(0))
            if self.opt("bnlms.innovation_distribution", "gaussian") != "gaussian":
                raise MatlabError("the distribution for the structural innovations is not (yet) implemented!")
            try:
                exo = _bw.draw_innovations(self.rng, m.Sigma_e, T)
            except _bw.BackwardError as exc:
                raise MatlabError(str(exc)) from None
            pos = np.flatnonzero(np.diag(np.asarray(m.Sigma_e, float)) > 0)
            bn = oo.get("bnlms").clone() if isinstance(oo.get("bnlms"), Struct) else Struct()
            bn["shocks"] = R.mat(exo[:, pos]) if pos.size else R.empty()
            oo["bnlms"] = bn
        else:
            if not isinstance(innov, _ds.DSeries):
                raise MatlabError("simul_static_model: the innovations must be a dseries object")
            if innov.nobs < T:
                raise MatlabError(f"Time span in third argument is too short (should not be less than {T}, the value of the second argument)")
            exo = np.zeros((T, nx))
            for i, nm in enumerate(m.exo_names):
                if nm in innov.names:
                    exo[:, i] = innov.data[:T, innov.names.index(nm)]
            init = innov.init
        try:
            endo, failed = _bw.static_simulate(m, exo, m.params.copy(), o, self.rng)
        except _bw.BackwardError as exc:
            raise Unsupported(f"simul_static_model: {exc}", "function") from None
        for t in failed:
            self.warnings.append(f"simul_static_mode: Nonlinear solver failed in period {t}.")
        oo["endo_simul"] = R.mat(endo)
        oo["exo_simul"] = R.mat(exo)
        sim = _ds.DSeries(np.hstack([endo[:n_orig].T, exo]), list(m.endo_names[:n_orig]) + list(m.exo_names), init)
        self.set_global("oo_", oo)
        return [sim, oo, np.bool_(bool(failed))][:max(nargout, 1)]

    # ------------------------------------------------------------------------------ model inversion
    def fn_model_inversion(self, args, nargout):
        """``[endogenousvariables, exogenousvariables] = model_inversion(constraints, exogenousvariables, initialconditions,
        M_, options_, oo_)``: the models without leads (backward_model.inversion, static_model_inversion).  A model with
        leads (a perfect-foresight problem with controlled paths per period) is refused."""
        if len(args) != 6:
            raise MatlabError("model_inversion: This routine require six input arguments!")
        constraints, exogenous, init = args[0], args[1], args[2]
        if not isinstance(constraints, _ds.DSeries):
            raise MatlabError("model_inversion: First input argument must be a dseries object!")
        if not isinstance(exogenous, _ds.DSeries):
            raise MatlabError("model_inversion: Second input argument must be a dseries object!")
        if not _empty(init) and not isinstance(init, _ds.DSeries):
            raise MatlabError("model_inversion: Third input argument must be a dseries object!")
        if not isinstance(args[3], Struct):
            raise MatlabError("model_inversion: Last input argument must be structures (M_)!")
        m = self.model
        if m.info["maximum_lead"]:
            raise Unsupported("model_inversion of a model with leads (a perfect-foresight problem with controlled paths for "
                              "each informational period)", "function")
        if m.n_exo_det:
            raise Unsupported("model_inversion: models with varexo_det", "function")
        crange = constraints.dates_list()
        instruments = exogenous.ml_index([_ds.DateArray(crange)])
        free = [nm for j, nm in enumerate(instruments.names) if np.all(np.isnan(instruments.data[:, j]))]
        if len(free) != constraints.vobs:
            raise MatlabError("The number of instruments must be equal to the number of constrained variables!")
        self.pull()
        self._pf_options()
        pf = _pfmod()
        o = pf._resolve_options(m, False, {})
        live = self._sim_live()
        ss, params = live["ss"], m.params.copy()
        n, nx = m.n_endo, m.n_exo
        exo_all = exogenous.extract(list(m.exo_names))                  # exogenousvariables{exo_names{:}}
        static = not m.info["maximum_lag"]
        y_init = None
        if not static:
            if _empty(init):
                raise MatlabError("model_inversion: initial conditions are needed for a model with lags")
            # checkdatabase: the auxiliary variables, then the endogenous variables the period before the first constraint
            endo0 = np.full((n, init.nobs), np.nan)
            for i, nm in enumerate(m.endo_names):
                if nm in init.names:
                    endo0[i] = init.data[:, init.names.index(nm)]
            exo0 = np.zeros((init.nobs, nx))
            for i, nm in enumerate(m.exo_names):
                if nm in init.names:
                    exo0[:, i] = init.data[:, init.names.index(nm)]
            aux = [a for a in m.raw.get("aux_vars", []) if a.get("type") != 6]
            if any(m.endo_names[a["endo_index"] - 1] not in init.names for a in aux):
                with np.errstate(all="ignore"):
                    filled = _bw.aux_series(m, endo0, np.where(np.isnan(exo0), 0.0, exo0), params, ss)
                for a in aux:
                    if m.endo_names[a["endo_index"] - 1] not in init.names:
                        endo0[a["endo_index"] - 1] = filled[a["endo_index"] - 1]
            k = (crange[0] - 1) - init.init
            if k < 0 or k >= init.nobs:
                raise MatlabError("dseries::subsref: Indices are out of bounds!")
            y_init = endo0[:, k]
        first = crange[0]
        k0 = first - exo_all.init
        if k0 < 0 or k0 + constraints.nobs > exo_all.nobs:
            raise MatlabError("model_inversion: the exogenous variables do not cover the periods of the constraints")
        try:
            Y, X = _bw.invert(m, constraints.data, constraints.names, exo_all.data[k0:k0 + constraints.nobs], m.exo_names, free,
                              y_init, params, ss, o, static=static, rng=self.rng)
        except _bw.BackwardError as exc:
            raise MatlabError(f"model_inversion: {exc}") from None
        endo = _ds.DSeries(Y.T, list(m.endo_names), first)
        exo = _ds.DSeries(X, list(m.exo_names), first)
        return [endo, exo][:max(nargout, 1)]
