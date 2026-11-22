# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# The names and layouts of the fields of M_, options_ and oo_, and the order of the operations of the
# commands, follow Dynare 7.2 (matlab/*.m, Copyright (C) Dynare Team, GPL-3.0-or-later).
"""MATLAB-style results: ``dynare()`` runs a .mod file as Dynare does and leaves ``M_``, ``options_`` and ``oo_``.

``pydyn.dynare("model.mod")`` runs the statements of the file in file order. Dynare statements run where
they stand and fill ``oo_`` with Dynare's field names and shapes (``oo_.steady_state`` a column in
declaration order, ``oo_.dr.ghx`` in decision-rule order with ``order_var``/``inv_order_var``,
``oo_.irfs.y_e`` a row, ``oo_.endo_simul`` an ``n_endo x T`` matrix, ...). MATLAB lines of the file run in
the same interpreter (:mod:`pydyn.matlab`) and see the live ``M_``, ``options_`` and ``oo_``: they can read
them (Dynare's own test files check their results this way), write them (``M_.params(2) = 0.5``,
``options_.irf = 20``, ``oo_.exo_simul = ...``) and call Dynare's functions on the live workspace without
preprocessing again (``set_param_value``, ``steady``, ``check``, ``stoch_simul``,
``perfect_foresight_setup``/``perfect_foresight_solver``, ``occbin_solver``, ...).

    ws = pydyn.dynare("rbc.mod")
    ws.oo_.dr.ghx             # Dynare's oo_.dr.ghx
    ws.M_.params; ws.options_.irf; ws.oo_.irfs.y_e
    ws.set_param_value("rho", 0.5); ws.stoch_simul(order=1, irf=20)      # re-run on the live workspace
    ws.vars                   # the MATLAB workspace of the file (variables the MATLAB lines defined)

The structs are :class:`pydyn.matlab.Struct` (attribute access works) holding numpy matrices with
MATLAB's shapes (a 1 x 1 matrix is a ``numpy.float64``). ``pydyn.Model`` and the rest of the package are
unchanged: ``ws.model`` is the usual :class:`pydyn.Model`, and ``dynare()`` is an additional mode.
What pydyn does not compute faithfully is refused: reading a field Dynare would have filled and
pydyn does not raises :class:`ResultsError` naming the field; a statement pydyn cannot run
raises it too (see ``dynare()``).
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np

from . import matlab as _ml
from .matlab import (Cell, Interpreter, MatlabError, Struct, Tainted, Unsupported, arr, farr, ret, _F, _HOST_GLOBALS)
from .model import Model, ModelError
from .results_oo import CommandsMixin
from .results_est import EstimationMixin

__all__ = ["dynare", "Results", "ResultsError"]


class ResultsError(Exception):
    """A statement or MATLAB line of the .mod file cannot be run (or failed) in the results mode."""

    def __init__(self, msg, matlab_message=None):
        super().__init__(msg)
        self.matlab_message = matlab_message


# ======================================================================================= value helpers
def mat(a):
    """A 2-D float matrix as the interpreter holds it (1 x 1 becomes a scalar)."""
    a = np.asarray(a, float)
    if a.ndim == 0:
        return _F(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return ret(a.copy())


def col(a):
    """An n x 1 column (a scalar for n = 1; ``zeros(0, 0)`` for an empty list)."""
    a = np.asarray(a, float).ravel(order="F")
    return ret(a.reshape(-1, 1).copy()) if a.size else np.zeros((0, 0))


def row(a):
    """A 1 x n row (a scalar for n = 1; ``zeros(0, 0)`` for an empty list)."""
    a = np.asarray(a, float).ravel(order="F")
    return ret(a.reshape(1, -1).copy()) if a.size else np.zeros((0, 0))


def empty():
    return np.zeros((0, 0))


def cellstr(names, column=True):
    names = [str(n) for n in names]
    if not names:
        return Cell([], (0, 0))
    return Cell(names, (len(names), 1) if column else (1, len(names)))


def S(**kw):
    s = Struct()
    s.update(kw)
    return s


def struct_set(s, path, value):
    """A copy of struct ``s`` with the field at ``path`` (a name or dotted name) set; structs along the
    way are copied (values are never changed in place: a MATLAB copy ``x = oo_`` stays what it was)."""
    first, _, rest = path.partition(".")
    out = s.clone() if isinstance(s, Struct) else Struct()
    out[first] = struct_set(out.get(first), rest, value) if rest else value
    return out


def struct_get(s, path, default=None):
    for part in path.split("."):
        if not isinstance(s, Struct) or part not in s:
            return default
        s = s[part]
    return s


def py_value(v):
    """A MATLAB value as the Python value pydyn's option readers expect (JSON-like): integral doubles
    become ints, 1 x 1 stays scalar, vectors lists, char strings, cell arrays lists of strings."""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.float64, float)):
        f = float(v)
        return int(f) if np.isfinite(f) and f == int(f) and abs(f) < 1e15 else f
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, Cell):
        return [py_value(x) for x in v.items]
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return []
        flat = v.ravel(order="F")
        if v.dtype == bool:
            return [bool(x) for x in flat]
        return [py_value(_F(x)) for x in flat]
    return v


def flatten(s, prefix=""):
    """``{dotted name: value}`` of the leaves of a struct (nested structs are walked)."""
    out = {}
    for k, v in s.items():
        if isinstance(v, Struct):
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def same_value(a, b):
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True)) and a.dtype == b.dtype
    if isinstance(a, Cell) and isinstance(b, Cell):
        return a.shape == b.shape and len(a.items) == len(b.items) and all(same_value(x, y) for x, y in zip(a.items, b.items))
    if isinstance(a, Struct) and isinstance(b, Struct):
        return list(a) == list(b) and all(same_value(a[k], b[k]) for k in a)
    try:
        if isinstance(a, (np.floating, float)) and isinstance(b, (np.floating, float)):
            return bool(a == b) or (a != a and b != b)
        return type(a) is type(b) and bool(a == b)
    except Exception:  # noqa: BLE001
        return False


# ======================================================================================= options_ defaults
_DEFAULT_OPTIONS = None


def _dec(v):
    """Decode the JSON written by tools/make_default_options.py."""
    if isinstance(v, dict):
        kind = v.get("$")
        if kind is None:
            return Struct({k: _dec(x) for k, x in v.items()})
        if kind == "s0":
            return np.zeros((0, 0))
        if kind == "x":
            return Tainted("a struct array that pydyn does not represent")
        shape = tuple(v["shape"])
        if kind == "c":
            return Cell([_dec(x) for x in v["data"]], shape)
        data = [float(x) if isinstance(x, str) else x for x in v["data"]]
        a = np.array(data, dtype=bool if kind == "b" else float)
        a = a.reshape(shape, order="F") if a.size else np.zeros(shape, a.dtype)
        return ret(a) if a.dtype == float else (np.bool_(a[0, 0]) if a.shape == (1, 1) else a)
    if isinstance(v, bool):
        return np.bool_(v)
    if isinstance(v, (int, float)):
        return _F(v)
    return v


def default_options() -> Struct:
    """Dynare 7.2's ``options_`` after the driver of a model without commands (a fresh copy)."""
    global _DEFAULT_OPTIONS
    if _DEFAULT_OPTIONS is None:
        p = Path(__file__).parent / "data" / "default_options.json"
        _DEFAULT_OPTIONS = _dec(json.loads(p.read_text()))
    return _copy_tree(_DEFAULT_OPTIONS)


def _copy_tree(s):
    return Struct({k: _copy_tree(v) if isinstance(v, Struct) else v for k, v in s.items()})


def option_value(v):
    """The MATLAB value of an option as the preprocessor's driver writes it (see Statement.cc,
    OptionsList::writeOutputCommon): numbers are doubles, booleans logicals, strings char, a list of
    integers a column vector, other numeric lists a row vector, a list of names a cell column."""
    if isinstance(v, bool):
        return np.bool_(v)
    if isinstance(v, (int, float)):
        return _F(v)
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "symbol_list" in v:
            return cellstr(v["symbol_list"], column=False) if len(v["symbol_list"]) <= 1 else cellstr(v["symbol_list"])
        return Struct({k: option_value(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        if not v:
            return empty()
        if all(isinstance(x, str) for x in v):
            if len(v) == 1:
                return Cell(list(v), (1, 1))
            return cellstr(v)
        try:
            vals = [float(x) for x in v]
        except (TypeError, ValueError):
            return Cell([option_value(x) for x in v], (len(v), 1))
        if all(x == int(x) for x in vals):
            return col(vals) if len(vals) > 1 else _F(vals[0])
        return row(vals)
    return v


# ======================================================================================= structs
class RStruct(Struct):
    """A struct of ``M_``/``options_``/``oo_``: attribute access in Python, and a tag that lets the
    interpreter tell a field Dynare has and pydyn does not fill (refused) from a field Dynare has not."""

    def __getattr__(self, name):
        if name.startswith("__") or name == "_dyn":
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return list(self.keys())


def rstruct(tag, **fields):
    s = RStruct()
    s.update(fields)
    s._dyn = tag
    return s


def tag_tree(s, tag):
    """Tag struct ``s`` (and the structs below it) with the path ``tag`` for the interpreter's messages."""
    if not isinstance(s, RStruct):
        r = RStruct()
        r.update(s)
        s = r
    s._dyn = tag
    for k, v in list(s.items()):
        if isinstance(v, Struct):
            s[k] = tag_tree(v, f"{tag}.{k}")
    return s


# ======================================================================================= interpreter
class _LiveHost:
    """What ``set_param_value`` & co. see in the interpreter (see pydyn.matlab.Host)."""

    def __init__(self, res):
        self.res = res
        self.model = res.model
        self.readonly = False
        self.M = self.options = self.oo = None          # the structs live in the interpreter's globals

    def set_param(self, name, value):
        i = self.model._ip.get(name)
        if i is None:
            raise MatlabError(f"set_param_value: parameter name {name} doesn't exist")
        v = float(_ml.scalar(value, "parameter value"))
        self.res.set_M("params", _setitem(self.res.M_["params"], i, v))

    def get_param(self, name):
        i = self.model._ip.get(name)
        if i is None:
            raise MatlabError(f"get_param_by_name: parameter name {name} doesn't exist")
        return _F(np.asarray(farr(self.res.M_["params"])).ravel(order="F")[i])

    def set_shock_stderr(self, name, value):
        i = self.model._ix.get(name)
        if i is None:
            raise MatlabError(f"Shock name {name} doesn't exist")
        S = np.array(farr(self.res.M_["Sigma_e"]), float)
        S[i, i] = float(_ml.scalar(value)) ** 2
        self.res.set_M("Sigma_e", mat(S))


def _setitem(v, i, x):
    a = np.array(farr(v), float)
    flat = a.ravel(order="F")
    flat[i] = x
    return ret(flat.reshape(a.shape, order="F"))


class LiveInterp(Interpreter):
    """The interpreter of the results mode: ``M_``, ``options_`` and ``oo_`` are ordinary structs, Dynare
    statements run where they stand, and Dynare's functions act on the live workspace."""

    def __init__(self, directory, host, res):
        super().__init__(directory, host)
        self.res = res
        self.globals = {}

    # --- Dynare statements inside the file
    def exec(self, s, sc):
        if type(s) is _ml.Dyn:
            return self.res.exec_dyn(s.k)
        return super().exec(s, sc)

    # --- the three globals are plain structs here
    def setvar(self, sc, name, value):
        if name in sc.global_names and name in _HOST_GLOBALS:
            self.res.set_global(name, value)
            return
        super().setvar(sc, name, value)

    def getfield(self, v, name):
        if isinstance(v, Struct) and name not in v:
            tag = getattr(v, "_dyn", None)
            if tag is not None:
                why = self.res.refused.get((tag, name))
                if why is not None:
                    raise Unsupported(f"{tag}.{name}: Dynare fills this field and pydyn does not ({why})", "results")
        return super().getfield(v, name)

    # --- Dynare's functions and the functions the results mode replaces
    def _live(self, name, sc):
        if name in sc.funcs or name in self.dir_files():
            return None
        return self.res.functions.get(name)

    def call_node(self, name, argnodes, nargout, sc):
        if self._live(name, sc) is not None:
            args = [":" if type(a) is _ml.ColonAll else self.ev(a, sc) for a in argnodes]
            return self.call(name, args, nargout, sc)
        if name in self.res.noops and name not in sc.funcs and name not in self.dir_files():
            for a in argnodes:                         # arguments are evaluated: they may fail or have effects
                if type(a) not in (_ml.Str, _ml.Num, _ml.ColonAll):
                    self.ev_multi(a, sc, 1)
            if nargout > 0:
                raise Unsupported(f"{name}: its output is not available (pydyn draws nothing)", "display")
            return []
        return super().call_node(name, argnodes, nargout, sc)

    def call(self, name, args, nargout, sc):
        f = self._live(name, sc)
        if f is not None:
            self.scope_stack.append(sc)
            try:
                r = f(args, nargout)
            finally:
                self.scope_stack.pop()
            return r if type(r) is list else ([] if r is None else [r])
        return super().call(name, args, nargout, sc)

    def command(self, s, sc):
        name = s.name
        if name in self.res.functions and name not in sc.funcs and name not in self.dir_files():
            words = [a.strip("'\"") for a in s.args.split()]
            return self.call(name, words, 0, sc)
        return super().command(s, sc)

    def load_file(self, args, sc):
        return self.res.fn_load([a.strip("'\"") for a in args], 0, sc=sc)


# ======================================================================================= MAT files
def mat_value(v):
    """A value read from a MAT-file (``scipy.io.loadmat`` with ``struct_as_record=True``) as the interpreter holds it."""
    if isinstance(v, np.ndarray) and v.dtype.names:
        if v.size == 1:
            s = Struct()
            for n in v.dtype.names:
                s[n] = mat_value(v[n].ravel()[0] if v[n].size == 1 else v[n])
            return s
        return Tainted("a struct array read from a MAT-file (pydyn has no struct arrays)")
    if isinstance(v, np.ndarray) and v.dtype == object:
        return Cell([mat_value(x) for x in v.ravel(order="F")], v.shape if v.ndim == 2 else (1, v.size))
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            if v.ndim <= 1 or v.shape[0] == 1:
                return str("".join(v.ravel())) if v.size else ""
            return Cell([str(x).rstrip() for x in v], (v.shape[0], 1))
        if v.dtype.kind == "b":
            return ret(v.reshape(v.shape if v.ndim == 2 else (1, -1))) if v.size else np.zeros((0, 0), bool)
        if v.dtype.kind in "fiu":
            if np.iscomplexobj(v):
                return Tainted("a complex array read from a MAT-file")
            a = v.astype(float) if v.ndim == 2 else v.astype(float).reshape(1, -1) if v.ndim == 1 else v.astype(float)
            if a.ndim > 2:
                return Tainted("an N-dimensional array read from a MAT-file")
            return ret(a)
        return Tainted(f"a MAT-file value of class {v.dtype}")
    if isinstance(v, (bool, np.bool_)):
        return np.bool_(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return _F(v)
    if isinstance(v, str):
        return v
    return Tainted(f"a MAT-file value of type {type(v).__name__}")


# ======================================================================================= the workspace
#: Dynare statements that only declare (they run through ``Model._exec_statement``)
DECLARATIONS = {"param_init", "initval", "endval", "histval", "shocks", "mshocks", "varobs", "estimated_params",
                "estimated_params_init", "estimated_params_bounds", "estimated_params_remove", "occbin_constraints",
                "observation_trends", "estimation_data", "planner_objective", "osr_params", "optim_weights",
                "osr_params_bounds", "homotopy", "load_params_and_steady_state", "heteroskedastic_shocks",
                "conditional_forecast_paths", "perfect_foresight_controlled_paths", "initval_file", "histval_file",
                "set_time", "shock_groups", "init2shocks", "filter_initial_state"}
#: statements that print or write files only: skipped
IGNORED = {"write_latex_dynamic_model", "write_latex_static_model", "write_latex_original_model",
           "write_latex_steady_state_model", "collect_latex_files", "write_latex_definitions",
           "write_latex_parameter_table", "write_latex_prior_table", "model_info", "rplot", "resid", "model_diagnostics",
           "plot_shock_decomposition", "plot_conditional_forecast", "occbin_graph", "dsample", "dynatype", "dynasave",
           "occbin_write_regimes_xls", "squeeze_shock_decomposition"}
#: statements whose options Dynare's driver writes into options_ (the rest declare)
_NO_OPTIONS_KINDS = {"param_init", "initval", "endval", "histval", "shocks", "mshocks", "estimated_params",
                     "estimated_params_init", "estimated_params_bounds", "varobs"}
#: options of ``steady`` (and ``check``, which takes them) that the steady-state code reads
STEADY_KEYS = {"solve_algo", "solve_tolf", "solve_tolx", "steadystate.nocheck", "steady.maxit", "homotopy_mode",
               "homotopy_steps", "homotopy_force_continue", "homotopy_from_initval_to_endval", "qz_criterium"}
#: options_ fields whose pydyn readers have another default than Dynare: always passed on
_ALWAYS_OPTIONS = ("order",)


class _Core(CommandsMixin):
    """The workspace of a .mod file run by :func:`dynare`: ``M_``, ``options_``, ``oo_``, the MATLAB variables
    (``vars``), the :class:`pydyn.Model` (``model``) and Dynare's commands as methods."""

    def __init__(self, model: Model, mod_file, *, shocks=None, seed=None):
        from .octave_rng import OctaveRandom
        from .results_m import build_M
        self.model = model
        self.mod_file = Path(mod_file)
        self.name = self.mod_file.stem
        self.refused: dict = {}
        self.touched = set(_ALWAYS_OPTIONS)
        self._json_opts: dict = {}
        self._flat: dict = {}
        self.shocks = dict(shocks or {})            # {k: exo path}: shocks reused by the k-th stoch_simul with periods > 0
        self._n_simul = 0
        self.seed = 0 if seed is None else seed
        self.rng = OctaveRandom(self.seed)         # Dynare's default stream under Octave: randn('state', 0)
        self.last_k = -1
        self.current = None
        self.warnings = []
        self.host = _LiveHost(self)
        mod_dir = model.json_dir.parents[2]
        self.I = LiveInterp(mod_dir, self.host, self)
        self.I.session = self
        self.I.base.vars = _ml.Workspace(model)
        self.I.base.global_names = set(_HOST_GLOBALS)
        model._matlab = _LiveMatlab(self)
        opts = default_options()
        opts["dirname"] = self.name
        opts["dynare_version"] = "7.2"
        opts["nograph"] = np.bool_(True)
        if model.linear:
            opts["linear"] = np.bool_(True)
        self.I.globals["M_"] = tag_tree(build_M(model, self.name), "M_")
        self.I.globals["options_"] = tag_tree(opts, "options_")
        self.I.globals["oo_"] = tag_tree(self._initial_oo(), "oo_")
        self._flat = flatten(self.options_)
        self.functions = self._build_functions()
        self.noops = _NOOP_NAMES
        self._last_ss = None
        self._stmt_keys = {}
        self._push_all()

    # ------------------------------------------------------------------ the three structs
    @property
    def M_(self):
        return self.I.globals["M_"]

    @property
    def options_(self):
        return self.I.globals["options_"]

    @property
    def oo_(self):
        return self.I.globals["oo_"]

    @property
    def vars(self):
        """The MATLAB workspace of the .mod file (dict)."""
        return self.I.base.vars

    def set_global(self, name, value):
        if not isinstance(value, Struct):
            raise Unsupported(f"{name} is replaced by a value that is not a struct", "results")
        if not isinstance(value, RStruct) or getattr(value, "_dyn", None) != name:
            value = tag_tree(value, name)
        self.I.globals[name] = value

    def set_M(self, path, value):
        self.I.globals["M_"] = struct_set(self.M_, path, value)

    def set_oo(self, path, value):
        self.I.globals["oo_"] = struct_set(self.oo_, path, value)

    def set_options(self, path, value):
        self.I.globals["options_"] = struct_set(self.options_, path, value)

    def refuse(self, tag, names, why):
        for n in names:
            self.refused[(tag, n)] = why

    def unrefuse(self, tag, names):
        for n in names:
            self.refused.pop((tag, n), None)

    def _initial_oo(self):
        m = self.model
        oo = rstruct("oo_")
        oo["dynare_version"] = "7.2"
        oo["exo_simul"] = empty()
        oo["endo_simul"] = empty()
        oo["dr"] = empty()
        oo["exo_steady_state"] = col(np.zeros(m.n_exo)) if m.n_exo else empty()
        oo["exo_det_steady_state"] = col(np.zeros(m.n_exo_det)) if m.n_exo_det else empty()
        oo["exo_det_simul"] = empty()
        oo["initial_steady_state"] = empty()
        oo["initial_exo_steady_state"] = empty()
        oo["gui"] = Struct(ran_estimation=np.bool_(False), ran_stoch_simul=np.bool_(False),
                           ran_calib_smoother=np.bool_(False), ran_perfect_foresight=np.bool_(False),
                           ran_shock_decomposition=np.bool_(False), ran_realtime_shock_decomposition=np.bool_(False))
        oo["steady_state"] = col(np.zeros(m.n_endo))
        return oo

    # ------------------------------------------------------------------ MATLAB <-> model
    def pull(self):
        """MATLAB-side state (M_, options_, oo_) into the model, before a Dynare statement or function runs."""
        m, M = self.model, self.M_
        try:
            p = np.asarray(farr(M["params"]), float).ravel(order="F")
            if p.size != len(m.param_names):
                raise ResultsError(f"M_.params must have {len(m.param_names)} elements (it has {p.size})")
            m.params[:] = p
            S = np.asarray(farr(M["Sigma_e"]), float)
            if S.shape != (m.n_exo, m.n_exo):
                raise ResultsError(f"M_.Sigma_e must be {m.n_exo}x{m.n_exo}")
            m.Sigma_e[:] = S
            C = np.asarray(farr(M["Correlation_matrix"]), float)
            if C.shape == (m.n_exo, m.n_exo):
                m.Correlation_matrix = C.copy()
            H = np.asarray(farr(M["H"]), float)
            if m.varobs and H.shape == (len(m.varobs), len(m.varobs)):
                m.measurement_errors = {v: float(H[i, i]) for i, v in enumerate(m.varobs) if H[i, i] != 0}
        except Unsupported:
            raise
        self._pull_steady()
        self._pull_options()

    def _pull_steady(self):
        """oo_.steady_state / oo_.exo_steady_state into the model's initval (guess) and its record of oo_."""
        m, oo = self.model, self.oo_
        ys = np.asarray(farr(oo["steady_state"]), float).ravel(order="F")
        xs = np.asarray(farr(oo["exo_steady_state"]), float).ravel(order="F") if m.n_exo else np.zeros(0)
        if ys.size != m.n_endo or xs.size != m.n_exo:
            raise ResultsError("oo_.steady_state / oo_.exo_steady_state have the wrong size")
        last = self._last_ss
        if last is not None and (not np.array_equal(last[0], ys, equal_nan=True) or not np.array_equal(last[1], xs, equal_nan=True)):
            # MATLAB code assigned oo_.steady_state / oo_.exo_steady_state: later steady-state computations start
            # from it and the perfect-foresight replay sees the new values (as the loader's oo_write does)
            names = list(m.endo_names) if not np.array_equal(last[0], ys, equal_nan=True) else list(m.exo_names)
            vals = ys if names == list(m.endo_names) else xs
            ev = {k: float(v) for k, v in zip(names, vals)}
            m._oo.update(ev)
            m._oo_assign(ev)
            m._oo_events.append((self.last_k, "steady_state" if names == list(m.endo_names) else "exo_steady_state", ev))
        for v, x in zip(m.endo_names, ys):
            m._oo[v] = float(x)
            m.initval[v] = float(x)
        for v, x in zip(m.exo_names, xs):
            m._oo[v] = float(x)
            m.initval[v] = float(x)
        self._last_ss = (ys.copy(), xs.copy())

    def _pull_options(self):
        """options_ is the source of truth for re-runs: what any statement or MATLAB line wrote becomes the
        options the model's statement-options dicts hold (pydyn.options.accumulated)."""
        flat = flatten(self.options_)
        for k, v in flat.items():
            old = self._flat.get(k, _MISSING)
            if old is _MISSING or not same_value(v, old):
                self.touched.add(k)
        self._flat = flat
        legacy = {}
        for k in self.touched:
            if k not in flat:
                continue
            v = flat[k]
            orig = self._json_opts.get(k)
            if orig is not None and same_value(v, orig[0]):
                legacy[k] = orig[1]
                continue
            pv = py_value(v)
            if isinstance(pv, list) and not pv:
                continue                                # [] is "not set" in options_
            legacy[k] = pv
        m = self.model
        steady = {k: v for k, v in legacy.items() if k in STEADY_KEYS}
        for kind in ("steady", "check"):
            m.options[kind] = dict(steady)
            m._options_parsed[kind] = m.options[kind]
        for kind in ("stoch_simul", "estimation", "calib_smoother"):
            m.options[kind] = dict(legacy)
            m._options_parsed[kind] = m.options[kind]
        m.options_sequence = [(kind, m.options[kind]) for kind in ("steady", "check", "stoch_simul", "estimation",
                                                                    "calib_smoother")]
        # options_.x = ... assignments, which some readers (forecast, perfect foresight, steady-state files) look at
        m.matlab_options = _tree(flat)

    def push(self):
        """The model's calibration into M_ (after a statement that may have changed it)."""
        m = self.model
        self.set_M("params", col(m.params))
        S = mat(m.Sigma_e)
        self.set_M("Sigma_e", S)
        self.set_M("Correlation_matrix", mat(m.Correlation_matrix))
        self.set_M("sigma_e_is_diagonal", np.bool_(not np.any(m.Sigma_e - np.diag(np.diag(m.Sigma_e)))))
        if m.varobs:
            h = np.zeros((len(m.varobs), len(m.varobs)))
            for i, v in enumerate(m.varobs):
                h[i, i] = m.measurement_errors.get(v, 0.0)
            self.set_M("H", mat(h) if np.any(h != 0) else _F(0.0))      # Dynare: the scalar 0 without measurement errors

    def _push_all(self):
        self.push()
        self.push_steady()

    def push_steady(self):
        m = self.model
        ys = np.array([m._oo.get(v, 0.0) for v in m.endo_names], float)
        xs = np.array([m._oo.get(v, 0.0) for v in m.exo_names], float)
        self.set_oo("steady_state", col(ys))
        if m.n_exo:
            self.set_oo("exo_steady_state", col(xs))
        self._last_ss = (ys.copy(), xs.copy())

    def write_options(self, opts):
        """Dynare's driver writes the options of a statement into options_ (``options_.x = ...``)."""
        for key, val in (opts or {}).items():
            v = option_value(val)
            self.set_options(key, v)
            self._json_opts[key] = (v, val)
            self.touched.add(key)
        self._flat = flatten(self.options_)

    # ------------------------------------------------------------------ the run
    def run(self):
        m = self.model
        sts = m.statements
        if any(st["statementName"] in ("native", "verbatim") for st in sts):
            lines = [st["string"].replace("occbin.unpack_simulations", "occbin_unpack_simulations")
                     if st["statementName"] in ("native", "verbatim") else ("dyn", k)
                     for k, st in enumerate(sts)]
            try:
                body, funcs = _ml.parse(lines)
            except _ml.ParseError as exc:
                raise ResultsError(f"MATLAB code in the .mod file cannot be parsed: {exc}") from None
        else:
            body, funcs = [_ml.Dyn(k, None) for k in range(len(sts))], {}
        self.I.base.funcs = funcs
        with np.errstate(**_ml._ERRSTATE):
            self._guard(lambda: self.I.exec_block(body, self.I.base), top=True)
        return self

    def _guard(self, fn, top=False, what=""):
        """Run ``fn``, turning the interpreter's exceptions into ResultsError."""
        try:
            return fn()
        except _ml._ReturnSig:
            if not top:
                raise
        except (_ml._BreakSig, _ml._ContinueSig):
            raise ResultsError("MATLAB break/continue outside a loop in the .mod file") from None
        except ResultsError:
            raise
        except MatlabError as exc:
            raise ResultsError(f"MATLAB error in the .mod file{self._where()}{what}: {exc}", str(exc)) from None
        except Unsupported as u:
            raise ResultsError(f"cannot run the .mod file{self._where()}{what}: {u}") from None
        except _ml.PY_ERRORS as exc:
            if _ml._matlab_error(exc, "error") is None:
                raise
            raise ResultsError(f"MATLAB error in the .mod file{self._where()}{what}: {exc}", str(exc)) from None
        except ArithmeticError as exc:
            raise ResultsError(f"arithmetic error in the .mod file{self._where()}{what}: {exc}") from None

    def _where(self):
        k = self.last_k
        if k is not None and 0 <= k < len(self.model.statements):
            return f" (after the statement <{self.model.statements[k]['statementName']}> number {k + 1})"
        return ""

    def _histval_to_M(self, st):
        """M_.endo_histval / M_.exo_histval: one column per lag (Dynare: column maximum_lag + lag)."""
        m = self.model
        ml = max(int(m.info["maximum_lag"]), 1)
        endo, exo = np.zeros((m.n_endo, ml)), np.zeros((m.n_exo, ml))
        for (name, lag), v in m._stmt_values.get(id(st), {}).items():
            col_ = ml + int(lag) - 1
            if not 0 <= col_ < ml:
                continue
            if name in m._ie:
                endo[m._ie[name], col_] = v
            elif name in m._ix:
                exo[m._ix[name], col_] = v
        self.set_M("endo_histval", mat(endo))
        self.set_M("exo_histval", mat(exo))

    def exec_dyn(self, k):
        """One Dynare statement of the file."""
        m = self.model
        st = m.statements[k]
        kind = st["statementName"]
        self.last_k = k
        self.pull()
        try:
            if kind in IGNORED:
                pass
            elif kind in DECLARATIONS:
                self.declare(st)
            else:
                handler = getattr(self, "st_" + kind, None)
                if handler is None:
                    raise ResultsError(f"the Dynare statement <{kind}> is not supported by pydyn.dynare()")
                handler(st)
        except ModelError as exc:
            raise ResultsError(f"<{kind}>: {exc}") from None
        self.push()

    def declare(self, st):
        m = self.model
        kind = st["statementName"]
        n0 = len(m.unhandled_statements)
        if kind == "endval" and "learnt_in" not in st:
            self.set_oo("initial_steady_state", self.oo_["steady_state"])
            if m.n_exo:
                self.set_oo("initial_exo_steady_state", self.oo_["exo_steady_state"])
        m._exec_statement(st)
        if len(m.unhandled_statements) > n0:
            raise ResultsError(f"the Dynare statement <{kind}> is not supported by pydyn.dynare() "
                               f"({m.unhandled_statements[-1]})")
        if kind not in _NO_OPTIONS_KINDS:
            self.write_options(st.get("options"))
        if kind in ("initval", "endval", "load_params_and_steady_state", "homotopy") or \
                (kind == "steady" and st.get("options", {}).get("homotopy_mode")):
            self.push_steady()
        if kind == "histval" and "learnt_in" not in st:
            self._histval_to_M(st)
        if kind == "param_init":
            self.vars[st["name"]] = _F(m.params[m._ip[st["name"]]])      # the driver also defines the variable
        if kind == "varobs":
            names = list(st.get("varobs", m.varobs))
            self.set_options("varobs", cellstr(names))
            self.set_options("varobs_id", col([m._ie[v] + 1 for v in names]))


_MISSING = object()
_NOOP_NAMES = set(_ml.PLOT_FUNCS) | {"hold", "drawnow", "pause", "close", "clf", "figure", "fclose", "dyn_figure", "addpath", "rmpath"}
_NOOP_NAMES -= {"set", "get", "line", "text", "fill", "area", "bar", "hist", "histogram", "contour", "print", "type"}


def _tree(flat):
    """Nested struct from ``{dotted name: value}``."""
    out = Struct()
    for k, v in flat.items():
        parts = k.split(".")
        d = out
        for p in parts[:-1]:
            if not isinstance(d.get(p), Struct):
                d[p] = Struct()
            d = d[p]
        d[parts[-1]] = v
    return out


class _LiveMatlab:
    """What ``Model`` expects of its MATLAB session (``model._matlab``): the oo_ values an initval block reads, and
    the evaluation of expressions that need the interpreter."""

    def __init__(self, res):
        self.res = res
        self.oo_written = set()

    def oo_value(self, name):
        m = self.res.model
        return _F(m._oo.get(name, 0.0))

    def eval_value(self, text, local=None, vector=False):
        res = self.res
        m = res.model
        text = str(text)
        try:
            cached = _ml._EVAL_CACHE.get(text)
            body, funcs = cached if cached is not None else _ml.parse(text)
            if funcs or len(body) != 1 or type(body[0]) is not _ml.ExprStmt:
                raise _ml.ParseError("not an expression")
            if cached is None and len(_ml._EVAL_CACHE) < 5000:
                _ml._EVAL_CACHE[text] = (body, funcs)
            sc = _ml.Scope(dict(res.vars), funcs=res.I.base.funcs)
            sc.global_names = set(_HOST_GLOBALS)
            for name in {n.name for n in _ml.walk(body[0].e) if type(n) is _ml.Id}:
                if name in m._ip:
                    sc.vars[name] = _F(m.params[m._ip[name]])
            for k, v in (local or {}).items():
                sc.vars[k] = _F(v)
            with np.errstate(**_ml._ERRSTATE):
                res.I.scope_stack.append(sc)
                try:
                    v = res.I.ev(body[0].e, sc)
                    if vector:
                        a = np.asarray(farr(v), float)
                        if a.ndim == 2 and min(a.shape) > 1:
                            raise MatlabError("value must be a scalar or a vector")
                        return a.ravel(order="F")
                    return float(_ml.scalar(v, "value"))
                finally:
                    res.I.scope_stack.pop()
        except (_ml.ParseError, Unsupported) as exc:
            raise ModelError(f"cannot evaluate {text[:80]!r}: {exc}") from None
        except MatlabError as exc:
            raise ModelError(f"MATLAB error while evaluating {text[:80]!r}: {exc}") from None


# ======================================================================================= functions of the interpreter
def _names(arg):
    """A variable list given as a cell array of names, a name, or several names."""
    if isinstance(arg, Cell):
        return [str(x) for x in arg.items]
    if isinstance(arg, str):
        return arg.split()
    if isinstance(arg, np.ndarray) and arg.size == 0:
        return []
    raise Unsupported("a variable list must be a cell array of names")


class _FunctionsMixin:
    def _build_functions(self):
        return {
            "load": self.fn_load,
            "steady": self.fn_steady, "steady_": self.fn_steady,
            "check": self.fn_check,
            "stoch_simul": self.fn_stoch_simul,
            "set_dynare_seed": self.fn_seed,
            "perfect_foresight_setup": self.fn_command("perfect_foresight_setup"),
            "perfect_foresight_solver": self.fn_command("perfect_foresight_solver"),
            "occbin_solver": self.fn_command("occbin_solver"),
            "isfield": self.fn_isfield,
            "occbin_unpack_simulations": self.fn_unpack_simulations,
            "resol": self.fn_resol,
        }

    def fn_command(self, kind):
        def run(args, nargout):
            if args and isinstance(args[0], Struct):
                # oo_ = perfect_foresight_solver(M_, options_, oo_): the structs are values; the globals are untouched
                if len(args) < 3 or not all(isinstance(a, Struct) for a in args[:3]):
                    raise Unsupported(f"{kind}(M_, options_, oo_) needs the three structs")
                saved = {k: self.I.globals[k] for k in _HOST_GLOBALS}
                for k, v in zip(("M_", "options_", "oo_"), args[:3]):
                    self.set_global(k, v)
                try:
                    self.run_command({"statementName": kind, "options": {}})
                    out = self.oo_
                finally:
                    for k, v in saved.items():
                        self.I.globals[k] = v
                if nargout > 1:
                    raise Unsupported(f"{kind}: the second output (ts) is not available")
                return [out]
            self.run_command({"statementName": kind, "options": {}})
            return []
        return run

    def run_command(self, st):
        """Run a command called from MATLAB code like the statement ``st`` (options come from options_)."""
        self.last_k = self.last_k
        self.pull()
        handler = getattr(self, "st_" + st["statementName"], None)
        if handler is None:
            raise ResultsError(f"{st['statementName']} is not supported by pydyn.dynare()")
        try:
            handler(st)
        except ModelError as exc:
            raise ResultsError(f"{st['statementName']}: {exc}") from None
        self.push()

    def fn_steady(self, args, nargout):
        self.run_command({"statementName": "steady", "options": {}})
        return []

    def fn_check(self, args, nargout):
        if args and not isinstance(args[0], Struct):
            raise Unsupported("check(...) called with arguments that are not M_, options_, oo_")
        self.pull()
        eig, ok = self.compute_check()
        self.push()
        if nargout == 0 and not args:
            return []
        info = _F(0.0)
        return [col(eig), np.bool_(ok), info]

    def fn_stoch_simul(self, args, nargout):
        var_list = []
        if args and isinstance(args[0], Struct):
            # [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, var_list_)
            if len(args) < 3 or not all(isinstance(a, Struct) for a in args[:3]):
                raise Unsupported("stoch_simul(M_, options_, oo_, var_list) needs the three structs")
            var_list = _names(args[3]) if len(args) > 3 else []
            saved = {k: self.I.globals[k] for k in _HOST_GLOBALS}
            for k, v in zip(("M_", "options_", "oo_"), args[:3]):
                self.set_global(k, v)
            try:
                self.run_command({"statementName": "stoch_simul", "options": {}, "symbol_list": var_list})
                out = [_F(0.0), self.oo_, self.options_, self.M_]
            finally:
                for k, v in saved.items():
                    self.I.globals[k] = v
            return out[:max(nargout, 1)]
        for a in args:
            var_list += _names(a)
        self.run_command({"statementName": "stoch_simul", "options": {}, "symbol_list": var_list})
        return []

    def fn_seed(self, args, nargout):
        from .octave_rng import OctaveRandom
        a = args[0] if args else "default"
        if isinstance(a, str):
            if a.lower() == "default":
                self.seed = 0
            elif a.lower() == "reset":
                pass
            else:
                import time
                self.seed = int(time.time()) % (2 ** 32 - 1)
        else:
            self.seed = int(_ml.scalar(a))
        self.rng = OctaveRandom(self.seed)
        return []

    def fn_isfield(self, args, nargout):
        if len(args) == 2 and isinstance(args[0], Struct) and isinstance(args[1], str) and args[1] not in args[0]:
            tag = getattr(args[0], "_dyn", None)
            why = self.refused.get((tag, args[1])) if tag else None
            if why is not None:
                raise Unsupported(f"{tag}.{args[1]}: Dynare fills this field and pydyn does not ({why})", "results")
        return _ml.BUILTINS["isfield"](self.I, args, nargout)

    def fn_load(self, args, nargout, sc=None):
        """``s = load('file.mat')`` / ``load file.mat``: MAT-files next to the .mod file."""
        from scipy.io import loadmat
        names = [a for a in args if isinstance(a, str)]
        if not names or len(names) != len(args):
            raise Unsupported("load with computed arguments", "file")
        fname = names[0]
        if any(n.startswith("-") for n in names) or ".." in fname or fname.startswith("/"):
            raise Unsupported("load with options or a path", "file")
        base = self.model.json_dir.parents[2]
        path = next((p for p in (base / fname, base / (fname + ".mat")) if p.is_file()), None)
        if path is None:
            raise MatlabError(f"load: unable to find file {fname}")
        try:
            data = loadmat(str(path), squeeze_me=False, chars_as_strings=True, struct_as_record=True)
        except Exception as exc:  # noqa: BLE001
            raise Unsupported(f"load: {path.name} is not a MAT-file that can be read ({exc})", "file") from None
        wanted = [k for k in data if not k.startswith("__") and (len(names) == 1 or k in names[1:])]
        if nargout > 0:
            s = Struct()
            for k in wanted:
                s[k] = mat_value(data[k])
            return s
        sc = sc or self.I.scope_stack[-1]
        for k in wanted:
            if k in _HOST_GLOBALS:
                raise Unsupported(f"'load {fname}' would replace {k}", "file")
            sc.vars[k] = mat_value(data[k])
        return []



class Results(_Core, _FunctionsMixin, EstimationMixin):
    """The workspace of a .mod file run by :func:`dynare` (see the module docstring).

    Dynare's commands are methods acting on the live workspace: keyword arguments are written into
    ``options_`` first (``ws.stoch_simul(order=1, irf=20)`` is ``options_.order = 1; options_.irf = 20;
    stoch_simul``) and stay there, as in Dynare."""

    def _command(self, kind, opts, **extra):
        st = {"statementName": kind, "options": dict(opts), **extra}
        self.write_options(st["options"])
        self.run_command({**st, "options": {}})
        return self

    def set_param_value(self, name, value):
        """Dynare's ``set_param_value``: M_.params and the MATLAB variable of that name."""
        self._guard(lambda: self.host.set_param(name, _F(value)))
        self.vars[name] = _F(value)
        return self

    def get_param_by_name(self, name):
        return float(self.host.get_param(name))

    def steady(self, **opts):
        return self._guard_command("steady", opts)

    def check(self, **opts):
        return self._guard_command("check", opts)

    def stoch_simul(self, var_list=None, **opts):
        return self._guard_command("stoch_simul", opts, symbol_list=list(var_list or []))

    def perfect_foresight_setup(self, **opts):
        return self._guard_command("perfect_foresight_setup", opts)

    def perfect_foresight_solver(self, **opts):
        return self._guard_command("perfect_foresight_solver", opts)

    def occbin_solver(self, **opts):
        return self._guard_command("occbin_solver", opts)

    def _guard_command(self, kind, opts, **extra):
        return self._guard(lambda: self._command(kind, opts, **extra))


def dynare(mod_file, *, shocks=None, seed=None, preprocessor=None, functions=None, steady_state_fn=None,
           run=True) -> Results:
    """Run a .mod file as Dynare does and return its workspace (:class:`Results`: ``M_``, ``options_``, ``oo_``).

    ``mod_file`` is a path (the preprocessor runs on it, as ``pydyn.run_preprocessor`` does) or the folder of
    JSON files of an earlier run. ``shocks`` maps the number k of a ``stoch_simul`` with ``periods > 0`` to the
    exogenous path it should use (rows = periods): without it the shocks are Octave's ``randn`` draws from
    Dynare's default seed (``randn('state', 0)``), in Dynare's order, so that a run reproduces Dynare run under
    Octave; ``seed`` replaces that seed. ``functions`` and ``steady_state_fn`` are as for ``pydyn.Model``.

    Raises :class:`ResultsError` where Dynare would stop with an error (``error()`` in a MATLAB line, a failed
    solution) and where pydyn cannot run a statement or MATLAB line faithfully."""
    from .preprocessor import run_preprocessor
    p = Path(mod_file)
    if p.is_dir():
        json_dir = p
        mod_path = p.parents[2] / (p.parents[1].name + ".mod")
    else:
        json_dir = run_preprocessor(p, preprocessor=preprocessor)
        mod_path = p
    model = Model(json_dir, functions=functions, steady_state_fn=steady_state_fn, run_statements=False)
    res = Results(model, mod_path, shocks=shocks, seed=seed)
    return res.run() if run else res
