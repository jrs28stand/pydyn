# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# What Dynare 7.2's preprocessor writes for var_model, trend_component_model, var_expectation_model and pac_model
# (the M_.var / M_.trend_component / M_.var_expectation / M_.pac assignments of <model>/+<model>/driver.m and the
# functions varmatrices.m, trend_component_ar_a0.m, pac_target_coefficients.m; Copyright (C) Dynare Team,
# GPL-3.0-or-later).
"""The description of VAR, trend component, VAR expectation and PAC models.

The JSON output of the preprocessor holds only the declarations of these models (names and equation tags); what
Dynare's functions work with (the left- and right-hand side variables of each equation, the maximum lags, the
indices of the parameters of the PAC equation, ...) is written by the preprocessor as MATLAB code only.  So the
preprocessor runs a second time, in a scratch folder, to write the MATLAB files; the assignments to ``M_.var``,
``M_.trend_component``, ``M_.var_expectation`` and ``M_.pac`` are read from ``driver.m`` and the three small
functions that give the matrices of the models are kept as MATLAB text for the interpreter.  The two runs are
checked against each other (names of the endogenous variables and parameters)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import matlab as _ml
from .matlab import Cell, Struct, Tainted, Unsupported
from .preprocessor import find_preprocessor

PREFIXES = ("M_.var.", "M_.trend_component.", "M_.var_expectation.", "M_.pac.", "M_.exo_partitions", "M_.endo_partitions",
            "M_.param_partitions")
FUNCTION_FILES = ("varmatrices", "trend_component_ar_a0", "pac_target_coefficients",
                  # the model's own functions, callable from MATLAB code as <model>.<name>(...)
                  "set_auxiliary_variables", "dynamic_resid", "dynamic_resid_tt", "dynamic_g1", "dynamic_g1_tt",
                  "static_resid", "static_resid_tt", "static_g1", "static_g1_tt", "epilogue_dynamic", "epilogue_static")
#: the files among them that are the model's own functions (not the sub-model ones above)
MODEL_FUNCTIONS = FUNCTION_FILES[3:]


class SubModelError(Exception):
    pass


class StructArray(Tainted):
    """A MATLAB struct array (``M_.pac.<name>.components``).  The interpreter has none: pydyn's code reads ``elems``
    (a list of Struct); MATLAB code that touches the array is refused."""
    __slots__ = ("elems",)

    def __init__(self, elems=None):
        super().__init__("a struct array (M_.pac.*.components, ...): pydyn's interpreter has none")
        self.elems = list(elems or [])


class Generated:
    """The MATLAB output of the preprocessor for a model: the lines of ``driver.m`` that describe the sub-models and the
    text of the generated functions."""

    def __init__(self, lines, functions, endo_names, param_names):
        self.lines = lines
        self.functions = functions
        self.endo_names = endo_names
        self.param_names = param_names


_CACHE: dict = {}


def _driver_names(text):
    def names(var):
        out = []
        for m in re.finditer(r"^M_\.%s_names\((\d+)\)\s*=\s*\{'([^']*)'\}" % var, text, re.M):
            out.append((int(m.group(1)), m.group(2)))
        return [n for _, n in sorted(out)]
    return names("endo"), names("param")


def run_generator(mod_path, preprocessor=None):
    """Run the preprocessor on ``mod_path`` in a scratch folder and return the :class:`Generated` it writes."""
    mod_path = Path(mod_path).resolve()
    if not mod_path.exists():
        raise SubModelError(f"{mod_path} not found: the VAR / trend component / PAC models need the .mod file "
                            "(the JSON output of the preprocessor does not hold their matrices)")
    key = (str(mod_path), mod_path.stat().st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    exe = find_preprocessor(preprocessor)
    name = mod_path.stem
    tmp = Path(tempfile.mkdtemp(prefix="pydyn_sub_"))
    try:
        for entry in mod_path.parent.iterdir():
            if entry.name in (name, "+" + name):
                continue
            try:
                os.symlink(entry, tmp / entry.name)
            except OSError:
                pass
        res = subprocess.run([exe, mod_path.name, "json=compute", "nolog", "matlabroot=/usr", "mexext=mex"], cwd=tmp, capture_output=True, text=True)
        if res.returncode != 0 or not (tmp / ("+" + name) / "driver.m").exists():
            raise SubModelError(f"the preprocessor failed on {mod_path.name} (second run):\n{res.stdout[-500:]}\n{res.stderr[-500:]}")
        drv = (tmp / ("+" + name) / "driver.m").read_text(errors="replace")
        lines = [ln.strip() for ln in drv.splitlines() if ln.startswith(PREFIXES)]
        funcs = {}
        for fn in FUNCTION_FILES:
            f = tmp / ("+" + name) / (fn + ".m")
            if f.exists():
                funcs[fn] = f.read_text(errors="replace")
        endo, par = _driver_names(drv)
        gen = Generated(lines, funcs, endo, par)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _CACHE[key] = gen
    return gen


# ----------------------------------------------------------------------------------------- the driver's assignments
def _path_items(node):
    """Flatten the left-hand side of ``M_.a.b(2).c{1}.d`` into [('f','a'), ('f','b'), ('p',2), ('f','c'), ('b',1), ...]."""
    items = []
    while True:
        t = type(node)
        if t is _ml.Field:
            items.append(("f", node.name))
            node = node.base
        elif t is _ml.Index:
            if len(node.args) != 1 or type(node.args[0]) is not _ml.Num:
                raise SubModelError("unexpected index in the driver's description of a model")
            items.append(("b" if node.brace else "p", int(node.args[0].v)))
            node = node.base
        elif t is _ml.Id:
            items.append(("f", node.name))
            break
        else:
            raise SubModelError("unexpected left-hand side in the driver's description of a model")
    return items[::-1]


def _set(cur, items, value):
    kind, k = items[0]
    last = len(items) == 1
    if kind == "f":
        if not isinstance(cur, Struct):
            cur = Struct()
        cur[k] = value if last else _set(cur.get(k), items[1:], value)
        return cur
    if kind == "p":
        arr = cur if isinstance(cur, StructArray) else StructArray()
        while len(arr.elems) < k:
            arr.elems.append(Struct())
        arr.elems[k - 1] = value if last else _set(arr.elems[k - 1], items[1:], value)
        return arr
    # brace: a cell, row-shaped as MATLAB grows one by brace indexing
    cell = cur if isinstance(cur, Cell) else Cell([], (0, 0))
    its = list(cell.items)
    while len(its) < k:
        its.append(np.zeros((0, 0)))
    its[k - 1] = value if last else _set(its[k - 1], items[1:], value)
    return Cell(its, (1, len(its)))


def apply_lines(lines, M_):
    """Run the ``M_.var...`` assignments on a copy of the struct ``M_`` (returned)."""
    out = M_.clone() if isinstance(M_, Struct) else Struct()
    interp = _ml.Interpreter(None)
    for ln in lines:
        try:
            body, _ = _ml.parse(ln)
        except _ml.ParseError as exc:
            raise SubModelError(f"cannot parse {ln[:80]!r}: {exc}") from None
        if len(body) != 1 or type(body[0]) is not _ml.Assign or len(body[0].targets) != 1:
            raise SubModelError(f"unexpected statement {ln[:80]!r}")
        st = body[0]
        sc = _ml.Scope({"M_": out}, is_base=False)
        try:
            val = interp.ev(st.rhs, sc)
        except (_ml.MatlabError, Unsupported) as exc:
            raise SubModelError(f"cannot evaluate {ln[:80]!r}: {exc}") from None
        items = _path_items(st.targets[0])
        out = _set(out, items[1:], val)
    return out
