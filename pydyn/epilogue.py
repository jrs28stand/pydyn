# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 2019-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: the epilogue files the preprocessor writes
# (preprocessor/src/ModelEquationBlock.cc: Epilogue::writeDynamicEpilogueFile, writeStaticEpilogueFile)
# and the dseries "from ... to ... do" loop they use (matlab/dseries/src/utilities/from/from.m).
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""The ``epilogue`` block: variables computed from simulated or smoothed series after the fact.

The preprocessor's JSON output has no trace of the block, so it is read from the macro-expanded .mod
file (the preprocessor is run with ``onlymacro``). As in Dynare, ``Epilogue.dynamic`` mirrors
``+<model>/epilogue_dynamic.m``: the definitions are applied in order; each one runs from the first
period at which every variable it uses has been observed for at least its maximum lag
(``firstobservedperiod + max lag``) to the last period, recursively when the variable appears on the
right-hand side; a definition that fails (a lead beyond the sample, a variable that is never observed)
leaves the series untouched (NaN for a new variable). ``Epilogue.static`` mirrors ``epilogue_static.m``
(lags and leads removed, all periods at once).

Dynare's dseries ``from`` loop writes the assigned series back with ``sprintf('%f')``, which rounds every
epilogue variable to 6 decimals (and the rounded values feed the later definitions);
``dynare_rounding=True`` reproduces this, otherwise the values are exact.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .expr import NAMESPACE, Translator

_BLOCK = re.compile(r"(?ims)^\s*epilogue\s*;(.*?)^\s*end\s*;")


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*|%[^\n]*", "", text)


def parse(text: str) -> list:
    """``[(name, expression)]`` of the epilogue blocks of a (macro-expanded) .mod file."""
    out = []
    for block in _BLOCK.findall(_strip_comments(text)):
        for stmt in block.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            m = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", stmt, flags=re.S)
            if not m:
                raise ValueError(f"epilogue: cannot read {stmt!r} (expected NAME = expression)")
            out.append((m.group(1), " ".join(m.group(2).split())))
    return out


def macro_expanded_text(mod_file, defines=(), preprocessor=None) -> str:
    """The .mod file after macro processing (``dynare-preprocessor file.mod onlymacro savemacro=...``)."""
    from .preprocessor import find_preprocessor
    mod_file = Path(mod_file).resolve()
    text = mod_file.read_text(errors="replace")
    if "@#" not in text and not defines:
        return text
    exe = find_preprocessor(preprocessor)
    fd, tmp = tempfile.mkstemp(suffix=".mod")
    os.close(fd)
    try:
        res = subprocess.run([exe, mod_file.name, "onlymacro", f"savemacro={tmp}", *defines], cwd=mod_file.parent,
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"preprocessor failed:\n{res.stdout}\n{res.stderr}")
        return Path(tmp).read_text(errors="replace")
    finally:
        os.unlink(tmp)


class _OutOfSample(Exception):
    pass


class Epilogue:
    """The epilogue definitions of a model. ``mod_file`` defaults to the .mod file the model's JSON was
    produced from; ``defines`` are the ``-D`` macro definitions it was preprocessed with."""

    def __init__(self, model, mod_file=None, defines=(), equations=None):
        self.model = model
        if equations is None:
            if mod_file is None:
                jd = Path(model.json_dir)
                mod_file = jd.parents[2] / f"{jd.parents[1].name}.mod"
            equations = parse(macro_expanded_text(mod_file, defines))
        self.equations = list(equations)
        self.names = [n for n, _ in self.equations]
        known = set(model.endo_names) | set(model.exo_names) | set(self.names)
        self._ip = dict(model._ip)
        self._dyn, self._lags, self._leads, self._uses = [], [], [], []
        for name, expr in self.equations:
            shifts = {}

            def resolve(v, shift, shifts=shifts):
                if v in known:
                    shifts.setdefault(v, []).append(shift)
                    return f"_x({v!r}, {shift})"
                if v in self._ip:
                    return f"params[{self._ip[v]}]"
                return None

            src = Translator(resolve, shiftable=known)(expr)
            self._dyn.append(compile(src, f"<epilogue {name}>", "eval"))
            self._uses.append(list(shifts))
            self._lags.append({v: max(0, -min(s)) for v, s in shifts.items()})
            self._leads.append({v: max(0, max(s)) for v, s in shifts.items()})
            st = Translator(lambda v, shift: (f"_S[{v!r}]" if v in known else
                                              (f"params[{self._ip[v]}]" if v in self._ip else None)),
                            shiftable=known)(expr)
            self._static = getattr(self, "_static", []) + [compile(st, f"<epilogue static {name}>", "eval")]

    def dynamic(self, series: dict, params=None, dynare_rounding=False) -> dict:
        """``epilogue_dynamic(params, ds)``: ``series`` maps names to 1-d arrays of equal length (NaN where
        not observed). Returns a new dict with the epilogue variables added or updated."""
        S = {k: np.array(v, float).ravel() for k, v in series.items()}
        T = len(next(iter(S.values())))
        params = np.asarray(self.model.params if params is None else params, float)
        ns = dict(NAMESPACE)
        for (name, _), code, uses, lags, leads in zip(self.equations, self._dyn, self._uses, self._lags, self._leads):
            if name not in S:
                S[name] = np.full(T, np.nan)
            try:
                begin = []
                for v in uses:
                    if v not in S:
                        raise _OutOfSample(f"{v} is not in the data")
                    obs = np.flatnonzero(~np.isnan(S[v]))
                    if obs.size == 0:
                        raise _OutOfSample(f"{v} is never observed")
                    begin.append(obs[0] + lags[v])
                if not begin:
                    raise _OutOfSample("no variable")              # Dynare: max() of nothing fails
                t1, t2 = max(begin), T - 1
                if t1 > t2:
                    raise _OutOfSample("empty range")
                # from.m: the leads of variables other than the assigned one must stay in the sample
                for v in uses:
                    if v != name and t2 + leads[v] > T - 1:
                        raise _OutOfSample(f"lead of {v} beyond the sample")
                data = {v: S[v].copy() for v in uses}
                data.setdefault(name, S[name].copy())
                recursive = name in uses
                if recursive and leads.get(name, 0) > 0:
                    raise _OutOfSample("forward-looking epilogue variable")
                lead_sign = 1 if (recursive or not dynare_rounding) else -1      # from.m: t-lead in the vector case

                def _x(v, shift):
                    k = t + (shift if shift <= 0 else lead_sign * shift)
                    if k < 0 or k >= T:
                        raise _OutOfSample(f"{v}({shift}) outside the sample")
                    return data[v][k]

                ns["_x"], ns["params"] = _x, params
                target = data[name]
                for t in range(t1, t2 + 1):
                    with np.errstate(all="ignore"):
                        val = eval(code, ns)  # noqa: S307 - expression of the .mod file
                    if np.iscomplexobj(val):
                        val = complex(val).real if complex(val).imag == 0 else np.nan
                    target[t] = val
            except (_OutOfSample, ArithmeticError, ValueError, TypeError):
                continue                                        # Dynare: try ... catch end
            if dynare_rounding:
                target = np.array([float(f"{x:f}") if np.isfinite(x) else x for x in target])
            S[name] = target
        return S

    def static(self, values: dict, params=None) -> dict:
        """``epilogue_static(params, ds)``: every definition evaluated with lags and leads removed, on whole
        series (or scalars, e.g. a steady state)."""
        S = {k: np.asarray(v, float) for k, v in values.items()}
        params = np.asarray(self.model.params if params is None else params, float)
        ns = dict(NAMESPACE)
        ns["params"] = params
        n = max((np.size(v) for v in S.values()), default=1)
        for (name, _), code in zip(self.equations, self._static):
            ns["_S"] = S
            with np.errstate(all="ignore"):
                val = np.asarray(eval(code, ns), float)  # noqa: S307
            S[name] = val if val.size == n else np.full(n, float(val)) if n > 1 else val
        return S

    def from_paths(self, endo, exo=None, params=None, dynare_rounding=False) -> dict:
        """Epilogue of simulated or smoothed paths: ``endo`` (T, n_endo) in declaration order (e.g.
        ``pydyn.simulate`` output, ``PerfectForesightResult.endo``, ``SmootherResult.smoothed`` plus the
        steady state), ``exo`` (T, n_exo) optional."""
        m = self.model
        endo = np.asarray(endo, float)
        S = {v: endo[:, i] for i, v in enumerate(m.endo_names)}
        if exo is not None:
            exo = np.asarray(exo, float)
            S.update({v: exo[:, i] for i, v in enumerate(m.exo_names)})
        return self.dynamic(S, params, dynare_rounding)
