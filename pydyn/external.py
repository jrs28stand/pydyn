# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 1996-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: matlab/jacob_element.m, matlab/hess_element.m, and the
# preprocessor's treatment of external functions (ExprNode.cc: ExternalFunctionNode,
# FirstDerivExternalFunctionNode, SecondDerivExternalFunctionNode).
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""External functions (``external_function`` statement) in the model block.

The preprocessor writes, for every call of an external function, "TEF" terms in the JSON:

* ``TEF_k = f(args)``, or ``[TEF_k, TEFD_k] = f(args)`` / ``[TEF_k, TEFD_k, TEFDD_k] = f(args)`` when
  the function itself returns its gradient (and Hessian): ``first_deriv_provided`` /
  ``second_deriv_provided`` without a name;
* ``TEFD_def_k = g(args)`` / ``TEFDD_def_k = h(args)`` when separate derivative functions are named;
* ``TEFD_fdd_k_i`` and ``TEFDD_fdd_k_i_j``: numerical derivatives, which Dynare computes with
  ``jacob_element`` (centred difference, step 1e-6) and ``hess_element`` (the five-point and
  seven-point formulas below, step 1e-6). They are reproduced here exactly.

Gradients are indexed from 0 in the JSON (``TEFD_0[1]`` is the derivative with respect to the second
argument). The preprocessor cannot differentiate external functions three times, so models with
external functions are limited to order 2 (as in Dynare).

Functions are looked up, in this order, in ``Model(functions=...)`` (Python callables) and in MATLAB
function files next to the .mod file (``name.m``, or ``+pkg/name.m`` for ``pkg.name``), which
``pydyn.matlab`` translates or interprets. A Python callable is called with the arguments as floats
and returns a float; when the .mod file declares that the function returns its own derivatives
(``first_deriv_provided`` without a name), it must return a tuple ``(value, gradient)`` or
``(value, gradient, hessian)``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import matlab

H_STEP = 1e-6          # jacob_element.m and hess_element.m
ALIAS = "_pydyn_ef{}"
_EXT_ENTRY = re.compile(r'"(?:external_function|first_deriv_external_function|second_deriv_external_function)"')


def repair_json(text: str) -> str:
    """Dynare 7.2 writes empty array elements in ``external_functions_temporary_terms_*`` when a
    function returns its own derivatives (``[{...}, , ]``)."""
    text = re.sub(r",\s*(?=[,\]])", "", text)
    return re.sub(r"\[\s*,", "[", text)


def function_names(texts) -> list:
    """Names of the external functions (and derivative functions) called in the JSON texts."""
    names = set()
    for text in texts:
        if not _EXT_ENTRY.search(text):
            continue
        for mo in re.finditer(r'"(external_function|first_deriv_external_function|second_deriv_external_function)"'
                              r'\s*:\s*\{[^{}]*?"value"\s*:\s*"([A-Za-z_][\w.]*)\(', text):
            names.add(mo.group(2))
    return sorted(names)


def substitute(text: str, names) -> str:
    """Rename the function calls to Python-safe aliases and turn ``TEFD_0[1]`` / ``TEFDD_0[1,1]`` into
    calls of the indexing helper ``_efi`` (the expression translator has no brackets)."""
    for k, name in enumerate(names):
        text = re.sub(r"(?<![\w.])" + re.escape(name) + r"\(", ALIAS.format(k) + "(", text)
    text = re.sub(r"\b(TEF\w+)\[(\d+),(\d+)\]", r"_efi(\1,\2,\3)", text)
    return re.sub(r"\b(TEF\w+)\[(\d+)\]", r"_efi(\1,\2)", text)


def term_names(block) -> set:
    out = set()
    for key, entries in block.items():
        if key.startswith("external_functions_temporary_terms"):
            for e in entries:
                d = next(iter(e.values()))
                for f in ("external_function_term", "external_function_term_d", "external_function_term_dd"):
                    if f in d:
                        out.add(d[f])
    return out


def section_lines(entries, tr, names):
    """``[(defined names, python line)]`` for the TEF entries of one section."""
    out = []
    for e in entries:
        (kind, d), = e.items()
        src = tr(d["value"])
        mo = re.match(r"_pydyn_ef(\d+)\(", src)
        if not mo or not src.endswith(")"):
            raise NotImplementedError(f"unexpected external function term {d}")
        name = names[int(mo.group(1))]
        args = src[mo.end():-1]
        sep = ", " if args.strip() else ""
        if kind == "external_function":
            outs = [d["external_function_term"]] + [d[f] for f in ("external_function_term_d", "external_function_term_dd")
                                                   if f in d]
            if "external_function_term_dd" in d and "external_function_term_d" not in d:
                raise NotImplementedError(f"external function {name}: a function that returns its second "
                                          "derivatives must also return its first derivatives")
            lhs = ", ".join(f"_tt_{t}" for t in outs)
            rhs = f"_pydyn_ext.call({name!r}, {len(outs)}{sep}{args})"
            if len(outs) == 1:
                rhs = f"_pydyn_ext.value({name!r}{sep}{args})"
        elif kind == "first_deriv_external_function":
            outs = [d["external_function_term"]]
            lhs = f"_tt_{outs[0]}"
            if d.get("analytic_derivative"):
                rhs = f"_pydyn_ext.call({name!r}, 1{sep}{args})"
            else:
                rhs = f"_pydyn_ext.jacob({name!r}, {int(d['wrt'])}{sep}{args})"
        elif kind == "second_deriv_external_function":
            outs = [d["external_function_term"]]
            lhs = f"_tt_{outs[0]}"
            if d.get("analytic_derivative"):
                rhs = f"_pydyn_ext.call({name!r}, 1{sep}{args})"
            else:
                rhs = f"_pydyn_ext.hess({name!r}, {int(d['wrt1'])}, {int(d['wrt2'])}{sep}{args})"
        else:
            raise NotImplementedError(f"unexpected external function term {kind}")
        out.append((set(outs), f"    {lhs} = {rhs}"))
    return out


def order_lines(ext, temps):
    """Merge TEF lines and temporary-term lines of a section so that each line comes after the
    lines defining the ``_tt_`` names it uses (external-function arguments may use temporary terms)."""
    items = [(defs, ln) for defs, ln in ext] + [({t}, ln) for t, ln in temps]
    defined_here = set().union(*(d for d, _ in items)) if items else set()
    done, out, pending = set(), [], list(items)
    while pending:
        for i, (defs, ln) in enumerate(pending):
            rhs = ln.split("=", 1)[1]
            uses = {u for u in re.findall(r"_tt_(\w+)", rhs)} & defined_here
            if uses <= done:
                out.append(ln)
                done |= defs
                del pending[i]
                break
        else:
            raise NotImplementedError("circular external-function terms")
    return out


def _scalar(v):
    if isinstance(v, np.ndarray) and v.size > 1:
        raise TypeError("external functions are evaluated one period at a time")
    return v


class ExternalFunctions:
    """Callable table used by the generated model code."""

    def __init__(self, functions: dict, mod_dir=None):
        self.functions = functions
        self.mod_dir = Path(mod_dir) if mod_dir is not None else None
        self._cache = {}

    def lookup(self, name):
        f = self._cache.get(name)
        if f is not None:
            return f
        f = self.functions.get(name)
        if f is None and self.mod_dir is not None:
            parts = name.split(".")
            path = self.mod_dir.joinpath(*[f"+{p}" for p in parts[:-1]], parts[-1] + ".m")
            if path.is_file():
                f = matlab.MatlabFunction(path)
        if f is None:
            raise NameError(f"external function {name!r} is not available: pass a Python callable with "
                            f"Model(functions={{{name!r}: f}}) or put {name.split('.')[-1]}.m next to the .mod file")
        self._cache[name] = f
        return f

    def _raw(self, name, nout, args):
        args = [_scalar(a) for a in args]
        f = self.lookup(name)
        if isinstance(f, matlab.MatlabFunction):
            if nout == 1:
                out = f(*[float(a) for a in args])
                return (out,)
            with matlab.nargout(nout):
                return tuple(f(*[float(a) for a in args]))
        out = f(*args)
        if nout == 1:
            return (out[0] if isinstance(out, tuple) else out,)
        if not isinstance(out, (tuple, list)) or len(out) < nout:
            raise ValueError(f"external function {name!r} must return {nout} outputs (value, gradient"
                             f"{', hessian' if nout == 3 else ''}) as the .mod file declares")
        return tuple(out[:nout])

    def value(self, name, *args):
        v = np.asarray(self._raw(name, 1, args)[0], dtype=float)
        if v.size != 1:
            raise ValueError(f"external function {name!r} returned {v.size} values instead of one")
        return float(v.reshape(-1)[0])

    def call(self, name, nout, *args):
        out = self._raw(name, nout, args)
        if nout == 1:
            return np.asarray(out[0], dtype=float)
        res = [float(np.asarray(out[0], dtype=float).reshape(-1)[0])]
        res += [np.asarray(v, dtype=float) for v in out[1:]]
        return tuple(res)

    def jacob(self, name, element, *args):
        """``jacob_element.m``: centred difference with step 1e-6 in argument ``element`` (1-based)."""
        h = H_STEP
        p, m = list(args), list(args)
        p[element - 1] = p[element - 1] + h
        m[element - 1] = m[element - 1] - h
        return (self.value(name, *p) - self.value(name, *m)) / (2 * h)

    def hess(self, name, e1, e2, *args):
        """``hess_element.m`` (same formulas and evaluation order)."""
        h = H_STEP
        a = list(args)
        p10, p01, m10, m01, p11, m11 = (list(a) for _ in range(6))
        p10[e1 - 1] += h
        m10[e1 - 1] -= h
        p11[e1 - 1] += h
        m11[e1 - 1] -= h
        p01[e2 - 1] += h
        m01[e2 - 1] -= h
        p11[e2 - 1] += h
        m11[e2 - 1] -= h
        f = self.value
        if e1 == e2:
            return (16 * f(name, *p10) + 16 * f(name, *m10) - 30 * f(name, *a) - f(name, *p11)
                    - f(name, *m11)) / (12 * h ** 2)
        return (f(name, *p10) + f(name, *m10) + f(name, *p01) + f(name, *m01) - 2 * f(name, *a)
                - f(name, *p11) - f(name, *m11)) / (-2 * h ** 2)


def index(x, i, j=None):
    """``TEFD_k[i]`` / ``TEFDD_k[i,j]`` (0-based, as the preprocessor writes them)."""
    a = np.asarray(x, dtype=float)
    if j is None:
        return float(a.reshape(-1)[i])
    return float(np.atleast_2d(a)[i, j])


def bind(ns, names, table):
    ns["_pydyn_ext"] = table
    ns["_efi"] = index
    for k, name in enumerate(names):
        ns[ALIAS.format(k)] = (lambda nm: lambda *a: table.value(nm, *a))(name)
