# This file is part of pydyn.
#
# It is a small Python re-implementation of the part of Dynare 7.2's dseries class (matlab/dseries/src/@dseries:
# dseries, subsref, plus, minus, mtimes, times, mrdivide, rdivide, mpower, power, uminus, horzcat, extract, copy;
# Copyright (C) 2011-2025 Dynare Team, https://www.dynare.org, GPL-3.0-or-later) that the backward-model and model-
# inversion tests use: numeric data, names, a first date; no transformations, no operations history (ops), no tags.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""A minimal ``dseries`` for pydyn's interpreter (:mod:`pydyn.matlab`).

What exists: ``dseries(data, firstdate, names)``; the members ``data``, ``name``, ``dates``, ``init``, ``last``, ``nobs``,
``vobs``, ``freq``; selection of one variable by its name (``ds.Output``), of several (``ds{'a', 'b'}``) or by position
(``ds{1:2}``); selection of periods (``ds(dates)``; periods outside the sample are an error, as in Dynare);
``+ - * / ^ .* ./ .^`` and unary minus between series (aligned on their dates, NaN outside the common sample) and between a
series and a scalar or a vector; ``[a, b]`` (horizontal concatenation of series with distinct names); ``copy``,
``isdseries``, ``isempty``, ``size``.  Everything else (transformations such as ``log``/``diff``/``lag``, reading and writing
files, tags, ``ops``) is refused with a message that says so.

Dates are :class:`pydyn.dates.Date` values; a date given as a string (``'2000Q1'``) is parsed.
"""
from __future__ import annotations

import numpy as np

from .dates import Date, DateError

_F = np.float64


def _fail(msg):
    from .matlab import MatlabError
    raise MatlabError(msg)


def _unsupported(what):
    from .matlab import Unsupported
    raise Unsupported(f"dseries: {what} is not supported by pydyn's dseries (only data, names, dates, selection and "
                      "arithmetic are)", "dseries")


def as_date(v):
    if isinstance(v, Date):
        return v
    if isinstance(v, str):
        try:
            return Date.parse(v)
        except DateError as exc:
            _fail(f"dseries: {exc}")
    inner = getattr(v, "first_date", None)
    if inner is not None:
        return inner()
    if isinstance(v, (int, float, np.floating)) and float(v) == int(v):
        return Date(1, int(v))                       # a number is a year (annual date)
    _fail("dseries: the first date must be a date")


class DateArray:
    """The ``dates`` member of a series (and ``a:b`` between dates): a list of consecutive or arbitrary dates of one
    frequency."""

    def __init__(self, dates):
        self.list = list(dates)

    def get_field(self, name):
        if name in ("ndat", "length"):
            return _F(len(self.list))
        if name == "freq":
            return _F(self.list[0].freq)
        _unsupported(f"dates.{name}")

    def ml_index(self, args, brace=False):
        if brace or len(args) != 1:
            _fail("dates: invalid index")
        a = args[0]
        if isinstance(a, str) and a == ":":
            return self
        idx = np.asarray(a, float).ravel() if not isinstance(a, (int, float, np.floating)) else np.array([float(a)])
        if idx.dtype == bool:
            idx = np.flatnonzero(idx) + 1.0
        if np.any(idx != np.floor(idx)) or np.any(idx < 1) or np.any(idx > len(self.list)):
            _fail("dates: index out of bound")
        out = [self.list[int(i) - 1] for i in idx]
        return out[0] if len(out) == 1 else DateArray(out)

    def ml_binop(self, op, other, left):
        if op in ("+", "-") and left and isinstance(other, (int, float, np.floating)):
            k = int(other) if op == "+" else -int(other)
            return DateArray([d + k for d in self.list])
        _unsupported(f"dates {op}")

    def ml_shape(self):
        return (len(self.list), 1)

    def __len__(self):
        return len(self.list)


def dates_of(v):
    """A list of Date from a date, a DateArray, a list of dates or a date given as a string."""
    if isinstance(v, DateArray):
        return v.list
    if isinstance(v, Date):
        return [v]
    if isinstance(v, str):
        return [as_date(v)]
    if isinstance(v, (list, tuple)) and all(isinstance(d, Date) for d in v):
        return list(v)
    return None


class DSeries:
    """``data`` is ``nobs x vobs`` (float), ``names`` a list of str, ``init`` the first Date."""
    __slots__ = ("data", "names", "init")

    def __init__(self, data, names, init):
        d = np.asarray(data, float)
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        self.data, self.names, self.init = d, list(names), init
        if d.shape[1] != len(self.names):
            _fail("dseries: the number of names does not match the number of columns of the data")

    # --- structure
    @property
    def nobs(self):
        return self.data.shape[0]

    @property
    def vobs(self):
        return self.data.shape[1]

    def dates_list(self):
        return [self.init + k for k in range(self.nobs)]

    def copy(self):
        return DSeries(self.data.copy(), self.names, self.init)

    def _with(self, data):
        return DSeries(data, self.names if data.shape[1] == len(self.names) else self.names[:1] * data.shape[1], self.init)

    # --- the interpreter's protocol
    def get_field(self, name):
        from .matlab import Cell
        if name == "data":
            return self.data.copy() if self.data.shape != (1, 1) else _F(self.data[0, 0])
        if name == "name":
            return Cell(list(self.names), (len(self.names), 1))
        if name == "dates":
            return DateArray(self.dates_list())
        if name == "init":
            return self.init
        if name == "last":
            return self.init + (self.nobs - 1)
        if name == "nobs":
            return _F(self.nobs)
        if name == "vobs":
            return _F(self.vobs)
        if name == "freq":
            return _F(self.init.freq)
        if name == "save":
            return _Method(self, "save")
        if name in self.names:
            j = self.names.index(name)
            return DSeries(self.data[:, [j]], [name], self.init)
        if name in ("tex", "ops", "tags"):
            _unsupported(f"the member '{name}'")
        if name in _METHODS:
            _unsupported(f"the method '{name}'")
        _fail("dseries::subsref: Unknown public method, public member or variable!")

    def ml_index(self, args, brace=False):
        from .matlab import Cell
        if brace:
            if all(isinstance(a, str) for a in args):
                return self.extract(args)
            if len(args) == 1:
                idx = np.asarray(args[0], float).ravel()
                if np.any(idx != np.floor(idx)) or np.any(idx < 1) or np.any(idx > self.vobs):
                    _fail("dseries::subsref: Indices are out of bounds!")
                cols = [int(i) - 1 for i in idx]
                return DSeries(self.data[:, cols], [self.names[j] for j in cols], self.init)
            _fail("dseries::subsref: Invalid syntax")
        if len(args) != 1:
            _fail("dseries::subsref: Invalid syntax")
        a = args[0]
        ds = dates_of(a)
        if ds is not None:
            if not ds:
                _fail("dseries::subsref: empty set of dates")
            if any(d.freq != self.init.freq for d in ds):
                _fail("dseries::subsref: frequencies differ")
            first, last = self.init, self.init + (self.nobs - 1)
            if min(ds) < first:
                _fail(f"dseries::subsref: Indices are out of bounds! Subsample cannot start before {first}.")
            if max(ds) > last:
                _fail(f"dseries::subsref: Indices are out of bounds! Subsample cannot end after {last}.")
            keep = sorted({d - first for d in ds})
            return DSeries(self.data[keep, :], self.names, first + keep[0]) if keep == list(range(keep[0], keep[-1] + 1)) \
                else _unsupported("a subsample of non-consecutive dates")
        if isinstance(a, (np.bool_, bool)) or (isinstance(a, np.ndarray) and a.dtype == bool):
            m = np.asarray(a, bool).ravel()
            if m.size == self.nobs:
                rows = np.flatnonzero(m)
                if rows.size and np.all(np.diff(rows) == 1):
                    return DSeries(self.data[rows, :], self.names, self.init + int(rows[0]))
                _unsupported("a logical selection of non-consecutive periods")
            if m.size == self.vobs:
                cols = np.flatnonzero(m)
                return DSeries(self.data[:, cols], [self.names[j] for j in cols], self.init)
            _fail("dseries::subsref: length of the vector of logicals must match the number of variables or periods.")
        _unsupported("this kind of index (lead/lag by an integer, a file name, a matrix)")

    def extract(self, patterns):
        cols = []
        for p in patterns:
            if any(ch in p for ch in "@[]~"):
                _unsupported("implicit loops, regular expressions and wildcards in the names of ds{...}")
            if p not in self.names:
                _fail(f"dseries::extract: Variable {p} is unknown in dseries object!")
            cols.append(self.names.index(p))
        return DSeries(self.data[:, cols], [self.names[j] for j in cols], self.init)

    # --- arithmetic
    def ml_unop(self, op):
        if op == "-":
            return DSeries(-self.data, self.names, self.init)
        if op in ("+",):
            return self
        _unsupported(f"the operator {op}")

    def ml_binop(self, op, other, left):
        f = _OPS.get(op)
        if f is None:
            _unsupported(f"the operator {op}")
        if op in ("*", "/", "^") and not isinstance(other, DSeries):
            # matrix operators with a scalar are elementwise; with a vector Dynare's mtimes/mrdivide are elementwise too
            # (bsxfun); a general matrix product is not defined for a series
            pass
        if isinstance(other, DSeries):
            a, b = (self, other) if left else (other, self)
            return _combine(a, b, f)
        from .matlab import arr, Tainted
        if isinstance(other, Tainted):
            arr(other)
        v = np.asarray(other, float) if not isinstance(other, np.ndarray) else other.astype(float)
        if v.ndim == 0 or v.size == 1:
            v = v.reshape(1, 1)
        elif v.ndim == 2 and 1 in v.shape:
            v = v.reshape(-1, 1) if v.shape[0] == self.nobs and v.shape[1] == 1 else v.reshape(1, -1)
            if op in ("^", ".^"):
                if v.shape[0] != self.nobs and v.shape[0] != 1:
                    _fail("dseries::power: wrong dimensions")
        else:
            _fail(f"dseries::{op}: wrong inputs")
        out = f(self.data, v) if left else f(v, self.data)
        return DSeries(out, self.names, self.init)

    def ml_assign(self, interp, sc, chain, value):
        """``ds{'a', 'b'} = series``, ``ds{'a'}(dates) = series_or_numbers``, ``ds(dates) = ...`` (a new object: values are
        copied on assignment).  The periods must lie in the sample of ``ds``; a variable named for the first time is added
        (``ds{'a'} = series``)."""
        from .matlab import Index
        steps = []
        for node in chain:
            if type(node) is not Index:
                _unsupported("a field assignment (ds.name = ...)")
            steps.append((node.brace, interp.index_args(self, node.args, sc)))
        if len(steps) == 1 and steps[0][0] and all(isinstance(a, str) for a in steps[0][1]):
            names = list(steps[0][1])
            rows = None
        elif len(steps) == 2 and steps[0][0] and all(isinstance(a, str) for a in steps[0][1]) and not steps[1][0]:
            names, rows = list(steps[0][1]), steps[1][1]
        elif len(steps) == 1 and not steps[0][0]:
            names, rows = list(self.names), steps[0][1]
        else:
            _unsupported("this form of assignment into a series")
        out = self.copy()
        if rows is None:
            if not isinstance(value, DSeries):
                _unsupported("assignment of a value that is not a series to ds{names}")
            if value.vobs != len(names):
                _fail("dseries::subsasgn: the number of variables does not match")
            first, A, B = _align(out, value) if out.vobs else (value.init, np.zeros((value.nobs, 0)), value.data)
            data, nm = A, list(out.names)
            for k, name in enumerate(names):
                if name in nm:
                    data[:, nm.index(name)] = B[:, k]
                else:
                    data = np.hstack([data, B[:, [k]]])
                    nm.append(name)
            return DSeries(data, nm, first)
        if len(rows) != 1:
            _fail("dseries::subsasgn: Invalid syntax")
        ds = dates_of(rows[0])
        if ds is None:
            _unsupported("an index that is not a date or a set of dates")
        if any(d.freq != out.init.freq for d in ds):
            _fail("dseries::subsasgn: frequencies differ")
        last = out.init + (out.nobs - 1)
        if min(ds) < out.init or max(ds) > last:
            _unsupported("an assignment outside the sample of the series (the series would be extended)")
        r = [d - out.init for d in ds]
        cols = []
        for nm in names:
            if nm not in out.names:
                _unsupported("an assignment to a new variable over a subsample")
            cols.append(out.names.index(nm))
        if isinstance(value, DSeries):
            if value.vobs != len(cols) or value.nobs != len(r):
                _fail("dseries::subsasgn: dimension mismatch")
            v = value.data
        else:
            v = np.asarray(value, float)
            v = v.reshape(1, 1) if v.size == 1 else v
        out.data[np.ix_(r, cols)] = v
        return out

    # --- misc
    def as_mat_struct(self):
        """What ``save`` writes for a series: a struct (the MAT-file of Octave holds an object that only Dynare can read)."""
        from .matlab import Cell, Struct
        return Struct(data=self.data.copy() if self.data.shape != (1, 1) else _F(self.data[0, 0]),
                      name=Cell(list(self.names), (len(self.names), 1)), init=str(self.init), freq=_F(self.init.freq))

    def ml_shape(self):
        return (self.nobs, self.vobs)

    def ml_hcat(self, vals):
        return hcat(vals)

    def __repr__(self):
        return f"DSeries({self.names}, {self.init}, {self.nobs} obs)"


_METHODS = {"log", "exp", "diff", "lag", "lead", "cumsum", "cumprod", "abs", "mean", "std", "hpcycle", "hptrend", "plot",
            "save", "firstdate", "lastdate", "set_names", "rename", "tag", "subsample", "isnan", "ydiff", "qdiff", "mdiff",
            "ygrowth", "qgrowth", "mgrowth", "round", "center", "detrend", "chain", "remove", "pop", "insert", "struct"}

_OPS = {"+": np.add, "-": np.subtract, "*": np.multiply, ".*": np.multiply, "/": np.divide, "./": np.divide,
        "^": np.power, ".^": np.power}


def _align(a: DSeries, b: DSeries):
    if a.init.freq != b.init.freq:
        _fail("dseries: Cannot combine series of different frequencies!")
    first = min(a.init, b.init)
    last = max(a.init + (a.nobs - 1), b.init + (b.nobs - 1))
    n = (last - first) + 1

    def pad(s):
        out = np.full((n, s.vobs), np.nan)
        k = s.init - first
        out[k:k + s.nobs] = s.data
        return out
    return first, pad(a), pad(b)


def _combine(a: DSeries, b: DSeries, f):
    if a.vobs != b.vobs and a.vobs != 1 and b.vobs != 1:
        _fail("dseries: Cannot combine series (wrong number of variables)!")
    if a.nobs != b.nobs or a.init != b.init:
        first, A, B = _align(a, b)
    else:
        first, A, B = a.init, a.data, b.data
    with np.errstate(all="ignore"):
        out = f(A, B)
    names = a.names if a.vobs >= b.vobs else b.names
    return DSeries(out, names, first)


def hcat(items):
    """``[a, b, ...]`` for series: the names must differ; the samples are aligned (NaN where a series is missing)."""
    items = [x for x in items if not (isinstance(x, np.ndarray) and x.size == 0)]
    out = None
    for s in items:
        if not isinstance(s, DSeries):
            _unsupported("concatenation of a series with a value that is not a series")
        if out is None:
            out = s
            continue
        common = set(out.names) & set(s.names)
        if common:
            _fail(f"dseries::horzcat: I cannot concatenate dseries objects with common variable names ({', '.join(sorted(common))})!")
        first, A, B = _align(out, s)
        out = DSeries(np.hstack([A, B]), out.names + s.names, first)
    return out


#: the workspace (pydyn.results.Results) that runs the interpreter: it writes and reads the files of ``ts.save(...)`` and
#: ``dseries('file.mat')`` (set by pydyn.results_bwd; the files are those of the .mod file's own folder)
HOST = None


class _Method:
    """``ts.save`` before its arguments: ``ts.save('name', 'mat')``."""

    def __init__(self, ds, name):
        self.ds, self.name = ds, name

    def ml_index(self, args, brace=False):
        if brace or HOST is None:
            _unsupported(f"the method '{self.name}' in this form")
        return HOST.dseries_method(self.ds, self.name, args)


def make(args):
    """``dseries(data, firstdate, names)``."""
    from .matlab import Cell, arr
    if not args:
        return DSeries(np.zeros((0, 0)), [], Date(1, 1))
    if len(args) > 4:
        _fail("dseries: too many arguments")
    a0 = args[0]
    if isinstance(a0, str):
        if HOST is None:
            _unsupported("dseries('file') (reading a file)")
        return HOST.dseries_read(a0)
    if isinstance(a0, DSeries):
        return a0.copy()
    data = arr(a0).astype(float)
    if data.ndim != 2:
        _fail("dseries: the data must be a matrix")
    if data.shape[0] == 1 and len(args) > 2 and isinstance(args[2], Cell) and data.shape[1] != len(args[2].items):
        pass
    init = Date(1, 1)
    if len(args) > 1 and not (isinstance(args[1], np.ndarray) and args[1].size == 0):
        init = as_date(args[1])
    if len(args) > 2 and not (isinstance(args[2], np.ndarray) and args[2].size == 0):
        nm = args[2]
        if isinstance(nm, Cell):
            names = [str(x) for x in nm.items]
        elif isinstance(nm, str):
            names = [nm]
        else:
            _fail("dseries: the names must be a cell array of strings")
    else:
        names = [f"Variable_{i + 1}" for i in range(data.shape[1])]
    if len(names) != data.shape[1]:
        _fail("dseries: the number of names does not match the number of columns of the data")
    return DSeries(data, names, init)
