# This file is part of pydyn. Copyright (C) 2026; GPL-3.0-or-later (see LICENSE).
"""Char matrices (a char array with more than one row, or a 1-by-0 / n-by-0 one) for the MATLAB interpreter.

A one-row char array stays a Python ``str`` everywhere in the interpreter; :class:`CharMat` is the value of ``char('a','bcd')``,
``['ab';'cd']``, ``strvcat`` ... and of what indexing a char matrix leaves (``c(:,1)``).  It holds the character codes of its
rows, padded with blanks by ``char`` as Octave does.  The built-ins below accept it; every other built-in refuses it (the
base class :class:`pydyn.matlab._Obj` makes arithmetic and the like raise ``Unsupported``), so a char matrix is never taken for
a number or a string.  Every case is checked against Octave 8.4 (reference/matlab_octave_cases/pydyn_char_cases.txt)."""
from __future__ import annotations

import numpy as np

from . import matlab as M
from .matlab import (BUILTINS, Cell, MatlabError, Unsupported, _Obj, _index_arr, assign_index, ret, _F, _SCAL)


class CharMat(_Obj):
    obj_kind = "char matrix"
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = np.asarray(a, float).reshape(np.shape(a))        # 2-D character codes

    def ml_shape(self):
        return self.a.shape

    def ml_index(self, args, brace):
        if brace:
            raise MatlabError("'{' undefined for arguments of type 'char'")
        return mk(_index_arr(self.a, list(args), keep=True))

    def ml_assign(self, interp, sc, chain, value):
        acc = chain[0]
        if len(chain) != 1 or type(acc) is not M.Index or acc.brace:
            raise Unsupported("this assignment to a char matrix", "function")
        args = interp.index_args(self, acc.args, sc)
        return mk(np.asarray(M.arrn(assign_index(self.a, args, codes(value))), float))

    def snapshot(self):
        return self

    def rows(self):
        return ["".join(chr(int(c)) for c in r) for r in self.a]


def codes(v):
    """The 2-D array of character codes of a char value (``str`` or :class:`CharMat`); numbers are their own codes."""
    if isinstance(v, CharMat):
        return v.a
    if isinstance(v, str):
        return np.array([[ord(c) for c in v]], float) if v else np.zeros((0, 0))
    return np.asarray(M.arr(v), float)


def mk(a):
    """The value of a 2-D array of character codes: a ``str`` for one row, else a :class:`CharMat`."""
    a = np.asarray(a, float)
    if a.ndim != 2:
        raise Unsupported(M.ND_REFUSAL)
    if a.shape == (0, 0):
        return ""
    if a.shape[0] == 1 and a.shape[1] >= 1:
        return "".join(chr(int(c)) for c in a[0])
    return CharMat(a)


def is_char(v):
    return isinstance(v, (str, CharMat))


def _rows(v, who):
    """The rows (strings) a ``char``/``strvcat`` argument contributes."""
    if isinstance(v, str):
        return [v]
    if isinstance(v, CharMat):
        return v.rows()
    if isinstance(v, Cell):
        out = []
        for x in v.items:
            if isinstance(x, Cell):
                raise MatlabError(f"{who}: wrong type argument 'cell array'")
            out += _rows(x, who)
        return out
    if isinstance(v, (np.ndarray,) + _SCAL):
        a = M.arr(v)
        if a.size == 0:
            return [""]
        return ["".join(chr(M.as_index_int(c)) for c in r) for r in a]
    raise Unsupported(f"{who} of a {M._class(v)} value", "function")


def _stack(rows):
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    if len(rows) == 1 and w == 0:
        return ""
    a = np.array([[ord(c) for c in r.ljust(w)] for r in rows], float).reshape(len(rows), w)
    return CharMat(a) if len(rows) > 1 else mk(a)


_old_char = BUILTINS.get("char")


def _char(I, a, nout):
    if not a:
        return ""
    if len(a) == 1 and isinstance(a[0], str):
        return a[0]
    if not all(isinstance(x, (str, CharMat, Cell, np.ndarray) + _SCAL) for x in a):
        return _old_char(I, a, nout)                              # dates and the like
    if len(a) == 1 and isinstance(a[0], (np.ndarray,) + _SCAL):
        return _old_char(I, a, nout)
    rows = []
    for x in a:
        rows += _rows(x, "char")
    return _stack(rows)


def _strvcat(I, a, nout):
    if not a:
        return ""
    rows = []
    for x in a:
        rows += [r for r in _rows(x, "strvcat") if r != ""]
    return _stack(rows)


def cat_chars(vals, d):
    """``[a; b]`` (d = 0) and ``[a, b]`` (d = 1) of char arrays and char matrices (a number among them is refused)."""
    parts = []
    for v in vals:
        if not is_char(v):
            if isinstance(v, np.ndarray) and v.size == 0:
                continue
            raise Unsupported("concatenation of a char matrix with numbers", "function")
        c = codes(v)
        if c.size == 0 and c.shape == (0, 0):
            continue
        parts.append(c)
    if not parts:
        return ""
    o = 1 - d
    if d == 0 and len({p.shape[1] for p in parts}) > 1:             # Octave pads the rows with blanks (string_fill_char)
        w = max(p.shape[1] for p in parts)
        parts = [np.hstack([p, np.full((p.shape[0], w - p.shape[1]), 32.0)]) for p in parts]
    if len({p.shape[o] for p in parts}) > 1:
        what = "vertical" if d == 0 else "horizontal"
        raise MatlabError(f"{what} dimensions mismatch ({'x'.join(map(str, parts[0].shape))} vs "
                          f"{'x'.join(map(str, next(p for p in parts if p.shape[o] != parts[0].shape[o]).shape))})")
    return mk(np.concatenate(parts, axis=d))


def _wrap(name, fn):
    """The built-in ``name`` with ``fn`` handling the calls that have a :class:`CharMat` among their arguments."""
    old = BUILTINS.get(name)

    def f(I, a, nout):
        if any(isinstance(x, CharMat) for x in a):
            return fn(I, a, nout)
        if old is None:
            raise Unsupported(f"unknown function or variable '{name}'", "function")
        return old(I, a, nout)
    BUILTINS[name] = f


def _install():
    BUILTINS["char"] = _char
    BUILTINS["strvcat"] = _strvcat

    _old_cellstr = BUILTINS.get("cellstr")

    def cellstr(I, a, nout):
        if len(a) == 1 and isinstance(a[0], CharMat):
            items = [r.rstrip(" ") for r in a[0].rows()]            # Octave removes the trailing blanks only
            return Cell(items, (len(items), 1))
        if _old_cellstr is not None:
            return _old_cellstr(I, a, nout)
        if len(a) == 1 and isinstance(a[0], str):
            return Cell([a[0].rstrip(" ")], (1, 1))
        if len(a) == 1 and isinstance(a[0], Cell):
            return a[0]
        raise Unsupported("cellstr of this value", "function")
    BUILTINS["cellstr"] = cellstr

    # size, numel, rows, isempty ... use shape_of(), which asks ml_shape(): nothing to do for them
    _wrap("ischar", lambda I, a, nout: np.bool_(len(a) == 1))
    _wrap("double", lambda I, a, nout: ret(codes(a[0])))
    _wrap("isnumeric", lambda I, a, nout: np.bool_(False))
    _wrap("isreal", lambda I, a, nout: np.bool_(True))
    _wrap("islogical", lambda I, a, nout: np.bool_(False))
    _wrap("iscell", lambda I, a, nout: np.bool_(False))
    _wrap("iscellstr", lambda I, a, nout: np.bool_(False))
    _wrap("isfloat", lambda I, a, nout: np.bool_(False))

    def deblank(I, a, nout):
        c = a[0].a
        keep = np.nonzero(np.any(~np.isin(c, [32, 0, 9, 10, 11, 12, 13]), axis=0))[0]
        n = int(keep.max()) + 1 if keep.size else 0
        return mk(c[:, :n]) if n else ""
    _wrap("deblank", deblank)

    def strtrim(I, a, nout):
        c = a[0].a
        keep = np.nonzero(np.any(~np.isin(c, [32, 0, 9, 10, 11, 12, 13]), axis=0))[0]
        if not keep.size:
            return CharMat(c[:, :0])
        return mk(c[:, int(keep.min()):int(keep.max()) + 1])
    _wrap("strtrim", strtrim)

    def case(fn):
        def f(I, a, nout):
            return mk(np.array([[ord(fn(chr(int(c)))) for c in r] for r in a[0].a], float).reshape(a[0].a.shape))
        return f
    _wrap("upper", case(str.upper))
    _wrap("lower", case(str.lower))

    def transpose(I, a, nout):
        return mk(a[0].a.T)
    _wrap("transpose", transpose)
    _wrap("ctranspose", transpose)

    def flip(axis):
        return lambda I, a, nout: mk(np.flip(a[0].a, axis=axis))
    _wrap("flipud", flip(0))
    _wrap("fliplr", flip(1))

    def _eq(I, a, nout, cmp):
        if len(a) != 2:
            raise MatlabError("Invalid call to strcmp")
        x, y = a
        if not (is_char(x) and is_char(y)):
            raise Unsupported("strcmp of a char matrix with a cell array or a number", "function")
        cx, cy = codes(x), codes(y)
        return np.bool_(cx.shape == cy.shape and bool(np.all(cx == cy)))
    _wrap("strcmp", lambda I, a, nout: _eq(I, a, nout, True))

    def isequal(I, a, nout):
        if len(a) < 2:
            raise MatlabError("Invalid call to isequal")
        if not all(is_char(x) for x in a):
            raise Unsupported("isequal of a char matrix with other values", "function")
        c0 = codes(a[0])
        return np.bool_(all(codes(x).shape == c0.shape and bool(np.all(codes(x) == c0)) for x in a[1:]))
    _wrap("isequal", isequal)

    def repmat_chars(I, a, nout):
        reps = M._dims(a[1:], "repmat", nd=True)
        if len(reps) > 2:
            raise Unsupported(M.ND_REFUSAL)
        return mk(np.tile(a[0].a, reps))
    _wrap("repmat", repmat_chars)

    _old_repmat = BUILTINS["repmat"]

    def repmat_str(I, a, nout):
        if len(a) >= 2 and isinstance(a[0], str) and a[0]:
            reps = M._dims(a[1:], "repmat", nd=True)
            if len(reps) <= 2 and reps[0] != 1:
                return mk(np.tile(codes(a[0]), reps))
        return _old_repmat(I, a, nout)
    BUILTINS["repmat"] = repmat_str

    _old_strmatch = BUILTINS["strmatch"]

    def strmatch(I, a, nout):
        M._nargs(a, 2, 3, "strmatch")
        s, cand = a[0], a[1]
        if isinstance(s, Cell):                                  # a cell holding one string is that string (Octave)
            if len(s.items) != 1 or not isinstance(s.items[0], str):
                raise MatlabError("strmatch: a cell array S must contain only one string")
            s = s.items[0]
        if len(a) == 3 and a[2] not in ("exact",):
            raise Unsupported("strmatch with an option other than 'exact'", "function")
        exact = len(a) == 3
        if not isinstance(cand, CharMat):
            return _old_strmatch(I, [s, *a[1:]], nout)
        if not isinstance(s, str):
            raise MatlabError("strmatch: S must be a string or cell array of strings")
        if s == "":
            raise MatlabError("strmatch: bsxfun: nonconformant dimensions")
        if exact and s != s.rstrip(" \0"):
            raise Unsupported("strmatch 'exact' with a pattern that ends in blanks", "function")
        hit = []
        for i, row in enumerate(cand.rows()):
            ok = row.rstrip(" \0") == s if exact else row[:len(s)] == s and len(row) >= len(s)
            if ok:
                hit.append(i + 1.0)
        return ret(np.array(hit).reshape(-1, 1)) if hit else np.zeros((0, 0))
    BUILTINS["strmatch"] = strmatch


_install()
