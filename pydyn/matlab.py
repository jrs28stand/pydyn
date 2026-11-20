"""A small interpreter for the subset of MATLAB that Dynare ``.mod`` files and steady-state files use
to set up a calibration.

What this module is for
-----------------------
Dynare passes every line of a ``.mod`` file that it does not recognise to MATLAB.  pydyn has no
MATLAB, so the lines that matter for the calibration are interpreted here:

* ``ModSession`` runs the native lines of a ``.mod`` file *in order with* the Dynare statements,
  so that MATLAB control flow around parameter assignments behaves as in Dynare;
* ``MatlabFunction`` runs a function file (``name.m``), e.g. a helper called from a
  ``steady_state_model`` block;
* ``SteadyStateFile`` runs ``<model>_steadystate.m``; ``translate_steady_state`` turns such a file
  into Python source, which is what the automatic path executes when the file is simple enough.

Boundary (deliberate; it is better to refuse than to guess)
-----------------------------------------------------------
Supported: real double/logical scalars and 2-D matrices, char row vectors, simple cell arrays and
scalar structs; operators with MATLAB precedence and associativity; indexing and indexed assignment
(1-based, ``end``, ``:``, logical masks, growth on assignment); ``if/for/while/switch/try``,
``break/continue/return``; anonymous functions and handles; function files and scripts that sit next
to the ``.mod`` file; ``eval`` of a string the interpreter itself can compute; ``fzero``, ``fsolve``
and ``csolve`` mapped to scipy; the Dynare interface (``M_.params``, ``M_.Sigma_e``, ``M_.*_names``,
``set_param_value``, ``get_param_by_name``, ``set_shock_stderr_value``, ``options_.x = ...``).

Refused: complex numbers (an operation with a complex or indeterminate result is an error),
N-d arrays, struct arrays, char matrices, string objects in concatenations, classes, ``dseries``,
random numbers, ``evalin``/``assignin``, every file-system and shell function except ``load`` of a
``.mat`` file located next to the ``.mod`` file, and all Dynare internals other than those above.

A statement that cannot be interpreted is *skipped with a warning* only when it provably does not
write to ``M_`` or the parameters; the variables it would have assigned become "unknown", and any
later statement that needs one of them is skipped or refused by the same rule.  A statement that
cannot be interpreted and writes to the model raises ``ModelError``.

Sandbox
-------
MATLAB text is tokenised and parsed by this module and evaluated by walking the syntax tree; names
are looked up only in the interpreter's own variable tables and in the ``BUILTINS`` table below.
No MATLAB text is ever passed to Python's ``eval``/``exec``.  ``translate_steady_state`` generates
Python source from the *validated syntax tree* (identifiers are checked against a strict pattern
and renamed if they collide with anything; string contents are emitted with ``repr``); that source
is executed with an empty ``__builtins__`` and a namespace holding only numpy, this module's runtime
helpers and ``SteadyStateError``.  The only file access is reading ``.m`` files and ``.mat`` files
in the directory of the ``.mod`` file (no path components are accepted in names).  Loops are
bounded by ``MAX_STEPS`` evaluation steps per entry call.
"""
from __future__ import annotations

import keyword
import math
import re
import warnings
from pathlib import Path

import numpy as np
from scipy import optimize, special

__all__ = ["MatlabError", "Unsupported", "Interpreter", "ModSession", "MatlabFunction",
           "SteadyStateFile", "translate_steady_state", "parse", "ISOCTAVE", "TranslationError"]

#: value returned by ``isoctave``.  pydyn follows the MATLAB branch by default.
ISOCTAVE = False
#: upper bound on interpreter steps per entry call (guards against endless loops)
MAX_STEPS = 20_000_000
#: largest array the interpreter creates
MAX_ELEMENTS = 20_000_000


class MatlabError(Exception):
    """An error MATLAB itself would raise (``error(...)``, bad index, size mismatch, ...)."""


class Unsupported(Exception):
    """The interpreter cannot evaluate this code.  ``kind`` classifies the reason."""

    def __init__(self, msg, kind="feature"):
        super().__init__(msg)
        self.kind = kind


class ParseError(Exception):
    pass


class TranslationError(Exception):
    """``translate_steady_state`` cannot express the file as plain Python."""


#: exceptions that correspond to a MATLAB run-time error
MATLAB_ERRORS = (MatlabError, FloatingPointError, ZeroDivisionError, OverflowError)

_F = np.float64
_ERRSTATE = dict(invalid="raise", divide="ignore", over="ignore", under="ignore")


# ====================================================================== values
class Cell:
    """A cell array (2-D, stored column-major)."""
    __slots__ = ("items", "shape")

    def __init__(self, items, shape=None):
        self.items = list(items)
        self.shape = shape if shape is not None else ((1, len(self.items)) if self.items else (0, 0))

    def __repr__(self):
        return f"Cell({self.items!r}, {self.shape})"


class Struct(dict):
    """A scalar struct."""


class Handle:
    """A function handle: anonymous (``params``/``body``/``env``) or named (``name``)."""
    __slots__ = ("params", "body", "env", "name", "interp", "files")

    def __init__(self, interp, params=None, body=None, env=None, name=None, files=None):
        self.interp, self.params, self.body, self.env, self.name, self.files = interp, params, body, env, name, files

    def __call__(self, *args):
        return self.interp.call_handle(self, list(args), 1)[0]


class Tainted:
    """Placeholder for a variable whose value is unknown because the statement that assigned it
    could not be interpreted."""
    __slots__ = ("why",)

    def __init__(self, why):
        self.why = why


class _Colon:
    def __repr__(self):
        return ":"


COLON = _Colon()
_SCAL = (np.float64, float, int, np.bool_, bool)


def is_num(v):
    return isinstance(v, _SCAL) or isinstance(v, np.ndarray)


def arr(v):
    """2-D float (or bool) array view of a numeric value."""
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, _SCAL):
        return np.array([[v]])
    if isinstance(v, str):
        raise Unsupported("arithmetic or numeric use of a char array")
    if isinstance(v, Tainted):
        raise Unsupported(f"value depends on a statement that was not executed: {v.why}", "tainted")
    raise MatlabError(f"a {_class(v)} was used where a number is required")


def farr(v):
    a = arr(v)
    return a.astype(float) if a.dtype != float else a


def ret(a):
    """Normalise an array result: 1x1 becomes a scalar."""
    if a.ndim != 2:
        a = np.atleast_2d(a)
    if a.shape == (1, 1):
        return a[0, 0]
    return a


def scalar(v, what="value"):
    if isinstance(v, (np.float64, float, int)):
        return _F(v)
    if isinstance(v, (np.bool_, bool)):
        return _F(v)
    a = arr(v)
    if a.size != 1:
        raise MatlabError(f"{what} must be a scalar")
    return _F(a.flat[0])


def _class(v):
    if isinstance(v, str):
        return "char"
    if isinstance(v, Cell):
        return "cell"
    if isinstance(v, Struct) or hasattr(v, "get_field"):
        return "struct"
    if isinstance(v, Handle):
        return "function_handle"
    if isinstance(v, (np.bool_, bool)) or (isinstance(v, np.ndarray) and v.dtype == bool):
        return "logical"
    return "double"


def shape_of(v):
    if isinstance(v, np.ndarray):
        return v.shape
    if isinstance(v, str):
        return (1, len(v)) if v else (0, 0)
    if isinstance(v, Cell):
        return v.shape
    if isinstance(v, Tainted):
        arr(v)
    return (1, 1)


def truth(v):
    """MATLAB truth value of an ``if``/``while`` condition."""
    if isinstance(v, (np.float64, float, int, np.bool_, bool)):
        if v != v:
            raise MatlabError("NaN cannot be converted to a logical value")
        return bool(v)
    if isinstance(v, str):
        return len(v) > 0 and all(ord(c) != 0 for c in v)
    a = arr(v)
    if a.dtype != bool and np.isnan(a).any():
        raise MatlabError("NaN cannot be converted to a logical value")
    return a.size > 0 and bool(np.all(a != 0))


def as_index_int(x, what="index"):
    x = scalar(x, what)
    if x != np.floor(x) or not np.isfinite(x):
        raise MatlabError(f"{what} must be an integer")
    return int(x)


# ---------------------------------------------------------------------- arithmetic
def _bcast(a, b):
    if a.shape != b.shape:
        for p, q in zip(a.shape, b.shape):
            if p != q and p != 1 and q != 1:
                raise MatlabError(f"nonconformant arguments ({a.shape[0]}x{a.shape[1]} and {b.shape[0]}x{b.shape[1]})")


def _fl(v):
    return _F(v) if isinstance(v, (np.bool_, bool, int, float)) and not isinstance(v, np.float64) else v


def _ew(fn, a, b):
    ta, tb = type(a), type(b)
    if ta is np.float64 and tb is np.float64:
        return fn(a, b)
    if ta in _SCALT and tb in _SCALT:
        return fn(_F(a), _F(b))
    x, y = farr(a), farr(b)
    _bcast(x, y)
    return ret(fn(x, y))


_SCALT = frozenset(_SCAL)


def plus(a, b):
    return _ew(np.add, a, b)


def minus(a, b):
    return _ew(np.subtract, a, b)


def times(a, b):
    return _ew(np.multiply, a, b)


def rdivide(a, b):
    return _ew(np.divide, a, b)


def ldivide(a, b):
    return _ew(np.divide, b, a)


def _pow(x, y):
    # numpy raises (invalid) for a negative base with a fractional exponent: MATLAB would return
    # a complex number, which is outside the supported subset
    return np.power(x, y)


def power(a, b):
    return _ew(_pow, a, b)


def mtimes(a, b):
    if type(a) in _SCALT or type(b) in _SCALT:
        return times(a, b)
    x, y = farr(a), farr(b)
    if x.size == 1 or y.size == 1:
        return ret(x * y)
    if x.shape[1] != y.shape[0]:
        raise MatlabError(f"operator *: nonconformant arguments ({x.shape[0]}x{x.shape[1]} and {y.shape[0]}x{y.shape[1]})")
    return ret(x @ y)


def mrdivide(a, b):
    if type(b) in _SCALT:
        return rdivide(a, b)
    x, y = farr(a), farr(b)
    if y.size == 1:
        return ret(x / y)
    if x.shape[1] != y.shape[1]:
        raise MatlabError("operator /: nonconformant arguments")
    if y.shape[0] != y.shape[1]:
        raise Unsupported("least-squares solution with a non-square matrix")
    return ret(np.linalg.solve(y.T, x.T).T)


def mldivide(a, b):
    if type(a) in _SCALT:
        return ldivide(a, b)
    x, y = farr(a), farr(b)
    if x.size == 1:
        return ret(y / x)
    if x.shape[0] != y.shape[0]:
        raise MatlabError("operator \\: nonconformant arguments")
    if x.shape[0] != x.shape[1]:
        raise Unsupported("least-squares solution with a non-square matrix")
    try:
        return ret(np.linalg.solve(x, y))
    except np.linalg.LinAlgError:
        raise Unsupported("linear system with a singular matrix") from None


def mpower(a, b):
    if type(a) in _SCALT and type(b) in _SCALT:
        return power(a, b)
    x, y = farr(a), farr(b)
    if x.size == 1 and y.size == 1:
        return ret(_pow(x, y))
    if y.size == 1 and x.shape[0] == x.shape[1]:
        p = float(y.flat[0])
        if p == int(p):
            try:
                return ret(np.linalg.matrix_power(x, int(p)))
            except np.linalg.LinAlgError:
                raise Unsupported("negative power of a singular matrix") from None
        raise Unsupported("non-integer power of a matrix")
    raise MatlabError("operator ^: one argument must be a square matrix and the other a scalar; use .^ for element-wise power")


def uminus(a):
    if type(a) is np.float64:
        return -a
    if type(a) in _SCALT:
        return -_F(a)
    return ret(-farr(a))


def lnot(a):
    if type(a) in _SCALT:
        if a != a:
            raise MatlabError("NaN cannot be converted to a logical value")
        return np.bool_(not a)
    x = arr(a)
    if x.dtype != bool and np.isnan(x).any():
        raise MatlabError("NaN cannot be converted to a logical value")
    return ret(x == 0)


def transpose(a):
    if isinstance(a, np.ndarray):
        return a.T
    if isinstance(a, str) and len(a) > 1:
        raise Unsupported("transpose of a char array")
    if isinstance(a, Cell):
        return Cell(a.items, (a.shape[1], a.shape[0])) if 1 in a.shape or not a.items else _unsup("transpose of a cell matrix")
    return a


def _unsup(msg, kind="feature"):
    raise Unsupported(msg, kind)


def _cmp(fn):
    def f(a, b):
        ta, tb = type(a), type(b)
        if ta in _SCALT and tb in _SCALT:
            return np.bool_(fn(a, b))
        if ta is str or tb is str:
            if ta is str and tb is str and (len(a) == len(b) or len(a) == 1 or len(b) == 1):
                x = np.array([[ord(c) for c in a]], float)
                y = np.array([[ord(c) for c in b]], float)
                return ret(fn(x, y))
            raise Unsupported("comparison of char arrays with ==; use strcmp")
        x, y = arr(a), arr(b)
        _bcast(x, y)
        return ret(fn(x, y))
    return f


lt, gt, le, ge, eq, ne = (_cmp(f) for f in (np.less, np.greater, np.less_equal, np.greater_equal, np.equal, np.not_equal))


def _logical(a):
    x = arr(a)
    if x.dtype != bool and np.isnan(x).any():
        raise MatlabError("NaN cannot be converted to a logical value")
    return x != 0


def el_and(a, b):
    x, y = _logical(a), _logical(b)
    _bcast(x, y)
    return ret(x & y)


def el_or(a, b):
    x, y = _logical(a), _logical(b)
    _bcast(x, y)
    return ret(x | y)


def colon(a, b, c=None):
    """``a:b`` or ``a:b:c`` (second argument is the step when three are given)."""
    start = scalar(a, "range bound")
    if c is None:
        step, stop = _F(1.0), scalar(b, "range bound")
    else:
        step, stop = scalar(b, "range step"), scalar(c, "range bound")
    if step == 0 or (step > 0 and start > stop) or (step < 0 and start < stop) or not np.isfinite(step):
        return np.zeros((1, 0))
    if not np.isfinite(stop - start) or (stop - start) / step > MAX_ELEMENTS:
        raise Unsupported(f"range with more than {MAX_ELEMENTS} elements")
    n = int(np.floor((stop - start) / step * (1 + 4 * np.finfo(float).eps) + 1e-300)) + 1
    return (start + step * np.arange(n)).reshape(1, n)


def hcat(vals):
    vals = [v for v in vals if not (isinstance(v, np.ndarray) and v.size == 0) and not (isinstance(v, str) and v == "")]
    if not vals:
        return np.zeros((0, 0))
    if any(isinstance(v, str) for v in vals):
        if all(isinstance(v, str) for v in vals):
            return "".join(vals)
        raise Unsupported("concatenation of char arrays with numbers")
    if any(isinstance(v, Cell) for v in vals):
        items = []
        for v in vals:
            if isinstance(v, Cell):
                if v.shape[0] > 1:
                    raise Unsupported("horizontal concatenation of cell columns")
                items += v.items
            else:
                items.append(v)
        return Cell(items, (1, len(items)))
    if len(vals) == 1:
        return vals[0]
    parts = [arr(v) for v in vals]
    r = parts[0].shape[0]
    if any(p.shape[0] != r for p in parts):
        raise MatlabError("horizontal dimensions mismatch")
    out = np.concatenate(parts, axis=1)
    return ret(out if all(p.dtype == bool for p in parts) else out.astype(float))


def vcat(vals):
    vals = [v for v in vals if not (isinstance(v, np.ndarray) and v.size == 0) and not (isinstance(v, str) and v == "")]
    if not vals:
        return np.zeros((0, 0))
    if len(vals) == 1:
        return vals[0]
    if any(isinstance(v, str) for v in vals):
        raise Unsupported("char matrices (vertical concatenation of char arrays)")
    if any(isinstance(v, Cell) for v in vals):
        items = []
        for v in vals:
            if isinstance(v, Cell):
                if v.shape[1] > 1:
                    raise Unsupported("vertical concatenation of cell rows")
                items += v.items
            else:
                items.append(v)
        return Cell(items, (len(items), 1))
    parts = [arr(v) for v in vals]
    c = parts[0].shape[1]
    if any(p.shape[1] != c for p in parts):
        raise MatlabError("vertical dimensions mismatch")
    out = np.concatenate(parts, axis=0)
    return ret(out if all(p.dtype == bool for p in parts) else out.astype(float))


# ---------------------------------------------------------------------- indexing
def _lin_index(ix, n, grow=False):
    """Convert one MATLAB subscript into 0-based positions along a dimension of length ``n``."""
    if ix is COLON:
        return np.arange(n)
    if isinstance(ix, str):
        raise Unsupported("char array used as an index")
    a = arr(ix)
    if a.dtype == bool:
        if a.size > n and a.ravel(order="F")[n:].any():
            if not grow:
                raise MatlabError("logical index out of bounds")
        return np.flatnonzero(a.ravel(order="F"))
    f = a.ravel(order="F")
    if f.size and (np.any(f != np.floor(f)) or np.any(f < 1) or not np.all(np.isfinite(f))):
        raise MatlabError("subscript indices must be positive integers or logicals")
    k = f.astype(np.int64) - 1
    if not grow and k.size and k.max() >= n:
        raise MatlabError(f"index ({int(k.max()) + 1}) out of bounds (dimension has {n} element(s))")
    return k


def index(v, args):
    """``v(args)`` for arrays, strings and cells (returns a cell for cells)."""
    if isinstance(v, Tainted):
        arr(v)
    if isinstance(v, Cell):
        items, shp = v.items, v.shape
        a = np.arange(len(items)).reshape(shp, order="F") if items else np.zeros(shp, int)
        sub = _index_arr(a, args, keep=True)
        return Cell([items[int(i)] for i in sub.ravel(order="F")], sub.shape)
    if isinstance(v, str):
        a = np.array([[ord(c) for c in v]], float) if v else np.zeros((0, 0))
        sub = _index_arr(a, args, keep=True)
        if sub.shape[0] > 1:
            raise Unsupported("char matrices")
        return "".join(chr(int(c)) for c in sub.ravel())
    if isinstance(v, (Struct, Handle)) or hasattr(v, "get_field"):
        if all(type(x) in _SCALT and x == 1 for x in args):
            return v
        raise Unsupported("indexing a struct or function handle")
    return ret(_index_arr(arr(v), args))


def _index_arr(a, args, keep=False):
    n = len(args)
    if n == 0:
        return a
    if n == 1:
        ix = args[0]
        if type(ix) in (np.float64, float, int):      # fast path: scalar linear index
            k = ix - 1
            if ix != int(ix) or ix < 1:
                raise MatlabError("subscript indices must be positive integers or logicals")
            if k >= a.size:
                raise MatlabError(f"index ({int(ix)}) out of bounds (array has {a.size} element(s))")
            return a.reshape(-1, order="F")[int(k)].reshape(1, 1) if not keep else a.reshape(-1, order="F")[int(k):int(k) + 1].reshape(1, 1)
        flat = a.ravel(order="F")
        k = _lin_index(ix, flat.size)
        out = flat[k]
        if ix is COLON:
            return out.reshape(-1, 1)
        ia = arr(ix)
        if ia.dtype == bool:
            # logical mask: column result unless the source is a row vector
            return out.reshape(1, -1) if (a.shape[0] == 1 and a.shape[1] != 1) else out.reshape(-1, 1)
        if min(ia.shape) != 1 or (a.shape[0] != 1 and a.shape[1] != 1):
            return out.reshape(ia.shape, order="F")     # shape of the index
        return out.reshape(1, -1) if a.shape[0] == 1 else out.reshape(-1, 1)   # orientation of the source vector
    if n == 2:
        r = _lin_index(args[0], a.shape[0])
        c = _lin_index(args[1], a.shape[1])
        return a[np.ix_(r, c)]
    if all(type(x) in _SCALT and x == 1 for x in args[2:]):
        return _index_arr(a, args[:2], keep)
    raise Unsupported("arrays with more than two dimensions")


def index_brace(v, args):
    if isinstance(v, Tainted):
        arr(v)
    if not isinstance(v, Cell):
        raise MatlabError(f"'{{}}' indexing applied to a {_class(v)}")
    sub = index(v, args)
    if len(sub.items) != 1:
        raise Unsupported("comma-separated list from a cell array (c{:})")
    return sub.items[0]


def assign_index(v, args, rhs):
    """``v(args) = rhs``; ``v`` may be ``None`` (undefined).  Returns the new value."""
    if isinstance(v, Tainted):
        arr(v)
    if isinstance(v, Cell) or isinstance(rhs, Cell):
        if isinstance(rhs, Cell) and (v is None or isinstance(v, Cell)) and len(args) == 1 and len(rhs.items) == 1:
            return assign_brace(v, args, rhs.items[0])
        raise Unsupported("'()' assignment with cell arrays")
    if isinstance(v, str) or isinstance(rhs, (str, Struct, Handle)):
        raise Unsupported("indexed assignment with char arrays, structs or handles")
    a = np.zeros((0, 0)) if v is None else arr(v)
    r = arr(rhs)
    deleting = r.size == 0 and isinstance(rhs, np.ndarray)
    if a.dtype == bool and r.dtype != bool:
        a = a.astype(float)
    elif a.dtype != bool and r.dtype == bool:
        r = r.astype(float)
    n = len(args)
    if n == 1:
        k = _lin_index(args[0], a.size, grow=True)
        if deleting:
            if min(a.shape) > 1 and args[0] is not COLON:
                raise Unsupported("deleting elements of a matrix with a linear index")
            keep = np.ones(a.size, bool)
            keep[k] = False
            out = a.ravel(order="F")[keep]
            return ret(out.reshape(1, -1) if a.shape[0] == 1 else out.reshape(-1, 1))
        need = int(k.max()) + 1 if k.size else 0
        if need > a.size:
            if a.size == 0:
                a = np.zeros((1, need), a.dtype)
            elif a.shape[0] == 1:
                a = np.concatenate([a, np.zeros((1, need - a.shape[1]), a.dtype)], axis=1)
            elif a.shape[1] == 1:
                a = np.concatenate([a, np.zeros((need - a.shape[0], 1), a.dtype)], axis=0)
            else:
                raise MatlabError("a matrix cannot be resized with a linear index")
        else:
            a = a.copy()
        if r.size != 1 and r.size != k.size:
            raise MatlabError(f"A(I) = X: X must have the same size as I ({k.size} element(s) on the left, {r.size} on the right)")
        flat = a.reshape(-1, order="F")
        flat[k] = r.ravel(order="F") if r.size != 1 else r.flat[0]
        return ret(flat.reshape(a.shape, order="F"))
    if n == 2:
        ri = _lin_index(args[0], a.shape[0], grow=True) if args[0] is not COLON else None
        ci = _lin_index(args[1], a.shape[1], grow=True) if args[1] is not COLON else None
        if deleting:
            if ri is None and ci is not None:
                keep = np.ones(a.shape[1], bool)
                keep[ci] = False
                return ret(a[:, keep])
            if ci is None and ri is not None:
                keep = np.ones(a.shape[0], bool)
                keep[ri] = False
                return ret(a[keep, :])
            raise MatlabError("a null assignment can only have one non-colon index")
        if ri is None:
            ri = np.arange(a.shape[0] if a.size or ci is None else r.shape[0])
        if ci is None:
            ci = np.arange(a.shape[1] if a.size else r.shape[1])
        nr = max(a.shape[0], int(ri.max()) + 1 if ri.size else 0)
        nc = max(a.shape[1], int(ci.max()) + 1 if ci.size else 0)
        if (nr, nc) != a.shape:
            b = np.zeros((nr, nc), a.dtype)
            b[:a.shape[0], :a.shape[1]] = a
            a = b
        else:
            a = a.copy()
        if r.size == 1:
            a[np.ix_(ri, ci)] = r.flat[0]
        elif r.shape == (ri.size, ci.size):
            a[np.ix_(ri, ci)] = r
        elif r.size == ri.size * ci.size and (min(r.shape) == 1 and (ri.size == 1 or ci.size == 1)):
            a[np.ix_(ri, ci)] = r.reshape(ri.size, ci.size, order="F")
        else:
            raise MatlabError(f"A(I,J) = X: dimensions mismatch ({ri.size}x{ci.size} on the left, {r.shape[0]}x{r.shape[1]} on the right)")
        return ret(a)
    raise Unsupported("arrays with more than two dimensions")


def assign_brace(v, args, rhs):
    if v is None or (isinstance(v, np.ndarray) and v.size == 0):
        v = Cell([], (0, 0))
    if not isinstance(v, Cell):
        raise MatlabError(f"'{{}}' assignment to a {_class(v)}")
    empty = np.zeros((0, 0))
    if len(args) == 1:
        k = as_index_int(args[0]) - 1
        if k < 0:
            raise MatlabError("subscript indices must be positive integers")
        items = list(v.items)
        shp = v.shape
        if k >= len(items):
            if shp[0] > 1 and shp[1] > 1:
                raise MatlabError("a cell matrix cannot be resized with a linear index")
            items += [empty] * (k + 1 - len(items))
            shp = (len(items), 1) if shp[1] == 1 and shp[0] > 1 else (1, len(items))
        items[k] = rhs
        return Cell(items, shp)
    if len(args) == 2:
        i, j = as_index_int(args[0]) - 1, as_index_int(args[1]) - 1
        if i < 0 or j < 0:
            raise MatlabError("subscript indices must be positive integers")
        nr, nc = max(v.shape[0], i + 1), max(v.shape[1], j + 1)
        grid = [[empty] * nc for _ in range(nr)]
        for c in range(v.shape[1]):
            for r_ in range(v.shape[0]):
                grid[r_][c] = v.items[c * v.shape[0] + r_]
        grid[i][j] = rhs
        return Cell([grid[r_][c] for c in range(nc) for r_ in range(nr)], (nr, nc))
    raise Unsupported("cell arrays with more than two dimensions")


# ====================================================================== lexer
KEYWORDS = {"if", "elseif", "else", "end", "for", "while", "switch", "case", "otherwise", "break",
            "continue", "return", "function", "try", "catch", "global", "persistent", "endfunction",
            "endif", "endfor", "endwhile", "endswitch", "end_try_catch", "parfor", "classdef"}
_END_ALIASES = {"endfunction", "endif", "endfor", "endwhile", "endswitch", "end_try_catch"}
#: names accepted in command syntax (``hold on``, ``close all``, ``load file``) at the start of a line
COMMAND_WORDS = {"close", "clear", "clc", "hold", "format", "warning", "more", "figure", "load", "save",
                 "addpath", "rmpath", "diary", "echo", "pause", "disp", "dbstop", "profile", "delete",
                 "cd", "clearvars", "pkg", "graphics_toolkit", "drawnow", "grid", "box", "axis", "clf",
                 "dynare", "type", "which", "mex", "import", "syms", "tic", "toc", "legend", "colormap",
                 "shading", "rng", "beep", "commandwindow", "dbclear", "keyboard", "print"}
_CMD_RE = re.compile(r"^\s*([A-Za-z_]\w*)[ \t]+(?![=(;,]|[-+*/^<>~&|.:]?=|[-+*/^<>&|:](?:\s|$)|\.[*/^\\'])(\S[^;,%]*)(.*)$")
_NUM_RE = re.compile(r"(?:\d+(?:\.(?![*/^\\'])\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?")
_ID_RE = re.compile(r"[A-Za-z]\w*")      # MATLAB identifiers start with a letter
_OPS3 = ("...",)
_OPS2 = ("==", "~=", "!=", "<=", ">=", "&&", "||", ".*", "./", ".\\", ".^", ".'")
_OPS1 = "+-*/\\^'<>&|~!=()[]{},;:.@"


class Tok:
    __slots__ = ("k", "v", "ws", "line", "col")

    def __init__(self, k, v, ws, line, col=0):
        self.k, self.v, self.ws, self.line, self.col = k, v, ws, line, col

    def __repr__(self):
        return f"{self.k}:{self.v!r}"


def tokenize(lines):
    """``lines``: list of source lines (str) or ``("dyn", k)`` placeholders for Dynare statements."""
    toks = []
    depth = []                 # stack of open brackets
    cont = False               # previous line ended with ...
    block_comment = 0
    for ln, text in enumerate(lines):
        if not isinstance(text, str):
            toks.append(Tok("dyn", text[1], True, ln))
            toks.append(Tok("nl", "\n", True, ln))
            depth, cont = [], False
            continue
        st = text.strip()
        if block_comment:
            if st == "%}":
                block_comment -= 1
            elif st == "%{":
                block_comment += 1
            continue
        if st == "%{":
            block_comment = 1
            continue
        i, n = 0, len(text)
        if not depth and not cont:
            m = _CMD_RE.match(text)
            if m and m.group(1) in COMMAND_WORDS:
                toks.append(Tok("cmd", (m.group(1), m.group(2).strip()), True, ln))
                i = m.end(2)
        cont = False
        ws = True
        while i < n:
            c = text[i]
            if c in " \t\r\n\f\v":
                ws = True
                i += 1
                continue
            if c == "%":
                break
            if text.startswith("...", i):
                cont = True
                break
            prev = toks[-1] if toks and toks[-1].k != "nl" else None
            if c == "'" or c == '"':
                value_end = prev is not None and (prev.k in ("num", "id", "str") or (prev.k == "op" and prev.v in (")", "]", "}", "'", ".'"))
                                                  or (prev.k == "kw" and prev.v == "end"))
                in_br = bool(depth) and depth[-1] in "[{"
                if c == "'" and value_end and (not ws or not in_br):
                    toks.append(Tok("op", "'", ws, ln, i))
                    i += 1
                    ws = False
                    continue
                j = i + 1
                buf = []
                while True:
                    if j >= n:
                        toks.append(Tok("err", "unterminated string", ws, ln, i))
                        j = n
                        break
                    if text[j] == c:
                        if j + 1 < n and text[j + 1] == c:
                            buf.append(c)
                            j += 2
                            continue
                        toks.append(Tok("str" if c == "'" else "dstr", "".join(buf), ws, ln, i))
                        j += 1
                        break
                    buf.append(text[j])
                    j += 1
                i = j
                ws = False
                continue
            m = _NUM_RE.match(text, i)
            if m and (c.isdigit() or c == "."):
                s = m.group(0)
                j = m.end()
                if j < n and (text[j].isalpha() or text[j] == "_"):
                    toks.append(Tok("err", f"bad number or date literal {text[i:j + 3]!r}", ws, ln, i))
                else:
                    toks.append(Tok("num", float(s.replace("d", "e").replace("D", "e")), ws, ln, i))
                i = j
                ws = False
                continue
            m = _ID_RE.match(text, i)
            if m:
                s = m.group(0)
                if s in KEYWORDS and not (prev is not None and prev.k == "op" and prev.v == "."):
                    toks.append(Tok("kw", "end" if s in _END_ALIASES else s, ws, ln, i))
                else:
                    toks.append(Tok("id", s, ws, ln, i))
                i = m.end()
                ws = False
                continue
            two = text[i:i + 2]
            if two in _OPS2:
                toks.append(Tok("op", two, ws, ln, i))
                i += 2
                ws = False
                continue
            if c in _OPS1:
                if c in "([{":
                    depth.append(c)
                elif c in ")]}" and depth:
                    depth.pop()
                toks.append(Tok("op", c, ws, ln, i))
                i += 1
                ws = False
                continue
            toks.append(Tok("err", f"unexpected character {c!r}", ws, ln, i))
            i += 1
            ws = False
        if not cont:
            toks.append(Tok("nl", "\n", True, ln))
    toks.append(Tok("eof", None, True, len(lines)))
    return toks


# ====================================================================== syntax tree
class Node:
    __slots__ = ("line",)
    _fields = ()

    def __init__(self, *a):
        for k, v in zip(self._fields, a):
            setattr(self, k, v)
        self.line = None

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(repr(getattr(self, f)) for f in self._fields)})"


def _node(name, fields):
    fields = tuple(fields.split())
    return type(name, (Node,), {"__slots__": fields, "_fields": fields})


Num = _node("Num", "v")
Str = _node("Str", "v dq")
Id = _node("Id", "name")
ColonAll = _node("ColonAll", "")
End = _node("End", "")
Unop = _node("Unop", "op e")
Binop = _node("Binop", "op l r")
Range = _node("Range", "a b c")
Index = _node("Index", "base args brace")
Field = _node("Field", "base name dyn")
Matrix = _node("Matrix", "rows")
CellLit = _node("CellLit", "rows")
AnonFunc = _node("AnonFunc", "params body")
FuncHandle = _node("FuncHandle", "name")
BadExpr = _node("BadExpr", "text why")

ExprStmt = _node("ExprStmt", "e text")
Assign = _node("Assign", "targets rhs text")
If = _node("If", "clauses orelse text")
For = _node("For", "var e body text")
While = _node("While", "cond body text")
Switch = _node("Switch", "e cases default text")
Break = _node("Break", "text")
Continue = _node("Continue", "text")
Return = _node("Return", "text")
Try = _node("Try", "body ident handler text")
Global = _node("Global", "names text")
Command = _node("Command", "name args text")
Dyn = _node("Dyn", "k text")
Unparsed = _node("Unparsed", "text why")
FuncDef = _node("FuncDef", "name ins outs body text")

_BLOCK_END = {"end", "else", "elseif", "case", "otherwise", "catch", "function"}


class Parser:
    def __init__(self, lines):
        self.lines = lines
        self.toks = tokenize(lines)
        self.i = 0
        self.idx_depth = 0
        self.in_mat = [False]
        self.end_functions = self._functions_use_end()

    # ---- helpers
    @property
    def t(self):
        return self.toks[self.i]

    def peek(self, k=1):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def is_op(self, *ops):
        t = self.toks[self.i]
        return t.k == "op" and t.v in ops

    def is_kw(self, *kws):
        t = self.toks[self.i]
        return t.k == "kw" and t.v in kws

    def expect_op(self, op):
        if not self.is_op(op):
            raise ParseError(f"expected {op!r}, found {self._show(self.t)}")
        self.i += 1

    @staticmethod
    def _show(t):
        return "end of line" if t.k == "nl" else ("end of input" if t.k == "eof" else repr(t.v))

    def src(self, a, b=None):
        """Source text of lines a..b (token line numbers)."""
        b = a if b is None else b
        return "\n".join(x for x in self.lines[a:b + 1] if isinstance(x, str)).strip()

    def _functions_use_end(self):
        nfun = nopen = nend = 0
        br = 0
        for t in self.toks:
            if t.k == "op" and t.v in "([{" and len(t.v) == 1:
                br += 1
            elif t.k == "op" and t.v in ")]}" and len(t.v) == 1:
                br = max(0, br - 1)
            elif t.k == "nl":
                br = 0
            elif t.k == "kw" and br == 0:
                if t.v == "function":
                    nfun += 1
                elif t.v in ("if", "for", "while", "switch", "try", "parfor"):
                    nopen += 1
                elif t.v == "end":
                    nend += 1
        return nfun > 0 and nend >= nopen + nfun

    # ---- expressions
    def expr(self):
        left = self.andand()
        while self.is_op("||"):
            self.i += 1
            left = Binop("||", left, self.andand())
        return left

    def andand(self):
        left = self.elor()
        while self.is_op("&&"):
            self.i += 1
            left = Binop("&&", left, self.elor())
        return left

    def elor(self):
        left = self.eland()
        while self.is_op("|"):
            self.i += 1
            left = Binop("|", left, self.eland())
        return left

    def eland(self):
        left = self.cmp()
        while self.is_op("&"):
            self.i += 1
            left = Binop("&", left, self.cmp())
        return left

    def cmp(self):
        left = self.rng()
        while self.is_op("==", "~=", "!=", "<", "<=", ">", ">="):
            if self._mat_break():
                break
            op = self.t.v
            self.i += 1
            left = Binop("~=" if op == "!=" else op, left, self.rng())
        return left

    def rng(self):
        a = self.add()
        if self.is_op(":") and not self._colon_ends():
            self.i += 1
            b = self.add()
            if self.is_op(":") and not self._colon_ends():
                self.i += 1
                return Range(a, b, self.add())
            return Range(a, None, b)
        return a

    def _colon_ends(self):
        nx = self.peek()
        return nx.k == "op" and nx.v in (")", ",") or nx.k in ("nl", "eof")

    def _mat_break(self):
        """Inside [ ] or { }, ``a -b`` separates two elements while ``a - b`` and ``a-b`` do not."""
        if not self.in_mat[-1]:
            return False
        t, nx = self.t, self.peek()
        return t.ws and not nx.ws

    def add(self):
        left = self.mul()
        while self.is_op("+", "-"):
            if self._mat_break():
                break
            op = self.t.v
            self.i += 1
            left = Binop(op, left, self.mul())
        return left

    def mul(self):
        left = self.unary()
        while self.is_op("*", "/", "\\", ".*", "./", ".\\"):
            op = self.t.v
            self.i += 1
            left = Binop(op, left, self.unary())
        return left

    def unary(self):
        if self.is_op("-", "+", "~", "!"):
            op = self.t.v
            self.i += 1
            return Unop("~" if op == "!" else op, self.unary())
        return self.power()

    def power(self):
        left = self.postfix()
        while True:
            if self.is_op("'", ".'"):
                self.i += 1
                left = Unop("'", left)
            elif self.is_op("^", ".^"):
                op = self.t.v
                self.i += 1
                left = Binop(op, left, self.pow_operand())
            else:
                return left

    def pow_operand(self):
        if self.is_op("-", "+", "~", "!"):
            op = self.t.v
            self.i += 1
            return Unop("~" if op == "!" else op, self.pow_operand())
        return self.postfix()

    def args(self, close):
        self.idx_depth += 1
        self.in_mat.append(False)
        out = []
        try:
            while self.t.k == "nl" and close == "}":
                self.i += 1
            if self.is_op(close):
                self.i += 1
                return out
            while True:
                if self.is_op(":") and self.peek().k == "op" and self.peek().v in (",", close):
                    self.i += 1
                    out.append(ColonAll())
                else:
                    out.append(self.expr())
                if self.is_op(","):
                    self.i += 1
                    continue
                self.expect_op(close)
                return out
        finally:
            self.idx_depth -= 1
            self.in_mat.pop()

    def postfix(self):
        e = self.primary()
        while True:
            t = self.t
            if t.k != "op":
                return e
            if t.v == "(":
                if self.in_mat[-1] and t.ws:
                    return e
                self.i += 1
                e = Index(e, self.args(")"), False)
            elif t.v == "{":
                if self.in_mat[-1] and t.ws:
                    return e
                self.i += 1
                e = Index(e, self.args("}"), True)
            elif t.v == "." and not t.ws and self.peek().k in ("id", "kw") and not self.peek().ws:
                self.i += 1
                e = Field(e, self.t.v, None)
                self.i += 1
            elif t.v == "." and not t.ws and self.peek().k == "op" and self.peek().v == "(":
                self.i += 2
                self.in_mat.append(False)
                d = self.expr()
                self.in_mat.pop()
                self.expect_op(")")
                e = Field(e, None, d)
            else:
                return e

    def primary(self):
        t = self.t
        if t.k == "num":
            self.i += 1
            return Num(_F(t.v))
        if t.k == "str" or t.k == "dstr":
            self.i += 1
            return Str(t.v, t.k == "dstr")
        if t.k == "id":
            self.i += 1
            return Id(t.v)
        if t.k == "kw" and t.v == "end" and self.idx_depth > 0:
            self.i += 1
            return End()
        if t.k == "op":
            if t.v == "(":
                self.i += 1
                self.in_mat.append(False)
                e = self.expr()
                self.in_mat.pop()
                self.expect_op(")")
                return Unop("()", e)
            if t.v == "[":
                self.i += 1
                return Matrix(self.matrix("]"))
            if t.v == "{":
                self.i += 1
                return CellLit(self.matrix("}"))
            if t.v == "@":
                self.i += 1
                if self.is_op("("):
                    self.i += 1
                    params = []
                    while not self.is_op(")"):
                        if self.t.k == "id":
                            params.append(self.t.v)
                        elif self.is_op("~"):
                            params.append("~")
                        else:
                            raise ParseError("bad parameter list of an anonymous function")
                        self.i += 1
                        if self.is_op(","):
                            self.i += 1
                    self.i += 1
                    self.in_mat.append(False)
                    save, self.idx_depth = self.idx_depth, 0
                    body = self.expr()
                    self.idx_depth = save
                    self.in_mat.pop()
                    return AnonFunc(params, body)
                if self.t.k == "id":
                    name = self.t.v
                    self.i += 1
                    while self.is_op(".") and self.peek().k == "id":
                        name += "." + self.peek().v
                        self.i += 2
                    return FuncHandle(name)
                raise ParseError("bad function handle")
        if t.k == "err":
            raise ParseError(t.v)
        raise ParseError(f"unexpected {self._show(t)}")

    def matrix(self, close):
        rows, row = [], []
        self.in_mat.append(True)
        save = self.idx_depth          # 'end' keeps its meaning inside brackets within an index: a([1 end])
        try:
            while True:
                t = self.t
                if t.k == "op" and t.v == close:
                    self.i += 1
                    break
                if t.k == "nl" or (t.k == "op" and t.v == ";"):
                    self.i += 1
                    if row:
                        rows.append(row)
                        row = []
                    continue
                if t.k == "op" and t.v == ",":
                    self.i += 1
                    continue
                if t.k in ("eof", "dyn", "cmd"):
                    raise ParseError("unbalanced brackets")
                row.append(self.expr())
            if row:
                rows.append(row)
            return rows
        finally:
            self.in_mat.pop()
            self.idx_depth = save

    # ---- statements
    def program(self):
        """Parse everything; returns (statements, functions)."""
        body, funcs = [], {}
        while self.t.k != "eof":
            if self.is_kw("function"):
                f = self.funcdef()
                funcs.setdefault(f.name, f)
                continue
            if self.is_kw("end", "else", "elseif", "case", "otherwise", "catch"):
                raise ParseError(f"'{self.t.v}' without a matching block opener (line: {self.src(self.t.line)!r})")
            s = self.statement()
            if s is not None:
                body.append(s)
        return body, funcs

    def block(self, stop, opener_line=None):
        out = []
        while True:
            t = self.t
            if t.k == "eof":
                raise ParseError("unexpected end of input inside a block (missing 'end')")
            if t.k == "kw" and t.v in stop:
                return out
            if t.k == "kw" and t.v in _BLOCK_END:
                raise ParseError(f"unexpected '{t.v}' (line: {self.src(t.line)!r})")
            s = self.statement(opener_line)
            if s is not None:
                out.append(s)

    def _skip_seps(self):
        while self.t.k == "nl" or self.is_op(";", ","):
            self.i += 1

    def _end_stmt(self):
        t = self.t
        if t.k in ("nl", "eof") or (t.k == "op" and t.v in (";", ",")):
            while self.is_op(";", ","):
                self.i += 1
            if self.t.k == "nl":
                self.i += 1
            return
        if t.k == "kw" and t.v in _BLOCK_END:
            return
        raise ParseError(f"unexpected {self._show(t)}")

    def _header_expr(self, line0):
        """Condition of if/while/switch/case: a failure here makes the block undecidable, not unparseable."""
        start = self.i
        try:
            self.in_mat.append(False)
            e = self.expr()
            if not (self.t.k in ("nl", "eof", "id", "kw", "cmd", "dyn") or self.is_op(",", ";", "[", "(")):
                raise ParseError(f"unexpected {self._show(self.t)}")
            return e
        except ParseError as exc:
            self.i = start
            while self.t.k not in ("nl", "eof"):
                self.i += 1
            return BadExpr(self.src(line0), str(exc))
        finally:
            self.in_mat.pop()

    def statement(self, opener_line=None):
        t = self.t
        if t.k == "nl" or (t.k == "op" and t.v in (";", ",")):
            self.i += 1
            return None
        line0 = t.line
        if t.k == "dyn":
            self.i += 1
            s = Dyn(t.v, None)
            s.line = line0
            return s
        if t.k == "cmd":
            self.i += 1
            s = Command(t.v[0], t.v[1], self.src(line0))
            s.line = line0
            return s
        if t.k == "kw":
            s = self.compound(t)
            if s is not None:
                s.line = line0
                return s
        start = self.i
        idx_depth, in_mat = self.idx_depth, list(self.in_mat)
        try:
            s = self.simple()
            self._end_stmt()
            s.text = self.src(line0, self.toks[self.i - 1].line)
            s.line = line0
            return s
        except ParseError as exc:
            self.idx_depth, self.in_mat = idx_depth, in_mat
            self.i = start
            return self._recover(line0, opener_line, str(exc))

    def _recover(self, line0, opener_line, why):
        """Skip a statement that could not be parsed."""
        depth, j = 0, self.i
        # prefer the end of the logical line; brackets left open extend it over following lines
        end = None
        while True:
            t = self.toks[j]
            if t.k in ("eof", "dyn"):
                break
            if t.k == "op" and t.v in ("(", "[", "{"):
                depth += 1
            elif t.k == "op" and t.v in (")", "]", "}"):
                depth -= 1
            elif t.k == "nl" and depth <= 0:
                end = j
                break
            elif t.k == "kw" and depth <= 0 and t.v in ("end", "else", "elseif", "case", "otherwise", "catch") \
                    and opener_line is not None and t.line == opener_line and j > self.i:
                end = j - 1
                break
            elif t.k == "nl" and t.line - line0 > 400:
                break
            j += 1
        if end is None:                     # unbalanced: give up at the first physical line end
            j = self.i
            while self.toks[j].k not in ("nl", "eof", "dyn"):
                j += 1
            end = j
        last_line = self.toks[end].line if self.toks[end].k != "eof" else self.toks[end - 1].line
        text = self.src(line0, last_line)
        kws = [t.v for t in self.toks[self.i:end + 1] if t.k == "kw" and t.v not in ("end",)]
        self.i = end + 1 if self.toks[end].k == "nl" else end + (0 if self.toks[end].k in ("eof", "dyn") else 1)
        if any(k in ("if", "for", "while", "switch", "try", "function", "parfor", "classdef") for k in kws):
            raise ParseError(f"cannot parse a line that opens a block: {text[:100]!r} ({why})")
        s = Unparsed(text, why)
        s.line = line0
        return s

    def compound(self, t):
        line0 = t.line
        kw = t.v
        if kw == "if":
            self.i += 1
            clauses, orelse = [], []
            cond = self._header_expr(line0)
            while True:
                body = self.block({"elseif", "else", "end"}, line0)
                clauses.append((cond, body))
                if self.is_kw("elseif"):
                    l1 = self.t.line
                    self.i += 1
                    cond = self._header_expr(l1)
                    continue
                if self.is_kw("else"):
                    self.i += 1
                    orelse = self.block({"end"}, line0)
                self.i += 1          # end
                break
            return If(clauses, orelse, self.src(line0))
        if kw in ("for", "parfor"):
            self.i += 1
            paren = self.is_op("(")
            start = self.i
            try:
                if paren:
                    self.i += 1
                if self.t.k != "id":
                    raise ParseError("bad for statement")
                var = self.t.v
                self.i += 1
                self.expect_op("=")
                self.in_mat.append(False)
                try:
                    e = self.expr()
                finally:
                    self.in_mat.pop()
                if paren:
                    self.expect_op(")")
            except ParseError as exc:
                self.i = start
                m = re.match(r"\s*\(?\s*([A-Za-z_]\w*)\s*=", self.src(line0)[3:])
                var = m.group(1) if m else None
                while self.t.k not in ("nl", "eof"):
                    self.i += 1
                e = BadExpr(self.src(line0), str(exc))
            body = self.block({"end"}, line0)
            self.i += 1
            return For(var, e, body, self.src(line0))
        if kw == "while":
            self.i += 1
            cond = self._header_expr(line0)
            body = self.block({"end"}, line0)
            self.i += 1
            return While(cond, body, self.src(line0))
        if kw == "switch":
            self.i += 1
            e = self._header_expr(line0)
            self._skip_seps()
            cases, default = [], None
            while True:
                if self.is_kw("case"):
                    l1 = self.t.line
                    self.i += 1
                    ce = self._header_expr(l1)
                    cases.append((ce, self.block({"case", "otherwise", "end"}, line0)))
                elif self.is_kw("otherwise"):
                    self.i += 1
                    default = self.block({"case", "otherwise", "end"}, line0)
                elif self.is_kw("end"):
                    self.i += 1
                    break
                else:
                    raise ParseError("bad switch statement")
            return Switch(e, cases, default, self.src(line0))
        if kw == "try":
            self.i += 1
            if self.is_op(","):
                self.i += 1
            body = self.block({"catch", "end"}, line0)
            ident, handler = None, []
            if self.is_kw("catch"):
                cl = self.t.line
                self.i += 1
                nx = self.peek()
                if self.t.k == "id" and self.t.line == cl and (nx.k == "nl" or (nx.k == "op" and nx.v in (";", ","))):
                    ident = self.t.v
                    self.i += 1
                handler = self.block({"end"}, line0)
            self.i += 1
            return Try(body, ident, handler, self.src(line0))
        if kw in ("break", "continue", "return"):
            self.i += 1
            self._end_stmt()
            return {"break": Break, "continue": Continue, "return": Return}[kw](self.src(line0))
        if kw in ("global", "persistent"):
            self.i += 1
            names = []
            while self.t.k == "id":
                names.append(self.t.v)
                self.i += 1
            self._end_stmt()
            if kw == "persistent":
                return Unparsed(self.src(line0), "persistent variables are not supported")
            return Global(names, self.src(line0))
        if kw == "function":
            raise ParseError("function definition inside a block")
        if kw == "classdef":
            raise ParseError("classdef is not supported")
        return None

    def simple(self):
        t = self.t
        if t.k == "op" and t.v == "[":
            # multiple assignment?  find the matching bracket
            depth, j = 0, self.i
            while True:
                tj = self.toks[j]
                if tj.k in ("eof", "nl", "dyn"):
                    j = None
                    break
                if tj.k == "op" and tj.v in ("[", "(", "{"):
                    depth += 1
                elif tj.k == "op" and tj.v in ("]", ")", "}"):
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j is not None and self.toks[j + 1].k == "op" and self.toks[j + 1].v == "=":
                self.i += 1
                targets = []
                self.in_mat.append(True)
                try:
                    while not self.is_op("]"):
                        if self.is_op(","):
                            self.i += 1
                            continue
                        if self.is_op("~", "!"):
                            self.i += 1
                            targets.append(None)
                            continue
                        e = self.postfix()
                        self._check_lvalue(e)
                        targets.append(e)
                finally:
                    self.in_mat.pop()
                self.i += 2
                return Assign(targets, self.expr(), None)
        e = self.expr()
        if self.is_op("="):
            self._check_lvalue(e)
            self.i += 1
            return Assign([e], self.expr(), None)
        return ExprStmt(e, None)

    def _check_lvalue(self, e):
        while True:
            if isinstance(e, Id):
                return
            if isinstance(e, (Index, Field)):
                e = e.base
                continue
            raise ParseError("invalid target of an assignment")

    def funcdef(self):
        line0 = self.t.line
        self.i += 1
        outs, ins = [], []
        # [o1, o2] = name(...) | o = name(...) | name(...)
        if self.is_op("["):
            self.i += 1
            while not self.is_op("]"):
                if self.t.k == "id":
                    outs.append(self.t.v)
                elif not self.is_op(","):
                    raise ParseError("bad function declaration")
                self.i += 1
            self.i += 1
            self.expect_op("=")
        elif self.t.k == "id" and self.peek().k == "op" and self.peek().v == "=":
            outs.append(self.t.v)
            self.i += 2
        if self.t.k != "id":
            raise ParseError("bad function declaration")
        name = self.t.v
        self.i += 1
        if self.is_op("("):
            self.i += 1
            while not self.is_op(")"):
                if self.t.k == "id":
                    ins.append(self.t.v)
                elif self.is_op("~"):
                    ins.append("~")
                elif not self.is_op(","):
                    raise ParseError("bad function declaration")
                self.i += 1
            self.i += 1
        body = []
        while True:
            t = self.t
            if t.k == "eof" or (t.k == "kw" and t.v == "function"):
                break
            if t.k == "kw" and t.v == "end":
                if not self.end_functions:
                    raise ParseError("unexpected 'end'")
                self.i += 1
                break
            if t.k == "kw" and t.v in _BLOCK_END:
                raise ParseError(f"unexpected '{t.v}'")
            s = self.statement()
            if s is not None:
                body.append(s)
        return FuncDef(name, ins, outs, body, self.src(line0))


def parse(text_or_lines):
    """Parse MATLAB source; returns ``(statements, functions)``."""
    lines = text_or_lines.split("\n") if isinstance(text_or_lines, str) else list(text_or_lines)
    return Parser(lines).program()


# ====================================================================== built-in functions
BUILTINS = {}
LAZY = {}          # built-ins that receive their arguments unevaluated


def bi(*names):
    def deco(f):
        for n in names:
            BUILTINS[n] = f
        return f
    return deco


def lazy(*names):
    def deco(f):
        for n in names:
            LAZY[n] = f
        return f
    return deco


def _nargs(a, lo, hi=None, name="function"):
    hi = lo if hi is None else hi
    if not lo <= len(a) <= hi:
        raise MatlabError(f"{name}: wrong number of arguments ({len(a)})")


def _elem(fn, name):
    def f(I, a, nout):
        _nargs(a, 1, 1, name)
        x = a[0]
        if type(x) is np.float64:
            return fn(x)
        return ret(fn(farr(x)))
    BUILTINS[name] = f
    return f


def _mround(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _gamma(x):
    return _F(special.gamma(x)) if np.ndim(x) == 0 else special.gamma(x)


for _n, _f in {"exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs,
               "sign": np.sign, "floor": np.floor, "ceil": np.ceil, "fix": np.trunc, "round": _mround,
               "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
               "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "asinh": np.arcsinh, "acosh": np.arccosh,
               "atanh": np.arctanh, "gamma": _gamma, "gammaln": lambda x: _F(special.gammaln(x)) if np.ndim(x) == 0 else special.gammaln(x),
               "erf": lambda x: _F(special.erf(x)) if np.ndim(x) == 0 else special.erf(x),
               "erfc": lambda x: _F(special.erfc(x)) if np.ndim(x) == 0 else special.erfc(x),
               "isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite, "log1p": np.log1p, "expm1": np.expm1,
               "real": lambda x: x, "conj": lambda x: x, "double": lambda x: x, "uplus": lambda x: x,
               "factorial": lambda x: np.round(special.gamma(x + 1)),
               "norminv_std": special.ndtri}.items():
    _elem(_f, _n)
del BUILTINS["norminv_std"]

for _n, _f in {"plus": plus, "minus": minus, "times": times, "rdivide": rdivide, "ldivide": ldivide,
               "power": power, "mtimes": mtimes, "mrdivide": mrdivide, "mldivide": mldivide, "mpower": mpower,
               "lt": lt, "gt": gt, "le": le, "ge": ge, "eq": eq, "ne": ne, "and": el_and, "or": el_or,
               "atan2": lambda a, b: _ew(np.arctan2, a, b), "hypot": lambda a, b: _ew(np.hypot, a, b)}.items():
    BUILTINS[_n] = (lambda fn, nm: lambda I, a, nout: (_nargs(a, 2, 2, nm), fn(a[0], a[1]))[1])(_f, _n)

BUILTINS["double"] = lambda I, a, nout: (np.array([[float(ord(c)) for c in a[0]]]) if len(a[0]) != 1 else _F(ord(a[0]))) \
    if isinstance(a[0], str) and a[0] else (np.zeros((0, 0)) if isinstance(a[0], str) else ret(farr(a[0])))
BUILTINS["class"] = lambda I, a, nout: _class(a[0])
BUILTINS["uminus"] = lambda I, a, nout: uminus(a[0])
BUILTINS["not"] = lambda I, a, nout: lnot(a[0])
BUILTINS["transpose"] = BUILTINS["ctranspose"] = lambda I, a, nout: transpose(a[0])
BUILTINS["colon"] = lambda I, a, nout: colon(*a)
BUILTINS["horzcat"] = lambda I, a, nout: hcat(a)
BUILTINS["vertcat"] = lambda I, a, nout: vcat(a)
BUILTINS["pi"] = lambda I, a, nout: _const(np.pi, a)
BUILTINS["eps"] = lambda I, a, nout: _const(np.finfo(float).eps, a) if not a or not is_num(a[0]) or isinstance(a[0], np.ndarray) else np.spacing(abs(scalar(a[0])))
BUILTINS["Inf"] = BUILTINS["inf"] = lambda I, a, nout: _const(np.inf, a)
BUILTINS["NaN"] = BUILTINS["nan"] = lambda I, a, nout: _const(np.nan, a)
BUILTINS["true"] = lambda I, a, nout: ret(_const(1.0, a) != 0) if a else np.bool_(True)
BUILTINS["false"] = lambda I, a, nout: ret(_const(0.0, a) != 0) if a else np.bool_(False)
BUILTINS["zeros"] = lambda I, a, nout: _const(0.0, a)
BUILTINS["ones"] = lambda I, a, nout: _const(1.0, a)
BUILTINS["realmax"] = lambda I, a, nout: _F(np.finfo(float).max)
BUILTINS["realmin"] = lambda I, a, nout: _F(np.finfo(float).tiny)


def _dims(a, name="size"):
    a = [x for x in a if not isinstance(x, str) or _unsup(f"{name}: class or 'like' arguments")]
    if not a:
        return (1, 1)
    if len(a) == 1:
        d = arr(a[0])
        if d.size == 1:
            n = max(0, as_index_int(d.flat[0], "dimension"))
            return (n, n)
        d = [max(0, as_index_int(x, "dimension")) for x in d.ravel()]
    else:
        d = [max(0, as_index_int(x, "dimension")) for x in a]
    if len(d) > 2 and any(x != 1 for x in d[2:]):
        raise Unsupported("arrays with more than two dimensions")
    if d[0] * d[1] > MAX_ELEMENTS:
        raise Unsupported(f"array with more than {MAX_ELEMENTS} elements")
    return (d[0], d[1])


def _const(value, a):
    if not a:
        return _F(value)
    return ret(np.full(_dims(a), value))


@bi("eye")
def _eye(I, a, nout):
    r, c = _dims(a)
    return ret(np.eye(r, c))


@bi("rand", "randn", "randi", "randperm", "normrnd", "unifrnd", "mvnrnd", "gamrnd", "betarnd")
def _random(I, a, nout):
    raise Unsupported("random numbers are not reproduced by pydyn", "random")


@bi("i", "j", "complex", "imag", "angle")
def _complex(I, a, nout):
    raise Unsupported("complex numbers")


@bi("mod")
def _mod(I, a, nout):
    _nargs(a, 2, 2, "mod")
    x, y = farr(a[0]), farr(a[1])
    _bcast(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(y == 0, x + 0 * y, np.mod(x, np.where(y == 0, 1, y)))
    return ret(r)


@bi("rem")
def _rem(I, a, nout):
    _nargs(a, 2, 2, "rem")
    x, y = farr(a[0]), farr(a[1])
    _bcast(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(y == 0, np.nan, np.fmod(x, np.where(y == 0, 1, y)))
    return ret(r)


@bi("nthroot")
def _nthroot(I, a, nout):
    _nargs(a, 2, 2, "nthroot")
    x, n = farr(a[0]), scalar(a[1])
    if n != int(n):
        raise MatlabError("nthroot: N must be an integer")
    if n % 2 == 0 and (x < 0).any():
        raise MatlabError("nthroot: negative X with even N")
    return ret(np.sign(x) * np.abs(x) ** (1.0 / n))


def _dim_arg(x, a, k):
    """Dimension to reduce along: argument k if given, else the first non-singleton one."""
    if len(a) > k:
        d = as_index_int(a[k], "dimension")
        if d not in (1, 2):
            if d > 2:
                return None
            raise MatlabError("dimension must be a positive integer")
        return d - 1
    return 1 if x.shape[0] == 1 else 0


def _reducer(name, fn, empty):
    def f(I, a, nout):
        _nargs(a, 1, 2, name)
        x = farr(a[0])
        if x.size == 0 and x.shape == (0, 0):
            return _F(empty)
        ax = _dim_arg(x, a, 1)
        if ax is None:
            return ret(x)
        with np.errstate(invalid="ignore"):
            return ret(fn(x, axis=ax, keepdims=True))
    BUILTINS[name] = f


_reducer("sum", np.sum, 0.0)
_reducer("prod", np.prod, 1.0)
_reducer("mean", np.mean, np.nan)
_reducer("median", np.median, np.nan)


@bi("std", "var")
def _std(I, a, nout, _which=None):
    x = farr(a[0])
    w = 0
    if len(a) > 1 and not (isinstance(a[1], np.ndarray) and a[1].size == 0):
        w = scalar(a[1])
        if w not in (0, 1):
            raise Unsupported("weighted variance")
    ax = _dim_arg(x, a, 2)
    if x.size == 0:
        return _F(np.nan)
    if ax is None:
        return ret(np.zeros(x.shape))
    n = x.shape[ax]
    ddof = 0 if (w == 1 or n == 1) else 1
    with np.errstate(invalid="ignore"):
        return ret(np.var(x, axis=ax, ddof=ddof, keepdims=True))


BUILTINS["std"] = lambda I, a, nout: ret(np.sqrt(farr(_std(I, a, nout))))


@bi("cumsum", "cumprod")
def _cum(I, a, nout):
    raise Unsupported("cumsum/cumprod")


def _minmax(name, red, argred, ew):
    def f(I, a, nout):
        _nargs(a, 1, 3, name)
        if len(a) >= 2 and not (isinstance(a[1], np.ndarray) and a[1].size == 0):
            if len(a) == 3:
                raise MatlabError(f"{name}: with a dimension argument the second argument must be []")
            if nout > 1:
                raise MatlabError(f"{name}: two outputs are not available with two inputs")
            return _ew(ew, a[0], a[1])
        x = farr(a[0])
        if x.size == 0:
            return [np.zeros((0, 0)), np.zeros((0, 0))][:max(nout, 1)]
        ax = _dim_arg(x, a, 2)
        if ax is None:
            return [ret(x), ret(np.ones(x.shape))][:max(nout, 1)]
        allnan = np.all(np.isnan(x), axis=ax, keepdims=True)
        y = np.where(np.isnan(x), -np.inf if name == "max" else np.inf, x)
        val = np.where(allnan, np.nan, red(y, axis=ax, keepdims=True))
        if nout <= 1:
            return ret(val)
        idx = np.where(allnan, 0, argred(y, axis=ax, keepdims=True)) + 1.0
        return [ret(val), ret(idx)]
    BUILTINS[name] = f


_minmax("max", np.max, np.argmax, np.fmax)
_minmax("min", np.min, np.argmin, np.fmin)


def _anyall(name, fn):
    def f(I, a, nout):
        _nargs(a, 1, 2, name)
        x = _logical(a[0])
        if x.size == 0 and x.shape == (0, 0):
            return np.bool_(name == "all")
        ax = _dim_arg(x, a, 1)
        return ret(x) if ax is None else ret(fn(x, axis=ax, keepdims=True))
    BUILTINS[name] = f


_anyall("any", np.any)
_anyall("all", np.all)


@bi("find")
def _find(I, a, nout):
    _nargs(a, 1, 2, "find")
    if nout > 1:
        raise Unsupported("find with several outputs")
    x = arr(a[0])
    k = np.flatnonzero(x.ravel(order="F") != 0) + 1.0
    if len(a) == 2:
        k = k[:as_index_int(a[1])]
    return ret(k.reshape(1, -1) if (x.shape[0] == 1 and x.shape[1] != 1) else k.reshape(-1, 1)) if k.size else \
        (np.zeros((1, 0)) if (x.shape[0] == 1 and x.shape[1] != 1) else np.zeros((0, 1)))


@bi("sort")
def _sort(I, a, nout):
    _nargs(a, 1, 2, "sort")
    x = farr(a[0])
    desc = len(a) == 2 and isinstance(a[1], str) and a[1].lower() == "descend"
    if len(a) == 2 and not isinstance(a[1], str):
        raise Unsupported("sort along a given dimension")
    if min(x.shape) != 1 and x.size:
        raise Unsupported("sort of a matrix")
    f = x.ravel()
    key = np.where(np.isnan(f), np.inf, f)
    order = np.argsort(-key if desc else key, kind="stable")
    if desc and np.isnan(f).any():
        order = np.concatenate([order[np.isnan(f[order])], order[~np.isnan(f[order])]])
    out = f[order].reshape(x.shape)
    return [ret(out), ret((order + 1.0).reshape(x.shape))][:max(nout, 1)] if nout > 1 else ret(out)


@bi("unique")
def _unique(I, a, nout):
    if nout > 1 or len(a) != 1 or not is_num(a[0]):
        raise Unsupported("unique with several outputs or non-numeric input")
    x = farr(a[0])
    u = np.unique(x.ravel())
    return ret(u.reshape(1, -1) if x.shape[0] == 1 else u.reshape(-1, 1))


@bi("fliplr")
def _fliplr(I, a, nout):
    return a[0][::-1] if isinstance(a[0], str) else ret(arr(a[0])[:, ::-1])


@bi("flipud")
def _flipud(I, a, nout):
    return ret(arr(a[0])[::-1, :])


@bi("size")
def _size(I, a, nout):
    _nargs(a, 1, 2, "size")
    shp = shape_of(a[0])
    if len(a) == 2:
        d = as_index_int(a[1], "dimension")
        if d < 1:
            raise MatlabError("size: dimension must be positive")
        return _F(shp[d - 1]) if d <= 2 else _F(1)
    if nout <= 1:
        return np.array([[float(shp[0]), float(shp[1])]])
    return [_F(shp[0]), _F(shp[1])] + [_F(1)] * (nout - 2)


BUILTINS["length"] = lambda I, a, nout: _F(0 if 0 in shape_of(a[0]) else max(shape_of(a[0])))
BUILTINS["numel"] = lambda I, a, nout: _F(shape_of(a[0])[0] * shape_of(a[0])[1])
BUILTINS["ndims"] = lambda I, a, nout: _F(2)
BUILTINS["rows"] = lambda I, a, nout: _F(shape_of(a[0])[0])
BUILTINS["columns"] = lambda I, a, nout: _F(shape_of(a[0])[1])
BUILTINS["isempty"] = lambda I, a, nout: np.bool_(0 in shape_of(a[0]))
BUILTINS["isscalar"] = lambda I, a, nout: np.bool_(shape_of(a[0]) == (1, 1))
BUILTINS["isvector"] = lambda I, a, nout: np.bool_(1 in shape_of(a[0]) and 0 not in shape_of(a[0]))
BUILTINS["isrow"] = lambda I, a, nout: np.bool_(shape_of(a[0])[0] == 1)
BUILTINS["iscolumn"] = lambda I, a, nout: np.bool_(shape_of(a[0])[1] == 1)
BUILTINS["ismatrix"] = lambda I, a, nout: np.bool_(True)
BUILTINS["issquare"] = lambda I, a, nout: np.bool_(shape_of(a[0])[0] == shape_of(a[0])[1])
BUILTINS["isnumeric"] = BUILTINS["isfloat"] = lambda I, a, nout: np.bool_(_class(a[0]) == "double")
BUILTINS["isreal"] = lambda I, a, nout: np.bool_(_class(a[0]) in ("double", "logical", "char"))
BUILTINS["ischar"] = lambda I, a, nout: np.bool_(isinstance(a[0], str))
BUILTINS["isstring"] = lambda I, a, nout: np.bool_(False)
BUILTINS["iscell"] = lambda I, a, nout: np.bool_(isinstance(a[0], Cell))
BUILTINS["iscellstr"] = lambda I, a, nout: np.bool_(isinstance(a[0], Cell) and all(isinstance(x, str) for x in a[0].items))
BUILTINS["isstruct"] = lambda I, a, nout: np.bool_(_class(a[0]) == "struct")
BUILTINS["islogical"] = lambda I, a, nout: np.bool_(_class(a[0]) == "logical")
BUILTINS["is_function_handle"] = lambda I, a, nout: np.bool_(isinstance(a[0], Handle))
BUILTINS["squeeze"] = BUILTINS["full"] = lambda I, a, nout: a[0]
BUILTINS["vec"] = lambda I, a, nout: index(a[0], [COLON])
BUILTINS["nnz"] = lambda I, a, nout: _F(np.count_nonzero(arr(a[0])))


@bi("isa")
def _isa(I, a, nout):
    _nargs(a, 2, 2, "isa")
    c = _class(a[0])
    return np.bool_(a[1] == c or (a[1] in ("float", "numeric") and c == "double"))


@bi("logical")
def _tological(I, a, nout):
    return ret(_logical(a[0]))


@bi("isequal")
def _isequal(I, a, nout):
    if len(a) < 2:
        raise MatlabError("isequal: at least two arguments")

    def same(x, y):
        if isinstance(x, Tainted) or isinstance(y, Tainted):
            arr(x if isinstance(x, Tainted) else y)
        if isinstance(x, str) or isinstance(y, str):
            if isinstance(x, str) and isinstance(y, str):
                return x == y
            return False if (isinstance(x, (Cell, Struct)) or isinstance(y, (Cell, Struct))) else _unsup("isequal of char and numeric")
        if isinstance(x, Cell) or isinstance(y, Cell):
            return isinstance(x, Cell) and isinstance(y, Cell) and x.shape == y.shape and all(same(p, q) for p, q in zip(x.items, y.items))
        if isinstance(x, Struct) or isinstance(y, Struct):
            return isinstance(x, Struct) and isinstance(y, Struct) and set(x) == set(y) and all(same(x[k], y[k]) for k in x)
        if not (is_num(x) and is_num(y)):
            raise Unsupported("isequal on this kind of value")
        p, q = arr(x), arr(y)
        return p.shape == q.shape and bool(np.all(p == q))
    return np.bool_(all(same(a[0], y) for y in a[1:]))


@bi("linspace")
def _linspace(I, a, nout):
    _nargs(a, 2, 3, "linspace")
    n = as_index_int(a[2]) if len(a) == 3 else 100
    lo, hi = scalar(a[0]), scalar(a[1])
    if n < 2:
        return ret(np.array([[hi]])) if n == 1 else np.zeros((1, 0))
    out = lo + (hi - lo) * np.arange(n) / (n - 1)
    out[-1] = hi
    return out.reshape(1, n)


@bi("logspace")
def _logspace(I, a, nout):
    n = as_index_int(a[2]) if len(a) == 3 else 50
    return ret(10.0 ** farr(_linspace(I, [a[0], a[1], _F(n)], 1)))


@bi("repmat")
def _repmat(I, a, nout):
    _nargs(a, 2, 3, "repmat")
    r, c = _dims(a[1:])
    if isinstance(a[0], str):
        if r > 1:
            raise Unsupported("char matrices")
        return a[0] * c
    if isinstance(a[0], Cell):
        raise Unsupported("repmat of a cell array")
    if arr(a[0]).size * r * c > MAX_ELEMENTS:
        raise Unsupported(f"array with more than {MAX_ELEMENTS} elements")
    return ret(np.tile(arr(a[0]), (r, c)))


@bi("reshape")
def _reshape(I, a, nout):
    x = arr(a[0])
    dims = a[1:]
    if len(dims) == 1:
        dims = [_F(v) for v in arr(dims[0]).ravel()]
    if len(dims) > 2 and any((isinstance(d, np.ndarray) and d.size == 0) or scalar(d) != 1 for d in dims[2:]):
        raise Unsupported("arrays with more than two dimensions")
    dims = dims[:2]
    if len(dims) != 2:
        raise MatlabError("reshape: bad size")
    d = [None if (isinstance(v, np.ndarray) and v.size == 0) else as_index_int(v) for v in dims]
    if d.count(None) > 1:
        raise MatlabError("reshape: only one dimension can be []")
    if None in d:
        known = d[1 - d.index(None)]
        if known == 0 or x.size % known:
            raise MatlabError("reshape: the number of elements must not change")
        d[d.index(None)] = x.size // known
    if d[0] * d[1] != x.size:
        raise MatlabError("reshape: the number of elements must not change")
    return ret(x.reshape(d, order="F"))


@bi("diag")
def _diag(I, a, nout):
    _nargs(a, 1, 2, "diag")
    x = farr(a[0])
    k = as_index_int(a[1]) if len(a) == 2 else 0
    if min(x.shape) == 1:
        return ret(np.diag(x.ravel(), k))
    return ret(np.diag(x, k).reshape(-1, 1))


BUILTINS["triu"] = lambda I, a, nout: ret(np.triu(farr(a[0]), as_index_int(a[1]) if len(a) > 1 else 0))
BUILTINS["tril"] = lambda I, a, nout: ret(np.tril(farr(a[0]), as_index_int(a[1]) if len(a) > 1 else 0))
BUILTINS["trace"] = lambda I, a, nout: _F(np.trace(farr(a[0])))
BUILTINS["kron"] = lambda I, a, nout: ret(np.kron(farr(a[0]), farr(a[1])))


def _square(x, name):
    if x.shape[0] != x.shape[1]:
        raise MatlabError(f"{name}: argument must be a square matrix")
    if not np.all(np.isfinite(x)):
        raise Unsupported(f"{name} of a matrix with NaN or Inf")
    return x


@bi("inv")
def _inv(I, a, nout):
    x = _square(farr(a[0]), "inv")
    try:
        if np.linalg.cond(x) > 1e15:
            raise np.linalg.LinAlgError
        return ret(np.linalg.inv(x))
    except np.linalg.LinAlgError:
        raise Unsupported("inverse of a singular or badly conditioned matrix") from None


BUILTINS["det"] = lambda I, a, nout: _F(np.linalg.det(_square(farr(a[0]), "det"))) if arr(a[0]).size else _F(1)


@bi("norm")
def _norm(I, a, nout):
    _nargs(a, 1, 2, "norm")
    x = farr(a[0])
    p = a[1] if len(a) == 2 else _F(2)
    if x.size == 0:
        return _F(0)
    if isinstance(p, str):
        if p == "fro":
            return _F(np.linalg.norm(x, "fro"))
        if p.lower() == "inf":
            p = _F(np.inf)
        else:
            raise MatlabError("norm: unknown norm type")
    p = scalar(p)
    if min(x.shape) == 1:
        return _F(np.linalg.norm(x.ravel(), p))
    if p in (1, 2, np.inf):
        return _F(np.linalg.norm(x, p))
    raise MatlabError("norm: unsupported matrix norm")


@bi("chol")
def _chol(I, a, nout):
    if nout > 1 or len(a) != 1:
        raise Unsupported("chol with options or several outputs")
    x = _square(farr(a[0]), "chol")
    try:
        return ret(np.linalg.cholesky((x + x.T) / 2 if np.allclose(x, x.T) else x).T)
    except np.linalg.LinAlgError:
        raise MatlabError("chol: matrix must be positive definite") from None


@bi("dot")
def _dot(I, a, nout):
    x, y = farr(a[0]), farr(a[1])
    if min(x.shape) != 1 or min(y.shape) != 1 or x.size != y.size:
        raise Unsupported("dot of matrices")
    return _F(x.ravel() @ y.ravel())


@bi("cat")
def _cat(I, a, nout):
    d = as_index_int(a[0])
    if d == 1:
        return vcat(a[1:])
    if d == 2:
        return hcat(a[1:])
    raise Unsupported("arrays with more than two dimensions")


@bi("nchoosek")
def _nchoosek(I, a, nout):
    n, k = scalar(a[0]), scalar(a[1])
    return _F(round(special.comb(n, k)))


@bi("normcdf")
def _normcdf(I, a, nout):
    _nargs(a, 1, 3, "normcdf")
    mu = a[1] if len(a) > 1 else _F(0)
    sg = a[2] if len(a) > 2 else _F(1)
    z = rdivide(minus(a[0], mu), sg)
    return ret(0.5 * special.erfc(-farr(z) / math.sqrt(2)))


@bi("normpdf")
def _normpdf(I, a, nout):
    _nargs(a, 1, 3, "normpdf")
    mu = a[1] if len(a) > 1 else _F(0)
    sg = a[2] if len(a) > 2 else _F(1)
    z = farr(rdivide(minus(a[0], mu), sg))
    return ret(np.exp(-0.5 * z * z) / (farr(sg) * math.sqrt(2 * math.pi)))


@bi("norminv")
def _norminv(I, a, nout):
    _nargs(a, 1, 3, "norminv")
    mu = a[1] if len(a) > 1 else _F(0)
    sg = a[2] if len(a) > 2 else _F(1)
    return plus(mu, times(sg, ret(special.ndtri(farr(a[0])))))


@bi("sub2ind")
def _sub2ind(I, a, nout):
    _nargs(a, 3, 3, "sub2ind")
    r, c = _dims([a[0]])
    i, j = farr(a[1]), farr(a[2])
    if (i < 1).any() or (i > r).any() or (j < 1).any() or (j > c).any():
        raise MatlabError("sub2ind: index out of range")
    return ret((j - 1) * r + i)


# ---------------------------------------------------------------------- strings, cells, structs
def _str(v, name):
    if not isinstance(v, str):
        if isinstance(v, np.ndarray) and v.size == 0:
            return ""
        raise MatlabError(f"{name}: a char array is required") if not isinstance(v, Tainted) else Unsupported("unknown value", "tainted")
    return v


def _strcmp(fn):
    def f(I, a, nout):
        x, y = a[0], a[1]
        n = as_index_int(a[2]) if len(a) > 2 else None

        def one(p, q):
            return isinstance(p, str) and isinstance(q, str) and fn(p, q, n)
        if isinstance(x, Tainted) or isinstance(y, Tainted):
            raise Unsupported("unknown value", "tainted")
        if isinstance(x, Cell) and isinstance(y, Cell):
            if x.shape != y.shape:
                raise MatlabError("strcmp: cell arrays must have the same size")
            return ret(np.array([one(p, q) for p, q in zip(x.items, y.items)]).reshape(x.shape, order="F"))
        if isinstance(x, Cell):
            return ret(np.array([one(p, y) for p in x.items], bool).reshape(x.shape, order="F"))
        if isinstance(y, Cell):
            return ret(np.array([one(x, q) for q in y.items], bool).reshape(y.shape, order="F"))
        return np.bool_(one(x, y))
    return f


BUILTINS["strcmp"] = _strcmp(lambda p, q, n: p == q)
BUILTINS["strcmpi"] = _strcmp(lambda p, q, n: p.lower() == q.lower())
BUILTINS["strncmp"] = _strcmp(lambda p, q, n: len(p) >= n and len(q) >= n and p[:n] == q[:n])
BUILTINS["strncmpi"] = _strcmp(lambda p, q, n: len(p) >= n and len(q) >= n and p[:n].lower() == q[:n].lower())
BUILTINS["upper"] = BUILTINS["toupper"] = lambda I, a, nout: _str(a[0], "upper").upper()
BUILTINS["lower"] = BUILTINS["tolower"] = lambda I, a, nout: _str(a[0], "lower").lower()
BUILTINS["strtrim"] = lambda I, a, nout: _str(a[0], "strtrim").strip(" \t\n\v\f\r\0")
BUILTINS["deblank"] = lambda I, a, nout: _str(a[0], "deblank").rstrip(" \t\n\v\f\r\0")
BUILTINS["strrep"] = lambda I, a, nout: _str(a[0], "strrep").replace(_str(a[1], "strrep"), _str(a[2], "strrep")) if a[1] != "" else a[0]
BUILTINS["blanks"] = lambda I, a, nout: " " * as_index_int(a[0])
BUILTINS["strcat"] = lambda I, a, nout: "".join(_str(x, "strcat").rstrip(" \t\n\v\f\r") for x in a)
BUILTINS["int2str"] = lambda I, a, nout: _num2str(_mround(scalar(a[0], "int2str argument")))
BUILTINS["str2double"] = lambda I, a, nout: _str2double(a[0])
BUILTINS["filesep"] = lambda I, a, nout: "/"


def _str2double(s):
    if not isinstance(s, str):
        return _F(np.nan)
    t = s.strip().replace("d", "e").replace("D", "e")
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|[+-]?(inf|Inf|nan|NaN)", t):
        return _F(float(t))
    return _F(np.nan)


def _num2str(x, fmt=None):
    if isinstance(x, str):
        return x
    if isinstance(x, (np.bool_, bool)):
        return "1" if x else "0"
    if not isinstance(x, _SCAL):
        a = arr(x)
        if a.size != 1:
            raise Unsupported("num2str of an array")
        x = a.flat[0]
    x = float(x)
    if fmt is not None:
        if isinstance(fmt, str):
            return _sprintf(fmt, [_F(x)])
        return _cfmt("%" + f".{as_index_int(fmt)}g", x)
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == int(x):
        return "%d" % x if abs(x) < 1e15 else _cfmt("%g", x)
    dgt = int(math.floor(math.log10(abs(x)))) + 5
    dgt = min(max(dgt, 5), 16)
    return _cfmt(f"%{dgt + 7 + (x < 0)}.{dgt}g", x).strip()


def _cfmt(spec, x):
    """C-style formatting of one number with MATLAB's spelling of Inf/NaN."""
    conv = spec[-1]
    if x != x or math.isinf(x):
        s = "NaN" if x != x else ("Inf" if x > 0 else "-Inf")
        m = re.match(r"%([-+ 0#]*)(\d*)", spec)
        w = int(m.group(2)) if m.group(2) else 0
        return s.ljust(w) if "-" in m.group(1) else s.rjust(w)
    if conv in "di":
        if x != int(x):
            return (spec[:-1] + "e") % x if "." in spec else (re.sub(r"\.\d+", "", spec[:-1]) + "e") % x
        return (re.sub(r"\.\d+", "", spec[:-1]) + "d") % int(x)
    if conv in "xXou":
        if x != int(x) or x < 0:
            return (spec[:-1] + "e") % x
        return spec % int(x)
    if conv == "c":
        return (spec[:-1] + "s") % chr(int(x))
    return spec % x


_FMT_RE = re.compile(r"%([-+ 0#]*)(\*|\d+)?(?:\.(\*|\d+))?([diouxXfeEgGcs%])")
_ESC = {"n": "\n", "t": "\t", "\\": "\\", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def _sprintf(fmt, args):
    fmt = re.sub(r"\\(.)", lambda m: _ESC.get(m.group(1), m.group(0)), fmt)
    flat = []
    for v in args:
        if isinstance(v, str):
            flat.append(v)
        elif isinstance(v, (Cell, Struct, Handle)):
            raise MatlabError("sprintf: wrong type of argument")
        else:
            flat.extend(_F(x) for x in arr(v).ravel(order="F"))
    specs = list(_FMT_RE.finditer(fmt))
    nconv = sum(1 for m in specs if m.group(4) != "%")
    out, k, first = [], 0, True
    while first or (k < len(flat) and nconv):
        first = False
        pos = 0
        for m in specs:
            out.append(fmt[pos:m.start()])
            pos = m.end()
            if m.group(4) == "%":
                out.append("%")
                continue
            if m.group(2) == "*" or m.group(3) == "*":
                raise Unsupported("sprintf with '*' field width")
            if k >= len(flat):
                if flat or k:
                    pos = len(fmt) if k else pos
                    if k:
                        break
                out.append("")
                continue
            v = flat[k]
            k += 1
            spec = "%" + m.group(1) + (m.group(2) or "") + ("." + m.group(3) if m.group(3) is not None else "") + m.group(4)
            if isinstance(v, str):
                if m.group(4) in "sc":
                    out.append((spec[:-1] + "s") % v)
                elif len(v) == 1:
                    out.append(_cfmt(spec, float(ord(v))))
                else:
                    out.append(v)
            elif m.group(4) == "s":
                x = float(v)
                out.append((spec[:-1] + "s") % (("%d" % x) if x == int(x) and abs(x) < 1e15 else _shortg(x)))
            else:
                out.append(_cfmt(spec, float(v)))
        else:
            out.append(fmt[pos:])
            continue
        break
    return "".join(out)


def _shortg(x):
    s = "%.5g" % x if x == x and not math.isinf(x) else ("NaN" if x != x else ("Inf" if x > 0 else "-Inf"))
    return s


@bi("num2str")
def _b_num2str(I, a, nout):
    _nargs(a, 1, 2, "num2str")
    return _num2str(a[0], a[1] if len(a) == 2 else None)


@bi("sprintf")
def _b_sprintf(I, a, nout):
    if not a:
        raise MatlabError("sprintf: format missing")
    return _sprintf(_str(a[0], "sprintf"), a[1:])


@bi("mat2str")
def _mat2str(I, a, nout):
    x = arr(a[0])
    prec = as_index_int(a[1]) if len(a) > 1 else 15

    def one(v):
        if isinstance(v, (bool, np.bool_)):
            return "true" if v else "false"
        return _cfmt(f"%.{prec}g", float(v))
    if x.size == 1:
        return one(x.flat[0])
    return "[" + ";".join(" ".join(one(v) for v in row) for row in x) + "]"


@bi("char")
def _char(I, a, nout):
    if len(a) == 1 and isinstance(a[0], str):
        return a[0]
    if len(a) == 1 and isinstance(a[0], Cell) and len(a[0].items) == 1 and isinstance(a[0].items[0], str):
        return a[0].items[0]
    if len(a) == 1 and is_num(a[0]):
        x = arr(a[0])
        if x.shape[0] <= 1:
            return "".join(chr(as_index_int(v)) for v in x.ravel())
    raise Unsupported("char matrices")


@bi("strsplit")
def _strsplit(I, a, nout):
    s = _str(a[0], "strsplit")
    d = _str(a[1], "strsplit") if len(a) > 1 else " "
    return Cell(s.split(d))


@bi("strjoin")
def _strjoin(I, a, nout):
    if not isinstance(a[0], Cell):
        raise MatlabError("strjoin: a cell array is required")
    d = _str(a[1], "strjoin") if len(a) > 1 else " "
    d = re.sub(r"\\(.)", lambda m: _ESC.get(m.group(1), m.group(0)), d)
    return d.join(_str(x, "strjoin") for x in a[0].items)


@bi("strfind")
def _strfind(I, a, nout):
    s, p = _str(a[0], "strfind"), _str(a[1], "strfind")
    k = [i + 1.0 for i in range(len(s) - len(p) + 1) if s.startswith(p, i)] if p else []
    return ret(np.array([k])) if k else np.zeros((0, 0))


BUILTINS["contains"] = lambda I, a, nout: np.bool_(_str(a[1], "contains") in _str(a[0], "contains"))
BUILTINS["startsWith"] = lambda I, a, nout: np.bool_(_str(a[0], "startsWith").startswith(_str(a[1], "startsWith")))
BUILTINS["endsWith"] = lambda I, a, nout: np.bool_(_str(a[0], "endsWith").endswith(_str(a[1], "endsWith")))


@bi("strmatch")
def _strmatch(I, a, nout):
    _nargs(a, 2, 3, "strmatch")
    s = _str(a[0], "strmatch")
    exact = len(a) == 3 and a[2] == "exact"
    if isinstance(a[1], Cell):
        cands = a[1].items
    elif isinstance(a[1], str):
        cands = [a[1]]
    else:
        raise Unsupported("strmatch on a char matrix")
    k = [i + 1.0 for i, c in enumerate(cands) if isinstance(c, str) and (c == s if exact else c.startswith(s))]
    return ret(np.array(k).reshape(-1, 1)) if k else np.zeros((0, 1))


@bi("ismember")
def _ismember(I, a, nout):
    _nargs(a, 2, 2, "ismember")
    x, s = a
    if isinstance(x, str) and isinstance(s, Cell):
        hit = [i + 1.0 for i, c in enumerate(s.items) if c == x]
        return [np.bool_(bool(hit)), _F(hit[0] if hit else 0)][:max(nout, 1)] if nout > 1 else np.bool_(bool(hit))
    if isinstance(x, str) and isinstance(s, str):
        return np.bool_(x == s) if len(x) != 1 else np.bool_(x in s)
    if isinstance(x, Cell) and isinstance(s, Cell):
        if nout > 1:
            raise Unsupported("ismember with two outputs on cell arrays")
        return ret(np.array([c in s.items for c in x.items], bool).reshape(x.shape, order="F"))
    if is_num(x) and is_num(s):
        if nout > 1:
            raise Unsupported("ismember with two outputs")
        return ret(np.isin(farr(x), farr(s)))
    raise Unsupported("ismember on these arguments")


@bi("cell")
def _cell(I, a, nout):
    r, c = _dims(a)
    return Cell([np.zeros((0, 0))] * (r * c), (r, c))


@bi("struct")
def _struct(I, a, nout):
    if len(a) % 2:
        raise MatlabError("struct: arguments must be field/value pairs")
    s = Struct()
    for k, v in zip(a[::2], a[1::2]):
        if isinstance(v, Cell):
            raise Unsupported("struct arrays")
        s[_str(k, "struct")] = v
    return s


@bi("fieldnames")
def _fieldnames(I, a, nout):
    if hasattr(a[0], "field_names"):
        names = a[0].field_names()
    elif isinstance(a[0], Struct):
        names = list(a[0])
    else:
        raise MatlabError("fieldnames: a struct is required")
    return Cell(names, (len(names), 1))


@bi("isfield")
def _isfield(I, a, nout):
    _nargs(a, 2, 2, "isfield")
    s, f = a

    def one(name):
        if hasattr(s, "has_field"):
            return s.has_field(name)
        return isinstance(s, Struct) and isinstance(name, str) and name in s
    if isinstance(f, Cell):
        return ret(np.array([one(x) for x in f.items], bool).reshape(f.shape, order="F"))
    return np.bool_(one(f))


@bi("rmfield")
def _rmfield(I, a, nout):
    if not isinstance(a[0], Struct):
        raise Unsupported("rmfield on this value")
    s = Struct(a[0])
    for name in (a[1].items if isinstance(a[1], Cell) else [a[1]]):
        if name not in s:
            raise MatlabError(f"rmfield: no field named {name!r}")
        del s[name]
    return s


@bi("getfield")
def _getfield(I, a, nout):
    _nargs(a, 2, 2, "getfield")
    return I.getfield(a[0], _str(a[1], "getfield"))


@bi("setfield")
def _setfield(I, a, nout):
    _nargs(a, 3, 3, "setfield")
    if not isinstance(a[0], Struct):
        raise Unsupported("setfield on this value")
    s = Struct(a[0])
    s[_str(a[1], "setfield")] = a[2]
    return s


@bi("deal")
def _deal(I, a, nout):
    if len(a) == 1:
        return [a[0]] * max(nout, 1)
    if len(a) != max(nout, 1):
        raise MatlabError("deal: number of inputs and outputs must match")
    return list(a)


# ---------------------------------------------------------------------- display, errors, misc
@lazy("disp", "display", "fprintf", "printf", "fdisp", "dprintf", "skipline", "warning", "more", "beep", "fflush",
      "set_dynare_seed", "rng", "tic", "format", "clc", "diary", "echo", "commandwindow", "dbstop", "profile",
      "close", "hold", "drawnow", "pause", "fclose", "print_info")
def _noop(I, argnodes, sc, nout):
    if nout > 0:
        raise Unsupported("output of a display or timing function", "display")
    return []


PLOT_FUNCS = ("figure plot subplot title legend xlabel ylabel zlabel axis print saveas line bar histogram hist grid box "
              "plot3 surf mesh colorbar suptitle sgtitle xlim ylim annotation text orient hgsave savefig fill area "
              "scatter stairs semilogy semilogx loglog clf set get gca gcf colormap shading dyn_saveas dyn_figure "
              "errorbar patch contour quiver fplot ezplot rplot").split()


@lazy(*PLOT_FUNCS)
def _plot(I, argnodes, sc, nout):
    raise Unsupported("plotting", "plot")


FILE_FUNCS = ("save fopen fread fwrite fgetl fgets fscanf textscan importdata xlsread xlswrite csvread csvwrite "
              "dlmread dlmwrite readtable writetable readmatrix writematrix delete copyfile movefile mkdir rmdir "
              "addpath rmpath cd pwd dir ls system unix dos isfile isfolder isdir fileparts fullfile exist_file "
              "datatomfile matfile which websave urlread unzip zip tempdir tempname getenv setenv "
              "open winopen edit type input keyboard dbstack mfilename genpath path userpath").split()


@lazy(*FILE_FUNCS)
def _file(I, argnodes, sc, nout):
    raise Unsupported("file-system, shell or interactive functions are not available", "file")


@bi("toc", "clock", "now", "date", "datestr", "cputime", "etime", "datetime", "version", "ver", "computer",
    "ispc", "isunix", "ismac", "matlab_ver_less_than", "octave_ver_less_than", "user_has_matlab_license",
    "user_has_octave_package", "matlabroot", "OCTAVE_VERSION", "dynare_version", "feature", "gcp", "maxNumCompThreads")
def _env(I, a, nout):
    if nout == 0:
        return []
    raise Unsupported("the value depends on the MATLAB installation or the clock", "environment")


@bi("isoctave")
def _isoctave(I, a, nout):
    return np.bool_(ISOCTAVE)


@bi("error")
def _error(I, a, nout):
    if not a:
        raise MatlabError("error")
    if isinstance(a[0], Struct):
        raise MatlabError(str(a[0].get("message", "error")))
    fmt = a[0] if isinstance(a[0], str) else "error"
    rest = a[1:]
    if rest and re.fullmatch(r"[A-Za-z]\w*(:[\w-]+)+", fmt) and isinstance(rest[0], str):
        fmt, rest = rest[0], rest[1:]
    try:
        msg = _sprintf(fmt, rest) if (rest or "%" in fmt or "\\" in fmt) else fmt
    except Exception:  # noqa: BLE001 - the message is informative only
        msg = fmt
    raise MatlabError(msg)


@bi("assert")
def _assert(I, a, nout):
    if not a:
        raise MatlabError("assert: argument missing")
    if len(a) >= 2 and not isinstance(a[1], str):
        raise Unsupported("assert(observed, expected, tol) form")
    if not truth(a[0]):
        raise MatlabError("assertion failed" + (": " + a[1] if len(a) > 1 else ""))
    return []


@bi("exist")
def _exist(I, a, nout):
    name = _str(a[0], "exist")
    kind = a[1] if len(a) > 1 else None
    sc = I.scope_stack[-1]
    if I.has_var(sc, name):
        return _F(1)
    if kind == "var" and not I.open_world:
        return _F(0)
    raise Unsupported("exist() for files or functions depends on the MATLAB path", "environment")


@bi("isvarname")
def _isvarname(I, a, nout):
    return np.bool_(isinstance(a[0], str) and bool(re.fullmatch(r"[A-Za-z]\w*", a[0])) and a[0] not in KEYWORDS)


@bi("nargin", "nargout")
def _nargin(I, a, nout):
    raise MatlabError("nargin/nargout used outside a function")


# ---------------------------------------------------------------------- solvers
def _options(a):
    """Options struct of fzero/fsolve as a lower-case dict."""
    if a is None or (isinstance(a, np.ndarray) and a.size == 0):
        return {}
    if isinstance(a, Struct):
        return {k.lower(): v for k, v in a.items()}
    raise MatlabError("solver options must be a struct created by optimset or optimoptions")


@bi("optimset", "optimoptions")
def _optimset(I, a, nout):
    s = Struct()
    if a and isinstance(a[0], Struct):
        s.update(a[0])
        a = a[1:]
    elif a and len(a) % 2 == 1 and (isinstance(a[0], (str, Handle))):
        a = a[1:]                        # optimoptions('fsolve', ...) / optimset('fzero')
    if len(a) % 2:
        raise MatlabError("optimset: arguments must be name/value pairs")
    for k, v in zip(a[::2], a[1::2]):
        s[_str(k, "optimset")] = v
    return s


def _opt_num(o, *names):
    for n in names:
        v = o.get(n.lower())
        if v is not None and is_num(v) and arr(v).size == 1:
            return float(scalar(v))
    return None


class _SolverNaN(Exception):
    pass


def _callable(I, f):
    if isinstance(f, Handle) or isinstance(f, str):
        if I is None:
            raise MatlabError("a function handle is required")
        return I.callable(f)
    if callable(f):
        return f
    raise MatlabError("a function handle or function name is required")


@bi("fzero")
def _fzero(I, a, nout):
    _nargs(a, 2, 3, "fzero")
    f = _callable(I, a[0])
    opts = _options(a[2] if len(a) > 2 else None)
    tolx = _opt_num(opts, "TolX")

    def F(x):
        try:
            v = f(_F(x))
            v = scalar(v, "fzero: function value")
        except FloatingPointError:
            return np.nan                        # complex or indeterminate value
        return float(v)

    def fail(flag):
        return [_F(np.nan), _F(np.nan), _F(flag), Struct()][:max(nout, 1)] if nout > 1 else _F(np.nan)

    x0 = farr(a[1]).ravel()
    if x0.size == 2:
        lo, hi = float(x0[0]), float(x0[1])
        flo, fhi = F(lo), F(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            raise MatlabError("fzero: function values at the interval endpoints must be finite and real")
        if flo == 0 or fhi == 0:
            x = lo if flo == 0 else hi
            return [_F(x), _F(0.0), _F(1), Struct()][:max(nout, 1)] if nout > 1 else _F(x)
        if (flo > 0) == (fhi > 0):
            raise MatlabError("fzero: the function values at the interval endpoints must differ in sign")
    elif x0.size == 1:
        x = float(x0[0])
        fx = F(x)
        if not np.isfinite(fx):
            raise MatlabError("fzero: function value at starting guess must be finite and real")
        if fx == 0:
            return [_F(x), _F(0.0), _F(1), Struct()][:max(nout, 1)] if nout > 1 else _F(x)
        # MATLAB's search for an interval with a sign change around x0
        dx = x / 50 if x != 0 else 1 / 50
        lo = hi = x
        flo = fhi = fx
        while (flo > 0) == (fhi > 0):
            dx *= math.sqrt(2)
            lo = x - dx
            flo = F(lo)
            if not np.isfinite(flo) or not np.isfinite(lo):
                return fail(-3 if flo != flo or math.isinf(flo) else -6)
            if (flo > 0) != (fhi > 0):
                break
            hi = x + dx
            fhi = F(hi)
            if not np.isfinite(fhi) or not np.isfinite(hi):
                return fail(-3)
    else:
        raise MatlabError("fzero: second argument must be a scalar or a two-element vector")

    def G(x):
        v = F(x)
        if v != v:
            raise _SolverNaN
        return v
    try:
        x = optimize.brentq(G, lo, hi, xtol=tolx if tolx else 4e-16 * max(abs(lo), abs(hi), 1e-300), rtol=8.9e-16, maxiter=1000)
    except _SolverNaN:
        return fail(-3)
    except (RuntimeError, ValueError):
        return fail(-5)
    return [_F(x), _F(F(x)), _F(1), Struct()][:max(nout, 1)] if nout > 1 else _F(x)


def _vector_fun(I, fh, shape, extra=()):
    f = _callable(I, fh)

    def F(v):
        x = ret(np.asarray(v, float).reshape(shape, order="F"))
        try:
            out = farr(f(x, *extra)).ravel(order="F")
        except FloatingPointError:
            raise _SolverNaN from None
        if not np.all(np.isfinite(out)):
            raise _SolverNaN
        return out
    return F


def _root(F, x0):
    """Solve F(x) = 0 from x0; returns (x, f, converged)."""
    sol = optimize.root(F, x0, method="hybr", tol=1e-14)
    x = sol.x
    f = F(x)
    for _ in range(3):                    # polish: hybr can stop a little early
        if np.max(np.abs(f)) < 1e-13 or x.size != f.size:
            break
        try:
            alt = optimize.root(F, x, method="hybr", tol=1e-15)
        except _SolverNaN:
            break
        fa = F(alt.x)
        if np.max(np.abs(fa)) >= np.max(np.abs(f)):
            break
        x, f = alt.x, fa
    return x, f


@bi("fsolve")
def _fsolve(I, a, nout):
    _nargs(a, 2, 3, "fsolve")
    opts = _options(a[2] if len(a) > 2 else None)
    x0 = farr(a[1])
    F = _vector_fun(I, a[0], x0.shape)
    try:
        f0 = F(x0.ravel(order="F"))
    except _SolverNaN:
        raise MatlabError("fsolve: the objective function returns undefined values at the initial point") from None
    if f0.size != x0.size:
        raise Unsupported("fsolve for a non-square system")
    tolfun = _opt_num(opts, "TolFun", "FunctionTolerance") or 1e-6
    try:
        x, f = _root(F, x0.ravel(order="F"))
        flag = 1 if float(f @ f) <= math.sqrt(tolfun) else -2
    except _SolverNaN:
        x, f, flag = x0.ravel(order="F"), f0, -2
    out = [ret(x.reshape(x0.shape, order="F")), ret(f.reshape(-1, 1)), _F(flag), Struct()]
    return out[:max(nout, 1)] if nout > 1 else out[0]


@bi("csolve")
def _csolve(I, a, nout):
    """Dynare's csolve(FUN, x, gradfun, crit, itmax, varargin) -> [x, rc]."""
    if len(a) < 5:
        raise MatlabError("csolve: five arguments are required")
    x0 = farr(a[1])
    crit = float(scalar(a[3]))
    F = _vector_fun(I, a[0], x0.shape, tuple(a[5:]))
    try:
        x, f = _root(F, x0.ravel(order="F"))
        rc = 0 if float(np.sum(np.abs(f))) < crit else 4
    except _SolverNaN:
        x, rc = x0.ravel(order="F"), 4
    out = [ret(x.reshape(x0.shape, order="F")), _F(rc)]
    return out[:max(nout, 1)] if nout > 1 else out[0]


@bi("feval")
def _feval(I, a, nout):
    if not a:
        raise MatlabError("feval: function missing")
    if isinstance(a[0], Handle):
        return I.call_handle(a[0], list(a[1:]), nout)
    name = _str(a[0], "feval")
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise Unsupported(f"feval of {name!r}", "function")
    return I.call(name, list(a[1:]), nout, I.scope_stack[-1])


@bi("str2func")
def _str2func(I, a, nout):
    s = _str(a[0], "str2func").strip()
    if re.fullmatch(r"[A-Za-z_]\w*", s):
        return Handle(I, name=s, files=I.scope_stack[-1].funcs)
    body, _ = parse(s)
    if len(body) == 1 and isinstance(body[0], ExprStmt) and isinstance(body[0].e, AnonFunc):
        return I.ev(body[0].e, Scope())
    raise MatlabError("str2func: invalid function string")


@bi("func2str")
def _func2str(I, a, nout):
    if isinstance(a[0], Handle) and a[0].name:
        return a[0].name
    raise Unsupported("func2str of an anonymous function")


_EVAL_CACHE = {}       # parsed eval strings (steady-state files evaluate the same strings at every call)


@bi("eval")
def _eval(I, a, nout):
    _nargs(a, 1, 2, "eval")
    code = a[0]
    if isinstance(code, Tainted) or not isinstance(code, str):
        raise Unsupported("eval of a string that cannot be determined", "eval")
    sc = I.scope_stack[-1]
    cached = _EVAL_CACHE.get(code)
    try:
        body, funcs = cached if cached is not None else parse(code)
    except ParseError as exc:
        if len(a) == 2:
            return I.run_eval(a[1], sc, nout)
        raise Unsupported(f"eval: cannot parse {code[:60]!r}: {exc}", "eval") from None
    if cached is None and len(_EVAL_CACHE) < 5000:
        _EVAL_CACHE[code] = (body, funcs)
    if funcs or any(isinstance(s, Unparsed) for s in body):
        raise Unsupported(f"eval: cannot parse {code[:60]!r}", "eval")
    if len(a) == 2:
        try:
            return I.run_parsed(body, sc, nout)
        except MATLAB_ERRORS:
            return I.run_eval(a[1], sc, nout)
    return I.run_parsed(body, sc, nout)


@bi("evalin", "assignin", "evalc", "run", "source", "dynare", "inputname", "builtin", "cellfun", "arrayfun",
    "structfun", "str2num")
def _dynamic(I, a, nout):
    raise Unsupported("this function manipulates workspaces or runs code that cannot be analysed", "eval")


# ---------------------------------------------------------------------- Dynare interface
@bi("set_param_value")
def _set_param_value(I, a, nout):
    _nargs(a, 2, 2, "set_param_value")
    I.need_host("set_param_value").set_param(_str(a[0], "set_param_value"), a[1])
    I.base_assign(a[0], a[1])                 # set_param_value also does assignin('base', name, value)
    return []


@bi("get_param_by_name")
def _get_param_by_name(I, a, nout):
    _nargs(a, 1, 1, "get_param_by_name")
    return I.need_host("get_param_by_name").get_param(_str(a[0], "get_param_by_name"))


@bi("set_shock_stderr_value")
def _set_shock_stderr_value(I, a, nout):
    _nargs(a, 2, 2, "set_shock_stderr_value")
    I.need_host("set_shock_stderr_value").set_shock_stderr(_str(a[0], "set_shock_stderr_value"), a[1])
    return []


@bi("send_endogenous_variables_to_workspace", "send_exogenous_variables_to_workspace", "send_irfs_to_workspace")
def _send(I, a, nout):
    raise Unsupported("simulation results are not available while the model is loaded", "results")


#: functions known to modify the model or the workspace behind the interpreter's back: a statement
#: calling one of them cannot be skipped
MUTATORS = {"evalin", "assignin", "run", "source", "dynare", "set_shock_skew_value", "set_all_parameters",
            "set_parameters_locally", "load_params_and_steady_state", "dynare_estimation", "dynare_estimation_1",
            "set_dynare_random_generator_state", "initvalf", "histvalf", "global_initialization", "clearvars",
            "set_local_param_value", "set_shock_stderr_value", "set_param_value", "eval", "evalc", "builtin",
            "str2num", "cellfun", "arrayfun", "structfun", "feval", "set_measurement_errors", "set_exogenous_variables_for_simulation",
            # Dynare estimation routines that write their estimates into M_.params through the global
            "sur", "surgibbs", "dyn_ols", "olseqs", "pooled_ols", "pooled_fgls", "olsgibbs", "dynare_estimation_init",
            "set_prior", "osr1", "simulated_moments_estimation", "prior_posterior_statistics"}
#: packages whose functions write into M_ (pac.estimate.nls, var_expectation.update, ...)
MUTATOR_PACKAGES = {"pac", "var_expectation", "mom", "estimate", "olsgibbs", "occbin", "identification", "gsa"}
#: Dynare functions whose M_/M_.params output only reflects what the steady-state file changed
SOLVE_FUNCS = {"resol", "evaluate_steady_state", "dynare_resolve", "stoch_simul", "steady_", "resid", "check",
               "evaluate_steady_state_file", "dyn_first_order_solver", "stochastic_solvers", "compute_decision_rules"}


# ====================================================================== interpreter
class _BreakSig(Exception):
    pass


class _ContinueSig(Exception):
    pass


class _ReturnSig(Exception):
    pass


_MISSING = object()
_HOST_GLOBALS = ("M_", "options_", "oo_")


class Scope:
    __slots__ = ("vars", "global_names", "funcs", "is_base", "nargin", "nargout")

    def __init__(self, vars=None, funcs=None, is_base=False):
        self.vars = {} if vars is None else vars
        self.global_names = set(_HOST_GLOBALS) if is_base else set()
        self.funcs = funcs or {}
        self.is_base = is_base
        self.nargin = self.nargout = None


class Unit:
    """A parsed .m file."""
    __slots__ = ("path", "body", "funcs", "main")

    def __init__(self, path, body, funcs):
        self.path, self.body, self.funcs = path, body, funcs
        self.main = next(iter(funcs.values())) if funcs and not body else None

    @property
    def is_script(self):
        return self.main is None


def walk(node):
    """Yield every syntax-tree node below (and including) ``node``."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Node):
            yield n
            for f in n._fields:
                stack.append(getattr(n, f))
        elif isinstance(n, (list, tuple)):
            stack.extend(n)


class Interpreter:
    """Evaluates syntax trees.  ``host`` gives access to the model (``M_``, ``options_``, ``oo_``)."""

    def __init__(self, directory=None, host=None):
        self.dir = Path(directory) if directory is not None else None
        self.host = host
        self.globals = {}
        if host is not None:
            self.globals.update(M_=host.M, options_=host.options, oo_=host.oo)
        self.base = Scope(is_base=True)
        self.scope_stack = [self.base]
        self.end_stack = []
        self.steps = 0
        self.depth = 0
        self.open_world = False     # True once an unknown script or load may have created variables
        self._units = {}
        self._dir_files = None

    # ---- files
    def dir_files(self):
        if self._dir_files is None:
            self._dir_files = {}
            if self.dir is not None and self.dir.is_dir():
                for p in self.dir.iterdir():
                    if p.suffix == ".m" and re.fullmatch(r"[A-Za-z_]\w*", p.stem):
                        self._dir_files[p.stem] = p
        return self._dir_files

    def unit(self, name):
        """Parsed ``name.m`` from the model directory, or None."""
        if name in self._units:
            return self._units[name]
        p = self.dir_files().get(name)
        u = None
        if p is not None:
            try:
                body, funcs = parse(p.read_text(errors="replace"))
                u = Unit(p, body, funcs)
            except ParseError as exc:
                u = exc
        self._units[name] = u
        return u

    def need_host(self, what):
        if self.host is None:
            raise Unsupported(f"{what} needs a model", "function")
        return self.host

    def tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise Unsupported("step limit exceeded (endless loop?)")

    # ---- variables
    def has_var(self, sc, name):
        return name in sc.vars or (name in sc.global_names and name in self.globals)

    def getvar(self, sc, name):
        v = sc.vars.get(name, _MISSING)
        if v is _MISSING and name in sc.global_names:
            v = self.globals.get(name, _MISSING)
        return v

    def setvar(self, sc, name, value):
        if name in sc.global_names:
            if name in _HOST_GLOBALS and self.host is not None:
                if value is not self.globals.get(name):
                    raise Unsupported(f"replacing {name} as a whole")
                return
            self.globals[name] = value
        else:
            sc.vars[name] = value

    def base_assign(self, name, value):
        self.base.vars[name] = value

    # ---- expressions
    def ev(self, n, sc):
        return self._ev[type(n)](self, n, sc)

    def _num(self, n, sc):
        return n.v

    def _strn(self, n, sc):
        return n.v

    def _id(self, n, sc):
        v = sc.vars.get(n.name, _MISSING)
        if v is _MISSING:
            if n.name in sc.global_names and n.name in self.globals:
                return self.globals[n.name]
            if n.name == "nargin" and sc.nargin is not None:
                return _F(sc.nargin)
            if n.name == "nargout" and sc.nargout is not None:
                return _F(sc.nargout)
            r = self.call(n.name, [], 1, sc)
            if not r:
                raise MatlabError(f"'{n.name}' returns no value")
            return r[0]
        if type(v) is Tainted:
            raise Unsupported(f"'{n.name}' is unknown: {v.why}", "tainted")
        return v

    def _end(self, n, sc):
        if not self.end_stack:
            raise Unsupported("'end' inside a call to an unknown function or variable", "function")
        v, k, nargs = self.end_stack[-1]
        shp = (0, 0) if v is None else shape_of(v)
        if nargs == 1:
            return _F(shp[0] * shp[1])
        return _F(shp[k]) if k < 2 else _F(1)

    def _unop(self, n, sc):
        v = self.ev(n.e, sc)
        op = n.op
        if op == "()":
            return v
        if op == "-":
            return uminus(v)
        if op == "+":
            return ret(farr(v)) if not isinstance(v, np.float64) else v
        if op == "~":
            return lnot(v)
        return transpose(v)

    _BIN = {"+": plus, "-": minus, "*": mtimes, "/": mrdivide, "\\": mldivide, "^": mpower, ".*": times, "./": rdivide,
            ".\\": ldivide, ".^": power, "<": lt, ">": gt, "<=": le, ">=": ge, "==": eq, "~=": ne, "&": el_and, "|": el_or}

    def _binop(self, n, sc):
        op = n.op
        if op == "&&" or op == "||":
            a = self._sc_bool(self.ev(n.l, sc))
            if (op == "&&") != a:
                return np.bool_(a)
            return np.bool_(self._sc_bool(self.ev(n.r, sc)))
        return self._BIN[op](self.ev(n.l, sc), self.ev(n.r, sc))

    @staticmethod
    def _sc_bool(v):
        if isinstance(v, str):
            raise MatlabError("operands of && and || must be logical scalars")
        a = arr(v)
        if a.size != 1:
            raise MatlabError("operands of && and || must be convertible to logical scalar values")
        return truth(a.flat[0])

    def _range(self, n, sc):
        a = self.ev(n.a, sc)
        c = self.ev(n.c, sc)
        return colon(a, c) if n.b is None else colon(a, self.ev(n.b, sc), c)

    def _matrix(self, n, sc):
        rows = []
        for row in n.rows:
            for e in row:
                if isinstance(e, Str) and e.dq:
                    raise Unsupported("string objects (double quotes) in a concatenation")
            rows.append(hcat([self.ev(e, sc) for e in row]))
        return vcat(rows) if len(rows) != 1 else rows[0]

    def _celllit(self, n, sc):
        if not n.rows:
            return Cell([], (0, 0))
        rows = [[self.ev(e, sc) for e in row] for row in n.rows]
        w = len(rows[0])
        if any(len(r) != w for r in rows):
            raise MatlabError("cell array rows must have the same length")
        return Cell([rows[i][j] for j in range(w) for i in range(len(rows))], (len(rows), w))

    def _anon(self, n, sc):
        names = {x.name for x in walk(n.body) if isinstance(x, Id)} - set(n.params)
        env = {}
        for nm in names:
            v = self.getvar(sc, nm)
            if v is not _MISSING:
                env[nm] = v
        return Handle(self, params=n.params, body=n.body, env=env, files=sc.funcs)

    def _fhandle(self, n, sc):
        if not re.fullmatch(r"[A-Za-z_]\w*", n.name):
            raise Unsupported(f"handle to {n.name}", "function")
        return Handle(self, name=n.name, files=sc.funcs)

    def _bad(self, n, sc):
        raise Unsupported(f"cannot parse {n.text[:80]!r}: {n.why}", "syntax")

    def _colonall(self, n, sc):
        raise MatlabError("':' used outside an index")

    def index_args(self, v, argnodes, sc):
        out, n = [], len(argnodes)
        for k, a in enumerate(argnodes):
            if type(a) is ColonAll:
                out.append(COLON)
                continue
            if type(a) is Num:
                out.append(a.v)
                continue
            self.end_stack.append((v, k, n))
            try:
                out.append(self.ev(a, sc))
            finally:
                self.end_stack.pop()
        return out

    def _index(self, n, sc):
        base = n.base
        if type(base) is Id:
            v = self.getvar(sc, base.name)
            if v is _MISSING:
                if n.brace:
                    raise MatlabError(f"'{base.name}' undefined")
                r = self.call_node(base.name, n.args, 1, sc)
                if not r:
                    raise MatlabError(f"'{base.name}' returns no value")
                return r[0]
            if type(v) is Tainted:
                raise Unsupported(f"'{base.name}' is unknown: {v.why}", "tainted")
        else:
            v = self.ev(base, sc)
        if isinstance(v, Handle) and not n.brace:
            r = self.call_handle(v, [self.ev(a, sc) for a in n.args], 1)
            if not r:
                raise MatlabError("function handle returns no value")
            return r[0]
        args = self.index_args(v, n.args, sc)
        return index_brace(v, args) if n.brace else index(v, args)

    def getfield(self, v, name):
        if hasattr(v, "get_field"):
            return v.get_field(name)
        if isinstance(v, Struct):
            if name not in v:
                raise MatlabError(f"reference to non-existent field '{name}'")
            return v[name]
        if isinstance(v, Tainted):
            arr(v)
        raise Unsupported(f"field access '.{name}' on a {_class(v)} value (objects and packages are not supported)", "function")

    def _field(self, n, sc):
        b = n.base
        if type(b) is Id and self.getvar(sc, b.name) is _MISSING:
            raise Unsupported(f"unknown variable, package or object '{b.name}'", "results" if b.name in _HOST_GLOBALS else "function")
        v = self.ev(b, sc)
        name = n.name if n.name is not None else self.ev(n.dyn, sc)
        if not isinstance(name, str):
            raise MatlabError("dynamic field name must be a char array")
        return self.getfield(v, name)

    _ev = {}

    def ev_multi(self, n, sc, nargout):
        """Evaluate an expression that may return several values (a call)."""
        t = type(n)
        if t is Index and not n.brace:
            base = n.base
            if type(base) is Id and self.getvar(sc, base.name) is _MISSING:
                return self.call_node(base.name, n.args, nargout, sc)
            if type(base) is not Id or isinstance(self.getvar(sc, base.name), Handle):
                v = self.ev(base, sc)
                if isinstance(v, Handle):
                    return self.call_handle(v, [self.ev(a, sc) for a in n.args], nargout)
        elif t is Id and self.getvar(sc, n.name) is _MISSING and n.name not in ("nargin", "nargout"):
            return self.call(n.name, [], nargout, sc)
        return [self.ev(n, sc)]

    # ---- calls
    def call_node(self, name, argnodes, nargout, sc):
        if name in LAZY and name not in sc.funcs and name not in self.dir_files():
            return LAZY[name](self, argnodes, sc, nargout) or []
        args = []
        for a in argnodes:
            if type(a) is ColonAll:
                args.append(":")
            else:
                args.append(self.ev(a, sc))
        return self.call(name, args, nargout, sc)

    def call(self, name, args, nargout, sc):
        f = sc.funcs.get(name)
        if f is not None:
            return self.call_user(f, sc.funcs, args, nargout)
        if name in self.dir_files():
            u = self.unit(name)
            if isinstance(u, ParseError):
                raise Unsupported(f"{name}.m cannot be parsed: {u}", "syntax")
            if u.is_script:
                if args or nargout > 1:
                    raise MatlabError(f"{name} is a script")
                self.run_script(u, sc)
                return []
            return self.call_user(u.main, u.funcs, args, nargout)
        f = BUILTINS.get(name)
        if f is not None:
            self.scope_stack.append(sc)
            try:
                r = f(self, args, nargout)
            finally:
                self.scope_stack.pop()
            return r if type(r) is list else [r]
        if name in LAZY:
            return LAZY[name](self, [], sc, nargout) or []
        raise Unsupported(f"unknown function or variable '{name}'", "function")

    def call_user(self, f, funcs, args, nargout):
        varargin = bool(f.ins) and f.ins[-1] == "varargin"
        nfixed = len(f.ins) - (1 if varargin else 0)
        if len(args) > nfixed and not varargin:
            raise MatlabError(f"{f.name}: called with too many inputs")
        if nargout > len(f.outs) and not (f.outs and f.outs[-1] == "varargout"):
            raise MatlabError(f"{f.name}: called with too many outputs")
        if f.outs and f.outs[-1] == "varargout":
            raise Unsupported("varargout")
        sc = Scope(funcs=funcs)
        for nm, v in zip(f.ins[:nfixed], args):
            if nm != "~":
                sc.vars[nm] = v
        if varargin:
            sc.vars["varargin"] = Cell(args[nfixed:], (1, len(args) - nfixed)) if len(args) > nfixed else Cell([], (1, 0))
        sc.nargin, sc.nargout = len(args), nargout
        self.depth += 1
        if self.depth > 60:
            self.depth -= 1
            raise MatlabError("maximum recursion depth exceeded")
        self.scope_stack.append(sc)
        try:
            self.exec_block(f.body, sc)
        except _ReturnSig:
            pass
        except (_BreakSig, _ContinueSig):
            raise MatlabError("break/continue outside a loop") from None
        finally:
            self.depth -= 1
            self.scope_stack.pop()
        outs = []
        for i, o in enumerate(f.outs[:max(nargout, 1)]):
            v = sc.vars.get(o, _MISSING)
            if v is _MISSING:
                if i == 0 and nargout == 0:
                    break
                raise MatlabError(f"{f.name}: output argument '{o}' is not assigned")
            if type(v) is Tainted:
                arr(v)
            outs.append(v)
        return outs

    def call_handle(self, h, args, nargout):
        self.tick()
        if h.name is not None:
            return self.call(h.name, args, nargout, Scope(funcs=h.files or {}))
        if len(args) > len(h.params):
            raise MatlabError("anonymous function called with too many inputs")
        sc = Scope(dict(h.env), funcs=h.files or {})
        for nm, v in zip(h.params, args):
            if nm != "~":
                sc.vars[nm] = v
        self.scope_stack.append(sc)
        try:
            return self.ev_multi(h.body, sc, nargout)
        finally:
            self.scope_stack.pop()

    def callable(self, f):
        """Python callable for a function argument of fzero/fsolve (handle or function name)."""
        if isinstance(f, Handle):
            return lambda *a: self._one(self.call_handle(f, list(a), 1))
        if isinstance(f, str) and re.fullmatch(r"[A-Za-z_]\w*", f):
            sc = self.scope_stack[-1]
            return lambda *a: self._one(self.call(f, list(a), 1, sc))
        raise MatlabError("a function handle or function name is required")

    @staticmethod
    def _one(r):
        if not r:
            raise MatlabError("function returns no value")
        return r[0]

    # ---- statements
    def run_script(self, u, sc):
        saved = sc.funcs
        sc.funcs = {**saved, **u.funcs}
        try:
            self.exec_block(u.body, sc)
        except _ReturnSig:
            pass
        finally:
            sc.funcs = saved

    def run_parsed(self, body, sc, nout):
        """Body of an ``eval``."""
        if nout > 0:
            if len(body) != 1 or not isinstance(body[0], ExprStmt):
                raise MatlabError("eval: the string must be an expression when outputs are requested")
            return self.ev_multi(body[0].e, sc, nout)
        self.exec_block(body, sc)
        return []

    def run_eval(self, code, sc, nout):
        if not isinstance(code, str):
            raise Unsupported("eval of a non-string", "eval")
        try:
            body, funcs = parse(code)
        except ParseError as exc:
            raise Unsupported(f"eval: cannot parse {code[:60]!r}: {exc}", "eval") from None
        return self.run_parsed(body, sc, nout)

    def exec_block(self, stmts, sc):
        for s in stmts:
            self.exec(s, sc)

    def exec(self, s, sc):
        self.tick()
        t = type(s)
        if t is Assign or t is ExprStmt or t is Command or t is Global or t is Unparsed:
            return self.exec_simple(s, sc)
        if t is If:
            for cond, body in s.clauses:
                if truth(self.ev(cond, sc)):
                    return self.exec_block(body, sc)
            return self.exec_block(s.orelse, sc)
        if t is For:
            for v in self.for_values(s, sc):
                self.tick()
                self.setvar(sc, s.var, v)
                try:
                    self.exec_block(s.body, sc)
                except _BreakSig:
                    break
                except _ContinueSig:
                    continue
            return None
        if t is While:
            while truth(self.ev(s.cond, sc)):
                self.tick()
                try:
                    self.exec_block(s.body, sc)
                except _BreakSig:
                    break
                except _ContinueSig:
                    continue
            return None
        if t is Switch:
            body = self.switch_body(s, sc)
            return self.exec_block(body, sc) if body else None
        if t is Try:
            saved_scopes, saved_end, saved_depth = len(self.scope_stack), len(self.end_stack), self.depth
            try:
                return self.exec_block(s.body, sc)
            except MATLAB_ERRORS as exc:
                del self.scope_stack[saved_scopes:], self.end_stack[saved_end:]
                self.depth = saved_depth
                if s.ident:
                    self.setvar(sc, s.ident, Struct(message=str(exc), identifier="", stack=Struct()))
                return self.exec_block(s.handler, sc)
        if t is Break:
            raise _BreakSig
        if t is Continue:
            raise _ContinueSig
        if t is Return:
            raise _ReturnSig
        if t is Dyn:
            raise Unsupported("Dynare statement inside a function")
        raise Unsupported(f"statement {t.__name__}")

    def for_values(self, s, sc):
        v = self.ev(s.e, sc)
        if isinstance(v, Cell):
            return [Cell([x], (1, 1)) for x in v.items] if v.shape[0] <= 1 else _unsup("for over a cell matrix")
        if isinstance(v, str):
            return list(v)
        a = arr(v)
        if a.size == 0:
            return []
        if a.shape[0] == 1:
            return [_F(x) if a.dtype != bool else x for x in a[0]]
        return [ret(a[:, [j]]) for j in range(a.shape[1])]

    def switch_body(self, s, sc):
        v = self.ev(s.e, sc)

        def match(c):
            if isinstance(c, Cell):
                return any(match(x) for x in c.items)
            if isinstance(v, str) or isinstance(c, str):
                return isinstance(v, str) and isinstance(c, str) and v == c
            a, b = arr(v), arr(c)
            if a.size != 1:
                raise MatlabError("switch expression must be a scalar or a char array")
            return b.size == 1 and bool(a.flat[0] == b.flat[0])
        for ce, body in s.cases:
            if match(self.ev(ce, sc)):
                return body
        return s.default

    def exec_simple(self, s, sc):
        t = type(s)
        if t is Assign:
            tg = s.targets
            if len(tg) == 1:
                rhs = s.rhs
                if type(rhs) is Matrix and not rhs.rows and type(tg[0]) is Index and not tg[0].brace:
                    return self.assign_to(tg[0], np.zeros((0, 0)), sc)       # x(idx) = []
                vals = self.ev_multi(rhs, sc, 1)
                if not vals:
                    raise MatlabError("the right-hand side of the assignment returns no value")
                return self.assign_to(tg[0], vals[0], sc)
            vals = self.ev_multi(s.rhs, sc, len(tg))
            last = max((i for i, x in enumerate(tg) if x is not None), default=-1)
            if len(vals) <= last:
                raise MatlabError(f"the right-hand side returns {len(vals)} value(s), {last + 1} requested")
            for x, v in zip(tg, vals):
                if x is not None:
                    self.assign_to(x, v, sc)
            return None
        if t is ExprStmt:
            e = s.e
            vals = self.ev_multi(e, sc, 0)
            if vals and type(e) is not Id:
                sc.vars["ans"] = vals[0]
            return None
        if t is Command:
            return self.command(s, sc)
        if t is Global:
            for nm in s.names:
                if nm in sc.vars:
                    raise Unsupported(f"global declaration of the existing local variable '{nm}'")
                if nm in _HOST_GLOBALS and self.host is None:
                    raise Unsupported(f"global {nm} without a model", "function")
                sc.global_names.add(nm)
                self.globals.setdefault(nm, np.zeros((0, 0)))
            return None
        if t is Unparsed:
            raise Unsupported(f"cannot parse: {s.why}", "syntax")
        raise Unsupported(f"statement {t.__name__}")

    def command(self, s, sc):
        name, args = s.name, s.args.split()
        if name in ("clear", "clearvars"):
            if any(a.startswith("-") or a in ("all", "global", "functions", "classes", "mex", "java", "import") for a in args):
                # "clear all" and "clear global" also remove the global M_, options_ and oo_ in MATLAB
                raise Unsupported(f"'{' '.join(s.text.split())}' (clears more than local variables)", "eval")
            if not args or args == ["variables"]:
                for k in list(sc.vars):
                    del sc.vars[k]
            else:
                for a in args:
                    if "*" in a or "?" in a:
                        pat = re.compile(re.escape(a).replace(r"\*", ".*").replace(r"\?", ".") + r"\Z")
                        for k in [k for k in sc.vars if pat.match(k)]:
                            del sc.vars[k]
                    elif a in sc.vars:
                        del sc.vars[a]
            return None
        if name == "load":
            return self.load_file(args, sc)
        if name == "disp" or name in LAZY and LAZY[name] is _noop:
            return None
        if name in LAZY:
            return LAZY[name](self, [], sc, 0)
        raise Unsupported(f"command '{name}'", "function")

    def load_file(self, args, sc):
        raise Unsupported("load", "file")

    def assign_to(self, t, value, sc):
        if type(value) is Tainted:
            arr(value)
        if type(t) is Id:
            return self.setvar(sc, t.name, value)
        chain, e = [], t
        while type(e) is not Id:
            chain.append(e)
            e = e.base
        chain.reverse()
        cur = self.getvar(sc, e.name)
        if cur is _MISSING:
            cur = None
        new = self._assign_chain(cur, chain, 0, value, sc)
        self.setvar(sc, e.name, new)

    def _assign_chain(self, cur, chain, i, value, sc):
        if type(cur) is Tainted:
            arr(cur)
        acc = chain[i]
        last = i == len(chain) - 1
        if type(acc) is Field:
            name = acc.name if acc.name is not None else self.ev(acc.dyn, sc)
            if not isinstance(name, str):
                raise MatlabError("dynamic field name must be a char array")
            if hasattr(cur, "set_field"):
                if not last and i == len(chain) - 2 and type(chain[i + 1]) is Index and not chain[i + 1].brace \
                        and hasattr(cur, "assign_sub"):
                    sub = cur.peek_field(name, raw=True)
                    args = self.index_args(sub, chain[i + 1].args, sc)
                    cur.assign_sub(name, args, value)
                    return cur
                cur.set_field(name, value if last else self._assign_chain(cur.peek_field(name), chain, i + 1, value, sc))
                return cur
            if cur is None or (isinstance(cur, np.ndarray) and cur.size == 0):
                cur = Struct()
            if not isinstance(cur, Struct):
                raise MatlabError(f"field assignment to a {_class(cur)} value")
            s = Struct(cur)
            s[name] = value if last else self._assign_chain(s.get(name), chain, i + 1, value, sc)
            return s
        args = self.index_args(cur, acc.args, sc)
        if acc.brace:
            if last:
                return assign_brace(cur, args, value)
            try:
                sub = index_brace(cur, args) if isinstance(cur, Cell) else None
            except MatlabError:
                sub = None
            return assign_brace(cur, args, self._assign_chain(sub, chain, i + 1, value, sc))
        if last:
            return assign_index(cur, args, value)
        if (cur is None or isinstance(cur, Struct)) and all(type(x) in _SCALT and x == 1 for x in args):
            return self._assign_chain(cur, chain, i + 1, value, sc)
        raise Unsupported("struct arrays")


Interpreter._ev = {Num: Interpreter._num, Str: Interpreter._strn, Id: Interpreter._id, End: Interpreter._end,
                   Unop: Interpreter._unop, Binop: Interpreter._binop, Range: Interpreter._range,
                   Matrix: Interpreter._matrix, CellLit: Interpreter._celllit, AnonFunc: Interpreter._anon,
                   FuncHandle: Interpreter._fhandle, BadExpr: Interpreter._bad, ColonAll: Interpreter._colonall,
                   Index: Interpreter._index, Field: Interpreter._field}


def run(text, variables=None, directory=None):
    """Run MATLAB source in a fresh interpreter (no model); returns the workspace as a dict."""
    I = Interpreter(directory)
    I.base.vars.update(variables or {})
    body, funcs = parse(text)
    I.base.funcs = funcs
    with np.errstate(**_ERRSTATE):
        try:
            I.exec_block(body, I.base)
        except _ReturnSig:
            pass
    return I.base.vars


@bi("clear", "clearvars")
def _clear(I, a, nout):
    if nout > 0 or not all(isinstance(x, str) for x in a):
        raise MatlabError("clear: arguments must be char arrays")
    c = Command("clear", " ".join(a), "clear " + " ".join(a))
    return I.command(c, I.scope_stack[-1]) or []


@bi("load")
def _load(I, a, nout):
    if nout > 0 or not a or not all(isinstance(x, str) for x in a):
        raise Unsupported("load with an output or computed arguments", "file")
    return I.load_file(list(a), I.scope_stack[-1]) or []


# ====================================================================== the model seen from MATLAB
class _HostM:
    """``M_``: a live view of the model's calibration."""
    _SCALARS = {"param_nbr": lambda m: len(m.param_names), "endo_nbr": lambda m: m.n_endo, "exo_nbr": lambda m: m.n_exo,
                "orig_endo_nbr": lambda m: m.raw.get("orig_endo_nbr", m.n_endo), "exo_det_nbr": lambda m: 0,
                "maximum_lag": lambda m: m.info["maximum_lag"], "maximum_lead": lambda m: m.info["maximum_lead"],
                "maximum_endo_lag": lambda m: m.info["maximum_endo_lag"],
                "maximum_endo_lead": lambda m: m.info["maximum_endo_lead"]}
    _NAMES = {"param_names": "param_names", "endo_names": "endo_names", "exo_names": "exo_names"}

    def __init__(self, host):
        self.h = host
        self._cells = {}

    def field_names(self):
        return ["params", "Sigma_e", "fname", "dname"] + list(self._SCALARS) + list(self._NAMES)

    def has_field(self, name):
        if name in self.field_names():
            return True
        raise Unsupported(f"pydyn does not know whether M_ has a field '{name}'", "results")

    def get_field(self, name):
        return self.peek_field(name)

    def peek_field(self, name, raw=False):
        h, m = self.h, self.h.model
        if name == "params":
            if h.tainted and not raw:
                bad = [m.param_names[i] for i in sorted(h.tainted)][:5]
                raise Unsupported(f"M_.params: the value of {bad} is unknown ({h.taint_why})", "tainted")
            return h.params.reshape(-1, 1).copy() if h.params.size != 1 else _F(h.params[0])
        if name == "Sigma_e":
            if h.sigma_tainted and not raw:
                raise Unsupported(f"M_.Sigma_e is unknown ({h.taint_why})", "tainted")
            return ret(h.sigma.copy())
        if name in self._SCALARS:
            return _F(self._SCALARS[name](m))
        if name in self._NAMES:
            if name not in self._cells:
                names = list(getattr(m, self._NAMES[name]))
                self._cells[name] = Cell(names, (len(names), 1))
            return self._cells[name]
        if name in ("fname", "dname"):
            return m.json_dir.parents[1].name
        raise Unsupported(f"M_.{name} is not available in pydyn", "results")

    def set_field(self, name, value):
        h = self.h
        if name == "params":
            v = farr(value)
            if v.shape != (h.params.size, 1) and not (v.size == 1 and h.params.size == 1):
                raise MatlabError(f"M_.params must remain a {h.params.size}x1 vector (assigned {v.shape[0]}x{v.shape[1]})")
            h.params[:] = v.ravel()
            h.tainted.clear()
            h.nwrites += 1
        elif name == "Sigma_e":
            v = farr(value)
            n = h.sigma.shape[0]
            if v.shape != (n, n):
                raise MatlabError(f"M_.Sigma_e must remain a {n}x{n} matrix")
            h.sigma[:] = v
            h.sigma_tainted = False
            h.nwrites += 1
        else:
            raise Unsupported(f"assignment to M_.{name} is not supported")

    def assign_sub(self, name, args, value):
        h = self.h
        if name == "params":
            n = h.params.size
            col = h.params.reshape(-1, 1)
            new = arr(assign_index(col, args, value))
            if new.shape != (n, 1):
                raise MatlabError("M_.params must not change size")
            if len(args) == 1:
                k = _lin_index(args[0], n)
            elif len(args) == 2:
                k = _lin_index(args[0], n)
            else:
                raise MatlabError("bad index into M_.params")
            h.params[:] = new.ravel()
            h.tainted.difference_update(int(i) for i in k)
            h.nwrites += 1
        elif name == "Sigma_e":
            new = farr(assign_index(h.sigma, args, value))
            if new.shape != h.sigma.shape:
                raise MatlabError("M_.Sigma_e must not change size")
            h.sigma[:] = new
            h.nwrites += 1
        else:
            raise Unsupported(f"assignment to M_.{name} is not supported")


class _HostOptions:
    """``options_``: values assigned by the .mod file are stored, nothing is acted upon."""

    def __init__(self, store):
        self.store = store

    def field_names(self):
        return list(self.store)

    def has_field(self, name):
        if name in self.store:
            return True
        raise Unsupported(f"pydyn does not know Dynare's default for options_.{name}", "results")

    def get_field(self, name):
        if name in self.store:
            return self.store[name]
        raise Unsupported(f"options_.{name} has not been set in the .mod file (Dynare's defaults are not known to pydyn)", "results")

    def peek_field(self, name, raw=False):
        return self.store.get(name)

    def set_field(self, name, value):
        self.store[name] = value


class _HostOO:
    def has_field(self, name):
        raise Unsupported("oo_ (results) is not available while the model is loaded", "results")

    def get_field(self, name):
        raise Unsupported(f"oo_.{name}: results are not available while the model is loaded", "results")

    def peek_field(self, name, raw=False):
        raise Unsupported(f"oo_.{name}: results are not available while the model is loaded", "results")

    def set_field(self, name, value):
        raise Unsupported(f"assignment to oo_.{name}", "results")


class Host:
    """What MATLAB code can see of a model.  ``live`` hosts write through to the model."""

    def __init__(self, model, params=None, live=False):
        self.model = model
        self.params = model.params if live else np.array(model.params if params is None else params, float)
        self.sigma = model.Sigma_e if live else np.array(model.Sigma_e, float)
        self.sigma_tainted = False
        self.tainted = set()            # indices of parameters whose value in Dynare is unknown to pydyn
        self.taint_why = ""
        self.nwrites = 0
        if not hasattr(model, "matlab_options"):
            model.matlab_options = Struct()
        self.M, self.options, self.oo = _HostM(self), _HostOptions(model.matlab_options), _HostOO()

    def _pidx(self, name, what):
        i = self.model._ip.get(name)
        if i is None:
            raise MatlabError(f"{what}: parameter name {name} doesn't exist")
        return i

    def set_param(self, name, value):
        i = self._pidx(name, "set_param_value")
        self.params[i] = scalar(value, "parameter value")
        self.tainted.discard(i)
        self.nwrites += 1

    def get_param(self, name):
        i = self._pidx(name, "get_param_by_name")
        if i in self.tainted:
            raise Unsupported(f"the value of parameter {name} is unknown ({self.taint_why})", "tainted")
        return _F(self.params[i])

    def set_shock_stderr(self, name, value):
        i = self.model._ix.get(name)
        if i is None:
            raise MatlabError(f"Shock name {name} doesn't exist")
        self.sigma[i, i] = scalar(value) ** 2
        self.nwrites += 1


# ====================================================================== .mod-file session
class _Escape(Exception):
    def __init__(self, u):
        self.u = u


class _TryUnknown(Exception):
    def __init__(self, u):
        self.u = u


class _Info:
    def __init__(self):
        self.writes, self.calib, self.dyn, self.all, self.paths = set(), None, [], False, False
        self.has_return = self.has_break = False
        self.count = 0


class Workspace(dict):
    """Base workspace; scalar numeric variables are mirrored into ``model.workspace``."""

    def __init__(self, model):
        super().__init__()
        self._m = model

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if type(v) in (np.float64, np.bool_) and k not in self._m._ip:
            self._m.workspace[k] = float(v)
        else:
            self._m.workspace.pop(k, None)

    def __delitem__(self, k):
        super().__delitem__(k)
        self._m.workspace.pop(k, None)

    def update(self, other=(), **kw):
        for k, v in dict(other, **kw).items():
            self[k] = v


#: Dynare statements that only declare or store something (they never call the steady-state file)
DECL_KINDS = {"param_init", "initval", "endval", "histval", "shocks", "mshocks", "heteroskedastic_shocks", "varobs",
              "estimated_params", "estimated_params_init", "estimated_params_bounds", "estimated_params_remove",
              "occbin_constraints", "observation_trends", "native", "verbatim", "write_latex_dynamic_model",
              "write_latex_static_model", "write_latex_original_model", "write_latex_steady_state_model",
              "collect_latex_files", "write_latex_definitions", "set_time", "estimation_data", "planner_objective",
              "osr_params", "optim_weights", "matched_moments", "matched_irfs", "matched_irfs_weights", "var_model",
              "pac_model", "trend_component_model", "var_expectation_model", "perfect_foresight_setup",
              "perfect_foresight_solver", "initval_file", "histval_file", "homotopy", "occbin_setup",
              "load_params_and_steady_state", "save_params_and_steady_state", "options", "model_info"}
#: Dynare statements that compute or display: inside a MATLAB block whose condition is unknown they
#: are skipped (with a warning); any other Dynare statement there is an error
COMPUTE_KINDS = {"steady", "check", "stoch_simul", "estimation", "osr", "forecast", "calib_smoother", "identification",
                 "model_diagnostics", "model_info", "resid", "rplot", "perfect_foresight_setup",
                 "perfect_foresight_solver", "shock_decomposition", "realtime_shock_decomposition",
                 "plot_shock_decomposition", "initial_condition_decomposition", "conditional_forecast",
                 "plot_conditional_forecast", "sensitivity", "dynare_sensitivity", "extended_path", "simul",
                 "write_latex_dynamic_model", "write_latex_static_model", "write_latex_original_model",
                 "write_latex_steady_state_model", "collect_latex_files", "write_latex_definitions", "occbin_solver",
                 "occbin_graph", "occbin_write_regimes_xls", "method_of_moments", "ramsey_policy", "discretionary_policy",
                 "evaluate_planner_objective", "save_params_and_steady_state", "smoother2histval", "dynasave",
                 "dynatype", "bvar_density", "bvar_forecast", "bvar_irf", "ms_sbvar_estimation", "ms_sbvar_simulation",
                 "ms_sbvar_compute_mdd", "ms_sbvar_compute_probabilities", "ms_sbvar_irf", "ms_sbvar_forecast",
                 "ms_sbvar_variance_decomposition", "sbvar", "prior_posterior_function", "generate_irfs",
                 "perfect_foresight_with_expectation_errors_setup", "perfect_foresight_with_expectation_errors_solver",
                 "heterogeneity_solve", "heterogeneity_simulate", "heterogeneity_compute_steady_state"}
#: Dynare statements whose expressions pydyn evaluates (they may refer to MATLAB variables)
_EVALUATED_KINDS = {"param_init", "initval", "endval", "shocks", "load_params_and_steady_state"}
_ESTIMATING = {"estimation", "method_of_moments", "osr", "dynare_sensitivity", "sensitivity", "ms_sbvar_estimation"}
_OO_INPUTS = ("steady_state", "exo_steady_state", "exo_det_steady_state")


class _ModInterp(Interpreter):
    def load_file(self, args, sc):
        name = args[0].strip("'\"")
        names = [a.strip("'\"") for a in args[1:]]
        if any(n.startswith("-") for n in names) or not re.fullmatch(r"[\w.-]+", name) or ".." in name or self.dir is None:
            raise Unsupported("load with options or a path", "file")
        path = next((p for p in (self.dir / name, self.dir / (name + ".mat")) if p.is_file()), None)
        if path is None:
            raise Unsupported(f"load: {name} does not exist next to the .mod file (yet)", "file")
        try:
            from scipy.io import loadmat, whosmat
            listed = [w[0] for w in whosmat(str(path))]
        except Exception:  # noqa: BLE001 - not a MAT-file scipy can read
            raise Unsupported(f"load: {path.name} is not a MAT-file that can be read", "file") from None
        wanted = [n for n in listed if not names or n in names]
        if any(n in _HOST_GLOBALS for n in wanted):
            raise self.session.error(f"'load {name}' would replace {sorted(set(wanted) & set(_HOST_GLOBALS))}: "
                                     "pydyn cannot follow this")
        data = loadmat(str(path), variable_names=wanted, squeeze_me=False, chars_as_strings=True)
        for n in wanted:
            v = data.get(n)
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype.kind in "fiub" and not np.iscomplexobj(v):
                sc.vars[n] = ret(v.astype(float) if v.dtype.kind != "b" else v)
            elif isinstance(v, np.ndarray) and v.dtype.kind == "U" and v.size <= 1:
                sc.vars[n] = str(v.flat[0]) if v.size else ""
            else:
                sc.vars[n] = Tainted(f"loaded from {path.name} with a class pydyn does not read")
        return None


class ModSession:
    """Runs the statements of a ``.mod`` file: Dynare statements through ``model._exec_statement``,
    native MATLAB lines through the interpreter, in source order and under the same control flow."""

    def __init__(self, model, error_cls):
        self.model = model
        self.error_cls = error_cls
        self.host = Host(model, live=True)
        mod_dir = model.json_dir.parents[2]
        self.I = _ModInterp(mod_dir, self.host)
        self.I.session = self
        self.I.base.vars = Workspace(model)
        self.sc = self.I.base
        self.skipped = []               # (text, kind, reason, number of statements)
        self._skipped_keys = set()
        self.path_edits = False
        self.est_params = set()
        self.current = None
        self.in_try = 0
        self._ss_params = None

    # ---- reporting
    def error(self, msg):
        return self.error_cls(msg)

    def record(self, text, kind, why, n=1):
        key = (" ".join(str(text).split())[:120], kind)
        if key not in self._skipped_keys:        # a statement skipped in every pass of a loop is reported once
            self._skipped_keys.add(key)
            self.skipped.append((key[0], kind, str(why)[:160], n))

    def warn(self):
        if self.path_edits:
            warnings.warn("MATLAB code in the .mod file writes simulation paths or results (oo_.*); pydyn ignores "
                          "these assignments, so a perfect-foresight simulation may differ from Dynare's")
        if not self.skipped:
            return
        kinds = {}
        for _, k, _, n in self.skipped:
            kinds[k] = kinds.get(k, 0) + n
        label = {"plot": "plotting", "results": "use of results (oo_, options_ defaults, ...)", "function": "unknown functions",
                 "file": "file input/output", "random": "random numbers", "syntax": "unsupported syntax",
                 "tainted": "dependent on skipped statements", "eval": "dynamic code", "environment": "MATLAB environment",
                 "display": "display", "block": "blocks with an unknown condition", "feature": "unsupported features"}
        parts = ", ".join(f"{n} {label.get(k, k)}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))
        first = "; ".join(f"{t!r} ({w})" for t, _, w, _ in self.skipped[:3])
        warnings.warn(f"MATLAB code in the .mod file: {sum(kinds.values())} statement(s) not executed ({parts}). None of them "
                      f"writes to M_ or to the parameters. First: {first}. Full list: model.matlab_skipped")

    # ---- static analysis
    def _scan_expr(self, e, info, seen):
        nodes = list(walk(e))
        literal_eval = {id(n.base) for n in nodes if type(n) is Index and type(n.base) is Id and n.base.name == "eval"
                        and len(n.args) == 1 and type(n.args[0]) is Str}
        for n in nodes:
            if id(n) in literal_eval:
                continue
            name = None
            if type(n) is Index and type(n.base) is Id:
                name = n.base.name
            elif type(n) is Id:
                name = n.name
            elif type(n) is Field and type(n.base) is Id and n.base.name in MUTATOR_PACKAGES \
                    and not self.I.has_var(self.sc, n.base.name):
                info.calib = info.calib or f"it calls a function of the Dynare package '{n.base.name}'"
            if name is None or self.I.has_var(self.sc, name):
                continue
            if name == "eval" and type(n) is Index and len(n.args) == 1 and type(n.args[0]) is Str:
                try:
                    body, funcs = parse(n.args[0].v)
                    if funcs:
                        raise ParseError("function")
                    self._scan(body, info, seen)
                except ParseError:
                    info.calib = info.calib or f"eval of text that cannot be parsed"
                continue
            if name == "load":
                info.all = True
            if name in ("clear", "clearvars"):
                info.calib = info.calib or "it clears variables in a way pydyn cannot follow"
            if name in MUTATORS:
                info.calib = info.calib or f"it calls {name}"
            elif name in self.I.dir_files() and name not in seen:
                seen.add(name)
                u = self.I.unit(name)
                if isinstance(u, ParseError):
                    info.calib = info.calib or f"{name}.m cannot be parsed"
                elif u.is_script:
                    self._scan(u.body, info, seen)
                else:
                    sub = _Info()
                    for f in u.funcs.values():
                        self._scan(f.body, sub, seen)
                        if any(type(x) is Global for x in walk(f.body)):
                            sub.calib = sub.calib or f"{name}.m declares global variables"
                    info.calib = info.calib or sub.calib

    def _target(self, t, info):
        e, first = t, None
        while type(e) is not Id:
            if type(e) is Field and type(e.base) is Id:
                first = e.name
            e = e.base
        if e.name == "M_":
            info.calib = info.calib or "it assigns to M_"
        elif e.name == "oo_":
            if first in _OO_INPUTS:
                info.calib = info.calib or f"it assigns to oo_.{first}"
            info.paths = True
        elif e.name != "options_":
            info.writes.add(e.name)

    def _scan(self, stmts, info, seen):
        for s in stmts:
            t = type(s)
            if t is Dyn:
                info.dyn.append(s)
                continue
            info.count += 1
            if t is Assign:
                for x in s.targets:
                    if x is not None:
                        self._target(x, info)
                        self._scan_expr([a for n in walk(x) if type(n) is Index for a in n.args], info, seen)
                self._scan_expr(s.rhs, info, seen)
            elif t is ExprStmt:
                self._scan_expr(s.e, info, seen)
                if type(s.e) is Id and not self.I.has_var(self.sc, s.e.name) and s.e.name not in BUILTINS \
                        and s.e.name not in LAZY and s.e.name not in self.I.dir_files():
                    info.all = True          # could be a script that is not next to the .mod file
            elif t is Command:
                if s.name in ("load", "clear", "clearvars"):
                    info.all = True
                if s.name in ("dynare", "run") or (s.name in ("clear", "clearvars") and s.args.strip()):
                    info.calib = info.calib or f"it runs '{' '.join(s.text.split())[:60]}'"
            elif t is Unparsed:
                txt = re.sub(r"'[^']*'", "''", s.text)
                assign = re.search(r"(?<![=<>~!])=(?!=)", txt)
                if (assign and re.search(r"\bM_\b", txt[:assign.start()])) or re.search(r"\boo_\.(exo_|exo_det_)?steady_state\b[^=]*=(?!=)", txt) \
                        or any(re.search(rf"\b{m}\b", txt) for m in MUTATORS):
                    info.calib = info.calib or "it cannot be parsed and may write to the model"
                if assign:
                    lhs = txt[:assign.start()]
                    info.writes.update(n for n in re.findall(r"(?<![\w.])([A-Za-z_]\w*)", re.sub(r"\([^()]*\)|\{[^{}]*\}", "", lhs))
                                       if n not in _HOST_GLOBALS)
                    if re.search(r"\boo_\b", lhs):
                        info.paths = True
                elif re.fullmatch(r"\s*[A-Za-z_]\w*\s*;?\s*", txt):
                    info.all = True
            elif t is If:
                for c, b in s.clauses:
                    self._scan_expr(c, info, seen)
                    self._scan(b, info, seen)
                self._scan(s.orelse, info, seen)
            elif t is For:
                if s.var:
                    info.writes.add(s.var)
                self._scan_expr(s.e, info, seen)
                sub = _Info()
                self._scan(s.body, sub, seen)
                self._merge(info, sub, loop=True)
            elif t is While:
                self._scan_expr(s.cond, info, seen)
                sub = _Info()
                self._scan(s.body, sub, seen)
                self._merge(info, sub, loop=True)
            elif t is Switch:
                self._scan_expr(s.e, info, seen)
                for c, b in s.cases:
                    self._scan_expr(c, info, seen)
                    self._scan(b, info, seen)
                self._scan(s.default or [], info, seen)
            elif t is Try:
                self._scan(s.body, info, seen)
                self._scan(s.handler, info, seen)
                if s.ident:
                    info.writes.add(s.ident)
            elif t is Return:
                info.has_return = True
            elif t in (Break, Continue):
                info.has_break = True
            elif t is Global:
                pass

    @staticmethod
    def _merge(info, sub, loop=False):
        info.writes |= sub.writes
        info.calib = info.calib or sub.calib
        info.dyn += sub.dyn
        info.all |= sub.all
        info.paths |= sub.paths
        info.count += sub.count
        info.has_return |= sub.has_return
        if not loop:
            info.has_break |= sub.has_break

    def analyze(self, stmts):
        info = _Info()
        self._scan(stmts, info, set())
        return info

    # ---- taints
    def taint(self, info, why):
        ws = self.sc.vars
        if info.all:
            self.I.open_world = True
            for k in list(ws):
                ws[k] = Tainted(why)
        for k in info.writes:
            ws[k] = Tainted(why)
        if info.paths:
            self.path_edits = True

    def ss_params(self):
        """Indices of the parameters a steady-state computation may change."""
        if self._ss_params is None:
            m = self.model
            if getattr(m, "steady_state_fn", None) is not None and not isinstance(m.steady_state_fn, SteadyStateFile):
                idx = set(range(len(m.param_names)))
            elif m.steady_state_model is not None:
                names = set()
                for lhs, _ in m.steady_state_model:
                    names.update([lhs] if isinstance(lhs, str) else lhs)
                idx = {m._ip[n] for n in names if n in m._ip}
            elif m.steady_state_file is not None:
                idx = set(range(len(m.param_names)))
                try:
                    body, funcs = parse(Path(m.steady_state_file).read_text(errors="replace"))
                    main = next(iter(funcs.values()))
                    out = main.outs[1] if len(main.outs) >= 3 else None
                    direct = any(type(n) is Assign and any(x is not None and type(x) is not Id and _root_name(x) in (out, "M_")
                                                           for x in n.targets) for n in walk(main.body))
                    if not direct:
                        assigned = {_root_name(x) for n in walk(main.body) if type(n) is Assign for x in n.targets if x is not None}
                        idx = {m._ip[n] for n in assigned if n in m._ip}
                except (ParseError, StopIteration, OSError):
                    pass
            else:
                idx = set()
            self._ss_params = idx
        return self._ss_params

    def after_compute(self, kind, why):
        h = self.host
        new = set(self.ss_params())
        if kind in _ESTIMATING:
            new |= {self.model._ip[n] for n in self.est_params if n in self.model._ip}
            h.sigma_tainted = True
            h.taint_why = why
        if new:
            h.tainted |= new
            h.taint_why = why
        ws = self.sc.vars
        m = self.model
        for k in list(ws):
            if k in m._ie or k in m._ix or ("_" in k and k.rsplit("_", 1)[0] in m._ie):
                if type(ws[k]) is not Tainted and k not in m._ip:
                    ws[k] = Tainted(f"Dynare's '{kind}' may overwrite this workspace variable")

    # ---- execution
    def run(self, statements):
        self.statements = statements
        lines = [st["string"] if st["statementName"] in ("native", "verbatim") else ("dyn", k)
                 for k, st in enumerate(statements)]
        try:
            body, funcs = parse(lines)
        except ParseError as exc:
            raise self.error(f"MATLAB code in the .mod file cannot be parsed: {exc}") from None
        self.sc.funcs = funcs
        with np.errstate(**_ERRSTATE):
            for i, s in enumerate(body):
                try:
                    self.run_stmt(s)
                except _ReturnSig:
                    break
                except (_BreakSig, _ContinueSig):
                    raise self.error("MATLAB break/continue outside a loop in the .mod file") from None
                except _Escape as esc:
                    rest = body[i + 1:]
                    info = self.analyze(rest)
                    self._refuse_if_calibrating(info, "the rest of the file after a 'return' whose condition is unknown", esc.u)
                    self.taint(info, "not executed")
                    self.record("(rest of the file after a conditional return)", "block", esc.u, info.count)
                    break
                except Unsupported as u:
                    raise self.error(f"MATLAB code in the .mod file cannot be interpreted at {self._where()}: {u}") from None
                except MATLAB_ERRORS as exc:
                    raise self.error(f"MATLAB error in the .mod file at {self._where()}: {exc}") from None
        self.model.matlab_skipped = [(t, k, w) for t, k, w, _ in self.skipped]
        self.warn()

    def _where(self):
        s = self.current
        return repr(" ".join(str(getattr(s, "text", "")).split())[:100]) if s is not None else "?"

    def _refuse_if_calibrating(self, info, what, u):
        bad = [self.statements[d.k]["statementName"] for d in info.dyn
               if self.statements[d.k]["statementName"] not in COMPUTE_KINDS]
        if info.calib or bad:
            why = info.calib or f"it contains the Dynare statement(s) {sorted(set(bad))}"
            raise self.error(f"MATLAB code changes the model and cannot be interpreted: {what}: {why}. Reason: {u}")

    def undecidable(self, s, u):
        info = self.analyze([s])
        self._refuse_if_calibrating(info, f"block {' '.join(s.text.split())[:100]!r}", u)
        self.taint(info, f"block {s.text[:50]!r} was not executed")
        for d in info.dyn:
            kind = self.statements[d.k]["statementName"]
            self.after_compute(kind, f"'{kind}' inside a block that was not executed")
        self.record(s.text, "block", u, info.count)
        if info.has_return or (info.has_break and type(s) not in (For, While)):
            raise _Escape(u)

    def run_block(self, stmts):
        for s in stmts:
            self.run_stmt(s)

    def run_stmt(self, s):
        I, sc = self.I, self.sc
        I.tick()
        t = type(s)
        if t is Dyn:
            self.current = s
            return self.exec_dyn(s)
        if t is Assign or t is ExprStmt or t is Command or t is Global or t is Unparsed:
            self.current = s
            n0 = self.host.nwrites
            try:
                if t is Assign and type(s.rhs) is Index and type(s.rhs.base) is Id and s.rhs.base.name in SOLVE_FUNCS \
                        and not I.has_var(sc, s.rhs.base.name):
                    if self.in_try:
                        raise _TryUnknown(Unsupported(f"{s.rhs.base.name} is not run while loading", "function"))
                    info = self.analyze([s])
                    info.calib = None
                    self.taint(info, f"output of {s.rhs.base.name}")
                    self.after_compute("steady", f"{s.rhs.base.name} was called")
                    return self.record(s.text, "function", f"{s.rhs.base.name} is not run while loading")
                return I.exec_simple(s, sc)
            except Unsupported as u:
                if self.host.nwrites != n0:
                    raise self.error(f"MATLAB statement changed the model before failing: {self._where()}: {u}") from None
                info = self.analyze([s])
                if info.calib:
                    raise self.error(f"MATLAB statement changes the model and cannot be interpreted: {self._where()}: "
                                     f"{info.calib}. Reason: {u}") from None
                if self.in_try:
                    raise _TryUnknown(u) from None      # the statement might have raised an error in MATLAB
                self.taint(info, f"{' '.join(s.text.split())[:60]!r} was not executed")
                return self.record(s.text, u.kind, u)
        try:
            if t is If:
                for cond, body in s.clauses:
                    self.current = s
                    if truth(I.ev(cond, sc)):
                        return self.run_block(body)
                return self.run_block(s.orelse)
            if t is Switch:
                self.current = s
                body = I.switch_body(s, sc)
                return self.run_block(body) if body else None
        except Unsupported as u:
            if self.current is not s:
                raise
            return self.undecidable(s, u)
        if t is For:
            self.current = s
            try:
                values = I.for_values(s, sc)
            except Unsupported as u:
                return self.undecidable(s, u)
            for v in values:
                I.tick()
                sc.vars[s.var] = v
                try:
                    self.run_block(s.body)
                except _BreakSig:
                    break
                except _ContinueSig:
                    continue
                except _Escape as esc:
                    return self.undecidable(s, esc.u)
            return None
        if t is While:
            while True:
                I.tick()
                self.current = s
                try:
                    if not truth(I.ev(s.cond, sc)):
                        break
                except Unsupported as u:
                    return self.undecidable(s, u)
                try:
                    self.run_block(s.body)
                except _BreakSig:
                    break
                except _ContinueSig:
                    continue
                except _Escape as esc:
                    return self.undecidable(s, esc.u)
            return None
        if t is Try:
            n0 = self.host.nwrites
            self.in_try += 1
            try:
                return self.run_block(s.body)
            except _TryUnknown as tu:
                self.in_try -= 1
                if self.host.nwrites != n0:
                    raise self.error(f"MATLAB try block changed the model before a statement that cannot be interpreted: {tu.u}") from None
                try:
                    return self.undecidable(s, tu.u)
                finally:
                    self.in_try += 1
            except MATLAB_ERRORS as exc:
                self.in_try -= 1
                try:
                    if s.ident:
                        sc.vars[s.ident] = Struct(message=str(exc), identifier="", stack=Struct())
                    return self.run_block(s.handler)
                finally:
                    self.in_try += 1
            finally:
                self.in_try -= 1
        if t is Break:
            raise _BreakSig
        if t is Continue:
            raise _ContinueSig
        if t is Return:
            raise _ReturnSig
        raise self.error(f"unsupported MATLAB statement in the .mod file: {s.text!r}")

    def exec_dyn(self, s):
        st = self.statements[s.k]
        kind = st["statementName"]
        m, ws, h = self.model, self.sc.vars, self.host
        s.text = f"<{kind}{' ' + st['name'] if kind == 'param_init' else ''}>"
        if kind in _EVALUATED_KINDS:
            payload = st.get("value", "") if kind == "param_init" else st
            ids = set(re.findall(r"[A-Za-z_]\w*", payload if isinstance(payload, str) else _json_text(payload)))
            ids -= set(m._ie) | set(m._ix)
            bad = sorted(n for n in ids if type(ws.get(n)) is Tainted and n not in m._ip)
            if bad:
                raise self.error(f"the Dynare statement {s.text} uses the MATLAB variable(s) {bad}, whose value pydyn "
                                 f"could not compute: {ws[bad[0]].why}")
            badp = sorted(n for n in ids if n in m._ip and m._ip[n] in h.tainted)
            if badp:
                raise self.error(f"the Dynare statement {s.text} uses parameter(s) {badp}, whose value at this point "
                                 f"pydyn does not know: {h.taint_why}")
        try:
            m._exec_statement(st)
        except (NameError, TypeError, ValueError, SyntaxError) as exc:
            raise self.error(f"cannot evaluate the Dynare statement {s.text}: {exc}") from None
        if kind == "param_init":
            i = m._ip[st["name"]]
            h.tainted.discard(i)
            h.nwrites += 1
            ws[st["name"]] = _F(m.params[i])         # Dynare's driver also creates this workspace variable
        elif kind in ("estimated_params", "osr_params"):
            if st.get("overwrite"):
                self.est_params.clear()
            self.est_params |= {n for n in re.findall(r"[A-Za-z_]\w*", _json_text(st)) if n in m._ip}
        elif kind == "estimated_params_remove":
            pass
        if kind not in DECL_KINDS:
            self.after_compute(kind, f"it may have been changed by '{kind}'")

    def eval_value(self, text):
        """Value of an expression in a Dynare statement that plain Python evaluation could not handle
        (it indexes a MATLAB matrix or calls a MATLAB function)."""
        try:
            body, funcs = parse(str(text))
            if funcs or len(body) != 1 or type(body[0]) is not ExprStmt:
                raise ParseError("not an expression")
            with np.errstate(**_ERRSTATE):
                return float(scalar(self.I.ev(body[0].e, self.sc), "parameter value"))
        except (ParseError, Unsupported) as exc:
            raise self.error(f"cannot evaluate {str(text)[:80]!r}: {exc}") from None
        except MATLAB_ERRORS as exc:
            raise self.error(f"MATLAB error while evaluating {str(text)[:80]!r}: {exc}") from None


def _json_text(obj):
    out = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return " ".join(out)


def _root_name(t):
    while type(t) is not Id:
        t = t.base
    return t.name


# ====================================================================== function files
_NARGOUT = [None]


class nargout:
    """Context manager: number of outputs requested from a ``MatlabFunction`` called inside."""

    def __init__(self, n):
        self.n = n

    def __enter__(self):
        _NARGOUT.append(self.n)

    def __exit__(self, *exc):
        _NARGOUT.pop()


def _to_python(v):
    if isinstance(v, (np.float64, float, int)):
        return float(v)
    if isinstance(v, (np.bool_, bool)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.astype(float)
    raise MatlabError(f"a {_class(v)} value was returned where a number is expected")


def _to_matlab(v):
    if isinstance(v, (float, int, np.floating, np.integer)):
        return _F(v)
    if isinstance(v, (bool, np.bool_)):
        return np.bool_(v)
    if isinstance(v, str):
        return v
    a = np.asarray(v, float)
    return ret(a.reshape(-1, 1) if a.ndim == 1 else a)


class MatlabFunction:
    """A MATLAB function file as a Python callable (numbers in, numbers out).

    The file is translated to Python when it only uses scalar code; otherwise it is interpreted."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.stem
        self.mode = None            # "compiled" or "interpreted", set at the first call
        self.why_interpreted = None
        self._fn = self._unit = self._interp = None

    def _prepare(self):
        try:
            body, funcs = parse(self.path.read_text(errors="replace"))
        except ParseError as exc:
            raise Unsupported(f"{self.path.name} cannot be parsed: {exc}", "syntax") from None
        self._unit = Unit(self.path, body, funcs)
        if self._unit.is_script:
            raise Unsupported(f"{self.path.name} is a script, not a function", "function")
        self._interp = Interpreter(self.path.parent)
        try:
            src = _Translator(self._unit, self.path.name).function_module()
            self._fn = _compile(src, self.path)[_pyname(self._unit.main.name)]
            self.mode = "compiled"
        except TranslationError as exc:
            self.mode, self.why_interpreted = "interpreted", str(exc)

    def __call__(self, *args):
        if self.mode is None:
            self._prepare()
        n = _NARGOUT[-1] or 1
        with np.errstate(**_ERRSTATE):
            if self.mode == "compiled":
                try:
                    out = self._fn(*[_F(a) for a in args])
                    out = out if isinstance(out, tuple) else (out,)
                    if len(out) < n:
                        raise MatlabError(f"{self.name}: output argument {len(out) + 1} is not assigned")
                    out = [_to_python(v) for v in out[:n]]
                    return out[0] if _NARGOUT[-1] is None else tuple(out)
                except _FALLBACK as exc:
                    self.mode, self.why_interpreted = "interpreted", f"{type(exc).__name__}: {exc}"
            I = self._interp
            I.steps = 0
            out = I.call_user(self._unit.main, self._unit.funcs, [_to_matlab(a) for a in args], n)
            out = [_to_python(v) for v in out]
            return out[0] if _NARGOUT[-1] is None else tuple(out)


#: exceptions of translated code that mean "the translation's assumptions do not hold": the
#: interpreter takes over (it either gives the MATLAB result or the proper MATLAB error)
_FALLBACK = (NameError, KeyError, IndexError, TypeError, AttributeError, ValueError)


class SteadyStateFile:
    """``<model>_steadystate.m`` as a pydyn ``steady_state_fn``: ``f(model, params, exo) -> (ys, params)``."""

    def __init__(self, path):
        self.path = Path(path)
        self.mode = None
        self.why_interpreted = None
        self.source = None          # Python source used when mode == "compiled"
        self._fn = self._unit = None
        self._static = {}

    def _prepare(self):
        try:
            body, funcs = parse(self.path.read_text(errors="replace"))
        except ParseError as exc:
            raise Unsupported(f"{self.path.name} cannot be parsed: {exc}", "syntax") from None
        self._unit = Unit(self.path, body, funcs)
        main = self._unit.main
        if main is None or len(main.outs) not in (2, 3) or len(main.ins) < 2:
            raise Unsupported(f"{self.path.name} is not a steady-state function "
                              "([ys, params, check] = f(ys, exo, M_, options_) or [ys, check] = f(ys, exo))", "function")
        try:
            self.source = _Translator(self._unit, self.path.name).steady_state_module()
            self._fn = _compile(self.source, self.path)["steady_state"]
            self.mode = "compiled"
        except TranslationError as exc:
            self.mode, self.why_interpreted = "interpreted", str(exc)

    def __call__(self, model, params, exo):
        from .steady import SteadyStateError
        if self.mode is None:
            self._prepare()
        with np.errstate(**_ERRSTATE):
            if self.mode == "compiled":
                try:
                    return self._fn(model, params, exo)
                except _FALLBACK as exc:
                    self.mode, self.why_interpreted = "interpreted", f"{type(exc).__name__}: {exc}"
                except SteadyStateError:
                    raise
                except MATLAB_ERRORS as exc:
                    raise SteadyStateError(f"{self.path.name}: MATLAB error: {exc}") from None
            try:
                return self._interpret(model, params, exo)
            except MATLAB_ERRORS as exc:
                raise SteadyStateError(f"{self.path.name}: MATLAB error: {exc}") from None

    def _interpret(self, model, params, exo):
        from .steady import SteadyStateError
        main = self._unit.main
        host = Host(model, params=params)
        I = Interpreter(self.path.parent, host)
        M_ = Struct(params=host.M.get_field("params"), Sigma_e=host.M.get_field("Sigma_e"))
        for f in list(_HostM._SCALARS) + list(_HostM._NAMES) + ["fname", "dname"]:
            M_[f] = host.M.get_field(f)
        ys0 = np.array([model.initval.get(v, 0.0) for v in model.endo_names], float).reshape(-1, 1)
        args = [ret(ys0), ret(np.asarray(exo, float).reshape(-1, 1)), M_, host.options][:len(main.ins)]
        out = I.call_user(main, self._unit.funcs, args, len(main.outs))
        if len(main.outs) == 3:
            ys, new, check = out
        else:
            ys, check = out
            new = host.M.get_field("params")
        if truth(check):
            raise SteadyStateError(f"{self.path.name} signalled failure (check = {float(scalar(check)):g})")
        new = farr(new).ravel(order="F")
        if new.size != len(model.param_names):
            raise SteadyStateError(f"{self.path.name} returned {new.size} parameters, the model has {len(model.param_names)}")
        return farr(ys).ravel(order="F"), new


# ====================================================================== translation to Python
class _RT:
    """Run-time helpers of translated code (``rt.<name>``)."""
    MatlabError = MatlabError
    MATLAB_ERRORS = MATLAB_ERRORS
    exp, log, sqrt, abs = np.exp, np.log, np.sqrt, np.abs

    def __getattr__(self, name):
        f = BUILTINS.get(name)
        if f is None:
            raise AttributeError(name)

        def call(*args):
            r = f(None, list(args), 1)
            return r[0] if type(r) is list else r
        setattr(self, name, call)
        return call

    @staticmethod
    def multi(name, nout, *args):
        """Built-in called with ``nout`` outputs."""
        r = BUILTINS[name](None, list(args), nout)
        r = r if type(r) is list else [r]
        if len(r) < nout:
            raise MatlabError(f"{name}: too many outputs requested")
        return r[:nout]

    @staticmethod
    def true(x):
        return truth(x)

    @staticmethod
    def irange(a, b, step=1.0):
        r = colon(a, step, b)
        return [_F(v) for v in r.ravel()]

    @staticmethod
    def ix(k, n):
        """0-based position for the MATLAB index k into a vector of length n."""
        if k != int(k) or k < 1:
            raise MatlabError("subscript indices must be positive integers")
        if k > n:
            raise IndexError("index beyond the end of the vector")
        return int(k) - 1

    @staticmethod
    def initial_ys(model):
        return np.array([model.initval.get(v, 0.0) for v in model.endo_names], float)

    @staticmethod
    def orig_endo_nbr(model):
        return model.raw.get("orig_endo_nbr", model.n_endo)

    @staticmethod
    def collect_params(model, p, loc, names):
        """``params(i) = <variable named like parameter i>`` for every parameter."""
        out = np.empty(len(model.param_names))
        for i, n in enumerate(model.param_names):
            py = names.get(n)
            out[i] = loc[py] if py is not None and py in loc else p[n]
        return out

    @staticmethod
    def collect_ys(model, ys, loc, names, n):
        ys = np.array(ys, float)
        for i, v in enumerate(model.endo_names[:int(n)]):
            py = names.get(v)
            if py is None or py not in loc:
                raise MatlabError(f"the steady-state file does not define '{v}'")
            ys[i] = loc[py]
        return ys

    @staticmethod
    def outs(loc, *names):
        out = []
        for n in names:
            if n not in loc:
                break
            out.append(loc[n])
        return tuple(out)

    @staticmethod
    def finish(ys, params, check, name="steady-state file"):
        from .steady import SteadyStateError
        if truth(check):
            raise SteadyStateError(f"{name} signalled failure (check = {float(check):g})")
        return np.asarray(ys, float), np.asarray(params, float)


rt = _RT()
_PY_BUILTINS = {"float": float, "len": len, "locals": locals, "bool": bool, "int": int, "dict": dict, "zip": zip}
_RESERVED = set(keyword.kwlist) | set(_PY_BUILTINS) | {"np", "rt", "model", "params_in", "exo_in", "steady_state",
                                                       "True", "False", "None"}


def _pyname(name):
    if not re.fullmatch(r"[A-Za-z]\w*", name):
        raise TranslationError(f"bad identifier {name!r}")
    # injective: lambda -> lambda_, lambda_ -> lambda__ (names used by the generated code start with "_")
    return name + "_" if name.rstrip("_") in _RESERVED else name


def _compile(src, path):
    body = "\n".join(l for l in src.split("\n") if not l.startswith(("import ", "from ")))
    ns = {"__builtins__": {}, "np": np, "rt": rt, **_PY_BUILTINS}
    exec(compile(body, f"<pydyn.matlab translation of {Path(path).name}>", "exec"), ns)  # noqa: S102 - generated from a validated syntax tree
    return ns


_NUMFUNCS = {"exp", "log", "sqrt", "abs", "log2", "log10", "sign", "floor", "ceil", "fix", "round", "sin", "cos", "tan",
             "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "gamma", "gammaln", "erf", "erfc",
             "log1p", "expm1", "factorial", "normcdf", "normpdf", "norminv", "max", "min", "mod", "rem", "atan2",
             "nthroot", "hypot", "power", "times", "plus", "minus", "rdivide", "real", "double"}
_BOOLFUNCS = {"isnan", "isinf", "isfinite"}
_CONSTS = {"pi": "np.pi", "Inf": "np.inf", "inf": "np.inf", "NaN": "np.nan", "nan": "np.nan", "eps": "np.finfo(float).eps",
           "true": "True", "false": "False"}
_SILENT = {"disp", "display", "fprintf", "printf", "warning", "skipline", "dprintf", "tic", "fflush"}
_ATOM, _POW, _UNA, _MUL, _ADD, _CMP, _NOT, _AND, _OR, _LAM = 100, 90, 80, 70, 60, 50, 45, 40, 30, 10


class _Translator:
    """Emit Python for a function file whose code is scalar (see ``translate_steady_state``)."""

    def __init__(self, unit, filename):
        self.unit, self.filename = unit, filename
        self.funcs = unit.funcs
        if unit.is_script or unit.body:
            raise TranslationError("not a pure function file")
        for f in self.funcs:
            _pyname(f)

    # ---- helpers
    def fail(self, msg, node=None):
        raise TranslationError(msg)

    @staticmethod
    def par(x, level):
        return f"({x[0]})" if x[2] < level else x[0]

    def num(self, x):
        """Code of a numeric (float) expression."""
        if x[1] == "bool":
            return (f"float({x[0]})", "num", _ATOM)
        if x[1] != "num":
            self.fail(f"a {x[1]} value is used as a number")
        return x

    def cond(self, x):
        if x[1] == "bool":
            return x
        if x[1] == "num":
            return (f"rt.true({x[0]})", "bool", _ATOM)
        self.fail(f"a {x[1]} value is used as a condition")

    # ---- expressions: returns (code, type, precedence[, is_literal])
    def ex(self, n, env):
        t = type(n)
        if t is Num:
            v = float(n.v)
            return (repr(v), "num", _ATOM if v >= 0 else _UNA, True)
        if t is Str:
            if n.dq:
                self.fail("string object")
            return (repr(n.v), "str", _ATOM)
        if t is Id:
            ty = env.get(n.name)
            if ty is not None:
                return (_pyname(n.name), ty, _ATOM)
            if n.name in _CONSTS and n.name not in self.maybe_params:
                return (_CONSTS[n.name], "bool" if n.name in ("true", "false") else "num", _ATOM)
            if n.name in self.maybe_params:
                return (_pyname(n.name), "num", _ATOM)
            self.fail(f"'{n.name}' is not a known variable")
        if t is Unop:
            if n.op == "()":
                x = self.ex(n.e, env)
                return (f"({x[0]})", x[1], _ATOM) + tuple(x[3:])
            if n.op == "'":
                x = self.ex(n.e, env)
                if x[1] not in ("num", "bool"):
                    self.fail("transpose of a non-scalar")
                return x
            if n.op == "~":
                x = self.cond(self.ex(n.e, env))
                return (f"not {self.par(x, _NOT)}", "bool", _NOT)
            x = self.num(self.ex(n.e, env))
            if n.op == "+":
                return x
            return (f"-{self.par(x, _UNA)}", "num", _UNA) + tuple(x[3:])
        if t is Binop:
            op = n.op
            if op in ("&&", "||", "&", "|"):
                a, b = self.cond(self.ex(n.l, env)), self.cond(self.ex(n.r, env))
                if op in ("&&", "&"):
                    return (f"{self.par(a, _AND)} and {self.par(b, _AND + 1)}", "bool", _AND)
                return (f"{self.par(a, _OR)} or {self.par(b, _OR + 1)}", "bool", _OR)
            a, b = self.ex(n.l, env), self.ex(n.r, env)
            if op in ("==", "~=") and a[1] == "str" and b[1] == "str":
                self.fail("comparison of char arrays")
            a, b = self.num(a), self.num(b)
            lit = len(a) > 3 and len(b) > 3
            if op in ("<", ">", "<=", ">=", "==", "~="):
                return (f"{self.par(a, _CMP + 1)} {'!=' if op == '~=' else op} {self.par(b, _CMP + 1)}", "bool", _CMP)
            if op in ("+", "-"):
                return (f"{self.par(a, _ADD)} {op} {self.par(b, _ADD + 1)}", "num", _ADD) + ((True,) if lit else ())
            if op in ("*", ".*"):
                return (f"{self.par(a, _MUL)}*{self.par(b, _MUL + 1)}", "num", _MUL) + ((True,) if lit else ())
            if op in ("/", "./", "\\", ".\\"):
                if op in ("\\", ".\\"):
                    a, b = b, a
                left = f"np.float64({a[0]})" if lit else self.par(a, _MUL)
                return (f"{left}/{self.par(b, _MUL + 1)}", "num", _MUL)
            if op in ("^", ".^"):
                left = f"np.float64({a[0]})" if lit else self.par(a, _ATOM)
                return (f"{left}**{self.par(b, _UNA)}", "num", _POW)
            self.fail(f"operator {op}")
        if t is AnonFunc:
            return self.lam(n, env)
        if t is Field:
            return self.field(n, env)
        if t is Index:
            return self.index(n, env)
        if t is Matrix and len(n.rows) == 1 and len(n.rows[0]) == 1:
            return self.ex(n.rows[0][0], env)
        self.fail(f"{t.__name__} expression")

    def lam(self, n, env):
        inner = dict(env)
        for p in n.params:
            inner[p] = "num"
        body = self.num(self.ex(n.body, inner))
        return (f"lambda {', '.join(_pyname(p) for p in n.params)}: {body[0]}", "fn", _LAM)

    def field(self, n, env):
        if type(n.base) is Id and n.base.name == self.M and n.name is not None and n.base.name not in env:
            if n.name == "params":
                return ("_m_params", "vec", _ATOM)
            code = {"param_nbr": "float(len(_m_params))", "orig_endo_nbr": "float(rt.orig_endo_nbr(model))",
                    "endo_nbr": "float(model.n_endo)", "exo_nbr": "float(model.n_exo)"}.get(n.name)
            if code:
                return (code, "num", _ATOM)
        self.fail("field access other than M_.params and the M_ counters")

    def args_num(self, args, env):
        return ", ".join(self.num(self.ex(a, env))[0] for a in args)

    def solver_args(self, name, args, env):
        if len(args) < 2 or len(args) > 3:
            self.fail(f"{name} with {len(args)} arguments")
        f = self.ex(args[0], env) if type(args[0]) is not FuncHandle else self.handle(args[0])
        if f[1] != "fn":
            self.fail(f"first argument of {name} must be a function handle")
        out = [f"({f[0]})" if f[2] == _LAM else f[0], self.num(self.ex(args[1], env))[0]]
        if len(args) == 3:
            o = self.ex(args[2], env)
            if o[1] != "opts":
                self.fail("solver options must come from optimset")
            out.append(o[0])
        return ", ".join(out)

    def handle(self, n):
        if n.name in self.funcs:
            return (_pyname(n.name) if len(self.funcs[n.name].outs) == 1 else f"(lambda *a: {_pyname(n.name)}(*a)[0])", "fn", _ATOM)
        self.fail(f"handle to {n.name}")

    def optimset(self, args, env):
        parts = []
        for a in args:
            x = self.ex(a, env)
            if x[1] not in ("str", "num", "bool", "opts"):
                self.fail("optimset argument")
            parts.append(x[0])
        return (f"rt.optimset({', '.join(parts)})", "opts", _ATOM)

    def vec_len(self, n, env):
        """Code for the length of a vector-valued expression used in zeros(size(v)) etc."""
        x = self.ex(n, env)
        if x[1] != "vec":
            self.fail("size of a non-vector")
        return f"len({x[0]})"

    def index(self, n, env):
        if n.brace:
            self.fail("cell indexing")
        b = n.base
        if type(b) is Field:
            x = self.field(b, env)
            if x[1] == "vec" and len(n.args) == 1:
                return (f"{x[0]}[{self.pos(n.args[0], env, x[0])}]", "num", _ATOM)
            self.fail("indexing")
        if type(b) is not Id:
            self.fail("indexing of an expression")
        name, args = b.name, n.args
        ty = env.get(name)
        if ty == "vec":
            if len(args) == 2 and type(args[1]) is Num and args[1].v == 1:
                args = args[:1]
            if len(args) != 1 or type(args[0]) is ColonAll:
                self.fail("vector indexing other than v(k)")
            return (f"{_pyname(name)}[{self.pos(args[0], env, _pyname(name))}]", "num", _ATOM)
        if ty == "fn":
            return (f"{_pyname(name)}({self.args_num(args, env)})", "num", _ATOM)
        if ty is not None or name in self.maybe_params:
            if len(args) == 1 and type(args[0]) is Num and args[0].v == 1 and ty in (None, "num"):
                return (_pyname(name), "num", _ATOM)
            self.fail(f"indexing of the scalar '{name}'")
        if name in self.funcs:
            f = self.funcs[name]
            if len(args) > len(f.ins) or not f.outs:
                self.fail(f"call of {name}")
            code = f"{_pyname(name)}({self.args_num(args, env)})"
            return (code if len(f.outs) == 1 else code + "[0]", "num", _ATOM)
        if name in ("fzero", "fsolve"):
            return (f"rt.multi({name!r}, 1, {self.solver_args(name, args, env)})[0]", "num", _ATOM)
        if name in ("optimset", "optimoptions"):
            return self.optimset(args, env)
        if name in _NUMFUNCS or name in _BOOLFUNCS:
            if name in ("max", "min") and len(args) != 2:
                self.fail(f"{name} with {len(args)} arguments")
            return (f"rt.{name}({self.args_num(args, env)})", "bool" if name in _BOOLFUNCS else "num", _ATOM)
        if name == "isreal" and len(args) == 1:
            self.num(self.ex(args[0], env))
            return ("True", "bool", _ATOM)
        if name in ("length", "numel") and len(args) == 1:
            return (f"float({self.vec_len(args[0], env)})", "num", _ATOM)
        if name == "size" and len(args) == 2 and type(args[1]) is Num and args[1].v == 1:
            return (f"float({self.vec_len(args[0], env)})", "num", _ATOM)
        if name in ("zeros", "ones", "NaN", "nan", "Inf", "inf") and not args:
            return ({"zeros": "0.0", "ones": "1.0"}.get(name, _CONSTS.get(name)), "num", _ATOM)
        self.fail(f"call of '{name}'")

    def pos(self, a, env, vec):
        """0-based index code for the MATLAB index expression ``a`` into vector ``vec``."""
        if type(a) is Num:
            k = float(a.v)
            if k != int(k) or k < 1:
                self.fail("bad constant index")
            return str(int(k) - 1)
        if type(a) in (ColonAll, End, Range):
            self.fail("index with ':' or 'end'")
        return f"rt.ix({self.num(self.ex(a, env))[0]}, len({vec}))"

    def vec_expr(self, n, env):
        """Code of an expression that creates a column vector, or None."""
        if type(n) is Field:
            x = self.field(n, env)
            return f"{x[0]}.copy()" if x[1] == "vec" else None
        if type(n) is Id and env.get(n.name) == "vec":
            return f"{_pyname(n.name)}.copy()"
        if type(n) is Index and type(n.base) is Id and n.base.name in ("zeros", "ones", "NaN", "nan") and n.base.name not in env:
            fill = {"zeros": "0.0", "ones": "1.0"}.get(n.base.name, "np.nan")
            a = n.args
            if len(a) == 2 and type(a[1]) is Num and a[1].v == 1:
                return f"np.full(int({self.num(self.ex(a[0], env))[0]}), {fill})"
            if len(a) == 1 and type(a[0]) is Index and type(a[0].base) is Id and a[0].base.name == "size" and len(a[0].args) == 1:
                return f"np.full({self.vec_len(a[0].args[0], env)}, {fill})"
            if len(a) >= 1:
                self.fail("matrix creation")
        return None

    # ---- statements
    def setvar(self, name, ty, env):
        old = env.get(name)
        if old is not None and old != ty and not ({old, ty} <= {"num", "bool"}):
            self.fail(f"'{name}' changes type ({old} -> {ty})")
        env[name] = "num" if {old, ty} == {"num", "bool"} else ty
        self.assigned.add(name)

    def block(self, stmts, env, ind):
        out = []
        for s in stmts:
            out += self.stmt(s, env, ind)
        return out or [ind + "pass"]

    def stmt(self, s, env, ind):
        t = type(s)
        if t is Assign:
            return self.assign(s, env, ind)
        if t is ExprStmt:
            e = s.e
            head = e.base.name if type(e) is Index and type(e.base) is Id else (e.name if type(e) is Id else None)
            if head in _SILENT and head not in env:
                return [f"{ind}# {' '.join(s.text.split())[:90]}"]
            if head == "error" and type(e) is Index:
                parts = [self.ex(a, env) for a in e.args]
                if any(p[1] not in ("str", "num", "bool") for p in parts):
                    self.fail("error() arguments")
                return [f"{ind}rt.error({', '.join(p[0] for p in parts)})"]
            self.fail(f"statement {' '.join(s.text.split())[:50]!r}")
        if t is If:
            out = []
            for k, (c, body) in enumerate(s.clauses):
                out.append(f"{ind}{'if' if k == 0 else 'elif'} {self.cond(self.ex(c, env))[0]}:")
                out += self.block(body, env, ind + "    ")
            if s.orelse:
                out.append(f"{ind}else:")
                out += self.block(s.orelse, env, ind + "    ")
            return out
        if t is While:
            self.loop_depth += 1
            out = [f"{ind}while {self.cond(self.ex(s.cond, env))[0]}:"] + self.block(s.body, env, ind + "    ")
            self.loop_depth -= 1
            return out
        if t is For:
            idiom = self.idiom(s, env, ind) if self.ss else None
            if idiom is not None:
                return idiom
            e = s.e
            if type(e) is Unop and e.op == "()":
                e = e.e
            if type(e) is not Range or s.var is None:
                self.fail("for loop over something else than a range")
            a, c = self.num(self.ex(e.a, env))[0], self.num(self.ex(e.c, env))[0]
            rng = f"rt.irange({a}, {c})" if e.b is None else f"rt.irange({a}, {c}, {self.num(self.ex(e.b, env))[0]})"
            self.setvar(s.var, "num", env)
            self.loop_depth += 1
            out = [f"{ind}for {_pyname(s.var)} in {rng}:"] + self.block(s.body, env, ind + "    ")
            self.loop_depth -= 1
            return out
        if t is Try:
            if s.ident:
                self.fail("catch with an identifier")
            out = [f"{ind}try:"] + self.block(s.body, env, ind + "    ")
            out += [f"{ind}except rt.MATLAB_ERRORS:"] + self.block(s.handler, env, ind + "    ")
            return out
        if t is Break:
            return [ind + "break"]
        if t is Continue:
            return [ind + "continue"]
        if t is Return:
            return [ind + self.ret_code]
        if t is Command and s.name in _SILENT:
            return []
        self.fail(f"statement {' '.join(str(getattr(s, 'text', t.__name__)).split())[:50]!r}")

    def assign(self, s, env, ind):
        tg = s.targets
        rhs = s.rhs
        if len(tg) > 1:
            if type(rhs) is not Index or type(rhs.base) is not Id or rhs.base.name in env:
                self.fail("multiple assignment from something else than a function call")
            name = rhs.base.name
            names = []
            for x in tg:
                if x is None:
                    names.append("_")
                elif type(x) is Id:
                    names.append(_pyname(x.name))
                else:
                    self.fail("indexed target in a multiple assignment")
            if name in ("fzero", "fsolve"):
                code = f"rt.multi({name!r}, {len(tg)}, {self.solver_args(name, rhs.args, env)})"
                types = ["num", "num", "num", "opts"]
            elif name == "csolve":
                self.fail("csolve")
            elif name in self.funcs:
                f = self.funcs[name]
                if len(tg) > len(f.outs) or len(rhs.args) > len(f.ins):
                    self.fail(f"call of {name}")
                code = f"{_pyname(name)}({self.args_num(rhs.args, env)})[:{len(tg)}]"
                types = ["num"] * len(tg)
            else:
                self.fail(f"multiple outputs of '{name}'")
            for x, ty in zip(tg, types):
                if x is not None:
                    self.setvar(x.name, ty, env)
            return [f"{ind}{', '.join(names)} = {code}"]
        x = tg[0]
        if type(x) is Id:
            v = self.vec_expr(rhs, env)
            if v is not None:
                self.setvar(x.name, "vec", env)
                return [f"{ind}{_pyname(x.name)} = {v}"]
            if type(rhs) is AnonFunc:
                if self.loop_depth:
                    self.fail("function handle created inside a loop")
                self.stored_handles.append((x.name, {i.name for i in walk(rhs.body) if type(i) is Id} - set(rhs.params)))
            r = self.ex(rhs, env)
            if r[1] == "vec":
                self.fail("vector expression")
            for nm, captured in self.stored_handles:
                if x.name in captured and nm != x.name:
                    self.fail(f"'{x.name}' is reassigned after being captured by the function handle '{nm}'")
            if r[1] == "bool":
                r = (f"np.float64({r[0]})", "num", _ATOM)
            self.setvar(x.name, r[1], env)
            return [f"{ind}{_pyname(x.name)} = {r[0]}"]
        if type(x) is Index and not x.brace and type(x.base) is Id and env.get(x.base.name) == "vec":
            args = x.args
            if len(args) == 2 and type(args[1]) is Num and args[1].v == 1:
                args = args[:1]
            if len(args) != 1:
                self.fail("vector assignment other than v(k) = x")
            v = _pyname(x.base.name)
            return [f"{ind}{v}[{self.pos(args[0], env, v)}] = {self.num(self.ex(rhs, env))[0]}"]
        self.fail(f"assignment {' '.join(s.text.split())[:50]!r}")

    # ---- functions
    def function(self, f, ss=False):
        self.ss, self.loop_depth, self.assigned, self.stored_handles = ss, 0, set(), []
        self.maybe_params = set()
        self.M = None
        env = {}
        if any(i in ("varargin", "~") for i in f.ins) or "varargout" in f.outs:
            self.fail("varargin/varargout")
        for i in f.ins:
            env[i] = "num"
        if len(f.outs) == 1:
            self.ret_code = f"return {_pyname(f.outs[0])}"
        else:
            self.ret_code = "return rt.outs(locals(), " + ", ".join(repr(_pyname(o)) for o in f.outs) + ")"
        if any(n.name in ("nargin", "nargout") for n in walk(f.body) if type(n) is Id):
            self.fail("nargin/nargout")
        lines = [f"def {_pyname(f.name)}({', '.join(_pyname(i) for i in f.ins)}):"]
        lines += self.block(f.body, env, "    ")
        lines.append("    " + self.ret_code)
        return lines

    def header(self, what):
        return [f'"""Python translation of {self.filename}, generated by pydyn.matlab ({what}).',
                "", "Scalars are numpy float64 values, so that arithmetic follows IEEE rules as in MATLAB (x/0 is Inf).",
                'Run inside ``with np.errstate(invalid="raise", divide="ignore", over="ignore")`` to turn results',
                'that MATLAB would make complex or NaN (sqrt(-1), 0/0) into errors, as pydyn does."""',
                "import numpy as np", "from pydyn.matlab import rt", ""]

    def function_module(self):
        lines = self.header("function file")
        for f in self.funcs.values():
            lines += self.function(f) + ["", ""]
        return "\n".join(lines).rstrip() + "\n"

    # ---- steady-state files
    def bound(self, e, env):
        """Symbolic value of a loop bound: 'param', 'orig' or 'endo'."""
        if type(e) is Id:
            return self.symbols.get(e.name)
        if type(e) is Field and type(e.base) is Id and e.base.name == self.M:
            return {"param_nbr": "param", "orig_endo_nbr": "orig", "endo_nbr": "endo"}.get(e.name)
        if type(e) is Index and type(e.base) is Id and e.base.name in ("length", "numel") and len(e.args) == 1:
            a = e.args[0]
            if type(a) is Field and type(a.base) is Id and a.base.name == self.M:
                return {"params": "param", "param_names": "param", "endo_names": "endo"}.get(a.name)
        if type(e) is Index and type(e.base) is Id and e.base.name == "size" and len(e.args) == 2 \
                and type(e.args[1]) is Num and e.args[1].v == 1:
            a = e.args[0]
            if type(a) is Field and type(a.base) is Id and a.base.name == self.M:
                return {"params": "param", "param_names": "param", "endo_names": "endo"}.get(a.name)
        return None

    def idiom(self, s, env, ind):
        """The eval loops of the steady-state template; returns Python lines or None."""
        e = s.e
        if type(e) is Unop and e.op == "()":
            e = e.e
        if type(e) is not Range or e.b is not None or type(e.a) is not Num or e.a.v != 1 or not s.body:
            return None
        ev = s.body[-1]
        if not (type(ev) is ExprStmt and type(ev.e) is Index and type(ev.e.base) is Id and ev.e.base.name == "eval"
                and len(ev.e.args) == 1 and type(ev.e.args[0]) is Matrix and len(ev.e.args[0].rows) == 1):
            return None
        which = {}

        def names_of(x):
            if type(x) is Index and x.brace and len(x.args) == 1 and type(x.args[0]) is Id and x.args[0].name == s.var \
                    and type(x.base) is Field and type(x.base.base) is Id and x.base.base.name == self.M:
                return x.base.name
            return None
        for pre in s.body[:-1]:
            if type(pre) is Assign and len(pre.targets) == 1 and type(pre.targets[0]) is Id and names_of(pre.rhs):
                which[pre.targets[0].name] = names_of(pre.rhs)
            else:
                self.fail("unrecognised eval loop")
        pieces, lists = [], set()
        for p in ev.e.args[0].rows[0]:
            if type(p) is Str:
                pieces.append(p.v)
            elif type(p) is Id and p.name in which:
                pieces.append("\0N")
                lists.add(which[p.name])
            elif names_of(p):
                pieces.append("\0N")
                lists.add(names_of(p))
            elif type(p) is Index and type(p.base) is Id and p.base.name in ("int2str", "num2str") and len(p.args) == 1 \
                    and type(p.args[0]) is Id and p.args[0].name == s.var:
                pieces.append("\0I")
            else:
                self.fail("unrecognised eval loop")
        text = "".join(pieces).strip()
        bound = self.bound(e.c, env)
        main = self.unit.main
        M = re.escape(self.M)
        for tmp in list(which) + [s.var]:
            self.setvar(tmp, "str" if tmp in which else "num", env)
            self.idiom_temps.add(tmp)
        if re.fullmatch(rf"\0N\s*=\s*{M}\.params\(\s*\0I\s*\)\s*;?", text) and lists == {"param_names"} and bound == "param":
            if ind != "    " or self.read_done:
                self.fail("parameter loop is not at the top level of the function")
            self.read_done = True
            return ["\0READ"]
        out_p, out_y = main.outs[1], main.outs[0]
        if re.fullmatch(rf"{re.escape(out_p)}\(\s*\0I\s*\)\s*=\s*\0N\s*;?", text) and lists == {"param_names"} and bound == "param":
            if not self.read_done:
                self.fail("parameters written back before being read")
            self.setvar(out_p, "vec", env)
            return [f"{ind}{_pyname(out_p)} = rt.collect_params(model, _m_p, locals(), _m_vars)"]
        if re.fullmatch(rf"{re.escape(out_y)}\(\s*\0I\s*\)\s*=\s*\0N\s*;?", text) and lists == {"endo_names"} and bound in ("orig", "endo"):
            n = "rt.orig_endo_nbr(model)" if bound == "orig" else "model.n_endo"
            return [f"{ind}{_pyname(out_y)} = rt.collect_ys(model, {_pyname(out_y)}, locals(), _m_vars, {n})"]
        self.fail("unrecognised eval loop")

    def steady_state_module(self):
        main = self.unit.main
        if len(main.outs) != 3 or len(main.ins) < 2 or len(main.ins) > 4:
            raise TranslationError("only [ys, params, check] = f(ys, exo, M_, options_) files are translated")
        lines = self.header("steady-state file; use as Model(..., steady_state_fn=steady_state)")
        for f in list(self.funcs.values())[1:]:
            lines += self.function(f) + ["", ""]
        self.ss, self.loop_depth, self.assigned, self.stored_handles = True, 0, set(), []
        self.read_done, self.idiom_temps, self.symbols = False, set(), {}
        self.M = main.ins[2] if len(main.ins) > 2 else "M_"
        opt = main.ins[3] if len(main.ins) > 3 else None
        nodes = list(walk(main.body))
        if opt and any(type(n) is Id and n.name == opt for n in nodes):
            raise TranslationError("the file reads options_")
        if any(type(n) is Global for n in nodes) or any(type(n) is Id and n.name in ("nargin", "nargout") for n in nodes):
            raise TranslationError("global variables or nargin")
        # names that may be parameters: read somewhere, and either never assigned or possibly read first
        assigned = {_root_name(x) for n in nodes if type(n) is Assign for x in n.targets if x is not None}
        assigned |= {n.var for n in nodes if type(n) is For and n.var}
        called = {n.base.name for n in nodes if type(n) is Index and type(n.base) is Id}
        read = {n.name for n in nodes if type(n) is Id}
        known_fn = set(self.funcs) | set(BUILTINS) | set(LAZY)
        special = {self.M, opt, main.ins[0], main.ins[1]}
        has_read_loop = any(type(n) is For and n.body and type(n.body[-1]) is ExprStmt and "eval" in
                            {x.base.name for x in walk(n.body[-1]) if type(x) is Index and type(x.base) is Id} for n in main.body)
        self.maybe_params = set()
        if has_read_loop:
            self.maybe_params = {n for n in read if n not in assigned and n not in special
                                 and not (n in called and n in known_fn) and (n not in known_fn or n in _CONSTS)}
            for n in self.maybe_params & called:
                if n not in _CONSTS:
                    raise TranslationError(f"'{n}' is used like a function but is not one")
        for n in self.maybe_params | assigned:
            _pyname(n)
        # symbols for loop bounds
        for n in main.body:
            if type(n) is Assign and len(n.targets) == 1 and type(n.targets[0]) is Id:
                b = self.bound(n.rhs, {})
                if b and sum(1 for m in nodes if type(m) is Assign and any(x is not None and _root_name(x) == n.targets[0].name for x in m.targets)) == 1:
                    self.symbols[n.targets[0].name] = b
        env = {main.ins[0]: "vec", main.ins[1]: "vec"}
        self.ret_code = f"return rt.finish({_pyname(main.outs[0])}, {_pyname(main.outs[1])}, {_pyname(main.outs[2])}, {self.filename!r})"
        body = []
        pre_declared, order = [], []
        for s in main.body:
            chunk = self.stmt(s, env, "    ")
            if chunk == ["\0READ"]:
                pre_declared = sorted(self.assigned - self.idiom_temps)
                body.append("\0READ")
                continue
            body += chunk
            order.append(s)
        # which variables assigned after the parameter loop may be read before they are assigned?
        cond_load = set(pre_declared)
        if self.read_done:
            definite, seen_read = set(pre_declared), False
            for s in main.body:
                if not seen_read:
                    seen_read = type(s) is For and bool(s.body) and type(s.body[-1]) is ExprStmt and any(
                        type(x) is Index and type(x.base) is Id and x.base.name == "eval" for x in walk(s.body[-1])) \
                        and self.read_done and not any(type(x) is Assign for x in s.body[-1:])
                    continue
                reads = _free_ids(s)
                if type(s) is Assign:
                    reads = _free_ids(s.rhs) | {n for x in s.targets if x is not None and type(x) is not Id for n in _free_ids(x)}
                cond_load |= {n for n in reads if n in assigned and n not in definite and n not in special}
                if type(s) is Assign:
                    definite |= {x.name for x in s.targets if type(x) is Id}
        cond_load -= self.idiom_temps | {main.outs[0], main.outs[1]}
        cond_load = {n for n in cond_load if env.get(n) in ("num", "bool")}
        load = []
        for n in sorted(self.maybe_params):
            if n in _CONSTS:
                load.append(f"    {_pyname(n)} = _m_p.get({n!r}, {_CONSTS[n]})")
            else:
                load.append(f"    {_pyname(n)} = _m_p[{n!r}]")
        for n in sorted(cond_load):
            load.append(f"    if {n!r} in _m_p:")
            load.append(f"        {_pyname(n)} = _m_p[{n!r}]")
        names = sorted((self.maybe_params | {n for n in self.assigned if env.get(n) in ("num", "bool")}) - self.idiom_temps)
        vars_line = "_m_vars = {" + ", ".join(f"{n!r}: {_pyname(n)!r}" for n in names) + "}"
        lines += ["#: MATLAB variable -> Python variable, for the loops that write parameters and variables back",
                  "\n".join(_wrap(vars_line)), "", "",
                  "def steady_state(model, params_in, exo_in):",
                  "    _m_params = np.array(params_in, float)             # M_.params",
                  "    _m_p = dict(zip(model.param_names, _m_params))     # parameters by name",
                  f"    {_pyname(main.ins[0])} = rt.initial_ys(model)                      # initial values of the endogenous variables",
                  f"    {_pyname(main.ins[1])} = np.array(exo_in, float)"]
        for l in body:
            if l == "\0READ":
                lines.append("    # parameters by name (the eval loop over M_.param_names)")
                lines += load
            else:
                lines.append(l)
        lines.append("    " + self.ret_code)
        return "\n".join(lines) + "\n"


def _free_ids(node):
    """Identifiers read in ``node``, without the parameters of anonymous functions."""
    if type(node) is AnonFunc:
        return _free_ids(node.body) - set(node.params)
    if type(node) is Id:
        return {node.name}
    out = set()
    if isinstance(node, Node):
        for f in node._fields:
            out |= _free_ids(getattr(node, f))
    elif isinstance(node, (list, tuple)):
        for x in node:
            out |= _free_ids(x)
    return out


def _wrap(line, width=110):
    out, cur = [], ""
    for part in line.split(", "):
        if len(cur) + len(part) > width and cur:
            out.append(cur.rstrip() + ",")
            cur = "           " + part
        else:
            cur = part if not cur else cur + ", " + part
    out.append(cur)
    return out


def translate_steady_state(path) -> str:
    """Python source equivalent to the MATLAB steady-state file ``path``.

    The source defines ``steady_state(model, params, exo) -> (ys, params)``, to be used as
    ``Model(..., steady_state_fn=steady_state)``.  Raises ``TranslationError`` when the file uses
    MATLAB features outside the scalar subset that can be expressed directly (pydyn then interprets
    the file instead)."""
    path = Path(path)
    try:
        body, funcs = parse(path.read_text(errors="replace"))
    except ParseError as exc:
        raise TranslationError(f"{path.name} cannot be parsed: {exc}") from None
    return _Translator(Unit(path, body, funcs), path.name).steady_state_module()


def helper_functions(mod_dir, names_used, taken=()):
    """``MatlabFunction`` objects for the function files next to the .mod file whose names occur in
    ``names_used`` (and are not in ``taken``)."""
    out = {}
    mod_dir = Path(mod_dir)
    if not mod_dir.is_dir():
        return out
    for p in sorted(mod_dir.glob("*.m")):
        n = p.stem
        if n in names_used and n not in taken and re.fullmatch(r"[A-Za-z_]\w*", n) and not n.endswith("_steadystate"):
            try:
                head = p.read_text(errors="replace")
            except OSError:
                continue
            if re.search(r"^\s*function\b", re.sub(r"(?m)^\s*%.*$", "", head).lstrip()[:200]):
                out[n] = MatlabFunction(p)
    return out
