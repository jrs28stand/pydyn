# This file is part of pydyn. Copyright (C) 2026; GPL-3.0-or-later (see LICENSE).
"""MATLAB graphics calls in .mod code (``figure``, ``subplot``, ``plot``, ``title``, ``legend``, ``hold``, ``set``, ``get``,
``close`` ...): the interpreter accepts them, keeps a table of figures and graphics objects so that the handles they return
behave like Octave's (figures are the integers 1, 2, ...; any other object has a negative fractional handle), and draws
nothing.  Graphics never change results.

What is refused rather than answered: anything that would return a value depending on drawn data or on a graphics toolkit
(``axis``/``xlim``/``ylim`` with an output, ``get`` of a property other than ``type``, ``hist``/``colormap`` with an output),
and everything that writes an image (``print``, ``saveas``, ``savefig``, ``hgsave``): pydyn does not render the figures of
.mod code, and an unfaithful image would be worse than none."""
from __future__ import annotations

import re

import numpy as np

from .matlab import (BUILTINS, Cell, Handle, MatlabError, Tainted, Unsupported, _SCAL, bi, ret)


class GState:
    """Figures and graphics objects of one interpreter."""

    def __init__(self):
        self.figs = {}               # figure handle -> {"axes": [handles], "cur": handle or None, "grid": {key: handle}}
        self.objs = {}               # object handle -> (type, parent handle)
        self.cur_fig = None
        self.k = 0

    def new_obj(self, kind, parent):
        while True:
            self.k += 1
            h = -((self.k * 0.6180339887498949) % 1.0)
            if h != 0.0 and h not in self.objs:
                break
        self.objs[h] = (kind, parent)
        return np.float64(h)

    def new_fig(self, n=None):
        if n is None:
            n = 1
            while n in self.figs:
                n += 1
        self.figs[n] = {"axes": [], "cur": None, "grid": {}}
        self.cur_fig = n
        return n

    def fig(self):
        if self.cur_fig is None or self.cur_fig not in self.figs:
            self.new_fig()
        return self.figs[self.cur_fig]

    def axes(self):
        f = self.fig()
        if f["cur"] is None:
            f["cur"] = self.new_obj("axes", self.cur_fig)
            f["axes"].append(f["cur"])
        return f["cur"]


def _gs(I):
    """The graphics state; only the interpreter of ``pydyn.dynare()`` results mode (``graphics_enabled``) draws, any other (the .mod file's load
    time, where the results the figures show do not exist yet) refuses as before."""
    if not getattr(I, "graphics_enabled", False):
        raise Unsupported("plotting", "plot")
    return _peek(I)


def _peek(I):
    g = getattr(I, "_gstate", None)
    if g is None:
        g = I._gstate = GState()
    return g


def _is_handle(g, v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return x == 0.0 or (x in g.figs) or (x in g.objs)


def _scalar(v, who):
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.reshape(-1)[0]
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    return None


def _check_args(a, who):
    for x in a:
        if isinstance(x, Tainted):
            raise Unsupported(f"{who}: an argument that pydyn could not compute", "plot")


def _no_output(nout, who):
    if nout > 0:
        raise Unsupported(f"{who}: with an output this returns values that depend on the graphics toolkit", "plot")


# ------------------------------------------------------------------------------------------------ figures
@bi("figure")
def _figure(I, a, nout):
    g = _gs(I)
    _check_args(a, "figure")
    n = None
    props = a
    if a and not isinstance(a[0], str):
        h = _scalar(a[0], "figure")
        if h is None:
            raise MatlabError("figure: N must be a scalar figure handle or integer")
        if h in g.figs:
            n = int(h)
        elif h >= 1 and h == int(h) and h not in g.objs:
            n = int(h)
        else:
            raise MatlabError("figure: N must be figure handle or integer >= 1")
        props = a[1:]
    if len(props) % 2:
        raise MatlabError("figure: invalid parameter/value pair arguments")
    for k in range(0, len(props), 2):
        if not isinstance(props[k], str):
            raise MatlabError("figure: invalid parameter/value pair arguments")
    if n is not None and n in g.figs:
        g.cur_fig = n
    else:
        n = g.new_fig(n)
    return [np.float64(n)] if nout > 0 else []


@bi("gcf")
def _gcf(I, a, nout):
    g = _gs(I)
    g.fig()
    return [np.float64(g.cur_fig)]


@bi("gca")
def _gca(I, a, nout):
    return [_gs(I).axes()]


@bi("subplot")
def _subplot(I, a, nout):
    g = _gs(I)
    _check_args(a, "subplot")
    if len(a) == 1:
        h = _scalar(a[0], "subplot")
        if h is not None and h in g.objs and g.objs[h][0] == "axes":
            g.fig()["cur"] = np.float64(h)
            return [np.float64(h)] if nout > 0 else []
        if h is None or h != int(h) or not 111 <= h <= 999:
            raise MatlabError("subplot: INDEX must be a three-digit integer")
        m, n, p = int(h) // 100, (int(h) // 10) % 10, int(h) % 10
    elif len(a) == 3:
        vals = [_scalar(x, "subplot") for x in a]
        if None in vals:
            raise Unsupported("subplot with a vector index", "plot")
        m, n, p = [int(v) for v in vals]
        if m < 1 or n < 1 or p < 1 or p > m * n:
            raise MatlabError("subplot: INDEX must be in the range [1, ROWS*COLS]")
    else:
        raise Unsupported("subplot with property arguments", "plot")
    f = g.fig()
    key = (m, n, p)
    if key not in f["grid"]:
        h = g.new_obj("axes", g.cur_fig)
        f["grid"][key] = h
        f["axes"].append(h)
    f["cur"] = f["grid"][key]
    return [f["cur"]] if nout > 0 else []


@bi("close")
def _close(I, a, nout):
    g = _gs(I)
    if not a:
        if g.cur_fig is not None and g.cur_fig in g.figs:
            _drop(g, g.cur_fig)
        return []
    if len(a) == 1 and isinstance(a[0], str):
        w = a[0].lower()
        if w in ("all", "hidden"):
            for n in list(g.figs):
                _drop(g, n)
            return []
        raise MatlabError(f"close: unrecognized string argument {a[0]!r}")
    if len(a) == 2 and isinstance(a[0], str) and a[0].lower() == "all" and isinstance(a[1], str) and a[1].lower() == "hidden":
        for n in list(g.figs):
            _drop(g, n)
        return []
    for x in a:
        if isinstance(x, str):
            if x.lower() == "force":
                continue
            raise MatlabError("close: invalid argument")
        arr = np.asarray(x, float).ravel() if isinstance(x, (np.ndarray,) + _SCAL) else None
        if arr is None:
            raise MatlabError("close: invalid argument")
        for h in arr:
            if h in g.figs:
                _drop(g, int(h))
            elif not _is_handle(g, h):
                raise MatlabError("close: invalid figure handle")
    return []


def _drop(g, n):
    f = g.figs.pop(n, None)
    if f is not None:
        for h in list(g.objs):
            if _owner_fig(g, h) == n:
                del g.objs[h]
    if g.cur_fig == n:
        g.cur_fig = max(g.figs) if g.figs else None


def _owner_fig(g, h):
    seen = 0
    while h in g.objs and seen < 10:
        h = g.objs[h][1]
        seen += 1
    return h


@bi("clf", "cla")
def _clf(I, a, nout):
    g = _gs(I)
    g.fig()
    return []


# ------------------------------------------------------------------------------------------------ drawing
_LINESPEC = re.compile(r"^[bgrcmykw\-:.+*ox^v<>sdph]+$|^--$|^-\.$|^#[0-9a-fA-F]{6}$")


def _series(a, who):
    """The number of lines plot(...) creates for its arguments (MATLAB's argument rules)."""
    a = list(a)
    if a and _scalar(a[0], who) is not None and len(a) > 1 and not isinstance(a[0], str) and np.asarray(a[0]).size == 1 \
            and not isinstance(a[1], str):
        pass                                                 # plot(y, ...) with a scalar y is a point: handled below
    count, i = 0, 0
    while i < len(a):
        x = a[i]
        if isinstance(x, str):
            break
        if not isinstance(x, (np.ndarray,) + _SCAL):
            raise Unsupported(f"{who}: an argument of type {type(x).__name__}", "plot")
        x = np.asarray(x, float) if not isinstance(x, np.ndarray) else x
        i += 1
        if i < len(a) and not isinstance(a[i], str):
            y = a[i]
            if not isinstance(y, (np.ndarray,) + _SCAL):
                raise Unsupported(f"{who}: an argument of type {type(y).__name__}", "plot")
            y = np.asarray(y, float) if not isinstance(y, np.ndarray) else y
            i += 1
            xv = x.ndim == 2 and min(x.shape) == 1
            yv = y.ndim == 2 and min(y.shape) == 1
            if yv or y.size <= 1:
                count += 1
            elif xv and x.size == y.shape[0]:
                count += y.shape[1]
            elif xv and x.size == y.shape[1]:
                count += y.shape[0]
            else:
                count += y.shape[1]
        else:
            yv = x.ndim == 2 and min(x.shape) == 1
            count += 1 if (yv or x.size <= 1) else x.shape[1]
        if i < len(a) and isinstance(a[i], str) and _LINESPEC.match(a[i]):
            i += 1
    rest = a[i:]
    if len(rest) % 2 and not (len(rest) == 1 and isinstance(rest[0], str)):
        raise MatlabError(f"{who}: properties must appear followed by a value")
    return count


def _draw(I, a, nout, who, lines=False):
    g = _gs(I)
    _check_args(a, who)
    if a and not isinstance(a[0], str) and _scalar(a[0], who) is not None and np.asarray(a[0]).size == 1 \
            and _is_handle(g, a[0]) and float(a[0]) in g.objs and g.objs[float(a[0])][0] == "axes":
        ax = np.float64(float(a[0]))                         # plot(ax, ...)
        a = a[1:]
    else:
        ax = g.axes()
    if who == "plot":
        n = _series(a, who)
    else:
        n = 1
    hs = [g.new_obj("line" if who in ("plot", "line", "plot3", "semilogx", "semilogy", "loglog", "stairs") else
                    ("hggroup" if who in ("bar", "barh", "area", "errorbar", "scatter", "hist", "histogram", "contour", "surf", "mesh") else "patch"),
                    ax) for _ in range(max(n, 1))]
    if nout == 0:
        return []
    if who not in ("plot", "line", "plot3", "semilogx", "semilogy", "loglog", "fill", "bar", "area", "stairs", "scatter",
                   "errorbar"):
        raise Unsupported(f"{who}: with an output this returns data computed from the arguments", "plot")
    return [ret(np.array(hs, float).reshape(-1, 1))]


def _mk_draw(name):
    def f(I, a, nout):
        return _draw(I, a, nout, name)
    return f


for _n in ("plot", "plot3", "semilogx", "semilogy", "loglog", "stairs", "bar", "barh", "area", "fill", "scatter", "errorbar",
           "hist", "histogram", "contour", "surf", "mesh", "patch", "quiver", "fplot", "ezplot", "line"):
    BUILTINS[_n] = _mk_draw(_n)


def _text_obj(who):
    def f(I, a, nout):
        g = _gs(I)
        _check_args(a, who)
        if who in ("title", "xlabel", "ylabel", "zlabel") and not a:
            raise MatlabError(f"Invalid call to {who}")
        h = g.new_obj("text", g.axes())
        return [h] if nout > 0 else []
    return f


for _n in ("title", "xlabel", "ylabel", "zlabel", "text", "suptitle", "sgtitle"):
    BUILTINS[_n] = _text_obj(_n)


@bi("legend")
def _legend(I, a, nout):
    g = _gs(I)
    _check_args(a, "legend")
    h = g.new_obj("axes", g.cur_fig if g.cur_fig is not None else 0)
    return [h] if nout > 0 else []


@bi("annotation", "colorbar")
def _annot(I, a, nout):
    g = _gs(I)
    _check_args(a, "annotation")
    h = g.new_obj("axes", g.cur_fig if g.cur_fig is not None else 0)
    return [h] if nout > 0 else []


@bi("hold")
def _hold(I, a, nout):
    _no_output(nout, "hold")
    g = _gs(I)
    _check_args(a, "hold")
    g.axes()
    for x in a:
        if isinstance(x, str) and x.lower() not in ("on", "off", "all", "reset"):
            raise MatlabError(f"hold: invalid hold state {x!r}")
    return []


@bi("axis", "xlim", "ylim", "zlim", "colormap", "caxis", "view", "daspect", "pbaspect")
def _axis_like(I, a, nout):
    _no_output(nout, "axis")
    _check_args(a, "axis")
    _gs(I).axes()
    return []


@bi("grid", "box", "shading", "orient", "drawnow", "pause", "refresh", "movegui", "uistack", "linkaxes", "set_legend")
def _gfx_noop(I, a, nout):
    _no_output(nout, "graphics call")
    _check_args(a, "graphics call")
    return []


@bi("print", "saveas", "savefig", "hgsave", "export_fig", "exportgraphics")
def _gfx_write(I, a, nout):
    raise Unsupported("print/saveas/savefig/hgsave: this would write an image of a figure of the .mod file, and pydyn does not "
                      "render those figures", "plot")


# ------------------------------------------------------------------------------------------------ handles
def _handles_of(g, v, who):
    if not isinstance(v, (np.ndarray,) + _SCAL):
        raise MatlabError(f"{who}: invalid handle")
    arr = np.asarray(v, float)
    if arr.size == 0 or not all(_is_handle(g, x) for x in arr.ravel()):
        raise MatlabError(f"{who}: invalid handle (= {float(arr.ravel()[0]) if arr.size else '[]'})")
    return arr


@bi("set")
def _set(I, a, nout):
    _no_output(nout, "set")
    g = _gs(I)
    _check_args(a, "set")
    if not a:
        raise MatlabError("Invalid call to set")
    _handles_of(g, a[0], "set")
    rest = a[1:]
    if len(rest) == 1 and isinstance(rest[0], (Cell,)):
        raise Unsupported("set with a cell of properties", "plot")
    if len(rest) % 2 and not (len(rest) == 1 and isinstance(rest[0], str)):
        raise MatlabError("set: invalid parameter/value pair arguments")
    return []


@bi("get")
def _get(I, a, nout):
    g = _gs(I)
    _check_args(a, "get")
    if len(a) != 2 or not isinstance(a[1], str):
        raise Unsupported("get with other than (handle, 'type')", "plot")
    h = _handles_of(g, a[0], "get")
    if a[1].lower() != "type" or h.size != 1:
        raise Unsupported(f"get(h, '{a[1]}'): pydyn keeps no graphics properties (only 'type' of a handle)", "plot")
    x = float(h.ravel()[0])
    if x == 0.0:
        return ["root"]
    if x in g.figs:
        return ["figure"]
    return [g.objs[x][0]]


@bi("ishandle", "ishghandle")
def _ishandle(I, a, nout):
    g = _peek(I)
    if len(a) != 1:
        raise MatlabError("Invalid call to ishandle")
    v = a[0]
    if isinstance(v, (str, Cell)) or not isinstance(v, (np.ndarray,) + _SCAL):
        return [np.bool_(False)]
    arr = np.asarray(v, float)
    out = np.array([_is_handle(g, x) for x in arr.ravel()], dtype=bool).reshape(arr.shape, order="C")
    return [ret(out) if out.ndim == 2 else out]


@bi("isfigure")
def _isfigure(I, a, nout):
    g = _peek(I)
    v = a[0]
    if isinstance(v, (str, Cell)) or not isinstance(v, (np.ndarray,) + _SCAL):
        return [np.bool_(False)]
    arr = np.asarray(v, float)
    out = np.array([x in g.figs for x in arr.ravel()], dtype=bool).reshape(arr.shape)
    return [ret(out)]


@bi("isaxes")
def _isaxes(I, a, nout):
    g = _peek(I)
    v = a[0]
    if isinstance(v, (str, Cell)) or not isinstance(v, (np.ndarray,) + _SCAL):
        return [np.bool_(False)]
    arr = np.asarray(v, float)
    out = np.array([x in g.objs and g.objs[x][0] == "axes" for x in arr.ravel()], dtype=bool).reshape(arr.shape)
    return [ret(out)]


#: graphics calls that may be written in command syntax (``hold on``, ``close all``)
COMMANDS = ("figure", "close", "hold", "grid", "box", "axis", "clf", "cla", "drawnow", "pause", "legend", "colormap", "shading",
            "orient")
