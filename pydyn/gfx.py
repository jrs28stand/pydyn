# This file is part of pydyn. Copyright (C) 2026; GPL-3.0-or-later (see LICENSE).
"""A small recording stand-in for the MATLAB graphics calls Dynare's figure routines make.

The figure code of :mod:`pydyn.figures` is written as a transliteration of the plotting parts of Dynare's ``.m`` files
and issues the same calls (``figure``, ``subplot``, ``plot``, ``bar``, ``area``, ``fill``, ``title``, ``legend``,
``xlim``, ``ylim``, ``text``, ``dyn_saveas``, ...) on a :class:`Gfx`.  A ``Gfx`` always *records* the calls (this is
what the tests compare with the calls Dynare makes under Octave, recorded by ``reference/fig/``'s stubs) and, when
asked to save a figure, replays the record with matplotlib (imported lazily, ``Agg`` backend) into the files that
``dyn_saveas`` would write: ``eps``, ``pdf`` (``graph_format``); ``fig`` cannot be written and is reported; ``none``
writes nothing.  The figures are not meant to match Dynare's pixel for pixel."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

#: calls that carry data or text and are compared with Dynare's
DATA_CALLS = ("subplot", "plot", "bar", "barh", "area", "fill", "title", "legend", "xlim", "ylim", "text", "xlabel",
              "ylabel", "errorbar", "hline")


class Ledger:
    """The record of the files pydyn.dynare() wrote (SHA-256 by path relative to the .mod file's folder, the same JSON file
    ``save`` keeps): a file that exists is overwritten only if pydyn wrote it and nobody changed it since."""

    def __init__(self, base, path):
        self.base, self.path = Path(base), Path(path)

    def _load(self):
        import json
        try:
            return json.loads(self.path.read_text()) if self.path.is_file() else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _sha(p):
        import hashlib
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def key(self, p):
        try:
            return str(Path(p).relative_to(self.base))
        except ValueError:
            return str(p)

    def may_write(self, p):
        p = Path(p)
        if not p.exists():
            return True
        return p.is_file() and not p.is_symlink() and self._load().get(self.key(p)) == self._sha(p)

    def record(self, p):
        import json
        rec = self._load()
        rec[self.key(p)] = self._sha(p)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rec, indent=0))


class Figure:
    __slots__ = ("name", "calls", "saved", "formats", "suptitle")

    def __init__(self, name=None):
        self.name = name
        self.calls = []
        self.saved = None
        self.formats = None
        self.suptitle = None


class Gfx:
    """Recorder and writer of figures.  ``enabled=False`` makes every call a no-op (``nograph``)."""

    def __init__(self, enabled=True, graph_format=("eps",), nodisplay=True, root=None, ledger=None):
        self.ledger = ledger
        self.enabled = bool(enabled)
        self.graph_format = [graph_format] if isinstance(graph_format, str) else list(graph_format)
        self.nodisplay = nodisplay
        self.root = Path(root) if root is not None else Path(".")
        self.figures: list[Figure] = []
        self.written: list[str] = []
        self._cur = None
        self._warned = False

    # ---------------------------------------------------------------- recording
    def figure(self, name=None):
        if not self.enabled:
            return None
        self._cur = Figure(name)
        self.figures.append(self._cur)
        return self._cur

    def _rec(self, kind, *args, **kw):
        if self.enabled:
            if self._cur is None:
                self.figure()
            self._cur.calls.append((kind, args, kw))

    def subplot(self, nr, nc, k):
        self._rec("subplot", int(nr), int(nc), int(k))

    def plot(self, x, y=None, fmt=None, **kw):
        if y is None:
            y, x = x, None
        self._rec("plot", None if x is None else np.array(x, float, ndmin=1), np.array(y, float, ndmin=1), fmt, **kw)

    def bar(self, x, y, **kw):
        self._rec("bar", np.array(x, float, ndmin=1), np.array(y, float), **kw)

    def area(self, x, y, **kw):
        self._rec("area", np.array(x, float, ndmin=1), np.array(y, float), **kw)

    def fill(self, x, y, color, **kw):
        self._rec("fill", np.array(x, float, ndmin=1), np.array(y, float, ndmin=1), color, **kw)

    def title(self, s, **kw):
        self._rec("title", str(s), **kw)

    def xlabel(self, s, **kw):
        self._rec("xlabel", str(s), **kw)

    def ylabel(self, s, **kw):
        self._rec("ylabel", str(s), **kw)

    def legend(self, labels, **kw):
        self._rec("legend", [str(s) for s in labels], **kw)

    def xlim(self, lim):
        self._rec("xlim", np.array(lim, float, ndmin=1))

    def ylim(self, lim):
        self._rec("ylim", np.array(lim, float, ndmin=1))

    def text(self, x, y, s, **kw):
        self._rec("text", float(x), float(y), str(s), **kw)

    def axis(self, lim):
        """``axis([xmin xmax ymin ymax])``."""
        self._rec("axis", np.array(lim, float, ndmin=1))

    def set_xlim(self, lim):
        """``set(ax, 'xlim', lim)``: not a call Dynare's comparable record keeps."""
        self._rec("set_xlim", np.array(lim, float, ndmin=1))

    def axes(self, position):
        """``axes('Position', pos)``: a new set of axes at ``pos`` (normalised figure units)."""
        self._rec("axes", np.array(position, float, ndmin=1))

    def axis_off(self, limits=(0, 1, 0, 1)):
        self._rec("axis_off", np.array(limits, float))

    def hold(self, *_):
        pass

    def write_mat(self, stem, data):
        """A MAT-file ``<stem>.mat`` (under the .mod file's folder) holding the variables of ``data``; returns the path or
        None when an existing file that pydyn did not write is left alone."""
        from scipy.io import savemat
        p = Path(stem)
        p = p if p.is_absolute() else self.root / p
        p = p.with_suffix(".mat") if p.suffix != ".mat" else p
        if self.ledger is not None and not self.ledger.may_write(p):
            warnings.warn(f"{p} exists and was not written by pydyn.dynare(); it is not overwritten", stacklevel=2)
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(p), data, do_compression=False, oned_as="column", long_field_names=True)
        if self.ledger is not None:
            self.ledger.record(p)
        return str(p)

    def write_text(self, name, text):
        """A text file ``name`` (a plain file name) under the .mod file's folder, with the ledger's rule; returns the path or
        None when an existing file pydyn did not write is left alone."""
        p = self.root / name
        if self.ledger is not None and not self.ledger.may_write(p):
            warnings.warn(f"{p} exists and was not written by pydyn.dynare(); it is not overwritten", stacklevel=2)
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        if self.ledger is not None:
            self.ledger.record(p)
        return str(p)

    # ---------------------------------------------------------------- writing
    def saveas(self, stem, formats=None):
        """``dyn_saveas(h, stem, nodisplay, graph_format)``: ``stem`` is the path without extension."""
        if not self.enabled or self._cur is None:
            return []
        fig = self._cur
        fig.saved = str(stem)
        fmts = self.graph_format if formats is None else ([formats] if isinstance(formats, str) else list(formats))
        fig.formats = fmts
        out = []
        todo = [f for f in fmts if f in ("eps", "pdf")]
        if "fig" in fmts and not self._warned:
            self._warned = True
            warnings.warn("graph_format fig: MATLAB .fig files cannot be written by pydyn; only eps and pdf are", stacklevel=2)
        if todo:
            p = Path(stem)
            p = p if p.is_absolute() else self.root / p
            p.parent.mkdir(parents=True, exist_ok=True)
            mfig = self._render(fig)
            try:
                for f in todo:
                    path = f"{p}.{f}"
                    if self.ledger is not None and not self.ledger.may_write(path):
                        warnings.warn(f"{path} exists and was not written by pydyn.dynare(); it is not overwritten", stacklevel=2)
                        continue
                    mfig.savefig(path, format=f)
                    if self.ledger is not None:
                        self.ledger.record(path)
                    out.append(path)
            finally:
                import matplotlib.pyplot as plt
                plt.close(mfig)
        self.written += out
        return out

    @staticmethod
    def _style(fmt, kw):
        k = dict()
        if fmt:
            k["fmt"] = fmt
        for a, b in (("linewidth", "linewidth"), ("LineWidth", "linewidth"), ("color", "color"), ("Color", "color")):
            if a in kw:
                k[b] = kw[a]
        return k

    def _render(self, fig):
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        mf = plt.figure(figsize=(8, 6))
        if fig.name:
            mf.suptitle(fig.name, fontsize=9)
        ax = None
        legend_pending = None
        for kind, a, kw in fig.calls:
            if kind == "subplot":
                nr, nc, k = a
                ax = mf.add_subplot(nr, nc, k)
            elif kind == "axes":
                ax = mf.add_axes(list(a[0]))
            elif kind == "axis":
                if ax is not None:
                    lim = a[0]
                    if lim.size == 4 and np.all(np.isfinite(lim)):
                        if lim[0] < lim[1]:
                            ax.set_xlim(lim[0], lim[1])
                        if lim[2] < lim[3]:
                            ax.set_ylim(lim[2], lim[3])
            elif kind == "axis_off":
                if ax is not None:
                    ax.axis("off")
                    ax.set_xlim(a[0][0], a[0][1])
                    ax.set_ylim(a[0][2], a[0][3])
            else:
                if ax is None:
                    ax = mf.add_subplot(1, 1, 1)
                if kind == "plot":
                    x, y, fmt = a
                    st = self._style(fmt, kw)
                    yy = np.asarray(y, float)
                    xx = np.arange(1, yy.shape[0] + 1) if x is None else x
                    if yy.ndim == 2 and yy.shape[0] != len(xx) and yy.shape[1] == len(xx):
                        yy = yy.T                                  # MATLAB plots the rows when only the columns fit x
                    if "color" in st and "fmt" in st:
                        st["fmt"] = "".join(ch for ch in st["fmt"] if ch not in "bgrcmykw")
                    fm = st.pop("fmt", "")
                    args = (xx, yy) + ((fm,) if fm else ())
                    try:
                        ax.plot(*args, **st)
                    except ValueError:
                        ax.plot(xx, yy)
                elif kind in ("bar", "area"):
                    x, y = a
                    yy = np.asarray(y, float)
                    if yy.ndim == 2 and kind == "bar":
                        bottom_pos = np.zeros(yy.shape[0])
                        bottom_neg = np.zeros(yy.shape[0])
                        for j in range(yy.shape[1]):
                            col = yy[:, j]
                            bottom = np.where(col >= 0, bottom_pos, bottom_neg)
                            ax.bar(x, col, bottom=bottom, width=0.8)
                            bottom_pos = bottom_pos + np.where(col >= 0, col, 0)
                            bottom_neg = bottom_neg + np.where(col < 0, col, 0)
                    elif kind == "bar":
                        ax.bar(x, yy)
                    else:
                        ax.stackplot(x, yy.T) if yy.ndim == 2 else ax.fill_between(x, yy)
                elif kind == "fill":
                    x, y, c = a
                    if isinstance(c, (int, np.integer)):          # a colormap index, as fill(x, y, k) in MATLAB
                        c = plt.get_cmap("tab20")((int(c) - 1) % 20)
                    try:
                        ax.fill(x, y, color=c, linewidth=0)
                    except (ValueError, TypeError):
                        ax.fill(x, y, linewidth=0)
                elif kind == "title":
                    ax.set_title(a[0], fontsize=8)
                elif kind == "xlabel":
                    ax.set_xlabel(a[0], fontsize=8)
                elif kind == "ylabel":
                    ax.set_ylabel(a[0], fontsize=8)
                elif kind == "legend":
                    legend_pending = a[0]
                    try:
                        ax.legend(a[0], fontsize=7, frameon=False)
                    except Exception:
                        pass
                elif kind in ("xlim", "set_xlim"):
                    lim = a[0]
                    if lim.size == 2 and np.all(np.isfinite(lim)) and lim[0] < lim[1]:
                        ax.set_xlim(lim)
                elif kind == "ylim":
                    lim = a[0]
                    if lim.size == 2 and np.all(np.isfinite(lim)) and lim[0] < lim[1]:
                        ax.set_ylim(lim)
                elif kind == "text":
                    ax.text(a[0], a[1], a[2], fontsize=7)
        if not any(c[0] == "axes" for c in fig.calls):
            mf.tight_layout(rect=(0, 0, 1, 0.95) if fig.name else None)
        return mf


def normalise(figures):
    """The comparable form of recorded figures: list of dicts {name, saved, calls: [(kind, payload...)]}, keeping the
    calls of ``DATA_CALLS`` (the form the Octave recorder writes too)."""
    out = []
    for f in figures:
        calls = []
        for kind, a, kw in f.calls:
            if kind in DATA_CALLS:
                calls.append((kind, a))
        out.append({"name": f.name, "saved": None if f.saved is None else Path(f.saved).name, "calls": calls})
    return out
