"""Comparison of the figures pydyn.dynare() records (pydyn.gfx.Gfx) with the drawing calls Dynare makes under Octave
(reference/fig/r_fig.m writes <case>_fig.mat).  Arrays and texts are compared, not styles."""
import numpy as np
from scipy.io import loadmat

KINDS = ("subplot", "plot", "bar", "barh", "area", "fill", "title", "legend", "xlim", "ylim", "text", "xlabel", "ylabel")


def _s(v):
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        return "".join(v.ravel())
    return str(v)


def _arr(v):
    return np.asarray(v, float)


def load_ref(path):
    """-> (figures: list of {name, calls:[(kind,args)]}, saved: [(fig, file)])"""
    g = loadmat(path, struct_as_record=False, squeeze_me=True)["zz_G"]
    names = list(np.atleast_1d(g.names)) if int(g.fig) else []
    figs = [{"name": _s(n), "calls": []} for n in names]
    calls = g.calls if isinstance(g.calls, np.ndarray) and g.calls.dtype == object else np.array([g.calls], dtype=object)
    for c in np.atleast_1d(calls):
        c = list(c)
        kind, fig, a = str(c[0]), int(c[1]), c[2:]
        if kind not in KINDS:
            continue
        f = figs[fig - 1]
        if kind == "subplot":
            if len(a) == 1:                                   # subplot(231)
                v = int(_arr(a[0]))
                a = [v // 100, (v // 10) % 10, v % 10]
            f["calls"].append(("subplot", tuple(int(_arr(x).ravel()[0]) for x in a[:3])))
        elif kind == "plot":
            nums = []
            fmt = None
            if len(a) >= 3 and not any(isinstance(v, str) for v in a[:3]) and _arr(a[0]).size == 1:   # plot(ax, x, y, ...)
                a = a[1:]
            for x in a:
                if isinstance(x, str):
                    if fmt is None and not nums == [] and x[:1] in "-:.bgrcmykw+*ox^v<>sdph" and len(x) <= 4 and fmt is None and len(nums) in (1, 2):
                        fmt = x
                    break
                nums.append(_arr(x))
            if len(nums) == 1:
                nums = [None, nums[0]]
            f["calls"].append(("plot", nums[0], nums[1] if len(nums) > 1 else None, fmt))
        elif kind in ("bar", "barh", "area"):
            nums = [_arr(x) for x in a if not isinstance(x, str)][:2]
            f["calls"].append((kind, nums[0], nums[1]) if len(nums) == 2 else (kind, None, nums[0]))
        elif kind == "fill":
            nums = [_arr(x) for x in a if not isinstance(x, str)][:2]
            f["calls"].append(("fill", nums[0], nums[1]))
        elif kind in ("title", "xlabel", "ylabel"):
            f["calls"].append((kind, _s(a[0])))
        elif kind == "legend":
            labs = []
            for x in a:
                if isinstance(x, str):
                    if x in ("Location", "Interpreter", "location", "interpreter", "Orientation", "FontSize", "Box", "box"):
                        break
                    labs.append(x)
                elif isinstance(x, np.ndarray) and x.dtype == object:
                    labs += [_s(y) for y in x.ravel()]
            f["calls"].append(("legend", labs))
        elif kind in ("xlim", "ylim"):
            if a:                                  # without an argument it reads the limits (graph_decomp.m: plot(..., ylim, ...))
                f["calls"].append((kind, _arr(a[0])))
        elif kind == "text":
            f["calls"].append(("text", float(_arr(a[0])), float(_arr(a[1])), _s(a[2]).rstrip()))
    saved = []
    sv = g.saved
    if isinstance(sv, np.ndarray) and sv.size:
        sv = [sv] if sv.dtype != object or (sv.ndim == 1 and not isinstance(sv[0], np.ndarray)) else list(sv)
        for s_ in sv:
            s_ = list(s_)
            if len(s_) >= 3:                       # dyn_saveas(h, stem, formats) as r_fig.m records it
                saved.append((int(_arr(s_[0])), _s(s_[1]) + "." + _s(s_[2]).split("|")[0]))
            else:                                  # print(h, file)
                saved.append((int(_arr(s_[0])), _s(s_[-1])))
    return figs, saved


def mine(res):
    out = []
    for f in res.gfx.figures:
        calls = []
        for kind, a, kw in f.calls:
            if kind == "subplot":
                calls.append(("subplot", tuple(a)))
            elif kind == "plot":
                calls.append(("plot", a[0], a[1], a[2]))
            elif kind in ("bar", "barh", "area"):
                calls.append((kind, a[0], a[1]))
            elif kind == "fill":
                calls.append(("fill", a[0], a[1]))
            elif kind in ("title", "xlabel", "ylabel"):
                calls.append((kind, a[0]))
            elif kind == "legend":
                calls.append(("legend", list(a[0])))
            elif kind in ("xlim", "ylim"):
                calls.append((kind, a[0]))
            elif kind == "text":
                calls.append(("text", a[0], a[1], a[2].rstrip()))
        out.append({"name": f.name, "saved": f.saved, "calls": calls})
    return out


def _same_arr(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size != b.size:
        return False
    a, b = a.ravel(order="F"), b.ravel(order="F")
    same = (np.isnan(a) & np.isnan(b)) | (a == b)
    return bool(np.all(same | (np.abs(a - b) <= tol * np.maximum(1.0, np.abs(a)))))


#: lines whose heights are the axis limits Octave's stubbed ``get(gca, 'ylim')`` / ``ylim`` cannot give (mode_check's mode line, NaN
#: markers and bound line, the red line where a forecast starts in graph_decomp.m): only their abscissae are compared
_Y_FREE = ("c", "o", "r--", "-r")


def _norm(c):
    if c[0] == "plot" and len(c) > 3 and c[3] in _Y_FREE:
        return ("plot", c[1], None, c[3])
    return c


def diff(res, ref_path, tol=1e-8, ignore_names=False, ignore_kinds=()):
    """List of differences between the figures of ``res`` and Dynare's (empty when they agree)."""
    ref, saved = load_ref(ref_path)
    me = mine(res)
    out = []
    if len(ref) != len(me):
        out.append(f"number of figures {len(ref)} (Dynare) vs {len(me)}")
        return out
    ref_saved = {k: v for k, v in saved}
    for k, (r, m) in enumerate(zip(ref, me), 1):
        if not ignore_names and r["name"] != m["name"]:
            out.append(f"figure {k}: name {r['name']!r} vs {m['name']!r}")
        rs = ref_saved.get(k)
        if rs is not None:
            mine_stem = (m["saved"] or "") + ".eps"
            if rs != mine_stem and rs.replace("\\", "/") != mine_stem:
                out.append(f"figure {k}: file {rs!r} vs {mine_stem!r}")
        elif m["saved"]:
            out.append(f"figure {k}: saved by pydyn only ({m['saved']})")
        rc = [_norm(c) for c in r["calls"] if c[0] not in ignore_kinds]
        mc = [_norm(c) for c in m["calls"] if c[0] not in ignore_kinds]
        if len(rc) != len(mc):
            out.append(f"figure {k}: {len(rc)} calls (Dynare) vs {len(mc)}")
            continue
        for j, (a, b) in enumerate(zip(rc, mc)):
            if a[0] != b[0]:
                out.append(f"figure {k} call {j}: {a[0]} vs {b[0]}")
                continue
            ok = True
            for x, y in zip(a[1:], b[1:]):
                if isinstance(x, (np.ndarray, float, int)) and not isinstance(x, bool) or x is None:
                    if isinstance(y, (str, list, tuple)) and not isinstance(y, np.ndarray):
                        ok = ok and x is None and y is None
                    else:
                        ok = ok and _same_arr(x, y, tol)
                else:
                    ok = ok and (x == y)
            if not ok:
                out.append(f"figure {k} call {j} ({a[0]}): differs")
    return out
