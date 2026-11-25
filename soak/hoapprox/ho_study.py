"""Order-2 / cubature study of the regime formulation (docs/OBC_REGIME_HO.md).  python3 soak/hoapprox/ho_study.py O2|CUB|COST [model ...]
O2: order-2 regime vs order-2 LCP (and vs the order-1 regime path) on the approx / exact model set, 300 periods.
CUB: cubature (level 9 or 11, regions all) regime vs LCP; failure shares.  COST: seconds per period."""
import sys, time, json, warnings, tempfile, shutil
from pathlib import Path
sys.path.insert(0, "soak"); sys.path.insert(0, "soak/approx"); sys.path.insert(0, "tests")
warnings.simplefilter("ignore")
import numpy as np, lib, gen, pydyn
from pydyn.obc import constraint_objects, OBCError
from pydyn.obc_ho import HigherOrderNews, CubatureFailure
from pydyn.obc_regime_ho import HigherOrderRegime

SPEC = {"zlbm_0": ("zlbm", 0), "zlbm_1e-3": ("zlbm", 1e-3), "zlbm_1e-2": ("zlbm", 1e-2), "zlbm_5e-2": ("zlbm", 5e-2),
        "zlb2m_0": ("zlb2m", 0), "zlb2m_1e-3": ("zlb2m", 1e-3), "zlb2m_1e-2": ("zlb2m", 1e-2), "zlb2m_5e-2": ("zlb2m", 5e-2),
        "nkocc_1e-3": ("nkocc", 1e-3), "nkocc_5e-3": ("nkocc", 5e-3), "nkocc_1.5e-2": ("nkocc", 1.5e-2)}
FILES = {"nkm": lib.SRC["nkm_est"], "nk": lib.SRC["nk"], "zlb": lib.SRC["zlb"], "zlb2": lib.SRC["zlb2"], "bf": lib.SRC["bf"], "sw": lib.SRC["sw"],
         "borrcon": lib.SRC["borrcon"]}
CFG = {"nkm": (1.0, 40), "zlb": (0.02, 20), "zlb2": (0.02, 20), "nk": (1.0, 32), "borrcon": (0.5, 40), "sw": (0.25, 32), "bf": (1.0, 32)}
_c = {}


def load(name):
    if name in _c:
        return _c[name]
    if name in SPEC:
        fam, d = SPEC[name]
        text = gen.FAMILIES[fam](d)
        dd = Path(tempfile.mkdtemp(prefix="hoa_")); p = dd / f"{name.replace('.', '_').replace('-', '_')}.mod"; p.write_text(text)
    else:
        src = FILES[name]; dd = Path(tempfile.mkdtemp(prefix="hoa_"))
        for f in src.parent.iterdir():
            if f.is_file() and f.suffix in (".mod", ".inc", ".mat", ".m", ".csv"):
                shutil.copy(f, dd)
        p = dd / src.name
    m = pydyn.Model(pydyn.run_preprocessor(p, extra_args=("output=second",)))
    _c[name] = m
    return m


def shocks(name, m, T, seed=4, scale=None):
    sc, hz = CFG.get(name, (1.0, 40))
    if scale is not None:
        sc = scale
    sd = np.sqrt(np.diag(m.Sigma_e)) if name not in ("zlb", "zlb2") else 1.0
    return np.random.default_rng(seed).normal(size=(T, m.n_exo)) * sd * sc


def nbind(r):
    return int((np.abs(r["wedges"]).max(axis=1) > 1e-10).sum())


def o2(name, T=300, scales=(1.0, 1.6), hz=40):
    m = load(name)
    out = []
    for sc in scales:
        S = shocks(name, m, T, scale=(CFG[name][0] * sc if name in CFG else sc))
        row = dict(model=name, scale=sc)
        try:
            ob_cls = [(c.cls, round(c.approx, 5)) for c in constraint_objects(HigherOrderRegime(m, order=1, horizon=hz, check_ahead=hz).ob)]
            row["cls"] = ob_cls
            t = time.perf_counter(); a = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz).simulate(S); row["t_regime"] = (time.perf_counter() - t) / T
            a1 = HigherOrderRegime(m, order=1, horizon=hz, check_ahead=hz).simulate(S)
            row["binds"] = nbind(a); row["binds_o1"] = nbind(a1)
            row["o2_vs_o1_regime"] = float(np.abs(a["with_bound"] - a1["with_bound"]).max())
            try:
                t = time.perf_counter(); b = HigherOrderNews(m, order=2, horizon=hz, check_ahead=hz, obc_formulation="lcp").simulate(S); row["t_lcp"] = (time.perf_counter() - t) / T
                d = np.abs(a["with_bound"] - b["with_bound"]).max(axis=1)
                row["lcp_vs_regime_periods>1e-8"] = int((d > 1e-8).sum()); row["lcp_vs_regime_max"] = float(d.max())
                first = np.flatnonzero(d > 1e-8)
                row["first_diff_period"] = int(first[0]) if first.size else None
                row["wedge_sign_diff_periods"] = int(((np.abs(a["wedges"]).max(axis=1) > 1e-10) != (np.abs(b["wedges"]).max(axis=1) > 1e-10)).sum())
            except Exception as e:
                row["lcp"] = f"{type(e).__name__}: {str(e)[:60]}"
        except OBCError as e:
            row["regime_error"] = f"{e.status} period {getattr(e, 'period', None)}"
        out.append(row)
    return out


def cub(name, T=100, level=9, tol=0.9, hz=None, scale=None):
    m = load(name)
    sc, h0 = CFG.get(name, (1.0, 40)); hz = hz or h0
    S = shocks(name, m, T, scale=scale)
    kw = dict(order=2, horizon=hz, check_ahead=hz, cubature=True, qmc_level=level, regions="all", cubature_failure_tolerance=tol)
    row = dict(model=name, level=level, T=T)
    res = {}
    for nm, mk in (("regime", lambda: HigherOrderRegime(m, **kw)), ("lcp", lambda: HigherOrderNews(m, obc_formulation="lcp", **kw))):
        try:
            h = mk()
        except ValueError as e:
            row[nm] = "refused"; continue
        t = time.perf_counter()
        try:
            r = h.simulate(S)
        except OBCError as e:
            row[nm] = dict(error=e.status, period=getattr(e, "period", None), counts=getattr(e, "counts", None), msg=str(e)[:100]); continue
        dt = (time.perf_counter() - t) / T
        res[nm] = r
        row[nm] = dict(t_per_period=dt, relevant=int(r["relevant"].sum()), dropped=int(r["dropped"].sum()), binds=nbind(r),
                       share_dropped=float(r["dropped"].sum() / max(1, r["relevant"].sum())))
    if "regime" in res:
        plain = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz).simulate(S) if True else None
        row["cub_effect_regime_max"] = float(np.abs(res["regime"]["with_bound"] - plain["with_bound"]).max())
        if "lcp" in res:
            plain_l = HigherOrderNews(m, order=2, horizon=hz, check_ahead=hz, obc_formulation="lcp").simulate(S)
            row["cub_effect_lcp_max"] = float(np.abs(res["lcp"]["with_bound"] - plain_l["with_bound"]).max())
            row["regime_vs_lcp_cub_max"] = float(np.abs(res["regime"]["with_bound"] - res["lcp"]["with_bound"]).max())
    return row


def cost(name, T=40, milp=1.0):
    """Seconds per period: order 1 / order 2 (no cubature) and cubature levels 9 and 11, regime vs LCP (tolerance 0.9 so that failing points do not stop the run)."""
    m = load(name)
    sc, hz = CFG.get(name, (1.0, 40))
    S = shocks(name, m, T)
    row = dict(model=name, T=T, milp_limit=milp)
    for label, kw in (("o1", dict(order=1)), ("o2", dict(order=2)), ("cub9", dict(order=2, cubature=True, qmc_level=9)),
                      ("cub11", dict(order=2, cubature=True, qmc_level=11))):
        for nm, mk in (("regime", lambda **k: HigherOrderRegime(m, **k)), ("lcp", lambda **k: HigherOrderNews(m, obc_formulation="lcp", **k))):
            kk = dict(horizon=hz, check_ahead=hz, **kw)
            if "cubature" in kw:
                kk.update(regions="all", cubature_failure_tolerance=0.9, cubature_milp_time_limit=milp)
            try:
                h = mk(**kk)
            except ValueError:
                row[f"{label}_{nm}"] = None
                continue
            reps = 5 if label in ("o1", "o2") else 1
            best = 1e9
            for _ in range(reps):
                t = time.perf_counter()
                try:
                    r = h.simulate(S if label in ("cub9", "cub11") else np.tile(S, (5, 1)))
                except OBCError as e:
                    best = None; break
                best = min(best, (time.perf_counter() - t) / (len(S) * (1 if label.startswith("cub") else 5)))
            row[f"{label}_{nm}"] = best
            if label.startswith("cub") and best is not None:
                row[f"{label}_{nm}_dropped_share"] = float(r["dropped"].sum() / max(1, r["relevant"].sum()))
    return row


if __name__ == "__main__":
    if sys.argv[1] == "COST":
        import os
        for nm in sys.argv[2:]:
            print(json.dumps(cost(nm, T=int(os.environ.get("T", 40)), milp=float(os.environ.get("MILP", 1.0))), default=str), flush=True)
        sys.exit()
    what, names = sys.argv[1], sys.argv[2:]
    for nm in names:
        try:
            r = o2(nm) if what == "O2" else cub(nm, level=int(__import__("os").environ.get("LEVEL", 9)), T=int(__import__("os").environ.get("T", 100)))
        except Exception as e:
            r = dict(model=nm, error=f"{type(e).__name__}: {str(e)[:100]}")
        print(json.dumps(r, default=lib._clean if hasattr(lib, "_clean") else str), flush=True)
