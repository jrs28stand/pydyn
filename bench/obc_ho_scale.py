"""Scaling / multi-constraint measurements for HigherOrderNews (docs/OBC_HO_PERF.md, section 'Scaling').
Usage: python3 bench/obc_ho_scale.py probe            which models HigherOrderNews accepts
"""
from __future__ import annotations
import os, shutil, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import pydyn
from pydyn.obc_ho import HigherOrderNews

TESTS = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/ref/dynare-7.2/examples")).parent / "tests"
MODS = {
    "nk": REPO / "reference/dynareobc_octave/nk_occ.mod",
    "nkm_est": REPO / "reference/occest_nkm.mod",
    "nkm": TESTS / "occbin/filter/NKM_0_std_shocks.mod",
    "zlb": REPO / "reference/obc_zlb.mod",
    "zlb2": REPO / "reference/obc_zlb2.mod",
    "sw": REPO / "reference/obc_sw_zlb.mod",
    "rbc_est": REPO / "reference/occest_rbc.mod",
    "borrcon": REPO / "reference/occ_borrcon.mod",
    "rbc1": REPO / "reference/occ_rbc1.mod",
}

def load(name, path=None):
    path = Path(path or MODS[name])
    d = Path(tempfile.mkdtemp(prefix="obchos_"))
    if path.parent.name == "filter" or path.parent.is_dir():
        for f in path.parent.iterdir():
            if f.is_file():
                shutil.copy(f, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(d / path.name))

def padded_mod(nblocks, dst):
    """reference/occest_nkm.mod with ``nblocks`` extra blocks (4 variables, 3 states each): an AR(1) driven by one of 5 extra
    shocks and three lags of it. Estimation lines removed. n = 24 + 4 nblocks."""
    L = (REPO / "reference/occest_nkm.mod").read_text().split("\n")
    L = L[:next(i for i, l in enumerate(L) if "Estimation ------" in l)]
    def last_semicolon_after(key):
        i0 = next(i for i, l in enumerate(L) if l.strip().startswith(key))
        return next(i for i in range(i0, len(L)) if L[i].strip() == ";")
    def end_after(key, nth=0):
        idx = [i for i, l in enumerate(L) if l.strip().startswith(key)]
        i0 = idx[nth]
        return next(i for i in range(i0, len(L)) if L[i].strip() == "end;")
    names = [f"pa{j} pl1_{j} pl2_{j} pl3_{j}" for j in range(nblocks)]
    eqs = []
    for j in range(nblocks):
        eqs += [f"pa{j} = 0.9*pa{j}(-1) + pe{j % 5};", f"pl1_{j} = pa{j}(-1);", f"pl2_{j} = pl1_{j}(-1);", f"pl3_{j} = pl2_{j}(-1);"]
    zero = [f"{v} = 0;" for nm in names for v in nm.split()]
    # insert from the back so earlier indices stay valid
    sh = end_after("shocks", 0); L[sh:sh] = [f"var pe{k} = 1;" for k in range(5)]
    ss = end_after("steady_state_model"); L[ss:ss] = zero
    md = end_after("model"); L[md:md] = eqs
    ve = last_semicolon_after("varexo"); L[ve:ve] = [f"pe{k}" for k in range(5)]
    vv = last_semicolon_after("var"); L[vv:vv] = names
    dst = Path(dst); dst.write_text("\n".join(L))
    return dst


def load_padded(nblocks):
    d = Path(tempfile.mkdtemp(prefix="obcpad_"))
    f = padded_mod(nblocks, d / "pad.mod")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(f))


def probe():
    for name in MODS:
        try:
            m = load(name)
        except Exception as e:
            print(f"{name:8s} LOAD FAIL {type(e).__name__}: {str(e)[:100]}"); continue
        for kw in (dict(order=1), dict(order=2), dict(order=2, cubature=True, qmc_level=7)):
            try:
                t = time.perf_counter()
                h = HigherOrderNews(m, horizon=32, check_ahead=32, **kw)
                print(f"{name:8s} n={m.n_endo:3d} nc={h.news.n_c} {kw}: ok {1e3*(time.perf_counter()-t):.0f} ms, ns={len(h.ist)}")
            except Exception as e:
                print(f"{name:8s} n={m.n_endo:3d} {kw}: REFUSED {type(e).__name__}: {str(e)[:140]}")

def sw_cp_model():
    """reference/obc_sw_zlb.mod with the second switching equation removed (rlag = rnot always: smoothing on the shadow rate), so the
    bound switches one equation and the news-shock class accepts it. Same 32 variables."""
    src = (REPO / "reference/obc_sw_zlb.mod").read_text()
    a = src.index("[name = 'Auxiliary notional interest rate',bind='zlb']")
    b = src.index("    [name='Output gap check']")
    d = Path(tempfile.mkdtemp(prefix="obcsw_"))
    (d / "sw_cp.mod").write_text(src[:a] + "[name = 'Auxiliary notional interest rate']\nrlag = rnot;\n" + src[b:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(d / "sw_cp.mod"))


def get_model(name):
    if name == "sw_cp":
        return sw_cp_model()
    if name.startswith("pad"):
        return load_padded(int(name[3:]))
    return load(name)


def shocks_for(m, name, n, scale, seed=4):
    from scipy.io import loadmat
    if name == "nk":
        return loadmat(REPO / "reference/dynareobc_octave/shocks.mat", squeeze_me=True)["ShockSequence"].reshape(-1, 1)[:n] * scale
    sd = np.sqrt(np.diag(m.Sigma_e)); sd[sd < 1e-9] = 0.01
    return np.random.default_rng(seed).normal(size=(n, m.n_exo)) * sd * scale


def calibrate(m, name, h_kw):
    """Largest scale (of 0.25 1 2 4 8 16 x base) at which order 2 solves 300 periods, with the share of binding periods."""
    from pydyn.obc import OBCError
    best = None
    for sc in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32):
        h = HigherOrderNews(m, order=2, **h_kw)
        try:
            r = h.simulate(shocks_for(m, name, 300, sc))
        except OBCError as e:
            best = best or (sc, 0.0, "fail@%g" % sc)
            break
        share = float((r["wedges"].max(axis=1) > 1e-10).mean())
        best = (sc, share, "ok")
        if share >= 0.3:
            break
    return best


def census(m, h2, qs, levels=(9,), budget=40.0):
    """For the base-path gap vectors qs: per level, relevant points per period and how perturbed points fare:
    scalar active set fails (a), then MILP finds a solution (b) / none (c)."""
    import pydyn.obc_ho as oh
    from pydyn.obc import solve_active_set, solve_news
    out = {}
    for lv in levels:
        h = HigherOrderNews(m, order=2, cubature=True, qmc_level=lv, regions="all", horizon=h2.T, check_ahead=h2.N - h2.T)
        rel = pts = a = b = c = 0
        nraise = 0
        tm = 0.0
        t_start = time.perf_counter()
        done = 0
        for q in qs:
            if time.perf_counter() - t_start > budget:
                break
            done += 1
            idx = h._relevant_points(q)
            rel += idx.size; pts += h.xi.shape[1]
            pr = q[:, None] + h.R @ h.xi[:, idx]
            bad = False
            for j in range(idx.size):
                p = pr[:, j]
                if np.all(p >= -1e-8 * max(1.0, np.abs(p).max())):
                    continue
                if solve_active_set(h.news.Mo, p, h.nws) is None:
                    a += 1
                    t = time.perf_counter()
                    y, _ = solve_news(h.news.Mo, p, h.nws, horizons=sorted({max(1, h.T // 4), max(1, h.T // 2), h.T}), n_c=h.n_c)
                    tm += time.perf_counter() - t
                    if y is None:
                        c += 1; bad = True
                    else:
                        b += 1
            nraise += bad
        out[lv] = dict(R_dim=h.R.shape[1], relevant_per_period=rel / max(done, 1), points=h.xi.shape[1], as_fail=a, as_fail_milp_ok=b,
                       no_solution=c, periods_raising=nraise, periods=done, milp_seconds=tm, relevant_total=rel)
    return out


def run(name, horizon=32, maxq=150, levels=(9, 12), cases=("regions=1", "regions=4", "regions=all"), reps=3, scale=None, out=None, budget=40.0):
    from pydyn.obc import OBCError
    import pydyn.obc_ho as oh
    from pydyn.korder import k_order
    m = get_model(name)
    kw = dict(horizon=horizon, check_ahead=horizon)
    rows = {"model": name, "n": m.n_endo, "n_exo": m.n_exo, "horizon": horizon}
    def tmin(f, r=reps):
        ts = []
        for _ in range(r):
            t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
        return min(ts)
    # construction
    rows["constr_o1_ms"] = 1e3 * tmin(lambda: HigherOrderNews(m, order=1, **kw))
    h2 = HigherOrderNews(m, order=2, **kw)
    rows["constr_o2_ms"] = 1e3 * tmin(lambda: HigherOrderNews(m, order=2, **kw))
    rows["ns"], rows["nz"], rows["Gz_MB"] = len(h2.ist), h2._Gz.shape[1], h2._Gz.nbytes / 1e6
    rows["gap_maps_ms"] = 1e3 * tmin(h2._build_gap_maps)
    rows["k_order_ms"] = 1e3 * tmin(lambda: k_order(m, order=2, ys=h2.ob.ys, params=h2.ob.params, pruning=True, Sigma_e=h2.Sigma))
    rows["N"], rows["nc"] = h2.N, h2.n_c
    oh._SOBOL_CACHE.clear()
    t = time.perf_counter(); hc = HigherOrderNews(m, order=2, cubature=True, qmc_level=9, **kw); rows["constr_cub9_first_ms"] = 1e3 * (time.perf_counter() - t)
    rows["constr_cub9_ms"] = 1e3 * tmin(lambda: HigherOrderNews(m, order=2, cubature=True, qmc_level=9, **kw))
    rows["R_dim"] = hc.R.shape[1]
    # shocks
    sc = scale
    if sc is None:
        sc, share, st = calibrate(m, name, kw); rows["calib"] = (sc, share, st)
    S = shocks_for(m, name, 300, sc)
    rows["shock_scale"] = sc
    # order 1, 2 per period
    for lbl, order in (("o1", 1), ("o2", 2)):
        h = HigherOrderNews(m, order=order, **kw)
        try:
            h.simulate(S[:20]); ts = []
            for _ in range(reps):
                t = time.perf_counter(); r = h.simulate(S); ts.append((time.perf_counter() - t) / len(S) * 1e6)
            rows[f"{lbl}_us"] = min(ts); rows[f"{lbl}_bind_share"] = float((r["wedges"].max(axis=1) > 1e-10).mean())
        except OBCError as e:
            rows[f"{lbl}_us"] = f"raise {e}"
    # base-path gap vectors
    h1 = HigherOrderNews(m, order=2, **kw)
    qs = []
    orig = h1.news_shocks
    h1.news_shocks = lambda q: (qs.append(q.copy()), orig(q))[1]
    try:
        h1.simulate(S)
    except OBCError:
        pass
    qs = qs[:maxq]
    rows["n_q"] = len(qs)
    rows["census"] = census(m, h2, qs[:min(len(qs), 60)], levels=levels, budget=budget)
    # cubature news_shocks per call
    for lv in levels:
        for cs in cases:
            rg = "all" if cs.endswith("all") else int(cs.split("=")[1])
            h = HigherOrderNews(m, order=2, cubature=True, qmc_level=lv, regions=rg, **kw)
            nq = len(qs) if (lv <= 9 or rg == 1) else min(len(qs), 40)
            for q in qs[:3]:                                    # warm-up
                try: h.news_shocks(q)
                except OBCError: pass
            h.stats.update(lcp=0, cubature_points=0, relevant_points=0, milp=0, no_solution=0, batch_fail=0, passes=0, groups=0, tail=0, cols=0, calls=0)
            ok_t, bad_t, nbad, ndone = 0.0, 0.0, 0, 0
            t_case = time.perf_counter()
            for q in qs[:nq]:
                if time.perf_counter() - t_case > budget:
                    break
                ndone += 1
                t = time.perf_counter()
                try:
                    h.news_shocks(q); ok_t += time.perf_counter() - t
                except OBCError:
                    bad_t += time.perf_counter() - t; nbad += 1
            nq = ndone
            d = dict(h.stats)
            nok = nq - nbad
            ops = sum(v.nbytes for k, v in h._fac_cache.items() if isinstance(k, tuple) and k[0] == "_ops")
            d.update(us_ok=1e6 * ok_t / max(nok, 1), us_raise=1e6 * bad_t / max(nbad, 1), raised=nbad, nq=nq,
                     sets=sum(1 for k in h._fac_cache if not isinstance(k, tuple)), cache_MB=ops / 1e6)
            rows[f"L{lv}_{cs}"] = d
    if out:
        import json
        with open(out, "a") as f:
            f.write(json.dumps(rows, default=str) + "\n")
    return rows


def pp(rows):
    for k, v in rows.items():
        if isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
        else:
            print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("model", nargs="?")
    ap.add_argument("--horizon", type=int, default=32); ap.add_argument("--maxq", type=int, default=150)
    ap.add_argument("--levels", default="9,12"); ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe()
    elif a.cmd == "run":
        pp(run(a.model, a.horizon, a.maxq, tuple(int(x) for x in a.levels.split(",")), scale=a.scale, out=a.out, reps=a.reps))
