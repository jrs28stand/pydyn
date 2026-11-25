"""Formulation study, part E (estimation) and F (simulation).  python3 soak/approx/study.py E|F [--budget S] [--replay i]
Same checkpoint design as soak/run.py: task i is a pure function of i; state in soak/approx/state/<part>.json, results in soak/approx/results/<part>.jsonl."""
import argparse, json, os, sys, time, warnings
from pathlib import Path
sys.path.insert(0, "soak"); sys.path.insert(0, "soak/approx")
import numpy as np
import lib, part_a, part_cd, gen, models
from pydyn.occbin import Occbin
from pydyn.obc import constraint_objects

OUT = Path("soak/approx"); (OUT / "state").mkdir(exist_ok=True); (OUT / "results").mkdir(exist_ok=True)
NAMES = ["nkm", "nkms"] + list(models.SPEC)
METHODS = [("occbin", None), ("lcp", "lcp"), ("regime", "regime"), ("hybrid", "hybrid")]


def est_of(name):
    return part_a.make_est(name) if name in ("nkm", "nkms") else models.get(name)


def ev(e, th, meth, form):
    e.obc_method = "occbin" if meth == "occbin" else "news"
    if form:
        e.obc_formulation = form
    e.obc_horizon = 40
    e.last_news_solver = None
    t = time.perf_counter()
    try:
        with lib.alarm(60):
            lp = float(e.log_posterior(th))
    except lib.Alarm:
        return dict(lp=None, code=-1, t=60.0)
    except Exception as exc:                                   # noqa: BLE001
        return dict(lp=None, code=-2, t=time.perf_counter() - t, exc=type(exc).__name__ + ": " + str(exc)[:100])
    r = dict(lp=lp if np.isfinite(lp) else None, code=int(e.last_code), t=round(time.perf_counter() - t, 4))
    if np.isfinite(lp) and e.last_result is not None:
        rh = e.last_result.regime_history
        try:
            r["nb"] = int(sum(1 for reg in rh if any(c[0][0] for c in reg)))
        except Exception:                                      # noqa: BLE001
            pass
    s = getattr(e, 'last_news_solver', None) if meth != 'occbin' else None
    if s is not None and form == "hybrid" and hasattr(s, "rb"):
        r["h"] = list(s.rb.hstats.values())
    return r


def vectors(e, rng, cat):
    if cat == 0:
        return [("prior", part_a.prior_vec(e, rng))]
    if cat == 1:
        sc = float(rng.choice([0.01, 0.03, 0.1, 0.3, 0.6]))
        return [(f"perturb{sc}", e.x0v * np.exp(sc * rng.standard_normal(e.x0v.size)))]
    x = e.x0v * np.exp(float(rng.choice([0.03, 0.1, 0.3])) * rng.standard_normal(e.x0v.size))
    y = part_a.prior_vec(e, rng) if rng.integers(2) else e.x0v * np.exp(0.3 * rng.standard_normal(e.x0v.size))
    return [("line", x + t * (y - x)) for t in np.linspace(-0.1, 1.1, 13)]


def task_E(i):
    name = NAMES[i % len(NAMES)]; j = i // len(NAMES)
    e = est_of(name); rng = lib.rng_for("E", i)
    out = []
    for kind, th in vectors(e, rng, j % 3):
        out.append(dict(kind=kind, th=[float(f"{v:.9g}") for v in th], **{m: ev(e, th, m, f) for m, f in METHODS}))
    return dict(i=i, model=name, v=out)


def task_F(i):
    """Long simulation: occbin vs lcp vs regime vs hybrid, 1000-period segments carried from the occbin path."""
    name = NAMES[i % len(NAMES)]; rng = lib.rng_for("F", i)
    if name in ("nkm", "nkms"):
        name = "nkm" if name == "nkms" else "nkm_est"
        m = lib.load_model(name); base = 1.0
    else:
        m = models.get(name).model; base = 1.0
    params = lib.perturb_params(m, rng, float(rng.choice([0.0, 0.0, 0.01, 0.03])))
    ob = Occbin(m, params)
    sd = np.sqrt(np.maximum(np.diag(np.array(m.Sigma_e, float)), 0.0))
    sscale = float(np.exp(rng.uniform(np.log(0.4), np.log(1.2))))
    rec = dict(i=i, model=name, sscale=round(sscale, 3), seg=[])
    init, t0 = None, time.time()
    for seg in range(30):
        if time.time() - t0 > 40:
            break
        S = rng.standard_normal((1000, m.n_exo)) * sd * sscale
        res = {}
        for meth, form in METHODS:
            kw = {} if meth == "occbin" else dict(obc_method="news", obc_horizon=40, obc_formulation=form)
            t = time.perf_counter()
            try:
                with lib.alarm(100):
                    r = ob.simulate(S, periods=1000, init=init, **kw)
                res[meth] = r
            except lib.Alarm:
                res[meth] = "timeout"
            except Exception as exc:                           # noqa: BLE001
                res[meth] = type(exc).__name__
            res[meth + "_t"] = round(time.perf_counter() - t, 3)
        a = res["occbin"]
        row = {k: res[k] for k in res if k.endswith("_t")}
        row["nb"] = int(sum(1 for h in a.regime_history if any(c[0][0] for c in h))) if not isinstance(a, str) else None
        for meth in ("lcp", "regime", "hybrid"):
            b = res[meth]
            if isinstance(a, str) or isinstance(b, str):
                row[meth] = ("occbin:" + a if isinstance(a, str) else "ok") + "|" + (b if isinstance(b, str) else "ok")
            else:
                d = np.abs(a.piecewise - b.piecewise).max(axis=1)
                row[meth] = dict(maxd=float(d.max()), nbad=int((d > 1e-8 * max(1.0, np.abs(a.piecewise - ob.ys).max())).sum()),
                                 rh=int(sum(1 for t_ in range(1000) if a.regime_history[t_] != b.regime_history[t_])))
        rec["seg"].append(row)
        if isinstance(a, str) and sum(isinstance(x.get('lcp'), str) and x['lcp'].startswith('occbin:') for x in rec['seg']) >= 3:
            break
        init = None if isinstance(a, str) else a.piecewise[-1] - ob.ys
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("part"); ap.add_argument("--budget", type=float, default=450); ap.add_argument("--replay", type=int)
    a = ap.parse_args(); fn = task_E if a.part == "E" else task_F
    if a.replay is not None:
        print(json.dumps(fn(a.replay), default=lib._clean)[:3000]); return
    sp = OUT / "state" / f"{a.part}.json"
    st = json.loads(sp.read_text()) if sp.exists() else dict(next=0, seconds=0.0)
    t_start = time.time()
    while time.time() - t_start < a.budget:
        i = st["next"]; t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = fn(i)
        rec["sec"] = round(time.time() - t0, 2)
        with open(OUT / "results" / f"{a.part}.jsonl", "a") as f:
            f.write(json.dumps(rec, default=lib._clean, separators=(",", ":")) + "\n"); f.flush()
        st["next"] = i + 1; st["seconds"] += time.time() - t0
        tmp = sp.with_suffix(".tmp"); tmp.write_text(json.dumps(st)); os.replace(tmp, sp)
    print(f"[{a.part}] next {st['next']} seconds {st['seconds']:.0f}")


if __name__ == "__main__":
    main()
