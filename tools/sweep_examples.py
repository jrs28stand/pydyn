"""Load every Dynare example with pydyn and compare with saved Dynare results where they exist.

Usage: python tools/sweep_examples.py <dir with one sub-folder per example group, containing the
.mod files and ref_<name>.mat files written by Dynare> 
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pydyn  # noqa: E402


def nk_fn():
    spec = importlib.util.spec_from_file_location("nk_ss", ROOT / "examples" / "nk_baseline_steadystate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.steady_state


def collard_helper(alpha, beta, delta, psi, theta):
    kh = ((1 / beta - (1 - delta)) / alpha) ** (1 / (alpha - 1))
    f = lambda h: 1 - (((kh * h) ** (alpha - 1)) * h ** (1 - alpha)  # noqa: E731
                       - ((1 - alpha) * kh ** alpha / (theta * h ** psi)) / (kh * h) + (1 - delta))
    return float(fsolve(f, 0.2, xtol=1e-14)[0])


EXTRA = {"nk_baseline": dict(steady_state_fn=nk_fn()),
         "collard_2001_analytical_steady_state":
             dict(functions={"collard_2001_analytical_steady_state_helper": collard_helper})}


def check(mod: Path):
    name = mod.stem
    out = {"name": f"{mod.parent.name}/{name}"}
    ref_file = mod.parent / f"ref_{name}.mat"
    ref = loadmat(ref_file, squeeze_me=True) if ref_file.exists() else {}
    out["dynare"] = "ok" if ref.get("ok") else ("partial" if "ys" in ref else "no reference")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = pydyn.Model(pydyn.run_preprocessor(mod), **EXTRA.get(name, {}))
        out["skipped"] = "; ".join(sorted({str(x.message)[:140] for x in w if "not handled" in str(x.message)}))
    except Exception as e:  # noqa: BLE001
        pac = "dseries" in str(e) or "pac" in mod.stem
        out["status"] = "not supported" if pac else "load failed"
        out["detail"] = "PAC / semi-structural models (dseries)" if pac else f"{type(e).__name__}: {str(e)[:110]}"
        return out
    kinds = {s["statementName"] for s in m.statements}
    if kinds & {"ramsey_model", "ramsey_policy", "discretionary_policy", "osr"}:
        out["status"], out["detail"] = "not supported", "optimal policy"
        return out
    if any(k.startswith("heterogeneity") for k in kinds) or "heterogeneity" in str(mod.parent):
        out["status"], out["detail"] = "not supported", "heterogeneous agents"
        return out
    try:
        params = np.asarray(ref["params"], float) if "params" in ref and np.all(np.isfinite(ref["params"])) else None
        ys, p = pydyn.steady_state(m, params=params)
    except Exception as e:  # noqa: BLE001
        out["status"], out["detail"] = "steady state failed", f"{type(e).__name__}: {str(e)[:110]}"
        return out
    notes = []
    opts = {**m.options.get("stoch_simul", {}), **m.options.get("estimation", {})}
    ref_ys = ref.get("dr_ys", ref.get("ys"))
    if ref_ys is not None and np.shape(ref_ys) == ys.shape:
        ref_ys = np.asarray(ref_ys, float)
        d = np.max(np.abs(ys - ref_ys))
        if d > 1e-3 and np.all(ys > 0) and np.max(np.abs(np.log(ys) - ref_ys)) < 1e-6:
            d = np.max(np.abs(np.log(ys) - ref_ys))            # Dynare stored logs (loglinear)
        notes.append(f"ys {d:.1e}")
        numerical = m.steady_state_model is None and m.steady_state_fn is None
        if numerical and 1e-10 < d < 1e-4:
            # Dynare's nonlinear solver stops early; evaluate the rules at its steady state
            ys = ref_ys
            notes.append("rules at Dynare's ys")
    if "perfect_foresight_with_expectation_errors_solver" in kinds:
        out["status"], out["detail"] = "not supported", "perfect foresight with expectation errors; " + ", ".join(notes)
        return out
    if "perfect_foresight_solver" in kinds:
        try:
            pf = pydyn.perfect_foresight(m)
            d = np.max(np.abs(pf.endo.T - ref["endo_simul"])) if "endo_simul" in ref else np.nan
            out["status"] = "matches Dynare" if d < 1e-6 else "MISMATCH"
            out["detail"] = f"perfect foresight path {d:.1e} ({pf.iterations} Newton steps)"
        except Exception as e:  # noqa: BLE001
            out["status"], out["detail"] = "solution failed", f"{type(e).__name__}: {str(e)[:110]}"
        return out
    try:
        sol = pydyn.first_order(m, ys=ys, params=p)
    except Exception as e:  # noqa: BLE001
        out["status"], out["detail"] = "solution failed", f"{type(e).__name__}: {str(e)[:110]}"
        return out
    if "ghx" in ref:
        ghx, ghu = sol.ghx, sol.ghu
        if opts.get("loglinear"):
            ghx = ghx * ys[sol.state_var][None, :] / ys[sol.order_var][:, None]
            ghu = ghu / ys[sol.order_var][:, None]
        rx, ru = np.atleast_2d(ref["ghx"]), np.asarray(ref["ghu"], float).reshape(ghu.shape)
        if rx.shape != ghx.shape:
            rx = rx.reshape(ghx.shape)
        d = max(np.max(np.abs(ghx - rx)), np.max(np.abs(ghu - ru)))
        notes.append(f"rules {d:.1e}")
        ok = d < 1e-6
    else:
        ok = None
    order = m.options.get("stoch_simul", {}).get("order", 1)
    if order >= 2 and "ghxx" in ref and m.max_deriv_order >= 2:
        try:
            dr = pydyn.k_order(m, order=2, ys=ys, params=p)
            d2 = max(np.max(np.abs(dr[k] - np.asarray(ref[k], float).reshape(dr[k].shape)))
                     for k in ("ghxx", "ghxu", "ghuu", "ghs2"))
            notes.append(f"order 2 {d2:.1e}")
            ok = ok and d2 < 1e-6
        except Exception as e:  # noqa: BLE001
            notes.append(f"order 2 failed: {str(e)[:60]}")
            ok = False
    out["status"] = {True: "matches Dynare", False: "MISMATCH", None: "solved, no reference"}[ok]
    out["detail"] = ", ".join(notes)
    return out


if __name__ == "__main__":
    base = Path(sys.argv[1])
    rows = [check(f) for f in sorted(base.glob("*/*.mod"))]
    for r in rows:
        print(f"{r['name']:58s} | {r['status']:20s} | {r.get('detail', '')}")
        if r.get("skipped"):
            print(f"{'':58s} |   {r['skipped']}")
