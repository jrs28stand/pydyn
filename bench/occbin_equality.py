"""Exact equality of the current pydyn.occbin.Occbin with the frozen copy bench/occbin_before.py, in one process:
50 parameter sets x 200 periods per model (obc_zlb, obc_zlb2, Dynare's NKM, reference/obc_sw_zlb.mod), at the shock scale of
bench/obc_bench.py and at twice that scale (which produces failures).  Compared with np.array_equal: piecewise and linear
paths, regime histories, iteration counts, periodic flags, the rules ``ss`` of every shock period, the diagnostics, and the error
(class, error flag, period) when a failure is raised.   python3 bench/occbin_equality.py [--sets 50] [--periods 200]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import obc_bench as B  # noqa: E402
import occbin_before as OB  # noqa: E402
from pydyn.occbin import Occbin  # noqa: E402


def run(obj, shocks, periods):
    try:
        return obj.simulate(shocks, periods=periods, check_ahead_periods=B.OCCBIN_CHECK_AHEAD), None
    except Exception as err:  # noqa: BLE001 - the frozen module has its own OccbinError classes
        if not hasattr(err, "error_flag"):
            raise
        return None, (type(err).__name__, err.error_flag, err.period, str(err))


def same(a, b):
    if a is None or b is None:
        return a is b
    ok = all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("piecewise", "linear", "shocks", "ys"))
    ok &= a.regime_history == b.regime_history and a.iterations == b.iterations and a.periodic == b.periodic
    ok &= a.error_flag == b.error_flag and a.exo_pos == b.exo_pos
    ok &= len(a.ss) == len(b.ss) and all(np.array_equal(x, y) for sa, sb in zip(a.ss, b.ss) for x, y in zip(sa, sb))
    ok &= len(a.diagnostics) == len(b.diagnostics) and all(
        all(np.array_equal(da[k], db[k]) or (k == "selected" and da[k] == db[k]) for k in da)
        for da, db in zip(a.diagnostics, b.diagnostics))
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=50)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--models", default="zlb,zlb2,nkm,sw")
    a = ap.parse_args()
    bad = 0
    for name in a.models.split(","):
        m = B.load_model(name)
        cfg = B.MODELS[name]
        sets = [None] + B.perturbation_sets(m, cfg["params"], a.sets - 1, B.SEED, size=0.02)
        for scale in (1.0, 2.0):
            shocks = B.draw_shocks(m, a.periods, cfg["s"] * scale, B.SEED)
            n_eq = n_err = n_bind = 0
            for p in sets:
                new = Occbin(m, p)
                old = OB.clone(new)
                ra, ea = run(new, shocks, a.periods)
                rb, eb = run(old, shocks, a.periods)
                ok = ea == eb and same(ra, rb)
                n_eq += ok
                n_err += ea is not None
                if ra is not None:
                    n_bind += B.binding_flags(ra).any()
                if not ok:
                    bad += 1
            print(f"{name} shock scale x{scale}: {n_eq}/{len(sets)} identical; {n_err} runs end in an OccbinError (same in both); "
                  f"{n_bind} runs with a binding period", flush=True)
    print("ALL IDENTICAL" if bad == 0 else f"DIFFERENCES: {bad}")
    sys.exit(bool(bad))


if __name__ == "__main__":
    main()
