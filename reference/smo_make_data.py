"""Simulate the data sets used by tests/test_smoother.py (fixed seeds) and write them as Dynare
``.m`` data files next to this script. Run from the repository root:

    DYNARE_PREPROCESSOR=... python3 reference/smo_make_data.py

The files are committed; rerunning is only needed to change the data.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pydyn  # noqa: E402


def write_m(path, series, header):
    lines = [f"% {h}" for h in header]
    for name, v in series.items():
        lines.append(f"{name} = [")
        lines += ["NaN" if np.isnan(x) else repr(float(x)) for x in v]
        lines.append("];")
    path.write_text("\n".join(lines) + "\n")


def simulate(mod, T, seed, burn=200):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(HERE / mod, tmp)
        m = pydyn.Model(pydyn.run_preprocessor(Path(tmp) / mod))
    sol = pydyn.first_order(m)
    rng = np.random.default_rng(seed)
    u = rng.multivariate_normal(np.zeros(m.n_exo), m.Sigma_e, size=T + burn)
    dev = sol.simulate(u)[burn:]
    return m, sol, dev, rng


def main():
    hdr = ["Simulated data for pydyn's smoother tests (reference/smo_make_data.py); not real data."]
    # --- cash-in-advance model, two observables, levels and logs
    m, sol, dev, _ = simulate("smo_fs.mod", 120, 20261)
    lev = {v: sol.ys[m.endo_names.index(v)] * np.exp(dev[:, m.endo_names.index(v)] / sol.ys[m.endo_names.index(v)])
           for v in ("gp_obs", "gy_obs")}
    write_m(HERE / "smo_fs_data.m", lev, hdr)
    write_m(HERE / "smo_fs_logdata.m", {k: np.log(v) for k, v in lev.items()}, hdr + ["Logs of smo_fs_data.m."])

    # --- linear NK model, three observables, measurement error on two, missing observations
    m, sol, dev, rng = simulate("smo_me.mod", 100, 20262)
    d = {v: dev[:, m.endo_names.index(v)].copy() for v in ("yobs", "piobs", "iobs")}
    for v, var in m.measurement_errors.items():
        d[v] += np.sqrt(var) * rng.standard_normal(100)
    d["yobs"] += 0.4          # non-zero sample means, used by the prefilter case
    d["piobs"] += 0.5
    d["iobs"] += 1.0
    d["yobs"][[2, 17, 18, 40, 59]] = np.nan
    d["piobs"][[7, 17, 40, 41, 99]] = np.nan
    d["iobs"][[0, 17, 30, 59]] = np.nan          # period 18 is missing altogether
    write_m(HERE / "smo_me_data.m", d, hdr)

    # --- unit-root model
    m, sol, dev, rng = simulate("smo_ur.mod", 90, 20263, burn=20)
    d = {v: dev[:, m.endo_names.index(v)].copy() + off for v, off in (("y", 0.0), ("x", 0.0), ("y2", 0.0), ("w", 0.0))}
    d["x"] += 0.15 * rng.standard_normal(90)      # measurement error, used by one case
    for v in ("y", "x", "y2"):
        d[v][[0, 9]] = np.nan                      # periods 1 and 10 are missing altogether
    d["x"][[3, 25, 26, 60]] = np.nan
    d["y"][[19, 60, 77]] = np.nan
    d["y2"][[5, 33]] = np.nan
    d["w"][[20]] = np.nan                         # with y missing in period 20: see smo_ur.mod, case 6
    write_m(HERE / "smo_ur_data.m", d, hdr)


if __name__ == "__main__":
    main()
