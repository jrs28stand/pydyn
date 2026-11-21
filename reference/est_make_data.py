"""Simulated data sets used by tests/test_estimation.py (est_nk_*.m, est_nk_csv.csv, est_ur_*.m).

Usage: DYNARE_PREPROCESSOR=... python reference/est_make_data.py OUTPUT_DIR
The files in reference/ were produced by this script; rerunning it reproduces them exactly.
After changing the data, regenerate the Dynare references with reference/r_estimation.m.
"""
import csv
import pathlib
import shutil
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pydyn  # noqa: E402


def write_m(path, cols, header):
    with open(path, "w") as fh:
        fh.write(f"% {header}\n")
        for name, v in cols.items():
            fh.write(f"{name} = [\n" + "\n".join("NaN" if np.isnan(x) else f"{x:.10f}" for x in v) + "\n];\n")


def main(out):
    out = pathlib.Path(out)
    rng = np.random.default_rng(20261120)
    tmp = pathlib.Path(tempfile.mkdtemp())
    shutil.copy(HERE / "est_nk.mod", tmp)
    shutil.copy(HERE / "est_nk_data.m", tmp)          # only so that the datafile option resolves
    # New Keynesian model: simulate at the calibration, add a measurement error to yobs
    m = pydyn.Model(pydyn.run_preprocessor(tmp / "est_nk.mod"))
    sol = pydyn.first_order(m)
    T = 120
    u = rng.multivariate_normal(np.zeros(3), m.Sigma_e, size=T + 200)
    y = sol.simulate(u)[200:] + sol.ys
    cols = {v: y[:, m.endo_names.index(v)].copy() for v in m.varobs}
    cols["yobs"] += 0.1 * rng.standard_normal(T)
    write_m(out / "est_nk_data.m", cols, "simulated data for est_nk.mod (pydyn, seed 20261120)")
    miss = {k: v.copy() for k, v in cols.items()}
    miss["yobs"][[3, 4, 17, 60]] = np.nan
    miss["piobs"][[4, 30, 31, 32]] = np.nan
    miss["robs"][[4, 75]] = np.nan
    write_m(out / "est_nk_miss.m", miss, "est_nk_data with missing observations")
    with open(out / "est_nk_csv.csv", "w", newline="") as fh:   # not est_nk_data.csv: Dynare refuses
        w = csv.writer(fh)                                      # two data files with the same stem
        w.writerow(list(cols))
        for t in range(T):
            w.writerow([f"{cols[k][t]:.10f}" for k in cols])
    # unit-root model (parameters of est_ur.mod)
    T = 100
    eg, ey, ex = 0.4 * rng.standard_normal(T), 0.2 * rng.standard_normal(T), 0.3 * rng.standard_normal(T)
    g, yy, x = np.zeros(T), np.zeros(T), np.zeros(T)
    gl, yl, xl = 0.0, 3.0, 0.0
    for t in range(T):
        g[t] = 0.6 * gl + eg[t]
        yy[t] = yl + g[t] + ey[t]
        x[t] = 0.5 * xl + 0.3 * g[t] + ex[t]
        gl, yl, xl = g[t], yy[t], x[t]
    cols = {"yobs": yy, "xobs": x + 0.05 * rng.standard_normal(T)}
    write_m(out / "est_ur_data.m", cols, "simulated data for est_ur.mod (pydyn, seed 20261120)")
    miss = {k: v.copy() for k, v in cols.items()}
    miss["yobs"][[0, 6, 40]] = np.nan
    miss["xobs"][[1, 6, 55]] = np.nan
    write_m(out / "est_ur_miss.m", miss, "est_ur_data with missing observations")


if __name__ == "__main__":
    main(sys.argv[1])
