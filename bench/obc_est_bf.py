"""Inversion-filter likelihood with the news solver against OccBin's on Dynare's bf.mod and bf_ppf.mod (two equations
switch).  Data: reference/occest_bf_data.mat.  Writes bench/results/est_bf.json; prints a Markdown table.

    PYDYN_DYNARE_TESTS=.../tests python3 bench/obc_est_bf.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pydyn  # noqa: E402
from pydyn.occbin_estimation import OccbinEstimation  # noqa: E402

T = Path(os.environ.get("PYDYN_DYNARE_TESTS", "/home/claude/ref/dynare-7.2/tests"))
FILES = {"bf.mod (estimation/slice)": T / "estimation/slice/bf.mod", "bf_ppf.mod (occbin/ppf)": T / "occbin/ppf/bf_ppf.mod"}


def main():
    d = loadmat(ROOT / "reference" / "occest_bf_data.mat", squeeze_me=True)
    data = np.column_stack([d["q"], d["r"]])
    kw = dict(posterior_sampling_method="random_walk_metropolis_hastings", posterior_sampler_options=())
    out, lines = {}, []
    for name, src in FILES.items():
        tmp = Path(tempfile.mkdtemp())
        shutil.copy(src, tmp / src.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pydyn.Model(pydyn.run_preprocessor(tmp / src.name))
            mk = lambda **k: OccbinEstimation(m, data={"q": data[:, 0], "r": data[:, 1]}, nobs=100, first_obs=1,  # noqa: E731
                                              occbin={"likelihood_inversion_filter": True}, **kw, **k)
            eo, en = mk(), mk(obc_method="news")
        rng = np.random.default_rng(1)
        x0 = np.array(eo.x0)
        rows = []
        for k in range(8):
            th = x0 if k == 0 else x0 * np.exp(0.05 * rng.standard_normal(5))
            to, tn = [], []
            for _ in range(3):
                t = time.perf_counter(); a = eo.log_posterior(th); to.append(time.perf_counter() - t)
                t = time.perf_counter(); b = en.log_posterior(th); tn.append(time.perf_counter() - t)
            ra, rb = eo.last_result, en.last_result
            row = dict(k=k, lp_occbin=a, lp_news=b, finite=bool(np.isfinite(a)), occbin_s=min(to), news_s=min(tn),
                       code_occbin=eo.last_code, code_news=en.last_code, trial_failures_news=dict(en.last_news_solver.errors))
            if np.isfinite(a):
                row.update(diff=abs(a - b), same_regimes=ra.regime_history == rb.regime_history,
                           max_shock_diff=float(np.abs(ra.filtered_errs - rb.filtered_errs).max()),
                           binding_periods=sum(r[0][0][0] for r in rb.regime_history))
            rows.append(row)
        out[name] = rows
        lines.append(f"\n{name}\n\n| vector | log posterior OccBin | news | abs. difference | same regimes | binding periods | max shock difference | OccBin s | news s |\n|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["finite"]:
                lines.append(f"| {r['k']} | {r['lp_occbin']:.8f} | {r['lp_news']:.8f} | {r['diff']:.1e} | {r['same_regimes']} | {r['binding_periods']} | "
                             f"{r['max_shock_diff']:.1e} | {r['occbin_s']:.3f} | {r['news_s']:.3f} |")
            else:
                lines.append(f"| {r['k']} | -inf (code {r['code_occbin']}) | -inf (code {r['code_news']}) | - | - | - | - | {r['occbin_s']:.3f} | {r['news_s']:.3f} |")
    (ROOT / "bench" / "results" / "est_bf.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
