"""Markdown tables for docs/OBC_ESTIMATION.md from bench/results/est_*.json.   python3 bench/make_est_tables.py [CASE START]"""
import json
import sys
from pathlib import Path

import numpy as np

import os
R = Path(os.environ.get("EST_RESULTS", Path(__file__).resolve().parent / "results"))
M = ("occbin", "news")


def f(x, nd=6):
    return "-" if x is None else (f"{x:.{nd}f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.3e}")


def vec(v, nd=5):
    return "(" + ", ".join(f"{a:.{nd}g}" for a in v) + ")"


def _pd(H):
    if H is None:
        return False
    H = np.array(H, float)
    return bool(np.all(np.isfinite(H)) and np.all(np.linalg.eigvalsh((H + H.T) / 2) > 0))


def mode_tables(case, start):
    p = R / f"est_{case}_{start}_mode.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    out = [f"#### Posterior mode, {case}, start = {start}", "",
           "| | OccBin | news |", "|---|---|---|"]
    rows = [("mode", lambda m: vec(d[m]["x"])), ("log posterior at the mode", lambda m: f(d[m]["log_posterior"], 7)),
            ("optimiser", lambda m: d[m]["message"]), ("likelihood evaluations (incl. Hessian)", lambda m: str(d[m]["evaluations"])),
            ("evaluations that failed (-inf)", lambda m: str(d[m]["failed_evaluations"])),
            ("wall time of the run (s)", lambda m: f"{d[m]['wall_s']:.1f}"),
            ("Hessian (finite differences, in the mode run) positive definite", lambda m: str(_pd(d[m].get("hessian")))),
            ("posterior sd from the Hessian", lambda m: vec(d[m]["std"]) if _pd(d[m].get("hessian")) else "n/a (Hessian not positive definite)"),
            ("Laplace approximation", lambda m: f(d[m]["laplace"], 6))]
    for name, fn in rows:
        out.append(f"| {name} | {fn('occbin')} | {fn('news')} |")
    c = d["comparison"]
    dx = np.abs(np.array(d["occbin"]["x"]) - np.array(d["news"]["x"]))
    sd = np.array(d["occbin"]["std"]) if d["occbin"]["std"] else None
    out += ["", f"Same path: {c['identical_path']}. Evaluations: OccBin {c['n_occbin']}, news {c['n_news']}. First evaluation at which the "
            f"iterates differ (> 1e-9): {c['first_evaluation_with_different_theta']}. |mode difference|: {vec(dx, 3)}"
            + (f", as a share of the posterior sd: {vec(dx / sd, 3)}." if sd is not None and np.all(np.isfinite(sd)) else "."),
            f"News evaluated at the {c['n_points_checked']} points OccBin's run visited: {c['n_equal']} agree (largest difference "
            f"{c['max_abs_lp_difference_where_equal']:.2e}), {len(c['occbin_trace_points_where_likelihoods_differ'])} differ "
            f"(largest difference {max([abs(x['diff']) for x in c['occbin_trace_points_where_likelihoods_differ']] or [0]):.4f})."]
    if c.get("first_difference_explained"):
        e = c["first_difference_explained"]
        i = c["occbin_trace_points_where_likelihoods_differ"][0]["index"]
        out.append(f"First differing evaluation: index {i} of OccBin's run, log posterior {e['lp_occbin']:.6f} (OccBin) against "
                   f"{e['lp_news']:.6f} (news); regimes differ in period(s) {e['periods_with_different_regime']}: "
                   + "; ".join(f"period {k}: OccBin {v['occbin']}, news {v['news']}" for k, v in e["regimes"].items()) + ".")
    t = c["per_evaluation_seconds_interleaved"]
    out += ["", f"Time per evaluation (evaluations that ran the filter; the two solvers called alternately at the same points): OccBin min {t['occbin_min']*1e3:.0f} ms, median "
            f"{t['occbin_median']*1e3:.0f} ms; news min {t['news_min']*1e3:.0f} ms, median {t['news_median']*1e3:.0f} ms (ratio of medians "
            f"{t['ratio_of_medians']:.1f}). Trial-point failures inside the filter during the mode search: OccBin {d['occbin']['trial_failures']}, "
            f"news {d['news']['trial_failures']}.", ""]
    return "\n".join(out)


def hess_tables(case, start):
    p = R / f"est_{case}_{start}_hessian.json"
    if not p.exists():
        return ""
    h = json.loads(p.read_text())
    out = [f"#### Hessian and Laplace approximation, {case}, start = {start}", "", "| | OccBin | news |", "|---|---|---|"]
    out.append("| evaluations / wall time (s) | " + " | ".join(f"{h[m]['evaluations']} / {h[m]['wall_s']:.1f}" for m in M) + " |")
    out.append("| positive definite | " + " | ".join(str(h[m]["positive_definite"]) for m in M) + " |")
    out.append("| posterior sd | " + " | ".join(vec(h[m]["std"]) if "std" in h[m] else "-" for m in M) + " |")
    out.append("| Laplace approximation | " + " | ".join(f(h[m].get("laplace"), 6) for m in M) + " |")
    Ho, Hn = np.array(h["occbin_at_occbin_mode"]["hessian"]), np.array(h["news_at_occbin_mode"]["hessian"])
    out += ["", f"Both Hessians at OccBin's mode: largest entrywise difference {np.abs(Ho - Hn).max():.3e} (largest entry {np.abs(Ho).max():.3e}, "
            f"relative {np.abs(Ho - Hn).max() / np.abs(Ho).max():.2e}).", ""]
    return "\n".join(out)


def mh_tables(case, start):
    p = R / f"est_{case}_{start}_mh.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    s = d["settings"]
    out = [f"#### Random-walk Metropolis, {case}", "",
           f"{s['chains']} chains x {s['replic']} draws, seed {s['seed']}, proposal: {s['cov']}, jscale {s['jscale']} (pilot: "
           + ", ".join(f"{a}: {b:.2f}" for a, b in s["pilot"]) + "), same random numbers and starting points for both solvers; first half of each chain dropped.", "",
           "| | OccBin | news |", "|---|---|---|"]
    rows = [("acceptance rate per chain", lambda m: ", ".join(f"{a:.3f}" for a in d[m]["acceptance_rate"])),
            ("wall time per chain (s)", lambda m: ", ".join(f"{a:.0f}" for a in d[m]["wall_s"])),
            ("total wall time (s)", lambda m: f"{d[m]['total_wall_s']:.0f}"),
            ("median seconds per evaluation", lambda m: f"{d[m]['median_seconds_per_evaluation']:.4f}"),
            ("likelihood evaluations", lambda m: str(d[m]["evaluations"])),
            ("draws rejected with -inf, by Dynare's code", lambda m: str(d[m]["rejected_by_error_code"])),
            ("trial-point failures inside the filter (solver flag: count)", lambda m: str(d[m]["trial_point_failures_inside_filter"])),
            ("posterior mean", lambda m: vec(d[m]["mean"])), ("posterior sd", lambda m: vec(d[m]["sd"]))]
    for name, fn in rows:
        out.append(f"| {name} | {fn('occbin')} | {fn('news')} |")
    out += ["", "Draw for draw:", ""]
    for c in d["comparison"]:
        e = c["first_evaluation_with_different_likelihood"]
        out.append(f"- chain {c['chain']}: identical draws: {c['identical']}; largest difference in a draw {c['max_abs_draw_difference']:.2e}; "
                   f"evaluations at the same parameter vector with different likelihood: {c['n_evaluations_with_different_likelihood_at_same_theta']}"
                   + (f"; first at evaluation {e['index']}: {e['occbin']:.6f} (OccBin) against {e['news']:.6f} (news), difference {e['diff']:.4f}, "
                      f"regimes differ in period(s) {e['explained']['periods_with_different_regime']}" if e else "") + ".")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    cases = [(sys.argv[1], sys.argv[2])] if len(sys.argv) > 2 else [(c, s) for c in ("nkms", "nkm") for s in ("prior", "stored")]
    for c, s in cases:
        for fn in (mode_tables, hess_tables, mh_tables):
            t = fn(c, s)
            if t:
                print(t)
