"""Regenerates docs/OBC_SOAK.md from soak/notes.md (hand-written: state, judgements) and the result/state files."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES, STATE = ROOT / "soak/results", ROOT / "soak/state"


def jl(name):
    p = RES / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def st(part):
    p = STATE / f"{part}.json"
    return json.loads(p.read_text()) if p.exists() else dict(next=0, seconds=0, kinds={}, warnings={}, counts={}, medians={})


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def tstats(v):
    return f"{len(v)} | {q(v, 50):.4g} | {q(v, 99):.4g} | {max(v) if len(v) else float('nan'):.4g}"


def main():
    S = {p: st(p) for p in "ABCD"}
    A, B, C, D = (jl(f"{p}.jsonl") for p in "ABCD")
    anoms = jl("anomalies.jsonl")
    out = []
    w = out.append
    w("## Coverage so far\n")
    w("| part | run time (h) | tasks | what |\n|---|---|---|---|")
    ev = collections.Counter()
    for k, v in S["A"]["counts"].items():
        p = k.split("|")
        if p[0] == "A" and len(p) == 6:
            ev[p[1]] += v
    nvec = collections.Counter((r["case"], r["kind"]) for r in A)
    w(f"| A | {S['A']['seconds'] / 3600:.2f} | {S['A']['next']} | evaluations of the log posterior (all configurations): " +
      ", ".join(f"{k} {v}" for k, v in sorted(ev.items())) + "; tasks by kind: " + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(nvec.items())) + " |")
    chains = [r for r in B if r.get("kind") == "mcmc"]
    modes = [r for r in B if r.get("kind") == "mode"]
    draws = len(chains) * 150
    replays = max([0] + [1 for _ in chains])
    cb = S["B"].get("chain") or {}
    w(f"| B | {S['B']['seconds'] / 3600:.2f} | {S['B']['next']} | {draws} MCMC draws (news) in {S['B'].get('chains_done', 0)} finished chains + the running one (chain {cb.get('c')}, draw {cb.get('k')}, "
      f"replays under occbin of this chain: {cb.get('replay')}); {len(modes)} mode searches ({sum(1 for r in modes if r.get('skipped'))} skipped: start not finite, "
      f"{sum(1 for r in modes if r.get('timeout'))} timeouts) |")
    per = collections.Counter(); bind = collections.Counter(); sims = collections.Counter()
    for r in C:
        if r.get("kind") == "sim":
            per[r["model"]] += r.get("periods", 0); bind[r["model"]] += r.get("nbind", 0); sims[r["model"]] += 1
    w(f"| C | {S['C']['seconds'] / 3600:.2f} | {S['C']['next']} | simulated periods per model (both methods each): " + ", ".join(f"{k} {v} ({sims[k]} runs, {bind[k]} binding)" for k, v in sorted(per.items())) + " |")
    dm = collections.Counter(); dok = collections.Counter()
    for r in D:
        dm[r["model"]] += 1
        dok[r["model"]] += bool(r.get("ok"))
    w(f"| D | {S['D']['seconds'] / 3600:.2f} | {S['D']['next']} | runs per model (ok/total): " + ", ".join(f"{k} {dok[k]}/{v}" for k, v in sorted(dm.items())) + " |")
    ver = {k: v for p in "ABCD" for k, v in S[p]["counts"].items() if k.startswith("verify|")}
    vsum = collections.Counter()
    for k, v in ver.items():
        vsum[k] += v
    w(f"\nLCP solutions checked with `verify_lcp` (tolerance 1e-6) at the `solve_active_set` / `solve_news` boundary, all parts: {dict(vsum)}.\n")
    nd = {p: S[p]["counts"].get(f"{p}|nondet_checks", 0) for p in "ACD"}
    w(f"Determinism checks (same call repeated): A same-process {nd['A']} (+{S['A']['counts'].get('A|nondet_fresh_checks', 0)} in a fresh estimation object), C {nd['C']}, D {nd['D']}.\n")

    # anomaly table
    w("## Anomaly catalogue\n")
    notes = json.loads((ROOT / "soak/notes.json").read_text()) if (ROOT / "soak/notes.json").exists() else {}
    kinds = collections.Counter()
    first = {}
    for p in "ABCD":
        for k, v in S[p]["kinds"].items():
            kinds[(k, p)] += v
    for a in anoms:
        first.setdefault((a["kind"], a["part"]), []).append(a)
    w("| kind | part | count | severity | first reproducer(s) | judgement |\n|---|---|---|---|---|---|")
    for (k, p), n in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0])):
        key = next((nk for nk in sorted(notes, key=len, reverse=True) if k.startswith(nk)), None)
        sev, com, repro = notes.get(key, ["?", "not yet looked at", ""]) if key else ["?", "not yet looked at", ""]
        reps = ", ".join(f"task {a['task']}" for a in first.get((k, p), [])[:3])
        w(f"| `{k}` | {p} | {n} | {sev} | {repro or 'soak/results/anomalies.jsonl'} ({reps}) | {com} |")
    w("")

    # failure tables A
    w("## Failure rates, part A (inversion filter, log posterior evaluations)\n")
    w("Distance bin = rms of the relative deviation from the model's initial values: a <0.02, b <0.1, c <0.3, d <1, e >=1; `line` = the 25 points of the straight-line sweeps (-0.1..1.1).\n")
    tab = collections.defaultdict(collections.Counter)
    for k, v in S["A"]["counts"].items():
        p = k.split("|")
        if p[0] == "A" and len(p) == 6:
            tab[(p[1], p[2])][(p[3], p[5])] += v
    w("| model | config | n | finite | failed (by code) | by distance bin: finite share |\n|---|---|---|---|---|---|")
    for (case, cfg), c in sorted(tab.items()):
        n = sum(c.values())
        ok = sum(v for (b, key), v in c.items() if key == "ok")
        codes = collections.Counter()
        for (b, key), v in c.items():
            if key != "ok":
                codes[key] += v
        bins = []
        for b in "abcde" + "l":
            bb = "line" if b == "l" else b
            nn = sum(v for (x, key), v in c.items() if x == bb)
            if nn:
                bins.append(f"{bb}:{sum(v for (x, key), v in c.items() if x == bb and key == 'ok') / nn:.2f} ({nn})")
        w(f"| {case} | {cfg} | {n} | {ok / max(n, 1):.3f} | {', '.join(f'{k} {v}' for k, v in sorted(codes.items()))} | {' '.join(bins)} |")
    w("")
    # agreement
    diffs = collections.defaultdict(list)
    for r in A:
        if r.get("kind") == "line":
            for pt in r["pts"]:
                o, nw = pt.get("occbin"), pt.get("news")
                if o and nw and isinstance(o[0], float) and isinstance(nw[0], float):
                    diffs[r["case"]].append(abs(o[0] - nw[0]))
        elif "r" in r:
            o, nw = r["r"].get("occbin"), r["r"].get("news")
            if o and nw and isinstance(o[0], float) and isinstance(nw[0], float):
                diffs[r["case"]].append(abs(o[0] - nw[0]))
    w("occbin vs news, both finite: |difference in log posterior|\n")
    w("| model | pairs | <=1e-9 (rel) | 1e-9..0.15 | >0.15 | max |\n|---|---|---|---|---|---|")
    for case, v in sorted(diffs.items()):
        v = np.array(v)
        w(f"| {case} | {len(v)} | {(v <= 1e-9 * 1500).sum()} | {((v > 1e-9 * 1500) & (v <= 0.15)).sum()} | {(v > 0.15).sum()} | {v.max():.3g} |")
    w("")
    # timing A
    w("## Timing distributions\n")
    w("Part A, seconds per finite log-posterior evaluation:\n")
    w("| model | config | n | median | p99 | max |\n|---|---|---|---|---|---|")
    tt = collections.defaultdict(list)
    for r in A:
        if r.get("kind") == "line":
            for pt in r["pts"]:
                for cfg, v in pt.items():
                    if isinstance(v[0], float):
                        tt[(r["case"], cfg)].append(v[2])
        else:
            for cfg, v in r.get("r", {}).items():
                if isinstance(v[0], float):
                    tt[(r["case"], cfg)].append(v[2])
    for (case, cfg), v in sorted(tt.items()):
        w(f"| {case} | {cfg} | {tstats(v)} |")
    w("\nPart C, seconds per simulated period (per 1000-period segment, averaged over a task):\n")
    w("| model | method | tasks | median | p99 | max |\n|---|---|---|---|---|---|")
    tc = collections.defaultdict(list)
    for r in C:
        if r.get("kind") == "sim" and r.get("periods") and not r.get("fails"):
            tc[(r["model"], "occbin")].append(r["t_occ"] / r["periods"])
            tc[(r["model"], "news")].append(r["t_news"] / r["periods"])
    for k, v in sorted(tc.items()):
        w(f"| {k[0]} | {k[1]} | {tstats(v)} |")
    w("\nPart D, seconds per period of `HigherOrderNews.simulate` (successful runs):\n")
    w("| model | config | runs | median | p99 | max |\n|---|---|---|---|---|---|")
    td = collections.defaultdict(list)
    for r in D:
        if r.get("ok"):
            td[(r["model"], r["cfg"])].append(r["t_sim"] / r["periods"])
    for k, v in sorted(td.items()):
        w(f"| {k[0]} | {k[1]} | {tstats(v)} |")
    # failure tables C, D
    w("\n## Failures, parts C and D (expected error types)\n")
    w("Part C: per model, segments of 1000 periods that failed, by method and error (count of segments):\n")
    fc = collections.defaultdict(collections.Counter)
    for k, v in S["C"]["counts"].items():
        p = k.split("|")
        if p[0] == "C" and len(p) >= 5 and p[2] == "fail":
            fc[p[1]][p[3] + ":" + "|".join(p[4:])] += v
    for m, c in sorted(fc.items()):
        w(f"- {m}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    w("\nPart D: runs per model and configuration (ok / error types):\n")
    fd = collections.defaultdict(collections.Counter)
    for r in D:
        fd[(r["model"], r["cfg"])][("ok" if r.get("ok") else r.get("err", "timeout/other"))] += 1
    w("| model | config | outcomes |\n|---|---|---|")
    for k, c in sorted(fd.items()):
        w(f"| {k[0]} | {k[1]} | {', '.join(f'{a} {b}' for a, b in sorted(c.items()))} |")
    # warnings
    w("\n## Warnings (distinct messages, numbers normalised, counts over all parts)\n")
    wc = collections.Counter()
    for p in "ABCD":
        for k, v in S[p]["warnings"].items():
            wc[k] += v
    w("\n".join(f"- {v} x `{k}`" for k, v in wc.most_common()) or "none")
    # memory
    w("\n## Memory (resident MB at the end of each task; one process per ~8 minute chunk, so growth is only meaningful within a chunk)\n")
    for p, R in zip("ABCD", (A, B, C, D)):
        rs = [r["rss"] for r in R if "rss" in r]
        if rs:
            w(f"- {p}: median {np.median(rs):.0f}, max {max(rs):.0f}")
    return "\n".join(out)


if __name__ == "__main__":
    notes = (ROOT / "soak/notes.md").read_text()
    body = main()
    (ROOT / "docs/OBC_SOAK.md").write_text(notes.replace("{{AUTO}}", body))
    print("written", len(body))
