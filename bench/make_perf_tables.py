"""Condensed markdown tables for docs/OBC_PERF.md from bench/results/*_final.json (and r1_*.json)."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def J(name):
    return json.load(open(R / name))["data"]


def mm(x, nd=1):
    return f"{x['min']:.{nd}f} / {x['median']:.{nd}f}"


def p1():
    d = J("p1_final.json")
    print("| model (s, share binding) | method | before min / median | after min / median | speed-up (min) | nothing binds before -> after | binds before -> after | max path diff after vs before | regimes same as before | max path diff vs OccBin | regimes same as OccBin |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, v in d.items():
        rows = v["rows"]
        occ = rows["occbin"]
        print(f"| {model} (s={v['s']}, {occ['share_binding']:.3f}) | occbin | {mm(occ['us_per_period'])} | same code | | {occ['split']['us_nothing_binds']:.0f} | {occ['split']['us_binds']:.0f} (one-period calls, include call overhead) | | | - | - |")
        for meth in ("news", "regime"):
            if f"{meth}/before" not in rows:
                continue
            b, a = rows[f"{meth}/before"], rows[f"{meth}/after"]
            print(f"| {model} | {meth} | {mm(b['us_per_period'])} | {mm(a['us_per_period'])} | {b['us_per_period']['min'] / a['us_per_period']['min']:.1f}x | "
                  f"{b['split']['us_nothing_binds']:.0f} -> {a['split']['us_nothing_binds']:.0f} | {b['split']['us_binds']:.0f} -> {a['split']['us_binds']:.0f} | "
                  f"{a['max_path_diff_vs_before']:.1e} | {a['regimes_equal_before']} | {a['max_path_diff_vs_occbin']:.1e} | {a['regime_histories_equal_occbin']} |")


def p2():
    d = J("p2_final.json")
    print("| model | method | steady ms | first-order ms | Occbin() incl. both ms | precompute before min / median | precompute after min / median | total before min / median | total after min / median | speed-up of the total (min) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for model, v in d.items():
        rows = v["rows"]
        o = rows["occbin"]
        print(f"| {model} | occbin | {o['steady']['min']:.2f} | {o['first_order']['min']:.2f} | {o['ctor']['min']:.2f} | none | none | {mm(o['total'], 2)} | same | |")
        for meth in ("news", "regime"):
            if f"{meth}/before" not in rows:
                continue
            b, a = rows[f"{meth}/before"], rows[f"{meth}/after"]
            print(f"| {model} | {meth} | {a['steady']['min']:.2f} | {a['first_order']['min']:.2f} | {a['ctor']['min']:.2f} | {mm(b['precompute'], 2)} | {mm(a['precompute'], 2)} | {mm(b['total'], 2)} | {mm(a['total'], 2)} | {b['total']['min'] / a['total']['min']:.1f}x |")


def p3():
    d = J("p3_final.json")
    print("| model | method | before total s (min / median) | after total s (min / median) | speed-up (min) | after: per iteration ms (min) | after: set-up share | max path diff after vs before | regimes same as before | max diff vs OccBin | regimes same as OccBin |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, v in d.items():
        rows = v["rows"]
        o = rows["occbin"]
        print(f"| {model} | occbin | {mm(o['total_s'], 2)} | same code | | {o['per_iteration_ms']['min']:.1f} | {o['setup_s']['min'] / o['total_s']['min']:.2f} | | | | |")
        for meth in ("news", "regime"):
            if f"{meth}/before" not in rows:
                continue
            b, a = rows[f"{meth}/before"], rows[f"{meth}/after"]
            print(f"| {model} | {meth} | {mm(b['total_s'], 2)} | {mm(a['total_s'], 2)} | {b['total_s']['min'] / a['total_s']['min']:.1f}x | {a['per_iteration_ms']['min']:.1f} | "
                  f"{a['setup_s']['min'] / a['total_s']['min']:.2f} | {a['max_path_diff_vs_before']:.1e} | {a['regimes_equal_before']} | {a['max_path_diff_vs_occbin']:.1e} | {a['regime_histories_equal_occbin']} |")


def p4():
    d = J("p4_final.json")
    print("| model | horizon | method | P1 before us/period (min / median) | P1 after | P2 before ms (min / median) | P2 after ms |")
    print("|---|---|---|---|---|---|---|")
    for model, v in d.items():
        for h, r in v.items():
            o1, o2 = r["p1"]["occbin"], r["p2"]["occbin"]
            print(f"| {model} | {h} | occbin | {mm(o1['us_per_period'])} | same | {mm(o2['total'], 2)} | same |")
            for meth in ("news", "regime"):
                if f"{meth}/before" not in r["p1"]:
                    continue
                b1, a1, b2, a2 = r["p1"][f"{meth}/before"], r["p1"][f"{meth}/after"], r["p2"][f"{meth}/before"], r["p2"][f"{meth}/after"]
                print(f"| {model} | {h} | {meth} | {mm(b1['us_per_period'])} | {mm(a1['us_per_period'])} | {mm(b2['total'], 2)} | {mm(a2['total'], 2)} |")


def r1(files):
    print("| model | s | OccBin solved / failed (types) | news solved / failed | regime solved / failed (types) | OccBin vs news: different regimes (of both solved) | OccBin vs regime: different regimes | OccBin only fails / other only fails / both fail (regime) | solved with binding (OccBin) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f in files:
        d = json.load(open(R / f))["data"]
        for model, v in d.items():
            for s, r in v.items():
                m = r["methods"]
                oc = m["occbin"]
                def cell(x):
                    ty = ", ".join(f"{k.split(':', 1)[1]}: {n}" if k.startswith("Occbin") else f"{k.split(':', 1)[1]}: {n}" for k, n in x["failures"].items())
                    return f"{x['solved']} / {x['failed']}" + (f" ({ty})" if ty else "")
                news = m.get("news/after")
                reg = m.get("regime/after")
                pn = r["pairs"].get("occbin vs news/after")
                pr = r["pairs"].get("occbin vs regime/after")
                print(f"| {model} | {s} | {cell(oc)} | {cell(news) if news else '-'} | {cell(reg)} | "
                      f"{str(pn['different_regimes']) + ' of ' + str(pn['both_solved']) if pn else '-'} | {str(pr['different_regimes']) + ' of ' + str(pr['both_solved'])} | "
                      f"{pr['only_first_fails']} / {pr['only_second_fails']} / {pr['both_fail']} | {oc['solved_with_binding']} |")


if __name__ == "__main__":
    {"p1": p1, "p2": p2, "p3": p3, "p4": p4}.get(sys.argv[1], lambda: r1(sys.argv[2:]))()
