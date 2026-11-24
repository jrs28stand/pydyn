"""Markdown tables from the JSON written by obc_bench.py.  Usage: obc_report.py p1|p2|p3|p4|r1 FILE [FILE...]
(several files are merged by row key, e.g. p1_before.json and p1_after.json)."""
import json
import sys


def load(files):
    out = {}
    for f in files:
        d = json.load(open(f))
        out[f] = d
    return out


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def p1(files):
    print("| model | method | impl | share binding | min us/period | median us/period | nothing binds us | binds us | max diff vs OccBin | regimes = OccBin | max diff vs before | regimes = before |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f, d in load(files).items():
        for model, v in d["data"].items():
            for key, r in v["rows"].items():
                meth, _, impl = key.partition("/")
                if "error" in r:
                    print(f"| {model} | {meth} | {impl or 'n/a'} | ERROR {r['error']} | | | | | | |")
                    continue
                sp = r["split"]
                tail = (f"{r['max_path_diff_vs_before']:.1e} | {r['regimes_equal_before']} |" if "max_path_diff_vs_before" in r
                        else " | |")
                print(f"| {model} | {meth} | {impl or '-'} | {r['share_binding']:.3f} | {r['us_per_period']['min']:.1f} | "
                      f"{r['us_per_period']['median']:.1f} | {fmt(sp['us_nothing_binds'])} | {fmt(sp['us_binds'])} | "
                      f"{r.get('max_path_diff_vs_occbin', float('nan')):.1e} | {r.get('regime_histories_equal_occbin')} | " + tail)


def p2(files):
    print("| model | method | impl | steady ms | first-order ms | Occbin() total ms | precompute ms (min / median) | total ms (min / median) |")
    print("|---|---|---|---|---|---|---|---|")
    for f, d in load(files).items():
        for model, v in d["data"].items():
            for key, r in v["rows"].items():
                meth, _, impl = key.partition("/")
                print(f"| {model} | {meth} | {impl or '-'} | {r['steady']['min']:.2f} | {r['first_order']['min']:.2f} | {r['ctor']['min']:.2f} | "
                      f"{r['precompute']['min']:.2f} / {r['precompute']['median']:.2f} | {r['total']['min']:.2f} / {r['total']['median']:.2f} |")


def p3(files):
    print("| model | method | impl | total s (min / median) | per iteration ms (min) | set-up share of min total | max diff vs OccBin | regimes = OccBin | max diff vs before | regimes = before |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f, d in load(files).items():
        for model, v in d["data"].items():
            for key, r in v["rows"].items():
                meth, _, impl = key.partition("/")
                if "error" in r:
                    print(f"| {model} | {meth} | {impl or '-'} | ERROR {r['error']} | | | | | | |")
                    continue
                print(f"| {model} | {meth} | {impl or '-'} | {r['total_s']['min']:.3f} / {r['total_s']['median']:.3f} | {r['per_iteration_ms']['min']:.2f} | "
                      f"{r['setup_s']['min'] / r['total_s']['min']:.2f} | {r.get('max_path_diff_vs_occbin', float('nan')):.1e} | {r.get('regime_histories_equal_occbin')} | "
                      f"{r.get('max_path_diff_vs_before', float('nan')):.1e} | {r.get('regimes_equal_before')} |")


def p4(files):
    print("| model | horizon | method | impl | P1 us/period (min / median) | P2 ms (min / median) |")
    print("|---|---|---|---|---|---|")
    for f, d in load(files).items():
        for model, v in d["data"].items():
            for h, r in v.items():
                for key, a in r["p1"].items():
                    meth, _, impl = key.partition("/")
                    b = r["p2"][key]
                    if "error" in a:
                        print(f"| {model} | {h} | {meth} | {impl or '-'} | ERROR {a['error'][:60]} | {b['total']['min']:.2f} / {b['total']['median']:.2f} |")
                        continue
                    print(f"| {model} | {h} | {meth} | {impl or '-'} | {a['us_per_period']['min']:.1f} / {a['us_per_period']['median']:.1f} | "
                          f"{b['total']['min']:.2f} / {b['total']['median']:.2f} |")


def r1(files):
    for f, d in load(files).items():
        print(f"\n{f}  (meta: {d['meta']['impls']})")
        print("| model | s | method | solved | failed | failure types | solved with binding | seconds |")
        print("|---|---|---|---|---|---|---|---|")
        for model, v in d["data"].items():
            for s, r in v.items():
                for key, x in r["methods"].items():
                    print(f"| {model} | {s} | {key} | {x['solved']} | {x['failed']} | {x['failures']} | {x['solved_with_binding']} | {x['seconds']:.1f} |")
        print("\n| model | s | pair | both solved | different regimes | only first fails | only second fails | both fail |")
        print("|---|---|---|---|---|---|---|---|")
        for model, v in d["data"].items():
            for s, r in v.items():
                for pair, x in r["pairs"].items():
                    print(f"| {model} | {s} | {pair} | {x['both_solved']} | {x['different_regimes']} | {x['only_first_fails']} | {x['only_second_fails']} | {x['both_fail']} |")
                if r.get("before_vs_after"):
                    print(f"| {model} | {s} | before vs after identical outcomes | {r['before_vs_after']} | | | | |")


if __name__ == "__main__":
    {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "r1": r1}[sys.argv[1]](sys.argv[2:])
