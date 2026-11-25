"""The logged NKM sweep vectors with occbin vs news difference > 0.12 (part A tasks), re-evaluated exactly (the task's own vector generation is reused by
replacing eval_vector) with occbin, lcp, regime and hybrid."""
import sys, json, warnings; sys.path.insert(0, "soak"); sys.path.insert(0, "soak/approx"); warnings.simplefilter("ignore")
import numpy as np, lib, part_a, study
tasks = {}
for l in open("soak/results/A.jsonl"):
    r = json.loads(l)
    if r["case"] not in ("nkm", "nkms"): continue
    prs = [r["r"]] if "r" in r else list(r.get("pts", []))
    for k, p in enumerate(prs):
        o, n = p.get("occbin"), p.get("news")
        if o and n and isinstance(o[0], float) and isinstance(n[0], float) and abs(o[0] - n[0]) > 0.12 and abs(o[0]) < 1e5:
            tasks.setdefault(r["i"], []).append((k, abs(o[0] - n[0])))
out = []
cur = {"i": None, "k": 0}
def my_eval(e, th, cfgs, anom, st, where, dbin):
    k = cur["k"]; cur["k"] += 1
    want = dict(tasks[cur["i"]]) if cur["i"] in tasks else {}
    if k in want:
        res = {m: study.ev(e, np.asarray(th), m, f) for m, f in study.METHODS}
        out.append(dict(i=cur["i"], model=e.case, k=k, logged_diff=want[k], th=list(map(float, th)), **res))
    return {}
part_a.eval_vector = my_eval
part_a.cmp_pairs = lambda *a, **k: None
part_a.nondet_check = lambda *a, **k: None
class A:
    def __call__(self, *a, **k): pass
st = dict(kinds={}, warnings={}, counts={}, medians={})
for i in sorted(tasks):
    cur.update(i=i, k=0)
    try: part_a.task_A(i, st, A())
    except KeyError: pass          # line-sweep post-processing needs the real records; the vectors are already evaluated
json.dump(out, open("soak/approx/results/nkm60.json", "w"), default=lib._clean)
print(len(tasks), "tasks", len(out), "vectors")
