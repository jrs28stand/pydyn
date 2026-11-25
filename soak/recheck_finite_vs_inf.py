"""Re-evaluate every logged finite_vs_inf:occbin_finite_news_code304 vector: news (default), news with obc_horizon_max=160, against OccBin's logged value."""
import sys, json, time; sys.path.insert(0, "soak")
import numpy as np, lib, part_a
rows = [json.loads(l) for l in open("soak/results/anomalies.jsonl")]
rows = [d for d in rows if d["kind"].startswith("finite_vs_inf:occbin_finite_news") and "theta" in d]
seen = set(); out = []
for d in rows:
    key = (d["task"], d["model"], tuple(d["theta"]))
    if key in seen: continue
    seen.add(key)
    e = part_a.make_est(d["model"]); e.obc_method = "news"
    res = []
    for hm in (None, 160):
        e.obc_horizon_max = hm; t = time.time(); lp = e.log_posterior(np.array(d["theta"]))
        res.append((lp, e.last_code, round(time.time() - t, 2), (e.last_failure or {}).get("primary")))
    print(d["task"], d["model"], "occbin", d["finite"], "news default", res[0], "news hmax160", res[1], "diff", abs(res[1][0] - d["finite"]) if np.isfinite(res[1][0]) else None, flush=True)
