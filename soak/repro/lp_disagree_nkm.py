"""occbin vs news inversion filter on NKM: log posterior differs by 0.2785 (known knife-edge class is <= ~0.12). soak task A#6, line t=0.1."""
import sys; sys.path.insert(0, "soak")
import numpy as np, lib, part_a
th = np.array([97.16138145643879, 1.9832925009213038, 0.5985245790721766, 0.8148560569026212, 0.8630555492751365, 0.6653615306976441, 0.005210288524517366, 0.004087483345003967, 0.0020922849220516176])
e = part_a.make_est("nkm"); r = {}
for m in ("occbin", "news"):
    e.obc_method = m; lp = e.log_posterior(th); r[m] = (lp, e.last_result)
print({m: v[0] for m, v in r.items()})
ra, rb = r["occbin"][1], r["news"][1]
bad = [t for t in range(len(ra.regime_history)) if ra.regime_history[t] != rb.regime_history[t]]
print("periods with different regimes (0-based):", bad)
for t in bad[:3]:
    print(t, ra.regime_history[t], rb.regime_history[t], "filtered shock diff", np.abs(ra.filtered_errs[t] - rb.filtered_errs[t]).max() if hasattr(ra, "filtered_errs") else "")
