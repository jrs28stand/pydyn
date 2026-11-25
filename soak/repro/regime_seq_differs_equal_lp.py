"""nkms (3 parameters), inversion filter: occbin and news give (almost) the same log posterior but different regime sequences (78 vs 77 binding periods).  soak task A#1342."""
import sys; sys.path.insert(0, "soak")
import numpy as np, lib, part_a
th = np.array([28.99120572508528, 1.708997640179066e-07, 4.2120060164811874e-08])
e = part_a.make_est("nkms"); r = {}
for m in ("occbin", "news"):
    e.obc_method = m; r[m] = (e.log_posterior(th), e.last_result)
print({m: v[0] for m, v in r.items()}, "diff", r["occbin"][0] - r["news"][0])
ra, rb = r["occbin"][1], r["news"][1]
bad = [t for t in range(len(ra.regime_history)) if ra.regime_history[t] != rb.regime_history[t]]
print("periods with different regimes:", bad[:10], "max filtered-shock diff", np.abs(ra.filtered_errs - rb.filtered_errs).max())
