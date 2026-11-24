"""bf, inversion filter: log posterior jumps -1283 -> -840 between t=-0.05 and t=0 on the line x + t (y - x) with the same regime sequence (soak task A#404).
Fine scan of that segment, with the number of binding periods and the largest filtered shock."""
import sys; sys.path.insert(0, "soak")
import numpy as np, lib, part_a
x = np.array([0.169806125747, 0.000545686767, 0.230805130616, 0.301787731056, 0.910976197795])
y = np.array([0.132726099822, 0.001347731353, 0.223340285686, 0.830650755471, 0.787300280527])
e = part_a.make_est("bf")
for m in ("occbin", "news"):
    e.obc_method = m
    for t in np.linspace(-0.05, 0.0, 11):
        lp = e.log_posterior(x + t * (y - x)); r = e.last_result
        print(m, round(t, 3), round(lp, 3), e.last_code, "bind", sum(1 for g in r.regime_history if g[0][0][0]) if r is not None and np.isfinite(lp) else None,
              "max|shock|", round(float(np.abs(r.filtered_errs).max()), 3) if r is not None and np.isfinite(lp) else None)
