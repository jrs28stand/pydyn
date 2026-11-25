"""csminwel mode search with obc_method='news' from a prior-ish start does not finish in 200 s (soak task B#203).  Prints evaluation count, time split."""
import sys, time; sys.path.insert(0, "soak")
import numpy as np, lib, part_a
x0 = np.array([62.52795549728511, 1.3749629121821647, 0.3152230456631081, 0.7452164378776344, 0.7919643598165762, 0.83968518938011, 0.005705702707354552, 0.003365348309073875, 0.0053222504798572405])
e = part_a.make_est("nkm"); e.obc_method = "news"
ts = []; orig = e.log_posterior
def f(th, bounds=None):
    t = time.time(); v = orig(th) if bounds is None else orig(th, bounds=bounds); ts.append((time.time() - t, v, e.last_code)); return v
e.log_posterior = f
try:
    with lib.alarm(90): r = e.mode(x0=x0, mode_compute=4); print("finished", r.log_posterior, r.nfev)
except lib.Alarm: print("not finished after 90 s")
t = np.array([a for a, _, _ in ts]); print("evaluations", len(t), "total", t.sum(), "median", np.median(t), "max", t.max(), "finite", sum(np.isfinite(v) for _, v, _ in ts), "last lp", ts[-1][1])
