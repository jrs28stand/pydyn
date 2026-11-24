"""NKM inversion filter: OccBin gives a finite log posterior, news returns -inf (code 304). soak task A#75."""
import sys; sys.path.insert(0, "soak")
import numpy as np, lib, part_a
th = np.array([114.66637644899114, 1.0892489266590764, 0.19766855658573093, 0.6874612608464392, 0.6985799052891387, 0.46604253593794787, 0.005397842157745668, 0.0016956255944929521, 0.0021002683787342213])
e = part_a.make_est("nkm")
for m in ("occbin", "news"):
    e.obc_method = m; e.last_news_solver = None
    print(m, e.log_posterior(th), e.last_code, e.last_info)
    if m == "news": s = e.last_news_solver; print("solver calls/fallbacks/errors", s.calls, s.fallbacks, s.errors)
    else: r = e.last_result; print("binding periods", sum(1 for reg in r.regime_history if reg[0][0][0]), "max |filtered shock|", np.abs(r.filtered_errs).max())
for h in (40, 80, 160):
    e.obc_method, e.obc_horizon = "news", h
    print("news horizon", h, e.log_posterior(th), e.last_code)
