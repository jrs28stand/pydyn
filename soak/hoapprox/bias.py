"""Effect of dropping undecided cubature points.  python3 soak/hoapprox/bias.py MODEL LEVEL T"""
import sys, json, warnings
sys.path.insert(0, "soak/hoapprox"); sys.path.insert(0, "soak"); sys.path.insert(0, "soak/approx")
warnings.simplefilter("ignore")
import numpy as np, ho_study as hs
from pydyn.obc import OBCError
from pydyn.obc_ho import HigherOrderNews
from pydyn.obc_regime_ho import HigherOrderRegime
name, lv, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = hs.load(name); S = hs.shocks(name, m, T); hz = hs.CFG.get(name, (1, 40))[1]
kw = dict(order=2, horizon=hz, check_ahead=hz, cubature=True, qmc_level=lv, regions="all")
def run(**k):
    h = HigherOrderRegime(m, **{**kw, **k}); return h, h.simulate(S)
def d(a, b): return float(np.abs(a["with_bound"] - b["with_bound"]).max())
out = dict(model=name, level=lv, T=T)
h0, A = run()
out["relevant"], out["dropped"] = int(A["relevant"].sum()), int(A["dropped"].sum())
out["periods_with_drops"] = int((A["dropped"] > 0).sum())
nocub = HigherOrderRegime(m, order=2, horizon=hz, check_ahead=hz).simulate(S)
out["cubature_effect_default_vs_none"] = d(A, nocub)
# (i) strict: run until it raises; compare on the periods before the first failure
try:
    HigherOrderRegime(m, cubature_undecided_tolerance=0, **kw).simulate(S); out["strict"] = "runs"
except OBCError as e:
    out["strict"] = f"raises at period {e.period}"
# (ii) a long MILP limit
h2, B = run(cubature_milp_time_limit=5.0)
out["long_limit_5s_dropped"] = int(B["dropped"].sum()); out["default_vs_long_limit"] = d(A, B)
# (iii) the LCP
try:
    C = HigherOrderNews(m, obc_formulation="lcp", **kw).simulate(S); out["default_vs_lcp"] = d(A, C); out["lcp_cubature_effect"] = d(C, HigherOrderNews(m, obc_formulation="lcp", order=2, horizon=hz, check_ahead=hz).simulate(S))
except Exception as e:
    out["lcp"] = str(e)[:60]
# dropped points counted as zero wedge instead of renormalising
h3 = HigherOrderRegime(m, **kw); orig = h3.news_wedges
def zero_fill(a, Y, A_, sel="occbin"):
    w, z, info = orig(a, Y, A_, sel); F = h3._dropped_last; n = h3.xi.shape[1]
    return w * (n - F) / n, z, info
h3.news_wedges = zero_fill; Z = h3.simulate(S); out["default_vs_zero_fill"] = d(A, Z)
# integration noise: sign patterns of the columns of R (equivalent to reflecting the quasi-Monte-Carlo points)
rng = np.random.default_rng(0); R0 = h0.R.copy(); paths = [A["with_bound"]]
for _ in range(7):
    h = HigherOrderRegime(m, **kw); h.R = R0 * rng.choice([-1.0, 1.0], size=R0.shape[1]); paths.append(h.simulate(S)["with_bound"])
P = np.array(paths); out["sign_pattern_spread_max_std"] = float(P.std(axis=0).max()); out["sign_pattern_spread_max_range"] = float((P.max(0) - P.min(0)).max())
print(json.dumps(out), flush=True)
