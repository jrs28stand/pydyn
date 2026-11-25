"""Distribution over periods of the share of relevant cubature points that fail (classes b and c, and a), per model and level; and how many periods a rule
'share <= 0.5%' would reject with and without a minimum of one point.  python3 soak/hoapprox/share_dist.py MODEL T [--lcp]"""
import sys, json, warnings
sys.path.insert(0, "soak/hoapprox"); sys.path.insert(0, "soak"); sys.path.insert(0, "soak/approx")
warnings.simplefilter("ignore")
import numpy as np, ho_study as hs
from pydyn.obc_ho import HigherOrderNews
from pydyn.obc_regime_ho import HigherOrderRegime
name, T = sys.argv[1], int(sys.argv[2]); lcp = "--lcp" in sys.argv
m = hs.load(name); S = hs.shocks(name, m, T); hz = hs.CFG.get(name, (1, 40))[1]
for lv in (9, 11, 13):
    kw = dict(order=2, horizon=hz, check_ahead=hz, cubature=True, qmc_level=lv, regions="all", cubature_failure_tolerance=0.99, cubature_milp_time_limit=0.05)
    h = HigherOrderNews(m, obc_formulation="lcp", **kw) if lcp else HigherOrderRegime(m, **kw)
    r = h.simulate(S)
    rel = r["relevant"]; bc = r["dropped_b"] + r["dropped_c"]; sh = bc / np.maximum(rel, 1)
    act = rel > 0
    print(json.dumps(dict(model=name, lcp=lcp, level=lv, periods=int(act.sum()), rel_median=int(np.median(rel[act])), rel_min=int(rel[act].min()),
          periods_with_bc=int((bc > 0).sum()), share_total=round(float(bc.sum() / rel.sum()), 5), share_max=round(float(sh.max()), 5),
          share_p95=round(float(np.percentile(sh[act], 95)), 5), a_points=int(r["dropped_a"].sum()),
          rejected_pure=int((bc > 0.005 * rel).sum()), rejected_min1=int((bc > np.maximum(0.005 * rel, 1)).sum()),
          rejected_min2=int((bc > np.maximum(0.005 * rel, 2)).sum()), max_count=int(bc.max()))), flush=True)
