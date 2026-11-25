"""Evidence for the default of cubature_milp_time_limit: per model and limit, the failed points by class, those solved by the full search, ms per period.
python3 soak/hoapprox/milp_limit.py MODEL LEVEL T   (regime formulation; the LCP class for exact-class models with --lcp)"""
import sys, time, json, warnings
sys.path.insert(0, "soak/hoapprox"); sys.path.insert(0, "soak"); sys.path.insert(0, "soak/approx")
warnings.simplefilter("ignore")
import numpy as np, ho_study as hs
from pydyn.obc_ho import HigherOrderNews
from pydyn.obc_regime_ho import HigherOrderRegime
name, lv, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); lcp = "--lcp" in sys.argv
m = hs.load(name); S = hs.shocks(name, m, T); hz = hs.CFG.get(name, (1, 40))[1]
for lim in ([float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else (0.02, 0.05, 0.1, 0.5, 1.0)):
    kw = dict(order=2, horizon=hz, check_ahead=hz, cubature=True, qmc_level=lv, regions="all", cubature_failure_tolerance=0.99, cubature_milp_time_limit=lim)
    h = HigherOrderNews(m, obc_formulation="lcp", **kw) if lcp else HigherOrderRegime(m, **kw)
    t = time.perf_counter(); r = h.simulate(S); dt = (time.perf_counter() - t) / T
    print(json.dumps(dict(model=name, lcp=lcp, level=lv, limit=lim, ms_per_period=round(1e3 * dt, 2), relevant=int(r["relevant"].sum()), a=int(r["dropped_a"].sum()),
          b=int(r["dropped_b"].sum()), c=int(r["dropped_c"].sum()), batch_fail=h.stats.get("batch_fail"), search_solved=h.stats.get("full_search_solved"))), flush=True)
