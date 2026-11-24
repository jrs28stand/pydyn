"""OccBin inversion-filter likelihood at an extreme NKM vector: 10 s and ~1.3 GB, because Occbin.rules' memo trie (occbin.py:260) is unbounded
(232,740 nodes in one Occbin instance).  Failure code 304.  soak task A#141 (obc_method='occbin')."""
import sys, resource; sys.path.insert(0, "soak")
import numpy as np, lib, part_a
from pydyn import occbin
th = np.array([109.65434996515319, 2.2349256172471215, 0.6453732314401828, 0.6609797328917931, 0.9999999229210729, 0.7881900975597803, 0.004289142824646319, 0.010112247559243531, 6.0572589762884507e-05])
e = part_a.make_est("nkm"); e.obc_method = "occbin"
sizes = []
orig = occbin.Occbin.rules
def rules(self, *a, **k):
    r = orig(self, *a, **k); sizes.append(len(self.__dict__.get("_rule_node", []))); return r
occbin.Occbin.rules = rules
print("lp", e.log_posterior(th), "code", e.last_code, "trie nodes", max(sizes), "peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
