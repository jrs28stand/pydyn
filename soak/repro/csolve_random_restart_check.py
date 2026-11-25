"""Does pydyn's inversion filter have a random restart whose draws differ from run to run (as found in the JDyn port)?
csolve's random search is seeded default_rng(0) afresh in every csolve call, so it is deterministic.  This counts how often the random branch is used on
prior draws / wide perturbations (soak part A inputs) and checks that evaluations in different order give bit-identical log posteriors."""
import sys; sys.path.insert(0, "soak")
import numpy as np, lib, part_a
from pydyn import occbin_filter as of
class Counting:
    n = 0
    def __init__(self): self.g = np.random.default_rng(0)
    def standard_normal(self, *a, **k): Counting.n += 1; return self.g.standard_normal(*a, **k)
orig = of.csolve
of.csolve = lambda *a, rng=None, **k: orig(*a, rng=Counting(), **k)
rng = np.random.default_rng(5)
for case in ("nkm", "nkms", "bf"):
    e = part_a.make_est(case); vecs = [part_a.prior_vec(e, rng) for _ in range(10)] + [part_a.perturbed(e, rng, 0.5) for _ in range(10)]
    for meth in ("occbin", "news"):
        e.obc_method = meth; Counting.n = 0
        a = [e.log_posterior(v) for v in vecs]; n1 = Counting.n
        b = [e.log_posterior(v) for v in vecs[::-1]][::-1]
        print(case, meth, "random restarts used:", n1, "finite", int(np.isfinite(a).sum()), "/20; identical on re-evaluation in reverse order:", all((x == y) or (np.isnan(x) and np.isnan(y)) for x, y in zip(a, b)))
