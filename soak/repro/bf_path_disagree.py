"""bf.mod (two equations switch), params below, 1000 periods of N(0,(0.5*sd)^2) shocks from rng_for('C',43) segment 3 (soak task C#43): occbin binds in period 146
(regime ([1,0],[1,2])) and news does not; paths differ by 0.006.  Prints rnot - rlb (bind condition: rnot <= rlb) in both paths at the period."""
import sys; sys.path.insert(0, "soak/repro"); sys.path.insert(0, "soak")
import numpy as np
from c_inputs import inputs
m, ob, params, S, h = inputs(43, 3)          # segment 3 (it ran from init=None: the previous segment failed)
a = ob.simulate(S, periods=1000); b = ob.simulate(S, periods=1000, obc_method="news", obc_horizon=h)
t = 146; j = m._ie["rnot"]; rlb = params[m._ip["rlb"]]
print("regimes", a.regime_history[t], b.regime_history[t], "max path diff", np.abs(a.piecewise - b.piecewise).max())
print("rnot-rlb occbin", a.piecewise[t, j] - rlb, "news", b.piecewise[t, j] - rlb, "(binding needs <= 0)")
