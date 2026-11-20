"""Dynare's shipped example models: load each one unedited and compare with Dynare 7.2's results
(reference/examples/<group>__<name>.mat, written by Dynare under Octave; see tools/sweep_examples.py)."""
import os
import shutil
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import sweep_examples  # noqa: E402

EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
REFS = ROOT / "reference" / "examples"

MATCH = "matches Dynare"
EXPECTED = {
    "estimation/gali_2015": MATCH,
    "estimation/rbc_irf_matching": "solved, no reference",      # Dynare's run stops before stoch_simul here
    "estimation/schorfheide_2000": MATCH,
    "macroprocessor/bkk_1992": MATCH,
    "occbin/rbc_occbin": MATCH,
    "perfect_foresight/perfect_foresight_rbc": MATCH,
    "reporting/collard_2001_reporting": MATCH,
    "stochastic_simulations/aguiar_gopinath_2007_trend": MATCH,
    "stochastic_simulations/collard_2001_analytical_steady_state": MATCH,     # uses its .m helper function
    "stochastic_simulations/collard_2001_simulated_moments": MATCH,
    "stochastic_simulations/collard_2001_theoretical_moments": MATCH,
    "stochastic_simulations/nk_baseline": MATCH,                              # uses its _steadystate.m file
    "stochastic_simulations/schorfheide_2000_nonstationary": MATCH,
    "heterogeneity/hank_one_asset": "not supported",
    "optimal_policy/nk_ramsey_osr": "not supported",
    "perfect_foresight/perfect_foresight_expectation_errors": MATCH,
    "semistructural/pac_model": "not supported",
}

pytestmark = pytest.mark.skipif(not EXAMPLES.exists(), reason="Dynare's examples folder not found (PYDYN_EXAMPLES)")


@pytest.mark.parametrize("example", sorted(EXPECTED))
def test_example(example, tmp_path):
    group, name = example.split("/")
    for f in (EXAMPLES / group).iterdir():                 # companion .m, data and include files
        if f.is_file():
            shutil.copy(f, tmp_path)
    ref = REFS / f"{group}__{name}.mat"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sweep_examples.check(tmp_path / f"{name}.mod", ref_file=ref)
    assert out["status"] == EXPECTED[example], out
    if EXPECTED[example] == MATCH:
        assert ref.exists()
