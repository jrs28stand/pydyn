"""Fast subset of the acceptance sweep (tools/accept_sweep.py) against Dynare 7.2 results stored by tools/accept_octave.py.
Skipped when the reference folder (env PYDYN_ACCEPT_REF, default /home/claude/accept_ref) or a file's result is absent."""
import os
import sys
from pathlib import Path

import pytest

REF = Path(os.environ.get("PYDYN_ACCEPT_REF", "/home/claude/accept_ref"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
FILES = ["stochastic_simulations/example1", "external_function/benchmark", "deterministic_simulations/rbc_det5",
         "occbin/model_borrcon/borrcon", "irfs/example1_unit_std"]


@pytest.mark.parametrize("name", FILES)
def test_against_dynare(name):
    if not (REF / f"{name}.mat").exists():
        pytest.skip("no Dynare reference for " + name)
    if not os.environ.get("DYNARE_PREPROCESSOR") and not Path("/home/claude/dl/src72/build-octave/preprocessor/src/dynare-preprocessor").exists():
        pytest.skip("Dynare 7.2 preprocessor not available")
    import accept_sweep as s
    out = s.run_one_to_file(name + ".mod", s.entries()[name + ".mod"], 300)
    assert out["verdict"] in ("match", "partial"), out["detail"]
