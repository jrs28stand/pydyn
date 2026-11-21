"""write_latex_* commands: no LaTeX output in pydyn, said once per model instead of silently."""
import warnings

import pydyn
from pydyn.options import UnsupportedOptionWarning

MOD = """var y; varexo e; parameters rho; rho = 0.5;
model; y = rho*y(-1) + e; end;
shocks; var e; stderr 1; end;
write_latex_dynamic_model;
write_latex_static_model;
write_latex_definitions;
"""


def test_latex_commands_warn(tmp_path):
    (tmp_path / "lx.mod").write_text(MOD)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "lx.mod"))
    msgs = [str(w.message) for w in rec if issubclass(w.category, UnsupportedOptionWarning)]
    for kind in ("write_latex_dynamic_model", "write_latex_static_model"):
        assert sum(kind in s and "LaTeX output is not implemented" in s for s in msgs) == 1
    # write_latex_definitions reaches pydyn as a MATLAB function call: reported with the MATLAB code not run
    assert any("write_latex_definitions" in str(w.message) or "not executed" in str(w.message) for w in rec)
    assert not list(tmp_path.glob("**/*.tex"))
    assert "write_latex_dynamic_model" not in m.unhandled_statements
