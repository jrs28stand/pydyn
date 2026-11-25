"""The default variable list comes from the last stoch_simul statement (Dynare's final results)."""
import pydyn
from pydyn.moments import _default_var_list

MOD = """var y c; varexo e; parameters a;
a = 0.5;
model; y = a*y(-1) + e; c = y; end;
shocks; var e; stderr 1; end;
stoch_simul(order=1, irf=0) {first};
stoch_simul(order=1, irf=0) {second};
"""


def _model(tmp_path, first, second):
    p = tmp_path / "vl.mod"
    p.write_text(MOD.format(first=first, second=second))
    return pydyn.Model(pydyn.run_preprocessor(p))


def test_last_statement_without_list_means_all_variables(tmp_path):
    assert _default_var_list(_model(tmp_path, "y", "")) == ["y", "c"]


def test_last_statement_list_wins(tmp_path):
    assert _default_var_list(_model(tmp_path, "", "c")) == ["c"]
