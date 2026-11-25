"""Model expressions cannot reach Python's builtins (exit, input, open, ...)."""
import pytest

import pydyn

MOD = """var y; varexo e; parameters a;
a = 0.5;
model; y = a*y(-1) + e; end;
steady_state_model; y = {rhs}; end;
"""


@pytest.fixture
def model(tmp_path):
    def make(rhs="0"):
        p = tmp_path / "sandbox.mod"
        p.write_text(MOD.format(rhs=rhs))
        return pydyn.Model(pydyn.run_preprocessor(p))
    return make


@pytest.mark.parametrize("expr", ["exit(3)", "quit()", "0*id(1)+1", "input()", "breakpoint()", "open"])
def test_eval_expr_has_no_python_builtins(model, expr):
    with pytest.raises(NameError):
        model().eval_expr(expr)


def test_eval_expr_still_evaluates_model_expressions(model):
    m = model()
    assert m.eval_expr("a*2 + max(1, 3) + exp(0)") == pytest.approx(5.0)


def test_steady_state_model_cannot_call_exit(model):
    m = model("exit(3)")
    with pytest.raises(NameError):
        pydyn.steady_state(m)


def test_numpy_warnings_inside_model_code_still_work(model):
    import warnings
    import numpy as np
    from pydyn.expr import NAMESPACE
    ns = dict(NAMESPACE, x=np.array([-1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(RuntimeWarning):
            eval("log(x)", ns)


def test_only_warnings_can_be_imported_from_model_code(model):
    from pydyn.expr import NAMESPACE
    with pytest.raises(ImportError):
        eval("__import__(x)", dict(NAMESPACE, x="os"))
