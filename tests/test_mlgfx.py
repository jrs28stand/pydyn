"""Graphics calls in MATLAB code (pydyn/mlgfx.py).  Octave 8.4 in the reference environment has no graphics toolkit, so
``figure`` cannot run there: the handle values below are Octave's documented ones (a figure is the integer 1, 2, ...; any
other graphics object has a negative fractional handle), and the Octave-checked part is ``ishandle`` and friends on values
that are not handles (reference/matlab_octave_cases/pydyn_gfx_cases.txt)."""
import numpy as np
import pytest

from pydyn import matlab as M


@pytest.fixture(autouse=True)
def _graphics_on(monkeypatch):
    monkeypatch.setattr(M.Interpreter, "graphics_enabled", True, raising=False)


def run(code):
    return M.run(code)


def test_plain_interpreter_still_refuses():
    import pydyn.matlab as m
    m.Interpreter.graphics_enabled = False
    with pytest.raises(M.Unsupported, match="plotting"):
        M.run("figure; plot(1:3)")


def test_figure_numbers():
    r = run("a = figure; b = figure; figure(a); c = gcf; close(b); d = figure; e = ishandle([a b d 9]);")
    assert (r["a"], r["b"], r["c"], r["d"]) == (1, 2, 1, 2)
    assert list(np.ravel(r["e"])) == [True, True, True, False]


def test_object_handles_are_negative_fractions():
    r = run("h = figure('Name', 'x'); s = subplot(2, 1, 1); p = plot(1:3); t = title('a'); l = legend('b'); "
            "k = ishandle([s p t l]); ty = get(s, 'type'); tp = get(p, 'type'); tt = get(t, 'type'); tf = get(h, 'type');")
    for k in ("s", "p", "t", "l"):
        assert -1 < float(r[k]) < 0
    assert len({float(r[k]) for k in ("s", "p", "t", "l")}) == 4
    assert all(np.ravel(r["k"]))
    assert (r["ty"], r["tp"], r["tt"], r["tf"]) == ("axes", "line", "text", "figure")


def test_subplot_reuses_axes_and_plot_counts_lines():
    r = run("a = subplot(2,2,1); b = subplot(2,2,1); c = subplot(2,2,2); p = plot([1 2 3; 4 5 6]'); q = plot(1:3, [1 2 3]);"
            "w = plot(1:2, [1 2], 'r-', 1:2, [3 4], 'b--'); n = [numel(p) numel(q) numel(w)];")
    assert r["a"] == r["b"] and r["a"] != r["c"]
    assert list(np.ravel(r["n"])) == [2, 1, 2]


def test_close_all_and_gcf_creates():
    r = run("figure; figure; close all; e = isempty(findobj_dummy_never_used_zero); ")if False else run(
        "figure; figure; close all; f = gcf; ")
    assert r["f"] == 1


def test_never_change_results_and_no_output_calls():
    r = run("x = 3; figure; hold on; plot(1:3); xlabel('a'); ylabel('b'); axis tight; grid on; set(gca, 'FontSize', 8); "
            "drawnow; close all; y = x + 1;")
    assert r["y"] == 4


@pytest.mark.parametrize("code", ["print('-dpng', 'a.png')", "saveas(gcf, 'a.png')", "savefig('a.fig')",
                                  "a = axis;", "a = xlim;", "x = get(gcf, 'position');", "a = hold;", "set(7, 'a', 1)"])
def test_refusals(code):
    with pytest.raises((M.Unsupported, M.MatlabError)):
        run(code)


def test_set_on_non_handle_is_an_error():
    with pytest.raises(M.MatlabError):
        run("set(7, 'a', 1)")
