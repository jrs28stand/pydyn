"""The single dseries implementation (pydyn.dseries.DSeries): field and brace assignment, merge order, lag/lead, the
interplay with dates objects, against Dynare 7.2's dseries class (reference/dseries/dseries_assign.mat from
reference/dseries/r_dseries_assign.m), and the pac.estimate refusals and optimisers that Octave cannot run."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pydyn import dseries as D
from pydyn import matlab as M

REF = Path(__file__).resolve().parent.parent / "reference" / "dseries"

SCRIPT = """
a = dseries([1 2 3; 4 5 6; 7 8 9], dates('2000Q1'), {'b','a','c'});
x = dseries([10;11;12], dates('2000Q2'), 'z');
a.z = x;
d1 = a.data; n1 = a.name;
a{'b'} = dseries([0;0;0;0],dates('2000Q1'),'q');
d2 = a.data;
a.init = dates('2001Q1'); i3 = a.init.char;
l = a.lag(1); ld = l.data;
a{'a'}(dates('2001Q2'):dates('2001Q3')) = [100;200]; d4 = a.data;
b = dseries([1 2; 3 4; 5 6], dates('2000Q1'), {'u','v'});
b.u(dates('2000Q2')) = 50; d5 = b.data;
b{'v'} = b{'u'}.lag(1); d6 = b.data; n6 = b.name;
"""


def test_assignment_matches_dynare():
    w = M.run(SCRIPT)
    r = loadmat(REF / "dseries_assign.mat", squeeze_me=True)
    for k in ("d1", "d2", "ld", "d4", "d5", "d6"):
        a, b = np.asarray(r[k], float), np.asarray(w[k], float)
        assert a.shape == b.shape, k
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=0, rtol=0), k
    assert str(w["i3"]) == str(r["i3"])
    assert list(w["n6"].items) == list(r["n6"])
    assert list(w["n1"].items) == list(r["n1"])


def test_one_class():
    assert not hasattr(M, "DSeriesV")
    w = M.run("a = dseries([1;2;3], dates('2000Q1'), 'x'); c = class(a); t = isdseries(a); e = isempty(a); n = length(a);\n"
              "i = a.init; l = a.last; ds = a.dates; s = size(a); ab = abs(dseries([-1;2], dates('2000Y'), 'q'));")
    assert w["c"] == "dseries" and bool(w["t"]) and not bool(w["e"]) and float(w["n"]) == 3
    assert w["i"].strings() == ["2000Q1"] and w["l"].strings() == ["2000Q3"] and w["ds"].ndat == 3     # dates objects
    assert list(np.ravel(w["s"])) == [3, 1]
    assert isinstance(w["a"], D.DSeries) and np.array_equal(w["ab"].data.ravel(), [1, 2])


@pytest.mark.parametrize("stmt, msg", [
    ("a.data = 1;", "You cannot overwrite data member"),
    ("a.z = 3;", "Can only assign dseries object"),
    ("a{'x', 'x'} = a;", "Wrong syntax"),
])
def test_assignment_errors(stmt, msg):
    with pytest.raises(M.MatlabError, match=msg):
        M.run("a = dseries([1;2;3], dates('2000Q1'), 'x');\n" + stmt)


def test_dseries_refusals():
    with pytest.raises(M.Unsupported):
        M.run("a = dseries([1;2;3], dates('2000Q1'), 'x'); a{'y'}(dates('2005Q1')) = 1;")
    with pytest.raises(M.Unsupported):
        M.run("a = dseries([1;2;3], dates('2000Q1'), 'x'); a{'y'} = [];")
