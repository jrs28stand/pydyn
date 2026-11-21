"""pydyn.octave_rng internals: the vectorised and scalar paths of the generator consume it identically, the generator
state is exact at any point, and the two front ends (samplers' OctaveRandom, filters' OctaveRandomStream) give the same
numbers. Agreement with Octave itself is tested in test_samplers.py and test_nonlinear_estimation.py."""
import numpy as np
import pytest

from pydyn import octave_rng, random_streams
from pydyn.octave_rng import MTStream, OctaveRandom, OctaveRandomStream


def _twin(seed, words=None):
    """Two identical streams; with ``words``, both read the given raw words first."""
    a, b = MTStream(seed), MTStream(seed)
    if words is not None:
        for s in (a, b):
            s._buf = np.concatenate([np.asarray(words, np.uint64), s._bg.random_raw(5000)])
            s._p = 0
    return a, b


@pytest.mark.parametrize("k", [0, 1, 7, 8, 9, 1000, 70000])
def test_vector_and_scalar_paths_agree(k):
    a, b = _twin(3)
    np.testing.assert_array_equal(a.uniforms(k), [b.u53() for _ in range(k)])
    np.testing.assert_array_equal(a.normals(k), [b.normal() for _ in range(k)])
    assert a.u53() == b.u53() and a.normal() == b.normal()


def test_zero_uniform_is_rejected_in_both_paths():
    # words 2-3 give a = b = 0, which randu53 rejects: both paths skip that pair
    words = [123456789, 987654321, 0x1F, 0x3F, 555555555, 444444444] * 3
    a, b = _twin(0, words)
    u = a.uniforms(10)
    np.testing.assert_array_equal(u, [b.u53() for _ in range(10)])
    assert a._p == b._p == 2 * 10 + 6
    assert 0.0 not in u


def test_state_is_exact_across_blocks_and_regenerations():
    for steps in (0, 1, 311, 623, 624, 625, 1001, 65535, 65536, 140000):
        m, ref = MTStream(5), MTStream(5)
        m.uniforms(steps // 2)                       # vectorised: reads a whole block ahead
        for _ in range(steps // 2):                  # scalar
            ref.u53()
        s = m.get_state()
        np.testing.assert_array_equal(s, ref.get_state())
        assert 1 <= s[624] <= 624
        again = MTStream(state=s)
        assert again.u53() == m.u53() == ref.u53()


def test_front_ends_agree_and_seed_rules():
    r, s = OctaveRandom(42), OctaveRandomStream(42)
    np.testing.assert_array_equal(r.randn((6, 4)), s.randn(6, 4))
    np.testing.assert_array_equal(r.rand(13), s.rand(13, 1).ravel())
    assert random_streams.OctaveRandomStream is OctaveRandomStream       # compatibility module re-exports
    assert random_streams.make_stream("octave").__class__ is OctaveRandomStream
    # the filters' stream reduces any seed as Octave does; the samplers' generator accepts only verified seeds
    np.testing.assert_array_equal(OctaveRandomStream(2.0 ** 32 - 1).rand(3, 1), OctaveRandomStream(0).rand(3, 1))
    np.testing.assert_array_equal(OctaveRandomStream(-1).rand(3, 1), OctaveRandomStream(2.0 ** 32 - 2).rand(3, 1))
    with pytest.raises(ValueError, match="integers"):
        OctaveRandom(2.5)
    assert octave_rng.OctaveRandom(0).rand((0,)).shape == (0,)
