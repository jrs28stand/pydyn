# This file is part of pydyn.
#
# It reproduces the random-number generators of GNU Octave (liboctave/numeric/randmtzig.cc and the state handling of
# liboctave/numeric/oct-rand.cc, Copyright (C) 2006-2024 The Octave Project Developers, GPL-3.0-or-later; the Mersenne
# Twister MT19937 and its init_by_array seeding are by Makoto Matsumoto and Takuji Nishimura, 1997-2002, BSD-3-Clause;
# the ziggurat method is that of Marsaglia and Tsang, 2000, as implemented in Octave by David Bateman) as used by
# Dynare 7.2 under Octave (matlab/set_dynare_seed_local_options.m, matlab/set_dynare_random_generator_state.m,
# Copyright (C) 2010-2026 Dynare Team, GPL-3.0-or-later). Written in 2026 from the published algorithms and checked
# against Octave 8.4.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Octave's ``rand``, ``randn`` and ``randperm``, bit for bit, so that pydyn reproduces Dynare's random numbers
(run under Octave) from Dynare's seeds alone. This is pydyn's only port of Octave's generator; it has two front ends.

Under Octave Dynare seeds with ``rand('state', s); randn('state', s)``. Octave keeps one Mersenne Twister state per
distribution, so the uniform and the normal stream are two independent MT19937 generators that start from the same
state (``init_by_array([s])``). :class:`MTStream` is one such generator:

* uniforms: 53-bit ``(a 2^26 + b) / 2^53`` with ``a``, ``b`` the top 27 and 26 bits of two MT outputs (never 0);
* normals: Octave's 256-level ziggurat (64-bit variant, 54 random bits per try; the tail and wedge tests draw
  uniforms from the normal generator).

Both have a scalar and a vectorised path that consume the generator identically.

Front end for the posterior samplers (:mod:`pydyn.samplers`), Octave-shaped calls::

    r = OctaveRandom(seed)          # rand('state', seed); randn('state', seed)
    r.rand(3); r.randn((4, 2)); r.randperm(6)

with :class:`NumpyRandom` (the same interface on a numpy generator: statistically equivalent, not Dynare's numbers),
:class:`ArrayRandom` (recorded numbers) and :func:`dynare_chain_seeds` (chain ``j`` of a run with seed ``k`` is seeded
with ``k + 1 + 2 + ... + j``, posterior_sampler_initialization.m). ``randperm(n, m)`` takes ``m`` uniforms from the
uniform stream and does a Knuth shuffle.

Front end for the particle filters (:mod:`pydyn.nonlinear_filters`), MATLAB's ``randn(r, c)``/``rand(r, c)`` calls:
:class:`OctaveRandomStream` (seed converted as Octave's ``double2uint32`` does; Dynare's filters reseed with 0 before
each likelihood evaluation, so a pydyn likelihood equals Dynare's under Octave bit for bit),
:class:`NumpyRandomStream` (numpy's PCG64), :class:`ReplayStream` (recorded numbers) and :func:`make_stream`.

Checked against Octave 8.4 on 2 million uniforms and 2 million normals (seeds 1 and 7), on generator state vectors,
on ``randperm`` interleaved with ``rand``/``randn`` (tests/test_samplers.py) and on four seeds of
reference/r_octave_rng.m (tests/test_nonlinear_estimation.py). :class:`OctaveRandom` emulates only integer seeds
0 <= s < 2^32 - 1; a 625-element state vector as returned by ``rand('state')`` (Dynare's
``record.InitialSeeds``/``LastSeeds``) can also be set.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["OctaveRandom", "NumpyRandom", "ArrayRandom", "MTStream", "dynare_chain_seeds", "init_by_array",
           "OctaveRandomStream", "NumpyRandomStream", "ReplayStream", "make_stream"]

_M32 = 0xFFFFFFFF


def init_by_array(key) -> np.ndarray:
    """MT19937 state (624 words) after ``init_by_array(key)`` (Matsumoto and Nishimura's reference code)."""
    key = [int(k) & _M32 for k in key]
    mt = [0] * 624
    mt[0] = 19650218
    for i in range(1, 624):
        mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & _M32
    i, j = 1, 0
    for _ in range(max(624, len(key))):
        mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525)) + key[j] + j) & _M32
        i += 1
        j += 1
        if i >= 624:
            mt[0] = mt[623]
            i = 1
        if j >= len(key):
            j = 0
    for _ in range(623):
        mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i) & _M32
        i += 1
        if i >= 624:
            mt[0] = mt[623]
            i = 1
    mt[0] = 0x80000000
    return np.array(mt, dtype=np.uint32)


# ziggurat tables of randmtzig.cc (normal distribution, ZIGGURAT_TABLE_SIZE 256, 64-bit integers)
_NOR_R = 3.6541528853610088
_NOR_INV_R = 0.27366123732975828
_NOR_AREA = 0.00492867323399
_NMANTISSA = 9007199254740992.0           # 2^53


def _ziggurat_tables():
    ki, wi, fi = [0] * 256, [0.0] * 256, [0.0] * 256
    x1 = _NOR_R
    wi[255] = x1 / _NMANTISSA
    fi[255] = math.exp(-0.5 * x1 * x1)
    ki[0] = int(x1 * fi[255] / _NOR_AREA * _NMANTISSA)
    wi[0] = _NOR_AREA / fi[255] / _NMANTISSA
    fi[0] = 1.0
    for i in range(254, 0, -1):
        x = math.sqrt(-2.0 * math.log(_NOR_AREA / x1 + fi[i + 1]))
        ki[i + 1] = int(x / x1 * _NMANTISSA)
        wi[i] = x / _NMANTISSA
        fi[i] = math.exp(-0.5 * x * x)
        x1 = x
    ki[1] = 0
    return ki, wi, fi


_KI, _WI, _FI = _ziggurat_tables()                       # Python lists (scalar path)
_KI_A, _WI_A = np.array(_KI, dtype=np.int64), np.array(_WI)   # arrays (vectorised path)


def _seed_word(seed) -> int:
    """Strict seed check of :class:`OctaveRandom`: only the seeds whose Octave behaviour was verified."""
    s = float(seed)
    if not (np.isfinite(s) and s == int(s) and 0 <= s < 4294967295.0):
        raise ValueError(f"Octave seed {seed!r}: only integers 0 <= seed < 2^32 - 1 are emulated")
    return int(s)


def _double2uint32(d) -> int:
    """Octave's conversion of a seed value to a 32-bit word (oct-rand.cc)."""
    if not math.isfinite(d):
        return 0
    top = float(_M32)
    d = math.fmod(float(d), top)
    if d < 0:
        d += top
    return int(d)


class MTStream:
    """One MT19937 generator with Octave's conversions (one Octave distribution state).

    Words are drawn from numpy's MT19937 in blocks; the generator state at the start of the current block is kept so
    that :meth:`get_state` is exact at any point."""

    _BLOCK = 65536
    _SMALL = 8                  # below this many numbers the scalar path is faster than the vectorised one

    def __init__(self, seed=None, state=None, *, word=None):
        self._bg = np.random.MT19937(0)
        if state is not None:
            self.set_state(state)
        elif word is not None:
            self._set(init_by_array([int(word) & _M32]), 624)
        else:
            self.seed(0 if seed is None else seed)

    # -- state
    def seed(self, seed):
        """``rand('state', seed)``."""
        self._set(init_by_array([_seed_word(seed)]), 624)

    def set_state(self, state):
        """A 625-element Octave state vector (``rand('state')``): 624 words and the count ``left``."""
        s = np.asarray(state, float).ravel()
        if s.size != 625 or not (0 < s[624] <= 624):
            raise ValueError("an Octave generator state has 625 elements, the last in 1..624")
        self._set(s[:624].astype(np.uint64).astype(np.uint32), 625 - int(s[624]))

    def get_state(self) -> np.ndarray:
        """The state vector Octave would return now (625 doubles)."""
        bg = np.random.MT19937(0)
        bg.state = self._origin
        if self._p:
            bg.random_raw(self._p)
        st = bg.state["state"]
        return np.concatenate([np.asarray(st["key"], np.uint32).astype(float), [625 - int(st["pos"])]])

    def _set(self, key, pos):
        st = self._bg.state
        st["state"]["key"] = np.asarray(key, np.uint32)
        st["state"]["pos"] = int(pos)
        self._bg.state = st
        self._origin = self._bg.state             # generator state at self._buf[0]
        self._buf = np.empty(0, np.uint64)
        self._p = 0

    # -- raw words
    def _refill(self, need):
        """Make at least ``need`` unread words available: rewind to the first unread word and draw a new block."""
        if self._p < len(self._buf):              # unread words: put the generator back at the first of them
            self._bg.state = self._origin
            if self._p:
                self._bg.random_raw(self._p)
        self._origin = self._bg.state
        self._buf = self._bg.random_raw(max(int(need), self._BLOCK))
        self._p = 0

    def _take(self, k):
        if self._p + k > len(self._buf):
            self._refill(k)
        out = self._buf[self._p:self._p + k]
        self._p += k
        return out

    def r32(self) -> int:
        if self._p >= len(self._buf):
            self._refill(1)
        v = int(self._buf[self._p])
        self._p += 1
        return v

    # -- scalar path
    def u53(self) -> float:
        while True:
            a = self.r32() >> 5
            b = self.r32() >> 6
            if a or b:
                return (a * 67108864.0 + b) / _NMANTISSA

    def normal(self) -> float:
        ki, wi, fi = _KI, _WI, _FI
        while True:
            lo = self.r32()
            hi = self.r32() & 0x3FFFFF
            r = (hi << 32) | lo
            rabs = r >> 1
            idx = rabs & 0xFF
            x = (-rabs if (r & 1) else rabs) * wi[idx]
            if rabs < ki[idx]:
                return x
            if idx == 0:
                while True:
                    xx = -_NOR_INV_R * math.log(self.u53())
                    yy = -math.log(self.u53())
                    if yy + yy > xx * xx:
                        break
                return (-_NOR_R - xx) if (rabs & 0x100) else (_NOR_R + xx)
            if (fi[idx - 1] - fi[idx]) * self.u53() + fi[idx] < math.exp(-0.5 * x * x):
                return x

    # -- vectorised path (same consumption of words as the scalar path)
    def uniforms(self, k: int) -> np.ndarray:
        """``k`` uniforms (1-D array)."""
        k = int(k)
        if k < self._SMALL:
            return np.array([self.u53() for _ in range(k)], float)
        out = np.empty(k)
        got = 0
        while got < k:
            m = k - got
            w = self._take(2 * m)
            a = (w[0::2] >> np.uint64(5)).astype(np.float64)
            b = (w[1::2] >> np.uint64(6)).astype(np.float64)
            bad = (a == 0) & (b == 0)
            if bad.any():                       # u53 rejects a zero (probability 2^-53)
                f = int(np.argmax(bad))
                self._p -= 2 * (m - f) - 2      # the rejected pair stays consumed
                out[got:got + f] = (a[:f] * 67108864.0 + b[:f]) / _NMANTISSA
                got += f
                continue
            out[got:] = (a * 67108864.0 + b) / _NMANTISSA
            got = k
        return out

    def normals(self, k: int) -> np.ndarray:
        """``k`` standard normals (1-D array)."""
        k = int(k)
        if k < self._SMALL:
            return np.array([self.normal() for _ in range(k)], float)
        out = np.empty(k)
        got = 0
        while got < k:
            m = min(k - got, 4096)
            w = self._take(2 * m)
            lo = w[0::2]
            hi = w[1::2] & np.uint64(0x3FFFFF)
            r = (hi << np.uint64(32)) | lo
            rabs = (r >> np.uint64(1)).astype(np.int64)
            idx = rabs & 0xFF
            x = np.where((r & np.uint64(1)) == 1, -rabs, rabs).astype(np.float64) * _WI_A[idx]
            ok = rabs < _KI_A[idx]
            if ok.all():
                out[got:got + m] = x
                got += m
                continue
            f = int(np.argmin(ok))               # first number that needs the slow path
            out[got:got + f] = x[:f]
            got += f
            self._p -= 2 * (m - f)               # give back its words; the scalar path redraws it
            out[got] = self.normal()
            got += 1
        return out


# ====================================================================== sampler front end (Octave-shaped calls)
def _shape(size):
    if size is None:
        return ()
    return (int(size),) if np.isscalar(size) else tuple(int(s) for s in size)


def _knuth_shuffle(rv, n, m):
    iv = list(range(n))
    for i in range(m):
        k = i + int(math.floor(rv[i] * (n - i)))
        iv[i], iv[k] = iv[k], iv[i]
    return np.array(iv[:m], int)                  # 0-based


class OctaveRandom:
    """Octave's uniform and normal generators after ``rand('state', seed); randn('state', seed)``.

    ``rand(size)`` / ``randn(size)`` return arrays filled in column-major order (``rand(3)`` is a vector of three
    numbers here, not Octave's 3 x 3 matrix); without ``size`` a float. ``randperm(n, m)`` as Octave's."""

    def __init__(self, seed=0, state_u=None, state_n=None):
        self.u = MTStream(seed, state_u)
        self.n = MTStream(seed, state_n)

    @staticmethod
    def _fill(shape, stream_vec, scalar):
        if not shape:
            return scalar()
        return stream_vec(int(np.prod(shape))).reshape(shape, order="F")

    def rand(self, size=None):
        return self._fill(_shape(size), self.u.uniforms, self.u.u53)

    def randn(self, size=None):
        return self._fill(_shape(size), self.n.normals, self.n.normal)

    def randperm(self, n, m=None):
        m = n if m is None else m
        return _knuth_shuffle([self.u.u53() for _ in range(m)], n, m)

    def get_state(self):
        return self.u.get_state(), self.n.get_state()


class NumpyRandom:
    """The interface of :class:`OctaveRandom` on a numpy ``Generator`` (pydyn's default random numbers)."""

    def __init__(self, rng=None):
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def rand(self, size=None):
        s = _shape(size)
        return float(self.rng.random()) if not s else self.rng.random(s)

    def randn(self, size=None):
        s = _shape(size)
        return float(self.rng.standard_normal()) if not s else self.rng.standard_normal(s)

    def randperm(self, n, m=None):
        m = n if m is None else m
        return _knuth_shuffle(self.rng.random(m), n, m)


class ArrayRandom:
    """Uniforms and normals taken in order from given arrays (random numbers recorded elsewhere); raises
    ``IndexError`` when one runs out."""

    def __init__(self, uniforms=(), normals=()):
        self._u = np.asarray(uniforms, float).ravel()
        self._z = np.asarray(normals, float).ravel()
        self.iu = self.iz = 0

    def _take(self, arr, attr, shape):
        n = int(np.prod(shape)) if shape else 1
        i = getattr(self, attr)
        if i + n > arr.size:
            raise IndexError("ArrayRandom: not enough recorded random numbers")
        setattr(self, attr, i + n)
        v = arr[i:i + n]
        return float(v[0]) if not shape else v.reshape(shape, order="F")

    def rand(self, size=None):
        return self._take(self._u, "iu", _shape(size))

    def randn(self, size=None):
        return self._take(self._z, "iz", _shape(size))

    def randperm(self, n, m=None):
        m = n if m is None else m
        rv = self.rand(m) if m else np.zeros(0)
        return _knuth_shuffle(np.atleast_1d(rv), n, m)


def dynare_chain_seeds(seed: int, nblocks: int) -> list:
    """Seeds of Dynare's chains 1..nblocks for a run with ``set_dynare_seed(seed)`` (default 0):
    ``seed + 1``, ``seed + 1 + 2``, ... (posterior_sampler_initialization.m adds ``j`` to the seed cumulatively)."""
    out, s = [], int(seed)
    for j in range(1, nblocks + 1):
        s += j
        out.append(s)
    return out


# ====================================================================== particle-filter front end (randn(r, c) calls)
class OctaveRandomStream:
    """The numbers of Octave's ``rand(r, c)`` and ``randn(r, c)`` after ``rand('state', seed)`` and
    ``randn('state', seed)`` (two independent Mersenne Twisters), filled in column-major order. Any finite seed is
    reduced to 32 bits as Octave does (``double2uint32``)."""

    def __init__(self, seed=0):
        word = _double2uint32(seed)
        self._u = MTStream(word=word)
        self._n = MTStream(word=word)

    def randn(self, r, c=1):
        return self._n.normals(int(r) * int(c)).reshape((int(r), int(c)), order="F")

    def rand(self, r, c=1):
        return self._u.uniforms(int(r) * int(c)).reshape((int(r), int(c)), order="F")


class NumpyRandomStream:
    """numpy's PCG64 generators (one for uniform, one for normal numbers), column-major fill."""

    def __init__(self, seed=0):
        su, sn = np.random.SeedSequence(seed).spawn(2)
        self._u = np.random.default_rng(su)
        self._n = np.random.default_rng(sn)

    def randn(self, r, c=1):
        return self._n.standard_normal((int(c), int(r))).T

    def rand(self, r, c=1):
        return self._u.random((int(c), int(r))).T


class ReplayStream:
    """Replays recorded normal and uniform numbers in order (column-major fill); raises ``RuntimeError`` when more
    are requested than were recorded."""

    def __init__(self, normals=(), uniforms=()):
        self.normals = np.asarray(normals, float).ravel(order="F")
        self.uniforms = np.asarray(uniforms, float).ravel(order="F")
        self.i_n = self.i_u = 0

    def randn(self, r, c=1):
        k = int(r) * int(c)
        if self.i_n + k > self.normals.size:
            raise RuntimeError("ReplayStream: not enough recorded normal numbers")
        out = self.normals[self.i_n:self.i_n + k]
        self.i_n += k
        return out.reshape((int(r), int(c)), order="F")

    def rand(self, r, c=1):
        k = int(r) * int(c)
        if self.i_u + k > self.uniforms.size:
            raise RuntimeError("ReplayStream: not enough recorded uniform numbers")
        out = self.uniforms[self.i_u:self.i_u + k]
        self.i_u += k
        return out.reshape((int(r), int(c)), order="F")


def make_stream(spec, seed=0):
    """A fresh stream from ``spec``: ``'octave'`` (default), ``'numpy'``, a callable returning a stream
    (called with no argument), or a stream instance (returned as is)."""
    if spec is None or spec == "octave":
        return OctaveRandomStream(seed)
    if spec == "numpy":
        return NumpyRandomStream(seed)
    if callable(spec) and not hasattr(spec, "randn"):
        return spec()
    if hasattr(spec, "randn") and hasattr(spec, "rand"):
        return spec
    raise ValueError(f"unknown random stream {spec!r}: use 'octave', 'numpy', a stream or a factory")
