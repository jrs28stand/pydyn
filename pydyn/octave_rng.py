# This file is part of pydyn.
#
# It reproduces the random-number generators of GNU Octave (liboctave/numeric/randmtzig.cc and the state handling of
# liboctave/numeric/oct-rand.cc, Copyright (C) 2006-2023 The Octave Project Developers, GPL-3.0-or-later; the Mersenne
# Twister MT19937 and its init_by_array seeding are by Makoto Matsumoto and Takuji Nishimura, 1997-2002, BSD licence;
# the ziggurat method is that of Marsaglia and Tsang, 2000) as used by Dynare 7.2 under Octave
# (matlab/set_dynare_seed_local_options.m, matlab/set_dynare_random_generator_state.m, Copyright (C) 2010-2026 Dynare
# Team, GPL-3.0-or-later). Written in 2026 from the published algorithms and checked against Octave 8.4.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Octave's ``rand``, ``randn`` and ``randperm``, bit for bit, so that pydyn's samplers can reproduce Dynare's chains
(run under Octave) from Dynare's seeds alone.

Under Octave Dynare seeds a chain with ``rand('state', s); randn('state', s)``. Octave keeps one Mersenne Twister
state per distribution, so the uniform and the normal stream are two independent MT19937 generators that start from
the same state (``init_by_array([s])``). :class:`OctaveRandom` holds both::

    r = OctaveRandom(seed)          # rand('state', seed); randn('state', seed)
    r.rand(3); r.randn(4); r.randperm(6)

* ``rand``: 53-bit uniforms ``(a 2^26 + b) / 2^53`` with ``a``, ``b`` the top 27 and 26 bits of two MT outputs (never 0);
* ``randn``: Octave's 256-level ziggurat (64-bit variant, 54 random bits per try; the tail and wedge tests draw
  uniforms from the normal generator);
* ``randperm(n, m)``: ``m`` uniforms from the uniform stream and a Knuth shuffle.

Checked against Octave 8.4 on 2 million uniforms and 2 million normals (seed 1) and on ``randperm`` interleaved with
``rand``/``randn`` (tests/test_samplers.py). Only integer seeds 0 <= s < 2^32 - 1 are emulated; a 625-element state
vector as returned by ``rand('state')`` (Dynare's ``record.InitialSeeds``/``LastSeeds``) can also be set.

:func:`dynare_chain_seeds` gives the seeds of Dynare's chains (chain ``j`` of a run with seed ``k`` is seeded with
``k + 1 + 2 + ... + j``, posterior_sampler_initialization.m) and :class:`NumpyRandom` the same interface on a numpy
generator (pydyn's default: statistically equivalent, not Dynare's numbers).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["OctaveRandom", "NumpyRandom", "ArrayRandom", "MTStream", "dynare_chain_seeds", "init_by_array"]

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
_NMANTISSA = 9007199254740992.0


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


_KI, _WI, _FI = _ziggurat_tables()


def _seed_word(seed) -> int:
    s = float(seed)
    if not (np.isfinite(s) and s == int(s) and 0 <= s < 4294967295.0):
        raise ValueError(f"Octave seed {seed!r}: only integers 0 <= seed < 2^32 - 1 are emulated")
    return int(s)


class MTStream:
    """One MT19937 generator with Octave's conversions (one Octave distribution state)."""

    _BLOCK = 4096

    def __init__(self, seed=None, state=None):
        self._bg = np.random.MT19937(0)
        self._buf = np.empty(0, np.uint64)
        self._p = 0
        if state is not None:
            self.set_state(state)
        else:
            self.seed(0 if seed is None else seed)

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
        st = self._bg.state["state"]
        key, pos = np.array(st["key"], np.uint32), int(st["pos"])
        unread = len(self._buf) - self._p
        if unread:                                # step the numpy generator back over the prefetched words
            pos -= unread
            if pos < 0:
                raise RuntimeError("cannot recover the state across a regeneration; call get_state before drawing")
        return np.concatenate([key.astype(float), [625 - pos]])

    def _set(self, key, pos):
        st = self._bg.state
        st["state"]["key"] = np.asarray(key, np.uint32)
        st["state"]["pos"] = int(pos)
        self._bg.state = st
        self._buf = np.empty(0, np.uint64)
        self._p = 0

    def _fill(self):
        # prefetch at most up to the next regeneration of the MT state, so that get_state stays exact
        pos = int(self._bg.state["state"]["pos"])
        n = 624 - pos if pos < 624 else 624
        self._buf = self._bg.random_raw(n)
        self._p = 0

    def r32(self) -> int:
        if self._p >= len(self._buf):
            self._fill()
        v = int(self._buf[self._p])
        self._p += 1
        return v

    def u53(self) -> float:
        while True:
            a = self.r32() >> 5
            b = self.r32() >> 6
            if a or b:
                return (a * 67108864.0 + b) / 9007199254740992.0

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


def _shape(size):
    if size is None:
        return ()
    return (int(size),) if np.isscalar(size) else tuple(int(s) for s in size)


def _octave_fill(shape, gen):
    """Arrays are filled in column-major order, as Octave does."""
    n = int(np.prod(shape)) if shape else 1
    vals = np.array([gen() for _ in range(n)], float)
    if not shape:
        return float(vals[0])
    return vals.reshape(shape, order="F")


class OctaveRandom:
    """Octave's uniform and normal generators after ``rand('state', seed); randn('state', seed)``.

    ``rand(size)`` / ``randn(size)`` return arrays filled in column-major order (``rand(3)`` is a vector of three
    numbers here, not Octave's 3 x 3 matrix); without ``size`` a float. ``randperm(n, m)`` as Octave's."""

    def __init__(self, seed=0, state_u=None, state_n=None):
        self.u = MTStream(seed, state_u)
        self.n = MTStream(seed, state_n)

    def rand(self, size=None):
        return _octave_fill(_shape(size), self.u.u53)

    def randn(self, size=None):
        return _octave_fill(_shape(size), self.n.normal)

    def randperm(self, n, m=None):
        m = n if m is None else m
        rv = [self.u.u53() for _ in range(m)]
        iv = list(range(n))
        for i in range(m):
            k = i + int(math.floor(rv[i] * (n - i)))
            iv[i], iv[k] = iv[k], iv[i]
        return np.array(iv[:m], int)               # 0-based

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
        rv = self.rng.random(m)
        iv = list(range(n))
        for i in range(m):
            k = i + int(math.floor(rv[i] * (n - i)))
            iv[i], iv[k] = iv[k], iv[i]
        return np.array(iv[:m], int)


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
        rv = np.atleast_1d(rv)
        iv = list(range(n))
        for i in range(m):
            k = i + int(math.floor(rv[i] * (n - i)))
            iv[i], iv[k] = iv[k], iv[i]
        return np.array(iv[:m], int)


def dynare_chain_seeds(seed: int, nblocks: int) -> list:
    """Seeds of Dynare's chains 1..nblocks for a run with ``set_dynare_seed(seed)`` (default 0):
    ``seed + 1``, ``seed + 1 + 2``, ... (posterior_sampler_initialization.m adds ``j`` to the seed cumulatively)."""
    out, s = [], int(seed)
    for j in range(1, nblocks + 1):
        s += j
        out.append(s)
    return out
