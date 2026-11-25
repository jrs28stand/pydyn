# This file is part of pydyn.
#
# It is a Python port of parts of Dynare's dseries toolbox shipped with Dynare 7.2 (Copyright (C)
# 2011-2026 Dynare Team, https://www.dynare.org), modified in 2026: the dates class
# (matlab/dseries/src/@dates: plus, minus, comparisons, colon) and
# matlab/dseries/src/utilities/convert/{string2date, date2string}.m.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Dynare dates: ``1990Y`` (or ``1990A``), ``1990S2`` (or ``1990H2``), ``1990Q1``, ``1990M3``,
``1990W12`` (ISO weeks) and ``1990-03-15`` (days), with Dynare's arithmetic.

A :class:`Date` is a frequency and an integer ``time`` as in Dynare (``year * freq + subperiod`` for
yearly to monthly dates, days or ISO weeks counted from a fixed origin otherwise), so that
``date + k`` moves by ``k`` periods and ``date1 - date2`` is the number of periods between them.
Dates of different frequencies cannot be compared or subtracted. No dseries: data files whose first
column holds dates are read by ``pydyn.estimation.load_data`` / ``pydyn.smoother.load_data``
(``dated=True``).
"""
from __future__ import annotations

import datetime as _dt
import re
from functools import total_ordering

FREQ_LETTER = {1: "Y", 2: "S", 4: "Q", 12: "M", 52: "W"}
_PATTERNS = [
    (1, re.compile(r"^\s*(-?\d+)\s*[YyAa]\s*$")),
    (2, re.compile(r"^\s*(-?\d+)\s*[HhSs]\s*([12])\s*$")),
    (4, re.compile(r"^\s*(-?\d+)\s*[Qq]\s*([1-4])\s*$")),
    (12, re.compile(r"^\s*(-?\d+)\s*[Mm]\s*(1[0-2]|0?[1-9])\s*$")),
    (52, re.compile(r"^\s*(\d+)\s*[Ww]\s*(\d{1,2})\s*$")),
    (365, re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")),
]
_WRAPPED = re.compile(r"""^\s*\(?\s*dates\(\s*['"]([^'"]+)['"]\s*\)\s*\)?\s*$""")
# Monday of ISO week 1 of year 1: the origin of the weekly time index
_W0 = _dt.date.fromisocalendar(1, 1, 1).toordinal()
# Dynare counts the weeks from year 0 (weeksince.m): year 0 has 52 ISO weeks, so the times are those counted from year 1 plus 52
_WOFF = 52


class DateError(ValueError):
    pass


@total_ordering
class Date:
    __slots__ = ("freq", "time")

    def __init__(self, freq: int, time: int):
        if freq not in (1, 2, 4, 12, 52, 365):
            raise DateError(f"unknown frequency {freq}")
        self.freq, self.time = int(freq), int(time)

    # ---- construction
    @classmethod
    def parse(cls, s) -> "Date":
        """``'1990Q1'``, ``"dates('1990Q1')"`` (as the preprocessor writes them) or a :class:`Date`."""
        if isinstance(s, Date):
            return s
        s = str(s)
        m = _WRAPPED.match(s)
        if m:
            s = m.group(1)
        for freq, pat in _PATTERNS:
            m = pat.match(s)
            if not m:
                continue
            if freq == 1:
                return cls(1, int(m.group(1)))
            if freq == 52:
                y, w = int(m.group(1)), int(m.group(2))
                try:
                    day = _dt.date.fromisocalendar(y, w, 1)
                except ValueError:
                    raise DateError(f"{s!r}: week {w} does not exist in {y}") from None
                return cls(52, (day.toordinal() - _W0) // 7 + 1 + _WOFF)
            if freq == 365:
                d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                return cls(365, d.toordinal() + 366)               # MATLAB's datenum
            return cls(freq, int(m.group(1)) * freq + int(m.group(2)))
        raise DateError(f"{s!r} is not a date (expected e.g. 1990Y, 1990S1, 1990Q1, 1990M1, 1990W1 or 1990-01-31)")

    @staticmethod
    def is_date(s) -> bool:
        try:
            Date.parse(s)
            return True
        except (DateError, ValueError):
            return False

    # ---- arithmetic (dates/plus.m, minus.m)
    def __add__(self, k):
        if isinstance(k, Date):
            raise TypeError("cannot add two dates")
        if int(k) != k:
            raise DateError("only integers can be added to a date")
        return Date(self.freq, self.time + int(k))

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Date):
            self._same_freq(other)
            return self.time - other.time
        if int(other) != other:
            raise DateError("only integers can be subtracted from a date")
        return Date(self.freq, self.time - int(other))

    def _same_freq(self, other):
        if not isinstance(other, Date):
            raise TypeError(f"cannot compare a date with {other!r}")
        if self.freq != other.freq:
            raise DateError(f"dates {self} and {other} have different frequencies")

    def __eq__(self, other):
        return isinstance(other, Date) and self.freq == other.freq and self.time == other.time

    def __lt__(self, other):
        self._same_freq(other)
        return self.time < other.time

    def __hash__(self):
        return hash((self.freq, self.time))

    def range_to(self, last: "Date") -> list:
        """``d1:d2`` (inclusive)."""
        self._same_freq(last)
        return [Date(self.freq, t) for t in range(self.time, last.time + 1)]

    # ---- display (date2string.m)
    @property
    def year(self) -> int:
        if self.freq == 1:
            return self.time
        if self.freq in (2, 4, 12):
            return (self.time - 1) // self.freq
        return self.to_datetime().isocalendar()[0] if self.freq == 52 else self.to_datetime().year

    @property
    def subperiod(self) -> int:
        if self.freq in (2, 4, 12):
            return self.time - self.year * self.freq
        if self.freq == 52:
            return self.to_datetime().isocalendar()[1]
        return 1

    def to_datetime(self) -> _dt.date:
        if self.freq == 365:
            return _dt.date.fromordinal(self.time - 366)
        if self.freq == 52:
            return _dt.date.fromordinal(_W0 + 7 * (self.time - 1 - _WOFF))
        raise DateError("only weekly and daily dates convert to a calendar day")

    def __str__(self):
        if self.freq == 1:
            return f"{self.time}Y"
        if self.freq == 365:
            return self.to_datetime().isoformat()
        return f"{self.year}{FREQ_LETTER[self.freq]}{self.subperiod}"

    def __repr__(self):
        return f"Date('{self}')"


def parse_period(v, origin: Date | None = None, what="period"):
    """A period of the .mod file: an integer, or a date converted to a period counted from ``origin``
    (period 1 is ``origin``). Returns an int."""
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer()):
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
        return int(v)
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return parse_period(v[0], origin, what)
    d = Date.parse(v)
    if origin is None:
        raise DateError(f"{what} is given as a date ({d}) but no first period date is known to convert it")
    return (d - origin) + 1


def is_date_value(v) -> bool:
    if isinstance(v, Date):
        return True
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    return isinstance(v, str) and not re.fullmatch(r"\s*-?\d+\s*", v) and Date.is_date(v)


# ----------------------------------------------------------------------------------- data files and the data command
def _resolve(path, base_dir=None):
    from pathlib import Path
    p = Path(str(path).strip("'\""))
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    if p.suffix.lower() not in (".m", ".mat", ".csv"):
        found = [q for q in (p.with_name(p.name + e) for e in (".m", ".mat", ".csv")) if q.exists()]
        if len(found) == 1:
            p = found[0]
    return p


def data_start_date(path, base_dir=None):
    """Date of the first observation of a data file, or None: the first column of a .csv file (as dseries
    writes it), or ``INIT__`` in a .m / .mat file."""
    p = _resolve(path, base_dir)
    if not p.exists():
        return None
    ext = p.suffix.lower()
    if ext == ".csv":
        import csv
        with open(p, newline="") as fh:
            rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
        if len(rows) > 1 and rows[1] and Date.is_date(rows[1][0].strip()):
            return Date.parse(rows[1][0].strip())
        return None
    if ext == ".m":
        m = re.search(r"""^\s*INIT__\s*=\s*['"]([^'"]+)['"]""", p.read_text(errors="replace"), re.M)
        return Date.parse(m.group(1)) if m else None
    if ext == ".mat":
        from scipy.io import loadmat
        mat = loadmat(str(p), squeeze_me=True)
        v = mat.get("INIT__")
        return Date.parse(str(v)) if v is not None and Date.is_date(str(v)) else None
    return None


def _dseries_default_start(path, base_dir=None):
    """load_m_file_data.m / load_mat_file_data.m: "default values for freq and init are 1 and dates(1,1)" for a .m or .mat
    data file that names neither FREQ__ nor INIT__ (the data command's dates then count from 1Y)."""
    p = _resolve(path, base_dir)
    if not p.exists():
        return None
    ext = p.suffix.lower()
    if ext == ".m":
        return None if re.search(r"^\s*(FREQ__|INIT__)\s*=", p.read_text(errors="replace"), re.M) else Date.parse("1Y")
    if ext == ".mat":
        from scipy.io import loadmat
        mat = loadmat(str(p), squeeze_me=True)
        return None if ("FREQ__" in mat or "INIT__" in mat) else Date.parse("1Y")
    return None


def apply_data_command(model, opts, base_dir=None):
    """The ``data`` command (``estimation_data`` statement; Dynare's options_.dataset): ``file``,
    ``first_obs`` / ``last_obs`` (dates, or observation numbers), ``nobs``. Rewrites ``datafile``,
    ``first_obs`` and ``nobs`` of the estimation options ``opts`` (observation numbers counted from 1).
    Returns the date of the first observation used (None if the file has no dates)."""
    st = [s for s in model.statements if s.get("statementName") == "estimation_data"]
    if not st:
        start = data_start_date(opts["datafile"], base_dir) if opts.get("datafile") else None
        if start is not None:
            return start + (int(opts["first_obs"]) - 1)
        return None
    d = dict(st[-1].get("options", {}))
    for k in ("series", "xls_sheet", "xls_range"):
        if k in d:
            raise NotImplementedError(f"data({k}=...) is not supported: use a .m, .mat or .csv file")
    if opts.get("datafile") and opts["datafile"] != d.get("file"):
        raise ValueError("You cannot simultaneously use the data command and the datafile option")
    if "file" not in d:
        raise ValueError("data command without a file")
    opts["datafile"] = d["file"]
    start = data_start_date(d["file"], base_dir)
    dstart = start if start is not None else _dseries_default_start(d["file"], base_dir)

    def index(v, what):
        v = v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
        if is_date_value(v):
            if dstart is None:
                raise DateError(f"data({what}={v}): the data file has no dates (a first column of dates in a .csv "
                                "file, or INIT__ in a .m/.mat file)")
            dd = Date.parse(v)
            if dd.freq != dstart.freq:
                raise DateError(f"data({what}={dd}): the data are at another frequency (first date {dstart})")
            return dd - dstart + 1
        return int(v)

    first = index(d["first_obs"], "first_obs") if "first_obs" in d else 1
    if first < 1:
        raise DateError(f"makedataset: first_obs cannot be less than the first date in the dataset ({start})")
    nobs = int(d["nobs"][0] if isinstance(d.get("nobs"), list) else d["nobs"]) if "nobs" in d else None
    if "last_obs" in d:
        last = index(d["last_obs"], "last_obs")
        if nobs is not None and last != first + nobs - 1:
            raise DateError("makedataset: Options last_obs, first_obs and nobs are not consistent!")
        nobs = last - first + 1
    opts["first_obs"] = first
    opts["nobs"] = nobs
    return None if start is None else start + (first - 1)
