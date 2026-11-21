# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# keep compiled-model cache files produced by the tests out of the user's own cache folder
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ.setdefault("PYDYN_CACHE_DIR", tempfile.mkdtemp(prefix="pydyn-test-cache-"))


def _korder_available() -> bool:
    try:
        import pydyn
        pydyn.korder.load_library()
        return True
    except FileNotFoundError:
        return False


def pytest_report_header(config):
    state = "found" if _korder_available() else "NOT BUILT: second- and third-order tests will be skipped"
    return f"pydyn k-order library: {state}"


def pytest_sessionstart(session):
    # Tests of orders 2 and 3 skip when the compiled library is missing. Set PYDYN_REQUIRE_KORDER=1
    # (as a validation run should) to make its absence an error instead of a silent gap.
    if os.environ.get("PYDYN_REQUIRE_KORDER", "0") == "1" and not _korder_available():
        raise SystemExit("PYDYN_REQUIRE_KORDER=1 but korder/libpydyn_korder.so is not built "
                         "(run korder/build.sh <dynare source>)")
