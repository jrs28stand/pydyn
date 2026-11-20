import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# keep compiled-model cache files produced by the tests out of the user's own cache folder
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ.setdefault("PYDYN_CACHE_DIR", tempfile.mkdtemp(prefix="pydyn-test-cache-"))
