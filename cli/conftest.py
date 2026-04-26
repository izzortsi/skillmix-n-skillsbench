"""pytest configuration: add project root to sys.path."""

import sys
from pathlib import Path

_project = Path(__file__).resolve().parent
if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))
