"""Public init."""

from . import plot  # noqa: F401
from .event_metadata import EventMetadata  # noqa: F401
from .result import SkyScanResult  # noqa: F401
from .constants import CATALOG_PATH

__all__ = [
    "EventMetadata",
    "plot",
    "SkyScanResult",
    "CATALOG_PATH",
]

# NOTE: `__version__` is not defined because this package is built using 'setuptools-scm' --
#   use `importlib.metadata.version(...)` if you need to access version info at runtime.
