"""Test-pattern generators.

Importing this package registers every built-in pattern with the registry.
"""

from __future__ import annotations

# Importing the modules below has the side effect of registering their patterns.
from . import (
    colorbars,  # noqa: E402,F401
    colorchecker,  # noqa: E402,F401
    geometry,  # noqa: E402,F401
    patches,  # noqa: E402,F401
    ramps,  # noqa: E402,F401
    rp219_2_2016,  # noqa: E402,F401
    solid,  # noqa: E402,F401
    starfield,  # noqa: E402,F401
    steps,  # noqa: E402,F401
    zoneplate,  # noqa: E402,F401
)
from .base import (
    Choice,
    Parameter,
    ParamType,
    Pattern,
    PatternInfo,
    PatternResult,
    SignalFormat,
    SignalRange,
)
from .registry import all_patterns, get_pattern, pattern_ids, register

__all__ = [
    "Choice",
    "Parameter",
    "ParamType",
    "Pattern",
    "PatternInfo",
    "PatternResult",
    "SignalFormat",
    "SignalRange",
    "all_patterns",
    "get_pattern",
    "pattern_ids",
    "register",
]
