"""Pattern registry.

Patterns register themselves via the :func:`register` decorator. The registry
is the single source of truth used by the CLI and the API.
"""

from __future__ import annotations

from .base import Pattern

_REGISTRY: dict[str, Pattern] = {}


def register(cls: type[Pattern]) -> type[Pattern]:
    """Class decorator that instantiates and registers a pattern."""
    if not cls.id:
        raise ValueError(f"{cls.__name__} must define a non-empty `id`")
    if cls.id in _REGISTRY:
        raise ValueError(f"Duplicate pattern id: {cls.id!r}")
    _REGISTRY[cls.id] = cls()
    return cls


def get_pattern(pattern_id: str) -> Pattern:
    try:
        return _REGISTRY[pattern_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown pattern {pattern_id!r}. Available: {sorted(_REGISTRY)}"
        ) from exc


def all_patterns() -> list[Pattern]:
    return sorted(_REGISTRY.values(), key=lambda p: (p.category, p.name))


def pattern_ids() -> list[str]:
    return sorted(_REGISTRY)
