"""Core abstractions for self-describing test patterns.

A :class:`Pattern` declares typed :class:`Parameter` objects and produces a
:class:`PatternResult`: a floating-point ``(H, W, 3)`` image together with a
:class:`SignalFormat` describing how the values are encoded (color space,
transfer function, signal range). Keeping this metadata with the pixels lets the
image writers tag files correctly and lets the UI explain what it is showing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"
    COLOR = "color"  # linear RGB triple in [0, 1]


class SignalRange(str, Enum):
    FULL = "full"
    LEGAL = "legal"


@dataclass(frozen=True)
class Choice:
    value: str
    label: str


@dataclass(frozen=True)
class DisabledWhen:
    """Marks a parameter inapplicable while another parameter holds one of ``values``.

    This is presentation metadata: the UI greys the control out. The generator
    still receives a value for the parameter and is responsible for ignoring it
    under the same condition, so the two can never disagree.
    """

    parameter: str
    values: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class Parameter:
    """Declarative description of a single pattern parameter."""

    name: str
    label: str
    type: ParamType
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: list[Choice] = field(default_factory=list)
    unit: str | None = None
    description: str = ""
    disabled_when: DisabledWhen | None = None

    def coerce(self, value: Any) -> Any:
        """Validate and coerce a raw incoming value to this parameter's type."""
        if value is None:
            return self.default
        if self.type is ParamType.INT:
            v = int(value)
        elif self.type is ParamType.FLOAT:
            v = float(value)
        elif self.type is ParamType.BOOL:
            v = bool(value)
        elif self.type is ParamType.CHOICE:
            v = str(value)
            valid = {c.value for c in self.choices}
            if valid and v not in valid:
                raise ValueError(f"{self.name}: {v!r} not in {sorted(valid)}")
            return v
        elif self.type is ParamType.COLOR:
            arr = [float(c) for c in value]
            if len(arr) != 3:
                raise ValueError(f"{self.name}: color must have 3 components")
            # Components are relative linear light, so out-of-range values have
            # no meaning. Clamping here rather than in the transfer functions
            # keeps the value the UI echoes back identical to the one rendered,
            # whichever curve is selected.
            lo = 0.0 if self.minimum is None else self.minimum
            hi = 1.0 if self.maximum is None else self.maximum
            return [min(max(c, lo), hi) for c in arr]
        else:  # pragma: no cover - defensive
            return value
        if self.minimum is not None and v < self.minimum:
            v = type(v)(self.minimum)
        if self.maximum is not None and v > self.maximum:
            v = type(v)(self.maximum)
        return v


@dataclass(frozen=True)
class SignalFormat:
    """Describes the encoding of a produced image."""

    color_space: str = "rec709"
    transfer_function: str = "linear"
    range: SignalRange = SignalRange.FULL
    peak_luminance: float | None = None


@dataclass
class PatternResult:
    """A generated image and the description of its encoding."""

    image: np.ndarray  # float (H, W, 3), values in [0, 1] after encoding
    signal_format: SignalFormat


@dataclass(frozen=True)
class PatternInfo:
    id: str
    name: str
    category: str
    description: str


class Pattern(ABC):
    """Base class for all test-pattern generators."""

    id: str = ""
    name: str = ""
    category: str = "General"
    description: str = ""
    parameters: list[Parameter] = []

    @property
    def info(self) -> PatternInfo:
        return PatternInfo(self.id, self.name, self.category, self.description)

    def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        """Coerce and fill defaults for the declared parameters."""
        resolved: dict[str, Any] = {}
        for p in self.parameters:
            resolved[p.name] = p.coerce(params.get(p.name))
        return resolved

    def render(self, width: int, height: int, params: dict[str, Any]) -> PatternResult:
        """Public entry point: validate params then generate."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        resolved = self.resolve(params or {})
        return self.generate(width, height, **resolved)

    @abstractmethod
    def generate(self, width: int, height: int, **params: Any) -> PatternResult:
        """Produce the pattern. Implemented by subclasses."""
        raise NotImplementedError
