"""Classic vertical color bars (EBU / SMPTE ordering)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Choice, Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import (
    canvas,
    colorspace_param,
    finalize_signal,
    range_param,
    transfer_param,
)

# White, Yellow, Cyan, Green, Magenta, Red, Blue (as linear RGB unit values).
_BAR_ORDER = [
    (1, 1, 1),
    (1, 1, 0),
    (0, 1, 1),
    (0, 1, 0),
    (1, 0, 1),
    (1, 0, 0),
    (0, 0, 1),
]


@register
class ColorBars(Pattern):
    id = "color-bars"
    name = "Color Bars"
    category = "Color Bars"
    description = "Seven-bar primary/secondary color bars at 100% or 75% amplitude."
    parameters = [
        Parameter(
            "amplitude",
            "Amplitude",
            ParamType.CHOICE,
            default="75",
            choices=[Choice("100", "100%"), Choice("75", "75%")],
        ),
        Parameter(
            "include_black",
            "Trailing black bar",
            ParamType.BOOL,
            default=False,
        ),
        colorspace_param(),
        transfer_param(default="gamma-2.4"),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        scale = 1.0 if p["amplitude"] == "100" else 0.75
        bars = [tuple(scale * c for c in bar) for bar in _BAR_ORDER]
        if p["include_black"]:
            bars.append((0.0, 0.0, 0.0))
        edges = np.linspace(0, width, len(bars) + 1).astype(int)
        for i, bar in enumerate(bars):
            img[:, edges[i] : edges[i + 1], :] = bar
        return finalize_signal(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            signal_range=p["signal_range"],
        )
