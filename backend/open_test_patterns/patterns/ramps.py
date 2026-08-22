"""Continuous ramps and gradients."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Choice, Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import (
    colorspace_param,
    finalize,
    peak_luminance_param,
    range_param,
    transfer_param,
)


@register
class Gradient(Pattern):
    id = "gradient"
    name = "Linear Gradient"
    category = "Ramps & Steps"
    description = "A smooth two-color gradient across the frame."
    parameters = [
        Parameter("color_a", "Start color (linear RGB)", ParamType.COLOR, default=[0.0, 0.0, 0.0]),
        Parameter("color_b", "End color (linear RGB)", ParamType.COLOR, default=[1.0, 1.0, 1.0]),
        Parameter(
            "direction",
            "Direction",
            ParamType.CHOICE,
            default="horizontal",
            choices=[Choice("horizontal", "Horizontal"), Choice("vertical", "Vertical")],
        ),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        a = np.asarray(p["color_a"], dtype=np.float64)
        b = np.asarray(p["color_b"], dtype=np.float64)
        if p["direction"] == "horizontal":
            t = np.linspace(0.0, 1.0, width).reshape(1, width, 1)
        else:
            t = np.linspace(0.0, 1.0, height).reshape(height, 1, 1)
        img = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
        img = np.broadcast_to(img, (height, width, 3)).copy()
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )


@register
class RampUpDown(Pattern):
    id = "ramp-up-down"
    name = "Symmetric Ramp (Up/Down)"
    category = "Ramps & Steps"
    description = "A ramp that rises to the center and falls back, mirrored left/right."
    parameters = [
        Parameter("color_a", "Edge color (linear RGB)", ParamType.COLOR, default=[0.0, 0.0, 0.0]),
        Parameter("color_b", "Center color (linear RGB)", ParamType.COLOR, default=[1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        a = np.asarray(p["color_a"], dtype=np.float64)
        b = np.asarray(p["color_b"], dtype=np.float64)
        half = width / 2.0
        cols = np.arange(width)
        t = 1.0 - np.abs(cols - half) / half  # 0 at edges, 1 at center
        t = np.clip(t, 0.0, 1.0).reshape(1, width, 1)
        img = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
        img = np.broadcast_to(img, (height, width, 3)).copy()
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
