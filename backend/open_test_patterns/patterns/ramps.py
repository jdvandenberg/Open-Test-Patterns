"""Continuous ramps and gradients."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Choice, Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import (
    authored_color,
    canvas,
    color_params,
    colorspace_param,
    finalize_authored,
    peak_luminance_param,
    range_param,
    transfer_param,
    value_mode_param,
)


@register
class Gradient(Pattern):
    id = "gradient"
    name = "Linear Gradient"
    category = "Ramps & Steps"
    description = "A smooth two-color gradient across the frame."
    parameters = [
        Parameter(
            "direction",
            "Direction",
            ParamType.CHOICE,
            default="horizontal",
            choices=[Choice("horizontal", "Horizontal"), Choice("vertical", "Vertical")],
        ),
        value_mode_param(),
        *color_params("color_a", "Start color", default=[0.0, 0.0, 0.0]),
        *color_params("color_b", "End color", default=[1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        a = authored_color(p, "color_a")
        b = authored_color(p, "color_b")
        if p["direction"] == "horizontal":
            t = np.linspace(0.0, 1.0, width).reshape(1, width, 1)
        else:
            t = np.linspace(0.0, 1.0, height).reshape(height, 1, 1)
        img = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
        img = np.broadcast_to(img, (height, width, 3)).copy()
        return finalize_authored(
            img,
            value_mode=p["value_mode"],
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
        value_mode_param(),
        *color_params("color_a", "Edge color", default=[0.0, 0.0, 0.0]),
        *color_params("color_b", "Center color", default=[1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        a = authored_color(p, "color_a")
        b = authored_color(p, "color_b")
        half = width / 2.0
        cols = np.arange(width)
        t = 1.0 - np.abs(cols - half) / half  # 0 at edges, 1 at center
        t = np.clip(t, 0.0, 1.0).reshape(1, width, 1)
        img = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
        img = np.broadcast_to(img, (height, width, 3)).copy()
        return finalize_authored(
            img,
            value_mode=p["value_mode"],
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )


def _band_edges(length: int, count: int = 4) -> np.ndarray:
    return np.linspace(0, length, count + 1).astype(int)


@register
class QuadRamp(Pattern):
    id = "quad-ramp"
    name = "Quad Ramp"
    category = "Ramps & Steps"
    description = (
        "Four ramps on one frame. Horizontal stacks them as rows ramping left "
        "to right; vertical places them as columns ramping top to bottom."
    )
    parameters = [
        Parameter(
            "direction",
            "Direction",
            ParamType.CHOICE,
            default="horizontal",
            choices=[Choice("horizontal", "Horizontal"), Choice("vertical", "Vertical")],
        ),
        value_mode_param(),
        *color_params("ramp1_start", "Ramp 1 start", default=[0.0, 0.0, 0.0]),
        *color_params("ramp1_end", "Ramp 1 end", default=[1.0, 0.0, 0.0]),
        *color_params("ramp2_start", "Ramp 2 start", default=[0.0, 0.0, 0.0]),
        *color_params("ramp2_end", "Ramp 2 end", default=[0.0, 1.0, 0.0]),
        *color_params("ramp3_start", "Ramp 3 start", default=[0.0, 0.0, 0.0]),
        *color_params("ramp3_end", "Ramp 3 end", default=[0.0, 0.0, 1.0]),
        *color_params("ramp4_start", "Ramp 4 start", default=[0.0, 0.0, 0.0]),
        *color_params("ramp4_end", "Ramp 4 end", default=[1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        starts = [authored_color(p, f"ramp{i}_start") for i in range(1, 5)]
        ends = [authored_color(p, f"ramp{i}_end") for i in range(1, 5)]
        img = canvas(width, height)
        if p["direction"] == "horizontal":
            t = np.linspace(0.0, 1.0, width).reshape(1, width, 1)
            edges = _band_edges(height)
            for i, (a, b) in enumerate(zip(starts, ends, strict=True)):
                y0, y1 = int(edges[i]), int(edges[i + 1])
                if y1 <= y0:
                    continue
                img[y0:y1] = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
        else:
            t = np.linspace(0.0, 1.0, height).reshape(height, 1, 1)
            edges = _band_edges(width)
            for i, (a, b) in enumerate(zip(starts, ends, strict=True)):
                x0, x1 = int(edges[i]), int(edges[i + 1])
                if x1 <= x0:
                    continue
                img[:, x0:x1] = a.reshape(1, 1, 3) * (1 - t) + b.reshape(1, 1, 3) * t
        return finalize_authored(
            img,
            value_mode=p["value_mode"],
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )

