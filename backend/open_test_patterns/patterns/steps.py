"""Stepped ramps: gray steps, PQ luminance steps, and per-color steps."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import (
    canvas,
    colorspace_param,
    finalize,
    finalize_signal,
    peak_luminance_param,
    range_param,
    transfer_param,
)


def _column_edges(width: int, steps: int) -> np.ndarray:
    return np.linspace(0, width, steps + 1).astype(int)


@register
class GraySteps(Pattern):
    id = "gray-steps"
    name = "Gray Steps"
    category = "Ramps & Steps"
    description = "Evenly spaced neutral steps in code value, for contouring/banding checks."
    parameters = [
        Parameter("steps", "Steps", ParamType.INT, default=11, minimum=2, maximum=256),
        colorspace_param(),
        transfer_param(default="gamma-2.4"),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        steps = p["steps"]
        edges = _column_edges(width, steps)
        for i in range(steps):
            value = i / (steps - 1)
            img[:, edges[i] : edges[i + 1], :] = value
        return finalize_signal(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            signal_range=p["signal_range"],
        )


@register
class PqLuminanceSteps(Pattern):
    id = "pq-luminance-steps"
    name = "PQ Luminance Steps"
    category = "Ramps & Steps"
    description = "Neutral steps at absolute luminance levels, PQ-encoded (ST 2084)."
    parameters = [
        Parameter("steps", "Steps", ParamType.INT, default=20, minimum=2, maximum=64),
        Parameter(
            "first_luminance",
            "First luminance",
            ParamType.FLOAT,
            default=5.0,
            minimum=0.0,
            maximum=10000.0,
            unit="cd/m²",
        ),
        Parameter(
            "increment",
            "Increment per step",
            ParamType.FLOAT,
            default=5.0,
            minimum=0.0,
            maximum=1000.0,
            unit="cd/m²",
        ),
        colorspace_param(default="rec2020"),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        steps = p["steps"]
        img = canvas(width, height)
        edges = _column_edges(width, steps)
        # Store absolute luminance then let finalize apply PQ with peak=1 nit
        # scaling; here values are already absolute, so use peak_luminance=1.
        for i in range(steps):
            lum = p["first_luminance"] + p["increment"] * i
            img[:, edges[i] : edges[i + 1], :] = lum
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function="pq",
            peak_luminance=1.0,
            signal_range=p["signal_range"],
        )


@register
class ColorSteps(Pattern):
    id = "color-steps"
    name = "Color Steps"
    category = "Ramps & Steps"
    description = "Steps of a single color from black to full amplitude."
    parameters = [
        Parameter(
            "color",
            "Color (linear RGB)",
            ParamType.COLOR,
            default=[1.0, 1.0, 1.0],
        ),
        Parameter("steps", "Steps", ParamType.INT, default=10, minimum=2, maximum=64),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        steps = p["steps"]
        color = np.asarray(p["color"], dtype=np.float64)
        edges = _column_edges(width, steps)
        for i in range(steps):
            img[:, edges[i] : edges[i + 1], :] = color * (i / (steps - 1))
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
