"""Window patches for luminance/measurement work."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import (
    canvas,
    colorspace_param,
    finalize,
    peak_luminance_param,
    range_param,
    transfer_param,
)


def _window_param() -> Parameter:
    return Parameter(
        "window_size",
        "Window size",
        ParamType.FLOAT,
        default=0.1,
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        description="Patch size as a fraction of the frame.",
    )


def _color_param(default: list[float]) -> Parameter:
    return Parameter("color", "Color (linear RGB)", ParamType.COLOR, default=default)


@register
class CenteredPatch(Pattern):
    id = "centered-patch"
    name = "Centered Window Patch"
    category = "Patches"
    description = "A centered rectangular patch on a black field (probe/measurement window)."
    parameters = [
        _window_param(),
        _color_param([1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        sw = round(width * p["window_size"])
        sh = round(height * p["window_size"])
        x0 = round(width / 2 - sw / 2)
        y0 = round(height / 2 - sh / 2)
        img[y0 : y0 + sh, x0 : x0 + sw, :] = p["color"]
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )


@register
class FourCorners(Pattern):
    id = "four-corners"
    name = "Four Corners + Center"
    category = "Patches"
    description = "Five patches (corners and center) for uniformity checks."
    parameters = [
        _window_param(),
        _color_param([1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        color = np.asarray(p["color"], dtype=np.float64)
        sw = round(width * p["window_size"])
        sh = round(height * p["window_size"])
        img[0:sh, 0:sw, :] = color
        img[0:sh, width - sw :, :] = color
        img[height - sh :, 0:sw, :] = color
        img[height - sh :, width - sw :, :] = color
        cx = round(width / 2 - sw / 2)
        cy = round(height / 2 - sh / 2)
        img[cy : cy + sh, cx : cx + sw, :] = color
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
