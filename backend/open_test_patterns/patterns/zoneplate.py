"""Circular zone plate for spatial-frequency and scaling/aliasing evaluation."""

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


@register
class ZonePlate(Pattern):
    id = "zone-plate"
    name = "Zone Plate"
    category = "Frequency"
    description = "Concentric-ring zone plate sweeping spatial frequency from the center."
    parameters = [
        Parameter(
            "max_frequency",
            "Max frequency",
            ParamType.FLOAT,
            default=0.8,
            minimum=0.05,
            maximum=1.0,
            step=0.01,
            description="Peak frequency as a fraction of Nyquist (km/π).",
        ),
        Parameter(
            "center_x",
            "Center X",
            ParamType.FLOAT,
            default=0.5,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
        ),
        Parameter(
            "center_y",
            "Center Y",
            ParamType.FLOAT,
            default=0.5,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
        ),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        yy, xx = np.mgrid[0:height, 0:width]
        xc = p["center_x"] * width
        yc = p["center_y"] * height
        x = xx - xc
        y = yy - yc
        r = np.sqrt(x**2 + y**2)

        km = p["max_frequency"] * np.pi
        rm = width / 2.0
        w = rm / 10.0

        term1 = np.sin((km * r**2) / (2 * rm))
        # tanh window rolls the pattern off near radius rm.
        term2 = 0.5 * np.tanh((rm - r) / w) + 0.5
        gray = (term1 * term2 + 1) / 2.0

        img = canvas(width, height)
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
