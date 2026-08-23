"""Starfield: a regular lattice of single pixels on black."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Choice, DisabledWhen, Parameter, ParamType, Pattern, PatternResult
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
class Starfield(Pattern):
    id = "starfield"
    name = "Starfield"
    category = "Frequency"
    description = (
        "Single-pixel stars on a black field, on a regular lattice. "
        "Spacing n=2 lights every other pixel in X and Y; larger n thins "
        "the field. Stars are either RGBW quadrants or a flat linear RGB colour."
    )
    parameters = [
        Parameter(
            "fill",
            "Fill",
            ParamType.CHOICE,
            default="rgbw",
            choices=[
                Choice("rgbw", "RGBW quadrants"),
                Choice("solid", "Solid color"),
            ],
        ),
        Parameter(
            "color",
            "Color (linear RGB)",
            ParamType.COLOR,
            default=[1.0, 1.0, 1.0],
            description="Used only when Fill is Solid color. Components in [0, 1].",
            disabled_when=DisabledWhen(
                parameter="fill",
                values=("rgbw",),
                reason="RGBW quadrants set the star colour, so this input is unused.",
            ),
        ),
        Parameter(
            "spacing",
            "Every n pixels",
            ParamType.INT,
            default=2,
            minimum=2,
            maximum=64,
            unit="px",
            description=(
                "Lattice period. 2 = every other pixel on both axes; "
                "n lights one pixel in each n×n cell, at the cell origin."
            ),
        ),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        n = p["spacing"]
        rows = np.arange(height)[:, None]
        cols = np.arange(width)[None, :]
        mask = (rows % n == 0) & (cols % n == 0)

        fill = np.zeros((height, width, 3), dtype=np.float64)
        if p["fill"] == "rgbw":
            hh, hw = height // 2, width // 2
            fill[:hh, :hw] = (1.0, 0.0, 0.0)
            fill[:hh, hw:] = (0.0, 1.0, 0.0)
            fill[hh:, :hw] = (0.0, 0.0, 1.0)
            fill[hh:, hw:] = (1.0, 1.0, 1.0)
        else:
            fill[:, :] = np.asarray(p["color"], dtype=np.float64)

        img[mask] = fill[mask]
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
