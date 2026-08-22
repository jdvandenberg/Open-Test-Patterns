"""X-Rite / Macbeth ColorChecker (24 patch) chart.

Uses the reference colorimetry shipped with colour-science instead of the
hard-coded XYZ table in the original code, and adapts to the chosen output color
space and white point.
"""

from __future__ import annotations

from typing import Any

import colour
import numpy as np

from ..color.colorspaces import get_colorspace
from .base import Choice, Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import (
    canvas,
    colorspace_param,
    finalize,
    peak_luminance_param,
    range_param,
    transfer_param,
)

_CHECKER_NAME = "ColorChecker24 - After November 2014"
_COLS, _ROWS = 6, 4


def _patch_linear_rgb(space_id: str) -> np.ndarray:
    """Return a (24, 3) array of relative-linear RGB patches for the target space."""
    checker = colour.CCS_COLOURCHECKERS[_CHECKER_NAME]
    target = get_colorspace(space_id).colourspace
    illuminant_xy = np.asarray(checker.illuminant, dtype=np.float64)

    rgb = []
    for _name, xyY in checker.data.items():
        XYZ = colour.xyY_to_XYZ(np.asarray(xyY, dtype=np.float64))
        linear = colour.XYZ_to_RGB(
            XYZ,
            target,
            illuminant=illuminant_xy,
            chromatic_adaptation_transform="Bradford",
            apply_cctf_encoding=False,
        )
        rgb.append(linear)
    return np.clip(np.asarray(rgb, dtype=np.float64), 0.0, None)


@register
class ColorChecker(Pattern):
    id = "colorchecker"
    name = "ColorChecker (24 Patch)"
    category = "Charts"
    description = "Macbeth/X-Rite ColorChecker 24 chart adapted to the output color space."
    parameters = [
        Parameter(
            "orientation",
            "Orientation",
            ParamType.CHOICE,
            default="landscape",
            choices=[Choice("landscape", "Landscape (6x4)"), Choice("portrait", "Portrait (4x6)")],
        ),
        Parameter(
            "border",
            "Border",
            ParamType.FLOAT,
            default=0.04,
            minimum=0.0,
            maximum=0.2,
            step=0.01,
            description="Gap around/between patches as a fraction of the frame.",
        ),
        colorspace_param(default="rec709"),
        transfer_param(default="gamma-2.4"),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        patches = _patch_linear_rgb(p["color_space"])

        cols, rows = (_COLS, _ROWS) if p["orientation"] == "landscape" else (_ROWS, _COLS)
        gap_x = p["border"] * width
        gap_y = p["border"] * height
        cell_w = (width - gap_x * (cols + 1)) / cols
        cell_h = (height - gap_y * (rows + 1)) / rows

        for idx in range(cols * rows):
            if p["orientation"] == "landscape":
                r, c = divmod(idx, cols)
            else:
                # Portrait keeps the canonical reading order down columns.
                c, r = divmod(idx, rows)
            x0 = int(round(gap_x + c * (cell_w + gap_x)))
            y0 = int(round(gap_y + r * (cell_h + gap_y)))
            x1 = int(round(x0 + cell_w))
            y1 = int(round(y0 + cell_h))
            img[y0:y1, x0:x1, :] = patches[idx]

        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
