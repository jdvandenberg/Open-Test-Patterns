"""Window patches for luminance/measurement work."""

from __future__ import annotations

from typing import Any

from .annotate import annotate_code_rgb
from .base import Parameter, ParamType, Pattern, PatternResult
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


@register
class CenteredPatch(Pattern):
    id = "centered-patch"
    name = "Centered Window Patch"
    category = "Patches"
    description = "A centered rectangular patch on a black field (probe/measurement window)."
    parameters = [
        _window_param(),
        value_mode_param(),
        *color_params(default=[1.0, 1.0, 1.0]),
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
        img[y0 : y0 + sh, x0 : x0 + sw, :] = authored_color(p)
        result = finalize_authored(
            img,
            value_mode=p["value_mode"],
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
        return annotate_code_rgb(result)


@register
class FourCorners(Pattern):
    id = "four-corners"
    name = "Four Corners + Center"
    category = "Patches"
    description = "Five patches (corners and center) for uniformity checks."
    parameters = [
        _window_param(),
        value_mode_param(),
        *color_params(default=[1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        color = authored_color(p)
        sw = round(width * p["window_size"])
        sh = round(height * p["window_size"])
        img[0:sh, 0:sw, :] = color
        img[0:sh, width - sw :, :] = color
        img[height - sh :, 0:sw, :] = color
        img[height - sh :, width - sw :, :] = color
        cx = round(width / 2 - sw / 2)
        cy = round(height / 2 - sh / 2)
        img[cy : cy + sh, cx : cx + sw, :] = color
        return finalize_authored(
            img,
            value_mode=p["value_mode"],
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
