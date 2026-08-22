"""Solid fields and quadrant patterns."""

from __future__ import annotations

from typing import Any

from .annotate import maybe_annotate, show_label_param
from .base import Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import (
    authored_color,
    canvas,
    color_params,
    colorspace_param,
    finalize,
    finalize_authored,
    peak_luminance_param,
    range_param,
    transfer_param,
    value_mode_param,
)


@register
class SolidColor(Pattern):
    id = "solid-color"
    name = "Solid Color"
    category = "Solid Fields"
    description = "A single flat color filling the frame."
    parameters = [
        value_mode_param(),
        *color_params(default=[1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
        show_label_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        img[:, :, :] = authored_color(p)
        result = finalize_authored(
            img,
            value_mode=p["value_mode"],
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
        return maybe_annotate(result, p["show_label"])


@register
class GrayField(Pattern):
    id = "gray-field"
    name = "Gray Field"
    category = "Solid Fields"
    description = "A neutral gray field at a chosen relative level."
    parameters = [
        Parameter(
            "level",
            "Level",
            ParamType.FLOAT,
            default=0.5,
            minimum=0.0,
            maximum=1.0,
            step=0.001,
            description="Relative linear gray level.",
        ),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
        show_label_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        img[:, :, :] = p["level"]
        result = finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
        return maybe_annotate(result, p["show_label"])


@register
class RGBWQuadrants(Pattern):
    id = "rgbw-quadrants"
    name = "RGBW Quadrants"
    category = "Solid Fields"
    description = "Red, green, blue, and white quadrants for primary/secondary checks."
    parameters = [
        Parameter(
            "level",
            "Level",
            ParamType.FLOAT,
            default=1.0,
            minimum=0.0,
            maximum=1.0,
            step=0.001,
        ),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        lvl = p["level"]
        hh, hw = height // 2, width // 2
        img[:hh, :hw, :] = (lvl, 0, 0)  # top-left: red
        img[:hh, hw:, :] = (0, lvl, 0)  # top-right: green
        img[hh:, :hw, :] = (0, 0, lvl)  # bottom-left: blue
        img[hh:, hw:, :] = (lvl, lvl, lvl)  # bottom-right: white
        return finalize(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
