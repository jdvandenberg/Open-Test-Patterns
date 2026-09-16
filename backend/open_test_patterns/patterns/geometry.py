"""Geometry / alignment patterns: frame, grid, checkerboard, center cross."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DisabledWhen, Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import canvas, colorspace_param, finalize_signal, range_param, transfer_param


def _line_param() -> Parameter:
    return Parameter(
        "line_level",
        "Line level",
        ParamType.FLOAT,
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        step=0.01,
    )


def _center_span(size: int) -> slice:
    """Pixel slice that is exactly centered on ``size``.

    An odd length has a unique middle pixel. An even length has no single
    middle pixel, so the line is two pixels wide, straddling the midpoint.
    """
    mid = size // 2
    if size % 2 == 0:
        return slice(mid - 1, mid + 1)
    return slice(mid, mid + 1)


def _thickness_param() -> Parameter:
    return Parameter(
        "thickness", "Line thickness", ParamType.INT, default=1, minimum=1, maximum=32, unit="px"
    )


def _stroke_circle(
    img: np.ndarray,
    *,
    cx: float,
    cy: float,
    radius: float,
    thickness: int,
    level: float,
) -> None:
    """Paint a ring of ``thickness`` pixels centred on ``(cx, cy)``."""
    height, width = img.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    half = thickness / 2.0
    mask = (dist >= max(radius - half, 0.0)) & (dist < radius + half)
    img[mask] = level


@register
class FrameChart(Pattern):
    id = "frame"
    name = "Frame / Border"
    category = "Geometry"
    description = "A single-pixel (or thicker) border around the active image area."
    parameters = [
        _line_param(),
        _thickness_param(),
        colorspace_param(),
        transfer_param(default="gamma-2.4"),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        t = p["thickness"]
        lvl = p["line_level"]
        img[0:t, :, :] = lvl
        img[-t:, :, :] = lvl
        img[:, 0:t, :] = lvl
        img[:, -t:, :] = lvl
        return finalize_signal(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            signal_range=p["signal_range"],
        )


@register
class GridChart(Pattern):
    id = "grid"
    name = "Alignment Grid"
    category = "Geometry"
    description = (
        "An evenly spaced grid with a border, for geometry and convergence checks. "
        "An optional centred circle is sized as a fraction of frame height."
    )
    parameters = [
        Parameter("columns", "Columns", ParamType.INT, default=16, minimum=1, maximum=256),
        Parameter("rows", "Rows", ParamType.INT, default=9, minimum=1, maximum=256),
        _line_param(),
        _thickness_param(),
        Parameter(
            "circle",
            "Circle",
            ParamType.BOOL,
            default=True,
            description="Draw a circle at the centre of the frame.",
        ),
        Parameter(
            "circle_diameter",
            "Circle diameter",
            ParamType.FLOAT,
            default=75.0,
            minimum=1.0,
            maximum=200.0,
            step=1.0,
            unit="%",
            description="Diameter as a percentage of frame height.",
            disabled_when=DisabledWhen(
                parameter="circle",
                values=("false",),
                reason="Circle is off.",
            ),
        ),
        colorspace_param(),
        transfer_param(default="gamma-2.4"),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        t = p["thickness"]
        lvl = p["line_level"]
        cols, rows = p["columns"], p["rows"]
        xs = np.linspace(0, width, cols + 1).astype(int)
        ys = np.linspace(0, height, rows + 1).astype(int)
        for x in xs:
            x0 = min(max(x - t // 2, 0), width - t)
            img[:, x0 : x0 + t, :] = lvl
        for y in ys:
            y0 = min(max(y - t // 2, 0), height - t)
            img[y0 : y0 + t, :, :] = lvl
        if p["circle"]:
            _stroke_circle(
                img,
                cx=(width - 1) / 2.0,
                cy=(height - 1) / 2.0,
                radius=(p["circle_diameter"] / 100.0) * height / 2.0,
                thickness=t,
                level=lvl,
            )
        return finalize_signal(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            signal_range=p["signal_range"],
        )


@register
class Checkerboard(Pattern):
    id = "checkerboard"
    name = "Checkerboard"
    category = "Geometry"
    description = "Alternating two-level checker squares (contrast / dynamic checks)."
    parameters = [
        Parameter("columns", "Columns", ParamType.INT, default=5, minimum=1, maximum=128),
        Parameter("rows", "Rows", ParamType.INT, default=5, minimum=1, maximum=128),
        Parameter(
            "high", "High level", ParamType.FLOAT, default=1.0, minimum=0.0, maximum=1.0, step=0.01
        ),
        Parameter(
            "low", "Low level", ParamType.FLOAT, default=0.0, minimum=0.0, maximum=1.0, step=0.01
        ),
        Parameter("invert", "Invert", ParamType.BOOL, default=False),
        colorspace_param(),
        transfer_param(default="gamma-2.4"),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        cols, rows = p["columns"], p["rows"]
        xs = np.linspace(0, width, cols + 1).astype(int)
        ys = np.linspace(0, height, rows + 1).astype(int)
        offset = 1 if p["invert"] else 0
        for r in range(rows):
            for c in range(cols):
                level = p["high"] if (r + c + offset) % 2 == 0 else p["low"]
                img[ys[r] : ys[r + 1], xs[c] : xs[c + 1], :] = level
        return finalize_signal(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            signal_range=p["signal_range"],
        )


@register
class CenterCross(Pattern):
    id = "center-cross"
    name = "Center Cross"
    category = "Geometry"
    description = (
        "A horizontal and a vertical line through the exact centre of the frame. "
        "On an even width or height the corresponding line is two pixels so it "
        "stays centred."
    )
    parameters = [
        _line_param(),
        colorspace_param(),
        transfer_param(default="gamma-2.4"),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        lvl = p["line_level"]
        img[:, _center_span(width), :] = lvl
        img[_center_span(height), :, :] = lvl
        return finalize_signal(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            signal_range=p["signal_range"],
        )
