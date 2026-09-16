"""Stepped ramps: gray steps, PQ luminance steps, and per-color steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageBufAlgo

from ..color import transfer
from .base import Parameter, ParamType, Pattern, PatternResult, SignalRange
from .registry import register
from .util import (
    authored_color,
    canvas,
    color_params,
    colorspace_param,
    finalize,
    finalize_authored,
    finalize_signal,
    peak_luminance_param,
    range_param,
    transfer_param,
    value_mode_param,
)


def _column_edges(width: int, steps: int) -> np.ndarray:
    return np.linspace(0, width, steps + 1).astype(int)


def format_pq_step_number(luminance: float) -> str:
    """Compact nits figure for a column: ``5``, ``2.5``, ``0.005``."""
    if luminance >= 100 or (luminance >= 1 and abs(luminance - round(luminance)) < 1e-4):
        return f"{int(round(luminance))}"
    if luminance < 0.01:
        return f"{luminance:.3f}".rstrip("0").rstrip(".")
    return f"{luminance:.2f}".rstrip("0").rstrip(".")


def format_pq_step_label(luminance: float) -> str:
    """``5 cd/m²``, compact enough to sit at the foot of a step."""
    return f"{format_pq_step_number(luminance)} cd/m²"


# Overlay type is a 50 cd/m² reference, not 10 000-nit PQ white.
LABEL_NITS = 50.0
_UNIT = "cd/m²"
TITLE = "Luminance values in Rec.2100 ST2084 (PQ)"
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _label_font() -> str:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return ""


def _pq_code(nits: float, signal_range: SignalRange) -> float:
    code = float(np.asarray(transfer.encode(max(nits, 0.0), "pq")).reshape(-1)[0])
    if signal_range is SignalRange.LEGAL:
        code = float(np.asarray(transfer.full_to_legal(code)).reshape(-1)[0])
    return code


def _label_padding(fontsize: int) -> tuple[int, int, int]:
    pad_x = max(8, round(fontsize * 0.35))
    pad_y = max(8, round(fontsize * 0.40))
    gap = max(2, round(fontsize * 0.12))
    return pad_x, pad_y, gap


def _two_line_box(
    number: str,
    unit: str,
    fontsize: int,
    font: str,
) -> tuple[Any, Any, int, int] | None:
    rn = ImageBufAlgo.text_size(number, fontsize, font)
    ru = ImageBufAlgo.text_size(unit, fontsize, font)
    if not rn.defined or not ru.defined:
        return None
    pad_x, pad_y, gap = _label_padding(fontsize)
    width = max(rn.width, ru.width) + 2 * pad_x
    height = rn.height + gap + ru.height + 2 * pad_y
    return rn, ru, width, height


def _labels_fit(
    numbers: list[str],
    unit: str,
    edges: np.ndarray,
    height: int,
    fontsize: int,
    font: str,
) -> bool:
    """False when any column is too narrow or the frame is too short."""
    min_col = int(min(edges[i + 1] - edges[i] for i in range(len(edges) - 1)))
    if min_col < 28 or height < 64:
        return False
    widest = 0
    tallest = 0
    for number in numbers:
        measured = _two_line_box(number, unit, fontsize, font)
        if measured is None:
            return False
        _, _, box_w, box_h = measured
        widest = max(widest, box_w)
        tallest = max(tallest, box_h)
    if widest > min_col - 2:
        return False
    if tallest > height * 0.35:
        return False
    return True


def _overlay_pq_title(img: np.ndarray, signal_range: SignalRange) -> None:
    """Burn ``TITLE`` in a 0-nit box at the top-left, type at 50 cd/m²."""
    height, width = img.shape[:2]
    if width < 160 or height < 64:
        return
    font = _label_font()
    fontsize = max(12, min(round(height * 0.021), round(width * 0.0135)))
    roi = ImageBufAlgo.text_size(TITLE, fontsize, font)
    while fontsize > 9 and roi.defined and roi.width + 48 > width:
        fontsize -= 1
        roi = ImageBufAlgo.text_size(TITLE, fontsize, font)
    if not roi.defined:
        return
    pad_x = max(5, round(fontsize * 0.45))
    pad_y = max(3, round(fontsize * 0.32))
    margin = max(10, min(width, height) // 80)
    x = margin - roi.xbegin
    y = margin - roi.ybegin
    box_x0 = max(0, x + roi.xbegin - pad_x)
    box_y0 = max(0, y + roi.ybegin - pad_y)
    box_x1 = min(width, x + roi.xend + pad_x)
    box_y1 = min(height, y + roi.yend + pad_y)
    if box_x1 - box_x0 < 8 or box_y1 - box_y0 < 8:
        return
    if box_x1 - box_x0 > width * 0.98:
        return
    box_level = _pq_code(0.0, signal_range)
    ink = _pq_code(LABEL_NITS, signal_range)
    img[box_y0:box_y1, box_x0:box_x1] = box_level
    buf = ImageBuf(np.ascontiguousarray(img, dtype=np.float32))
    fill = (ink, ink, ink)
    if not ImageBufAlgo.render_text(buf, x, y, TITLE, fontsize, font, fill, shadow=0):
        return
    img[:] = buf.get_pixels(oiio.FLOAT)


def _overlay_pq_labels(
    img: np.ndarray,
    edges: np.ndarray,
    luminances: list[float],
    signal_range: SignalRange,
) -> None:
    """Burn a two-line nits readout on a 0-nit box, type at 50 cd/m²."""
    height, _width = img.shape[:2]
    numbers = [format_pq_step_number(lum) for lum in luminances]
    font = _label_font()
    min_col = int(min(edges[i + 1] - edges[i] for i in range(len(edges) - 1)))
    fontsize = max(18, min(round(min_col * 0.42), round(height * 0.055)))
    while fontsize > 14 and not _labels_fit(
        numbers, _UNIT, edges, height, fontsize, font
    ):
        fontsize -= 1
    if not _labels_fit(numbers, _UNIT, edges, height, fontsize, font):
        return

    box_level = _pq_code(0.0, signal_range)
    ink_level = _pq_code(LABEL_NITS, signal_range)
    fill = (ink_level, ink_level, ink_level)
    _pad_x, pad_y, gap = _label_padding(fontsize)
    margin = max(6, fontsize // 5)
    inset = 2
    placements: list[tuple[int, int, str, int, int]] = []

    for i, number in enumerate(numbers):
        measured = _two_line_box(number, _UNIT, fontsize, font)
        if measured is None:
            return
        rn, ru, _box_w, _box_h = measured
        x0, x1 = int(edges[i]), int(edges[i + 1])
        cx = (x0 + x1 - 1) / 2.0
        y_unit = height - margin - ru.yend
        y_number = y_unit - ru.height - gap
        x_number = int(round(cx - rn.width / 2.0 - rn.xbegin))
        x_unit = int(round(cx - ru.width / 2.0 - ru.xbegin))
        box_x0 = x0 + inset
        box_x1 = x1 - inset
        box_y0 = max(0, y_number + rn.ybegin - pad_y)
        box_y1 = min(height - inset, y_unit + ru.yend + pad_y)
        if box_x1 <= box_x0 or box_y1 <= box_y0:
            return
        img[box_y0:box_y1, box_x0:box_x1] = box_level
        placements.append((x_number, y_number, number, x_unit, y_unit))

    buf = ImageBuf(np.ascontiguousarray(img, dtype=np.float32))
    for x_number, y_number, number, x_unit, y_unit in placements:
        if not ImageBufAlgo.render_text(
            buf, x_number, y_number, number, fontsize, font, fill, shadow=0
        ):
            return
        if not ImageBufAlgo.render_text(
            buf, x_unit, y_unit, _UNIT, fontsize, font, fill, shadow=0
        ):
            return
    img[:] = buf.get_pixels(oiio.FLOAT)


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
    description = (
        "Neutral steps at absolute luminance levels, PQ-encoded (ST 2084). "
        "Each column is labelled in cd/m² when there is room."
    )
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
        range_param(),
    ]

    # The steps are neutral, so the gamut cannot change a pixel; it would only
    # alter the metadata tag. BT.2100 pairs PQ with BT.2020 primaries, which is
    # what this chart is measured against.
    color_space = "rec2020"

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        steps = p["steps"]
        img = canvas(width, height)
        edges = _column_edges(width, steps)
        luminances = [p["first_luminance"] + p["increment"] * i for i in range(steps)]
        # Relative linear in [0, 1] against the brightest step, so PQ still
        # encodes the authored nits while the preview can map that peak to 1.0
        # instead of clipping every column to white.
        peak = max(max(luminances), 1.0)
        for i, lum in enumerate(luminances):
            img[:, edges[i] : edges[i + 1], :] = lum / peak
        result = finalize(
            img,
            color_space=self.color_space,
            transfer_function="pq",
            peak_luminance=peak,
            signal_range=p["signal_range"],
        )
        _overlay_pq_title(result.image, result.signal_format.range)
        _overlay_pq_labels(result.image, edges, luminances, result.signal_format.range)
        return result


@register
class ColorSteps(Pattern):
    id = "color-steps"
    name = "Color Steps"
    category = "Ramps & Steps"
    description = "Steps of a single color from black to full amplitude."
    parameters = [
        Parameter("steps", "Steps", ParamType.INT, default=10, minimum=2, maximum=64),
        value_mode_param(),
        *color_params(default=[1.0, 1.0, 1.0]),
        colorspace_param(),
        transfer_param(),
        peak_luminance_param(),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        img = canvas(width, height)
        steps = p["steps"]
        color = authored_color(p)
        edges = _column_edges(width, steps)
        for i in range(steps):
            img[:, edges[i] : edges[i + 1], :] = color * (i / (steps - 1))
        return finalize_authored(
            img,
            value_mode=p["value_mode"],
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            peak_luminance=p["peak_luminance"],
            signal_range=p["signal_range"],
        )
