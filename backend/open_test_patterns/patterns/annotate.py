"""Burn a measurement readout into a finished pattern.

The overlay reports the 12-bit RGB code of one pixel and the luminance a
probe should read if those same codes are displayed as Rec.2020 PQ, Rec.709
gamma 2.4, or theatrical DCI-P3. It is written *after* encoding, as a mid-grey
*code value*, so PQ cannot turn the text into a glare source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageBufAlgo

from ..color import colorspaces, transfer
from .base import PatternResult, SignalRange
from .util import CODE_12BIT_MAX

# ~18 nits on PQ — photographic mid-grey at a 100-nit reference — and a dim
# grey on SDR curves. Bright enough to read on a black field, dim enough not
# to become a glare source on a 1000-nit monitor.
LABEL_CODE = 0.35


@dataclass(frozen=True)
class DisplaySystem:
    """A display system under which the same RGB codes have a known luminance."""

    label: str
    color_space: str
    transfer_function: str
    # Relative systems map linear 1.0 to this many cd/m². Absolute systems
    # (PQ) already decode to nits, so peak is None.
    peak_luminance: float | None


# Same codes, three interpretations. Rec.709 peak is the 100 cd/m² studio
# reference; DCI-P3 is SMPTE ST 431-1 (gamma 2.6, 48 cd/m²).
DISPLAY_SYSTEMS: tuple[DisplaySystem, ...] = (
    DisplaySystem("Rec.2020 PQ", "rec2020", "pq", None),
    DisplaySystem("Rec.709 γ2.4", "rec709", "gamma-2.4", 100.0),
    DisplaySystem("DCI-P3 γ2.6", "dci-p3", "gamma-2.6", 48.0),
)


def format_code_rgb(rgb: np.ndarray) -> str:
    """Format a unit-range RGB triple as 12-bit integer code values."""
    codes = np.rint(np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0) * CODE_12BIT_MAX)
    r, g, b = (int(c) for c in codes)
    return f"R {r}  G {g}  B {b}"


def format_nits(luminance: float) -> str:
    """Compact cd/m² text that stays readable from 0.001 to 10000."""
    if luminance < 0.01:
        return f"{luminance:.3f}"
    if luminance < 10:
        return f"{luminance:.2f}"
    if luminance < 100:
        return f"{luminance:.1f}"
    return f"{luminance:.0f}"


def luminance_cdm2(rgb_code: np.ndarray, system: DisplaySystem) -> float:
    """Photometric luminance of ``rgb_code`` displayed on ``system``, in cd/m².

    ``rgb_code`` is full-range [0, 1]. The codes are decoded with that system's
    transfer function and weighted by its primaries; no gamut conversion is
    applied, because a probe reads the light the display actually emits.
    """
    linear = transfer.decode(rgb_code, system.transfer_function)
    y = float(np.asarray(colorspaces.RGB_to_XYZ(linear, system.color_space))[..., 1])
    if system.peak_luminance is None:
        return y
    return y * system.peak_luminance


def format_luminance_lines(rgb_code: np.ndarray) -> list[str]:
    """One line per display system: ``Rec.2020 PQ  92.4 cd/m²``."""
    lines = []
    for system in DISPLAY_SYSTEMS:
        nits = luminance_cdm2(rgb_code, system)
        lines.append(f"{system.label}  {format_nits(nits)} cd/m²")
    return lines


def measurement_lines(rgb_stored: np.ndarray, signal_range: SignalRange) -> list[str]:
    """RGB readout plus the three expected probe readings.

    Legal-range codes are expanded to full range before the EOTF, which is
    what a legal-range display does before it produces light.
    """
    stored = np.asarray(rgb_stored, dtype=np.float64)
    full = transfer.legal_to_full(stored) if signal_range is SignalRange.LEGAL else stored
    return [format_code_rgb(stored), *format_luminance_lines(full)]


def _stroke_width(fontsize: int) -> int:
    """Black halo wide enough to separate mid-grey glyphs from any field."""
    return max(2, round(fontsize * 0.1))


def _block_origin(
    lines: list[str], fontsize: int, width: int, height: int
) -> tuple[int, int] | None:
    """Baseline of the *last* line so the block sits bottom-left, or None if it
    will not fit."""
    rois = [ImageBufAlgo.text_size(line, fontsize) for line in lines]
    if not all(roi.defined for roi in rois):
        return None
    gap = max(2, round(fontsize * 0.25))
    stroke = _stroke_width(fontsize)
    block_width = max(roi.width for roi in rois) + 2 * stroke
    block_height = sum(roi.height for roi in rois) + gap * (len(lines) - 1) + 2 * stroke
    margin_x = max(12, width // 80)
    margin_y = max(10, height // 60)
    if block_width + 2 * margin_x > width or block_height + 2 * margin_y > height:
        return None
    last = rois[-1]
    x = margin_x - last.xbegin
    y = height - margin_y - last.yend
    return x, y


def _draw_lines(buf: ImageBuf, lines: list[str], fontsize: int, x: int, y_last: int) -> bool:
    """Draw ``lines`` stacked upward, last line at baseline ``y_last``.

    Each line is stroked in black first (OIIO's ``shadow``) so the mid-grey
    fill stays readable on white, mid-grey, or coloured fields.
    """
    grey = (LABEL_CODE, LABEL_CODE, LABEL_CODE)
    gap = max(2, round(fontsize * 0.25))
    stroke = _stroke_width(fontsize)
    y = y_last
    for line in reversed(lines):
        if not ImageBufAlgo.render_text(
            buf, x, y, line, fontsize, "", grey, shadow=stroke
        ):
            return False
        roi = ImageBufAlgo.text_size(line, fontsize)
        y -= roi.height + gap
    return True


def annotate_code_rgb(
    result: PatternResult, sample: tuple[int, int] | None = None
) -> PatternResult:
    """Overlay a bottom-left measurement readout of one pixel of ``result``.

    ``sample`` is ``(row, col)``; the frame centre is used when omitted. Tiny
    frames that cannot hold the label are left untouched, so unit tests that
    render a few pixels still see a uniform field.
    """
    img = result.image
    height, width = img.shape[:2]
    if sample is None:
        row, col = height // 2, width // 2
    else:
        row, col = sample
    lines = measurement_lines(img[row, col], result.signal_format.range)
    fontsize = max(14, round(min(width, height) * 0.022))
    origin = _block_origin(lines, fontsize, width, height)
    if origin is None:
        return result

    buf = ImageBuf(np.ascontiguousarray(img))
    if not _draw_lines(buf, lines, fontsize, *origin):
        return result

    result.image = np.ascontiguousarray(buf.get_pixels(oiio.FLOAT), dtype=np.float32)
    return result
