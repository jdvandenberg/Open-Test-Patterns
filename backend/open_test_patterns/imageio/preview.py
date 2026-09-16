"""Produce browser-friendly sRGB previews of patterns.

Patterns can be authored in wide gamuts and non-display transfer functions (PQ,
HLG, linear). To show them in a normal web page we decode the code values back
to light, convert to sRGB, and tone-map HDR down to the display range.

PQ is decoded to cd/m² and mapped onto a 100-nit sRGB display with a highlight
shoulder, so 100 nits is near-white without slamming every brighter code to 255.

The ``bypass`` transfer function opts out of all of that: its code values are
shown exactly as authored.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageBufAlgo, ImageSpec

from ..color import colorspaces, transfer
from ..patterns.base import PatternResult, SignalRange

# Browser sRGB is treated as a ~100 cd/m² display. 100 nits maps just below
# 1.0 so a 100-nit ramp still has headroom; light above that follows a
# Reinhard shoulder toward 1.0 at 10 000 nits instead of clipping to 255.
_SDR_WHITE_NITS = 100.0
_SDR_AT_WHITE = 0.90
_HDR_PEAK_NITS = 10000.0
_LUMA_709 = np.array([0.2126, 0.7152, 0.0722])


def _nits_to_sdr_linear(nits: np.ndarray) -> np.ndarray:
    """Map absolute cd/m² onto 0–1 linear light for an sRGB page."""
    nits = np.maximum(np.asarray(nits, dtype=np.float64), 0.0)
    pivot = _SDR_WHITE_NITS
    peak = _SDR_AT_WHITE
    low = nits * (peak / pivot)
    over = nits - pivot
    # Asymptote ``peak + (1 - peak)``; 10 000 nits lands near 0.98.
    width = (_HDR_PEAK_NITS - pivot) * 0.25
    high = peak + (1.0 - peak) * over / (over + width)
    return np.where(nits <= pivot, low, np.minimum(high, 1.0))


def _tonemap_nits_rgb(rgb_nits: np.ndarray) -> np.ndarray:
    """Compress PQ light to SDR, preserving chromaticity."""
    luma = np.tensordot(rgb_nits, _LUMA_709, axes=([-1], [0]))
    sdr = _nits_to_sdr_linear(luma)
    scale = sdr / np.maximum(luma, 1e-10)
    scale = np.where(luma <= 1e-10, 0.0, scale)
    return rgb_nits * scale[..., None]


def _soft_clip_highlights(rgb: np.ndarray) -> np.ndarray:
    """Leave in-range light alone; roll channels that would clip toward 1.0."""
    peak = np.max(rgb, axis=-1, keepdims=True)
    over = np.maximum(peak - 1.0, 0.0)
    mapped = np.where(peak <= 1.0, peak, 1.0 - 0.02 * over / (over + 1.0))
    scale = np.divide(mapped, np.maximum(peak, 1e-10))
    return rgb * scale


def to_display_srgb(result: PatternResult) -> np.ndarray:
    """Convert a pattern result to an 8-bit sRGB-encoded ``(H, W, 3)`` uint8 image."""
    sf = result.signal_format
    code = np.asarray(result.image, dtype=np.float64)

    tf = transfer.get_transfer_function(sf.transfer_function)
    if tf.is_bypass:
        # Bypass means no transform in either direction, including range scaling.
        # We cannot know what the code values represent, so decoding, matrixing
        # or expanding them would invent meaning; show them verbatim instead.
        return (np.clip(code, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    if sf.range is SignalRange.LEGAL:
        code = transfer.legal_to_full(code)

    light = tf.decode(code)
    light = np.clip(light, 0.0, None)

    if sf.color_space != "srgb":
        light = colorspaces.RGB_to_RGB(light, sf.color_space, "srgb")
        light = np.clip(light, 0.0, None)

    if tf.is_absolute:
        # PQ decodes to cd/m². Tone-map that onto a 100-nit sRGB display
        # instead of dividing by the pattern peak and clipping.
        light = _tonemap_nits_rgb(light)
    else:
        light = _soft_clip_highlights(light)

    light = np.clip(light, 0.0, 1.0)
    display = transfer.get_transfer_function("srgb").encode(light)
    return (np.clip(display, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _resize_preview(buf: ImageBuf, max_dim: int) -> ImageBuf:
    spec = buf.spec()
    w, h = spec.width, spec.height
    if max(w, h) <= max_dim:
        return buf
    scale = max_dim / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    dst = ImageBuf(ImageSpec(new_w, new_h, spec.nchannels, oiio.UINT8))
    # Nearest-neighbour: filtered resize smears single-pixel charts.
    ImageBufAlgo.resample(dst, buf, interpolate=False)
    return dst


def render_preview_png(result: PatternResult, max_dim: int | None = None) -> bytes:
    """Return PNG bytes of an sRGB preview.

    The working image is already capped for the browser. Encoding it 1:1 keeps
    Pixel Grid and similar charts sharp; a second filtered resize would blur them.
    """
    display = to_display_srgb(result)
    buf = ImageBuf(display)
    if max_dim is not None:
        buf = _resize_preview(buf, max_dim)

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        if not buf.write(tmp_path):
            raise RuntimeError(f"Preview encode failed: {buf.geterror()}")
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
