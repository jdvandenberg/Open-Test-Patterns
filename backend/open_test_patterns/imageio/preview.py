"""Produce browser-friendly sRGB previews of patterns.

Patterns can be authored in wide gamuts and non-display transfer functions (PQ,
HLG, linear). To show them in a normal web page we decode the code values back
to light, convert to sRGB, and tone-map HDR down to the display range.

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

    if tf.is_absolute:
        # PQ: decoded values are cd/m^2. Normalise by the pattern's peak so the
        # brightest intended level maps to 1.0 for display.
        peak = sf.peak_luminance if sf.peak_luminance else 100.0
        # peak_luminance for absolute patterns is the nits mapped to 1.0.
        light = light / max(peak, 1e-6)

    light = np.clip(light, 0.0, None)

    # Convert to sRGB/Rec.709 linear, then tone-map any remaining overshoot.
    if sf.color_space != "srgb":
        light = colorspaces.RGB_to_RGB(light, sf.color_space, "srgb")
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
