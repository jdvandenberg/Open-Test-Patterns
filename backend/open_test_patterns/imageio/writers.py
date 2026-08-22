"""Write generated patterns to production image formats via OpenImageIO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import OpenImageIO as oiio
from OpenImageIO import ImageOutput, ImageSpec

from ..patterns.base import PatternResult


@dataclass(frozen=True)
class CompressionOption:
    """A compression scheme offered for a format.

    ``id`` doubles as the value written to OpenImageIO's ``compression``
    attribute, so it must be a name OpenEXR recognises.
    """

    id: str
    label: str
    lossless: bool = True


# OpenEXR compression schemes. All three are lossless and safe for test
# patterns, where any lossy scheme (DWAA/DWAB, B44) would corrupt the exact code
# values the patterns exist to carry.
EXR_COMPRESSIONS: tuple[CompressionOption, ...] = (
    CompressionOption("none", "Uncompressed"),
    CompressionOption("zip", "ZIP (deflate, 16-scanline blocks)"),
    CompressionOption("piz", "PIZ (wavelet + Huffman)"),
)


@dataclass(frozen=True)
class ImageFormat:
    id: str
    extension: str
    label: str
    default_bit_depth: int
    allowed_bit_depths: tuple[int, ...]
    mime: str
    compressions: tuple[CompressionOption, ...] = ()
    default_compression: str | None = None


SUPPORTED_FORMATS: dict[str, ImageFormat] = {
    "exr": ImageFormat(
        "exr",
        "exr",
        "OpenEXR",
        16,
        (16, 32),
        "image/x-exr",
        compressions=EXR_COMPRESSIONS,
        default_compression="zip",
    ),
    "dpx": ImageFormat("dpx", "dpx", "DPX", 10, (10, 12, 16), "image/x-dpx"),
    "tiff": ImageFormat("tiff", "tif", "TIFF", 16, (8, 16), "image/tiff"),
    "png": ImageFormat("png", "png", "PNG", 16, (8, 16), "image/png"),
}


def _oiio_type(fmt: ImageFormat, bit_depth: int) -> oiio.BASETYPE:
    if fmt.id == "exr":
        return oiio.HALF if bit_depth == 16 else oiio.FLOAT
    if bit_depth == 8:
        return oiio.UINT8
    return oiio.UINT16


def resolve_compression(fmt: ImageFormat, compression: str | None) -> str | None:
    """Validate ``compression`` for ``fmt``, returning the name to write.

    OpenImageIO silently falls back to its default when handed an unknown
    compression name, so unrecognised values are rejected here instead of
    producing a file that quietly disagrees with what was asked for.
    """
    if not fmt.compressions:
        if compression:
            raise ValueError(f"{fmt.label} does not support a compression choice")
        return None
    if compression is None:
        return fmt.default_compression
    allowed = {c.id for c in fmt.compressions}
    if compression not in allowed:
        raise ValueError(
            f"{fmt.label} supports compressions {sorted(allowed)}, got {compression!r}"
        )
    return compression


def write_image(
    result: PatternResult,
    path: str,
    format_id: str,
    bit_depth: int | None = None,
    compression: str | None = None,
) -> str:
    """Write ``result`` to ``path`` in the requested format.

    Float [0, 1] code values are converted to the output data type by
    OpenImageIO. Color-space / transfer metadata is attached as attributes.
    """
    if format_id not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {format_id!r}. Options: {sorted(SUPPORTED_FORMATS)}")
    fmt = SUPPORTED_FORMATS[format_id]
    depth = bit_depth or fmt.default_bit_depth
    if depth not in fmt.allowed_bit_depths:
        raise ValueError(
            f"{fmt.label} supports bit depths {fmt.allowed_bit_depths}, got {depth}"
        )
    compression_name = resolve_compression(fmt, compression)

    pixels = np.ascontiguousarray(result.image, dtype=np.float32)
    height, width, channels = pixels.shape
    spec = ImageSpec(width, height, channels, _oiio_type(fmt, depth))

    if compression_name is not None:
        spec.attribute("compression", compression_name)

    # Signal towards true integer bit depth (e.g. 10/12-bit DPX).
    if fmt.id == "dpx":
        spec.attribute("oiio:BitsPerSample", depth)

    sf = result.signal_format
    # OpenImageIO may normalise the standard "oiio:ColorSpace" name via its OCIO
    # config, so also record the exact id under our own namespace.
    spec.attribute("oiio:ColorSpace", sf.color_space)
    spec.attribute("otp:colorSpace", sf.color_space)
    spec.attribute("otp:transferFunction", sf.transfer_function)
    spec.attribute("otp:signalRange", sf.range.value)
    if sf.peak_luminance is not None:
        spec.attribute("otp:peakLuminance", float(sf.peak_luminance))
    spec.attribute("Software", "Open Test Patterns")

    out = ImageOutput.create(path)
    if out is None:
        raise RuntimeError(f"Could not create image output for {path!r}: {oiio.geterror()}")
    if not out.open(path, spec):
        raise RuntimeError(f"Could not open {path!r}: {out.geterror()}")
    try:
        if not out.write_image(pixels):
            raise RuntimeError(f"Failed writing {path!r}: {out.geterror()}")
    finally:
        out.close()
    return path
