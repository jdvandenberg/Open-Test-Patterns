"""Transfer functions and signal-range helpers.

Encoding here means *linear light* -> *non-linear code value* (the inverse EOTF
/ OETF direction), because that is what test-pattern generators produce before
writing a file.

For PQ (SMPTE ST 2084) the "linear" input is **absolute luminance in cd/m^2**
(nits) in ``[0, 10000]``. For display-referred curves (gamma, sRGB) the input is
**relative** linear light in ``[0, 1]``. HLG uses scene-referred linear ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import colour
import numpy as np

ArrayLike = np.ndarray | list | tuple | float

# Legal (narrow) range anchors for 10-bit, expressed as normalised code values.
_LEGAL_BLACK = 64 / 1023
_LEGAL_WHITE = 940 / 1023
_LEGAL_SPAN = _LEGAL_WHITE - _LEGAL_BLACK


def _encode_pq(linear: ArrayLike, peak_luminance: float = 10000.0) -> np.ndarray:
    """Inverse EOTF of SMPTE ST 2084 (PQ).

    ``linear`` is absolute luminance in cd/m^2. ``peak_luminance`` scales a
    normalised ``[0, 1]`` input onto the PQ curve when a generator works in
    relative terms.
    """
    x = np.asarray(linear, dtype=np.float64)
    return np.clip(colour.models.eotf_inverse_ST2084(np.clip(x, 0.0, 10000.0)), 0.0, 1.0)


def _decode_pq(code: ArrayLike) -> np.ndarray:
    """EOTF of SMPTE ST 2084 (PQ): code value -> luminance (cd/m^2)."""
    return colour.models.eotf_ST2084(np.asarray(code, dtype=np.float64))


def _encode_gamma(linear: ArrayLike, gamma: float = 2.4) -> np.ndarray:
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return x ** (1.0 / gamma)


def _decode_gamma(code: ArrayLike, gamma: float = 2.4) -> np.ndarray:
    x = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    return x**gamma


def _encode_srgb(linear: ArrayLike) -> np.ndarray:
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return colour.models.eotf_inverse_sRGB(x)


def _decode_srgb(code: ArrayLike) -> np.ndarray:
    x = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    return colour.models.eotf_sRGB(x)


# colour-science evaluates both halves of the HLG piecewise curve and selects
# with np.where, so the logarithmic branch is computed for inputs below its
# domain and warns about the NaN it then discards. The result is unaffected.
def _encode_hlg(linear: ArrayLike) -> np.ndarray:
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(colour.models.oetf_BT2100_HLG(x), 0.0, 1.0)


def _decode_hlg(code: ArrayLike) -> np.ndarray:
    x = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return colour.models.oetf_inverse_BT2100_HLG(x)


# ACEScc and ACEScct are AP1-primaries log encodings, so they pair with the
# ACEScg color space rather than defining a gamut of their own. colour-science
# takes log2 of the linear input, which warns at exactly 0 before substituting
# the curve's floor, so the benign warning is silenced at the call.
def _encode_acescc(linear: ArrayLike) -> np.ndarray:
    x = np.asarray(linear, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(colour.models.log_encoding_ACEScc(x), 0.0, 1.0)


def _decode_acescc(code: ArrayLike) -> np.ndarray:
    x = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return colour.models.log_decoding_ACEScc(x)


def _encode_acescct(linear: ArrayLike) -> np.ndarray:
    x = np.asarray(linear, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(colour.models.log_encoding_ACEScct(x), 0.0, 1.0)


def _decode_acescct(code: ArrayLike) -> np.ndarray:
    x = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return colour.models.log_decoding_ACEScct(x)


def _identity(x: ArrayLike, **_: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class TransferFunctionInfo:
    """A named opto-electronic / electro-optical transfer function pair."""

    id: str
    name: str
    encode: Callable[..., np.ndarray]
    decode: Callable[..., np.ndarray]
    is_absolute: bool = False
    is_bypass: bool = False
    description: str = ""


TRANSFER_FUNCTIONS: dict[str, TransferFunctionInfo] = {
    "bypass": TransferFunctionInfo(
        "bypass",
        "Bypass (no processing)",
        _identity,
        _identity,
        is_bypass=True,
        description=(
            "Pass code values through untouched. Unlike Linear, the preview also "
            "skips decoding and gamut conversion, so the authored numbers are "
            "exactly what gets written and displayed."
        ),
    ),
    "linear": TransferFunctionInfo(
        "linear",
        "Linear",
        _identity,
        _identity,
        description="No encoding; values are linear light and are decoded as such for preview.",
    ),
    "pq": TransferFunctionInfo(
        "pq",
        "PQ (SMPTE ST 2084)",
        _encode_pq,
        _decode_pq,
        is_absolute=True,
        description="Perceptual Quantizer; input/output in absolute cd/m^2.",
    ),
    "hlg": TransferFunctionInfo(
        "hlg",
        "HLG (ITU-R BT.2100)",
        _encode_hlg,
        _decode_hlg,
        description="Hybrid Log-Gamma, scene-referred [0, 1].",
    ),
    "acescc": TransferFunctionInfo(
        "acescc",
        "ACEScc (AP1 log)",
        _encode_acescc,
        _decode_acescc,
        description=(
            "Pure logarithmic ACES encoding on AP1 primaries; pair with the "
            "ACEScg color space. Linear 1.0 encodes to 0.5548, and black clips "
            "because ACEScc places linear 0 below zero (-0.3584)."
        ),
    ),
    "acescct": TransferFunctionInfo(
        "acescct",
        "ACEScct (AP1 log with toe)",
        _encode_acescct,
        _decode_acescct,
        description=(
            "As ACEScc but with a linear toe near black, so linear 0 encodes to "
            "0.0729 and survives without clipping. Pair with ACEScg."
        ),
    ),
    "srgb": TransferFunctionInfo(
        "srgb", "sRGB (IEC 61966-2-1)", _encode_srgb, _decode_srgb
    ),
    "gamma-2.2": TransferFunctionInfo(
        "gamma-2.2",
        "Gamma 2.2",
        lambda x: _encode_gamma(x, 2.2),
        lambda x: _decode_gamma(x, 2.2),
    ),
    "gamma-2.4": TransferFunctionInfo(
        "gamma-2.4",
        "Gamma 2.4 (BT.1886-like)",
        lambda x: _encode_gamma(x, 2.4),
        lambda x: _decode_gamma(x, 2.4),
    ),
    "gamma-2.6": TransferFunctionInfo(
        "gamma-2.6",
        "Gamma 2.6 (DCI)",
        lambda x: _encode_gamma(x, 2.6),
        lambda x: _decode_gamma(x, 2.6),
    ),
}


def list_transfer_functions() -> list[TransferFunctionInfo]:
    return list(TRANSFER_FUNCTIONS.values())


def get_transfer_function(tf_id: str) -> TransferFunctionInfo:
    try:
        return TRANSFER_FUNCTIONS[tf_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown transfer function {tf_id!r}. "
            f"Available: {sorted(TRANSFER_FUNCTIONS)}"
        ) from exc


def encode(linear: ArrayLike, tf_id: str) -> np.ndarray:
    """Apply the encoding (inverse-EOTF / OETF) direction of ``tf_id``."""
    return get_transfer_function(tf_id).encode(linear)


def decode(code: ArrayLike, tf_id: str) -> np.ndarray:
    """Apply the decoding (EOTF / inverse-OETF) direction of ``tf_id``."""
    return get_transfer_function(tf_id).decode(code)


def full_to_legal(color_full: ArrayLike) -> np.ndarray:
    """Map full-range ``[0, 1]`` code values to 10-bit legal (narrow) range."""
    x = np.asarray(color_full, dtype=np.float64)
    return np.clip(x * _LEGAL_SPAN + _LEGAL_BLACK, _LEGAL_BLACK, _LEGAL_WHITE)


def legal_to_full(color_legal: ArrayLike) -> np.ndarray:
    """Map 10-bit legal (narrow) range code values back to full range ``[0, 1]``."""
    x = np.asarray(color_legal, dtype=np.float64)
    return np.clip((x - _LEGAL_BLACK) / _LEGAL_SPAN, 0.0, 1.0)
