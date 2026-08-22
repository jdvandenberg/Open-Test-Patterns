"""RGB color-space definitions and conversions.

We expose a curated set of color spaces relevant to cinema, broadcast, and HDR
display testing. Each maps to a well-defined :class:`colour.RGB_Colourspace`
from colour-science, which provides the tested normalised primary matrices and
white points. This replaces the hand-rolled primary→matrix math in the original
``pyColor.py`` while keeping the same set of gamuts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import colour
import numpy as np
from colour.models import RGB_COLOURSPACES

ArrayLike = np.ndarray | list | tuple


@dataclass(frozen=True)
class ColorSpaceInfo:
    """Metadata describing an RGB color space."""

    id: str
    name: str
    colour_name: str
    primaries: list[list[float]] = field(default_factory=list)
    whitepoint: list[float] = field(default_factory=list)

    @property
    def colourspace(self) -> colour.RGB_Colourspace:
        return RGB_COLOURSPACES[self.colour_name]


# Stable id -> colour-science colourspace name. Ids are what the API/UI use so
# that renaming upstream never breaks stored presets.
_DEFINITIONS: list[tuple[str, str, str]] = [
    ("srgb", "sRGB / Rec. 709", "sRGB"),
    ("rec709", "ITU-R BT.709", "ITU-R BT.709"),
    ("rec2020", "ITU-R BT.2020 / BT.2100", "ITU-R BT.2020"),
    ("p3-d65", "P3-D65", "P3-D65"),
    ("display-p3", "Display P3", "Display P3"),
    ("dci-p3", "DCI-P3 (Theater)", "DCI-P3"),
    ("aces2065-1", "ACES2065-1 (AP0)", "ACES2065-1"),
    ("acescg", "ACEScg (AP1)", "ACEScg"),
]


def _build() -> dict[str, ColorSpaceInfo]:
    spaces: dict[str, ColorSpaceInfo] = {}
    for space_id, label, colour_name in _DEFINITIONS:
        if colour_name not in RGB_COLOURSPACES:
            # Skip gracefully if a given colour-science version lacks a space.
            continue
        cs = RGB_COLOURSPACES[colour_name]
        spaces[space_id] = ColorSpaceInfo(
            id=space_id,
            name=label,
            colour_name=colour_name,
            primaries=np.asarray(cs.primaries).tolist(),
            whitepoint=np.asarray(cs.whitepoint).tolist(),
        )
    return spaces


COLOR_SPACES: dict[str, ColorSpaceInfo] = _build()


def list_colorspaces() -> list[ColorSpaceInfo]:
    """Return all available color spaces."""
    return list(COLOR_SPACES.values())


def get_colorspace(space_id: str) -> ColorSpaceInfo:
    """Look up a color space by its stable id."""
    try:
        return COLOR_SPACES[space_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown color space {space_id!r}. "
            f"Available: {sorted(COLOR_SPACES)}"
        ) from exc


def RGB_to_XYZ(rgb: ArrayLike, space_id: str) -> np.ndarray:
    """Convert linear RGB in ``space_id`` to CIE XYZ (D65-relative)."""
    cs = get_colorspace(space_id).colourspace
    return colour.RGB_to_XYZ(
        np.asarray(rgb, dtype=np.float64),
        cs,
        apply_cctf_decoding=False,
    )


def XYZ_to_RGB(xyz: ArrayLike, space_id: str) -> np.ndarray:
    """Convert CIE XYZ to linear RGB in ``space_id``."""
    cs = get_colorspace(space_id).colourspace
    return colour.XYZ_to_RGB(
        np.asarray(xyz, dtype=np.float64),
        cs,
        apply_cctf_encoding=False,
    )


def RGB_to_RGB(
    rgb: ArrayLike,
    source_id: str,
    target_id: str,
    chromatic_adaptation: str = "Bradford",
) -> np.ndarray:
    """Convert linear RGB between two color spaces with chromatic adaptation."""
    source = get_colorspace(source_id).colourspace
    target = get_colorspace(target_id).colourspace
    return colour.RGB_to_RGB(
        np.asarray(rgb, dtype=np.float64),
        source,
        target,
        chromatic_adaptation_transform=chromatic_adaptation,
    )
