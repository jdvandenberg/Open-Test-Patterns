"""SMPTE RP 219-2:2016 color bars.

Ultra High-Definition, 2048 x 1080 and 4096 x 2160 compatible color bar signal.

The signal is defined natively as **12-bit Y'CbCr unsigned integer code values** (Annex A
for UHDTV colorimetry, Annex B for conventional colorimetry) laid out over the
geometry of Annex C. This module builds the pattern in that native code-value
space and converts to R'G'B' with the matrix that matches the chosen
colorimetry.

Note (RP 219-2 Section 4.2): the R'G'B' values are identical for both
colorimetries; only the Y'CbCr values differ, because the R'G'B' to Y'CbCr
matrix equations differ. The colorimetry therefore selects both the code-value
table and the primaries the resulting R'G'B' is referred to.

Geometry (Annex C, Figure C.1), where ``a`` and ``b`` are the active image width
and height:

    a:b = 256:135  for 2048 x 1080 and 4096 x 2160
    a:b = 16:9     for 3840 x 2160 and 7680 x 4320

A central 4:3 region of width ``(4/3) b`` carries the bars, flanked by two gray
wings of width ``d = (a - (4/3) b) / 2``. Exact per-format widths are tabulated
below; other resolutions fall back to the ideal fractions of Figure 1.
"""

from __future__ import annotations

from typing import Any

import colour
import numpy as np

from .base import (
    Choice,
    Parameter,
    ParamType,
    Pattern,
    PatternResult,
    SignalFormat,
    SignalRange,
)
from .registry import register
from .util import effective_range, range_param, transfer_param

_BITS = 12
_MAX_CODE = 2**_BITS - 1  # 4095

# ---------------------------------------------------------------------------
# Digital coding values (12-bit Y'CbCr)
# ---------------------------------------------------------------------------

# Conventional colorimetry — SMPTE RP 219-2:2016 Annex B.
# Mandatory for 2048x1080 and 4096x2160; optional for 3840x2160 / 7680x4320.
CONVENTIONAL_12BIT: dict[str, tuple[int, int, int]] = {
    "white75": (2884, 2048, 2048),  # Table B.1
    "yellow75": (2694, 704, 2171),  # Table B.1
    "cyan75": (2325, 2356, 704),  # Table B.1
    "green75": (2136, 1012, 827),  # Table B.1
    "magenta75": (1004, 3084, 3269),  # Table B.1
    "red75": (815, 1740, 3392),  # Table B.1
    "blue75": (446, 3392, 1925),  # Table B.1
    "gray40": (1658, 2048, 2048),  # Table B.1
    "cyan100": (3015, 2459, 256),  # Table B.2 / B.3
    "blue100": (509, 3840, 1884),  # Table B.2 / B.3
    "white100": (3760, 2048, 2048),  # Table B.3 / B.4 / B.5
    "yellow100": (3507, 256, 2212),  # Table B.4
    "red100": (1001, 1637, 3840),  # Table B.4
    "black": (256, 2048, 2048),  # Table B.4 / B.5 (0% Black)
    "gray15": (782, 2048, 2048),  # Table B.5
    "minus2": (186, 2048, 2048),  # Table B.5 (-2% Black)
    "plus2": (326, 2048, 2048),  # Table B.5 (+2% Black)
    "plus4": (396, 2048, 2048),  # Table B.5 (+4% Black)
    "subblack": (16, 2048, 2048),  # Table B.5 (optional sub-black valley)
    "superwhite": (4079, 2048, 2048),  # Table B.5 (optional super-white peak)
}

# UHDTV colorimetry — SMPTE RP 219-2:2016 Annex A.
# Mandatory for UHDTV2 (7680x4320); optional for UHDTV1 (3840x2160).
UHDTV_12BIT: dict[str, tuple[int, int, int]] = {
    "white75": (2884, 2048, 2048),  # Table A.1
    "yellow75": (2728, 704, 2156),  # Table A.1
    "cyan75": (2194, 2423, 704),  # Table A.1
    "green75": (2038, 1079, 812),  # Table A.1
    "magenta75": (1102, 3017, 3284),  # Table A.1
    "red75": (946, 1673, 3392),  # Table A.1
    "blue75": (412, 3392, 1940),  # Table A.1
    "gray40": (1658, 2048, 2048),  # Table A.1
    "cyan100": (2839, 2548, 256),  # Table A.2 / A.3
    "blue100": (464, 3840, 1904),  # Table A.2 / A.3
    "white100": (3760, 2048, 2048),  # Table A.3 / A.4 / A.5
    "yellow100": (3552, 256, 2192),  # Table A.4
    "red100": (1177, 1548, 3840),  # Table A.4
    "black": (256, 2048, 2048),  # Table A.4 / A.5 (0% Black)
    "gray15": (782, 2048, 2048),  # Table A.5
    "minus2": (186, 2048, 2048),  # Table A.5 (-2% Black)
    "plus2": (326, 2048, 2048),  # Table A.5 (+2% Black)
    "plus4": (396, 2048, 2048),  # Table A.5 (+4% Black)
    "subblack": (16, 2048, 2048),  # Table A.5 (optional sub-black valley)
    "superwhite": (4079, 2048, 2048),  # Table A.5 (optional super-white peak)
}

# Colorimetry id -> (label, code-value table, Y'CbCr weights, RGB color space).
_COLORIMETRIES: dict[str, tuple[str, dict[str, tuple[int, int, int]], str, str]] = {
    "conventional": (
        "Conventional (ITU-R BT.709 primaries)",
        CONVENTIONAL_12BIT,
        "ITU-R BT.709",
        "rec709",
    ),
    "uhdtv": (
        "UHDTV (ITU-R BT.2020 primaries)",
        UHDTV_12BIT,
        "ITU-R BT.2020",
        "rec2020",
    ),
}

# ---------------------------------------------------------------------------
# Geometry (Annex C)
# ---------------------------------------------------------------------------

# Annex C offers three sets of widths per format:
#   "integer"    (a) ideal width rounded to the closest integer
#   "compatible" (b) conforms to 4:2:2 chroma sub-sampling and 2-sample
#                    interleave division (even samples; multiples of 4 for
#                    UHDTV1 / 4096x2160 and 8 for UHDTV2)
#   "modified"   (c) 75% White and Blue bars widened so there is no overlap at
#                    the 4:3 boundary when down-converted to SDTV
_WIDTH_SETS = ("integer", "compatible", "modified")

# Pattern 1 bar order within a row (Table C.1 / C.3 / C.4 / C.6):
#   d(gray) f(white) c(yellow) c(cyan) e(green) c(magenta) c(red) f(blue) d(gray)
# Stored as (d, [f, c, c, e, c, c, f]).
_PATTERN1: dict[tuple[int, int], dict[str, tuple[int, list[int]]]] = {
    (2048, 1080): {
        "integer": (304, [205, 206, 206, 206, 206, 206, 205]),
        "compatible": (304, [206, 206, 206, 204, 206, 206, 206]),
        "modified": (300, [210, 206, 206, 204, 206, 206, 210]),
    },
    (3840, 2160): {
        "integer": (480, [410, 412, 412, 412, 412, 412, 410]),
        "compatible": (480, [412, 412, 412, 408, 412, 412, 412]),
        "modified": (472, [420, 412, 412, 408, 412, 412, 420]),
    },
    (4096, 2160): {
        "integer": (608, [410, 412, 412, 412, 412, 412, 410]),
        "compatible": (608, [412, 412, 412, 408, 412, 412, 412]),
        "modified": (600, [420, 412, 412, 408, 412, 412, 420]),
    },
    (7680, 4320): {
        "integer": (960, [820, 824, 824, 824, 824, 824, 820]),
        "compatible": (960, [824, 824, 824, 816, 824, 824, 824]),
        "modified": (944, [840, 824, 824, 816, 824, 824, 840]),
    },
}

# Pattern 4 sub-bar widths within the central region (Table C.2 / C.5 / C.7):
#   k(0%) g(100%) h(0%) i1(-2%) i2(0%) i3(+2%) j1(0%) j2(+4%) m(0%)
_PATTERN4: dict[tuple[int, int], dict[str, list[int]]] = {
    (2048, 1080): {
        "integer": [309, 411, 171, 69, 68, 69, 68, 69, 206],
        "compatible": [308, 412, 170, 68, 70, 68, 70, 68, 206],
        "modified": [312, 412, 170, 68, 70, 68, 70, 68, 210],
    },
    (3840, 2160): {
        "integer": [618, 822, 342, 138, 136, 138, 136, 138, 412],
        "compatible": [616, 824, 340, 136, 140, 136, 140, 136, 412],
        "modified": [624, 824, 340, 136, 140, 136, 140, 136, 420],
    },
    (7680, 4320): {
        "integer": [1236, 1644, 684, 276, 272, 276, 272, 276, 824],
        "compatible": [1232, 1648, 680, 272, 280, 272, 280, 272, 824],
        "modified": [1248, 1648, 680, 272, 280, 272, 280, 272, 840],
    },
}
_PATTERN4[(4096, 2160)] = _PATTERN4[(3840, 2160)]

# Ideal fractions of the central 4:3 region (Figure 1), used for any resolution
# not tabulated in Annex C. Pattern 1 is seven equal bars of c = C/7; pattern 4
# is (3/2)c, 2c, (5/6)c, five of (1/3)c, and c.
_P1_IDEAL = [1 / 7] * 7
_P4_IDEAL = [3 / 14, 2 / 7, 5 / 42, 1 / 21, 1 / 21, 1 / 21, 1 / 21, 1 / 21, 1 / 7]


def _split(total: int, fractions: list[float]) -> list[int]:
    """Divide ``total`` by ``fractions`` with cumulative rounding (exact sum)."""
    edges = [0]
    acc = 0.0
    for fraction in fractions:
        acc += fraction
        edges.append(round(acc * total))
    edges[-1] = total
    return [edges[i + 1] - edges[i] for i in range(len(fractions))]


def rp219_metrics(width: int, height: int, width_set: str = "compatible") -> dict[str, Any]:
    """Return the Annex C geometry for an image of ``width`` x ``height``.

    ``d`` is the gray wing width, ``bars`` the seven pattern 1 bar widths, and
    ``pattern4`` the nine pattern 4 sub-bar widths. Tabulated formats use the
    exact Annex C values; anything else is derived from the ideal fractions.
    """
    key = (width, height)
    if key in _PATTERN1 and width_set in _PATTERN1[key]:
        d, bars = _PATTERN1[key][width_set]
        pattern4 = list(_PATTERN4[key][width_set])
    else:
        # a - 2d = central 4:3 region; absorb any odd remainder into the center
        # so the wings stay symmetric and the widths always sum to `width`.
        d = max((width - round(height * 4 / 3)) // 2, 0)
        central = width - 2 * d
        bars = _split(central, _P1_IDEAL)
        pattern4 = _split(central, _P4_IDEAL)

    return {
        "d": d,
        "bars": bars,
        "pattern4": pattern4,
        "central": sum(bars),
        "aspect": "256:135" if key in {(2048, 1080), (4096, 2160)} else "16:9",
    }


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------


def _decode(ycbcr: Any, weights: np.ndarray, legal: bool) -> np.ndarray:
    """Convert 12-bit Y'CbCr code values to normalised R'G'B' in [0, 1].

    With ``legal`` the result stays in 12-bit narrow-range code values scaled by
    4095, which is what keeps the optional sub-black (16) and super-white (4079)
    excursions representable. Otherwise 0% black maps to 0.0 and 100% white to
    1.0 and those excursions clip.
    """
    rgb = colour.YCbCr_to_RGB(
        np.asarray(ycbcr, dtype=np.float64),
        K=weights,
        in_bits=_BITS,
        in_legal=True,
        in_int=True,
        out_bits=_BITS,
        out_legal=legal,
        out_int=legal,
        clamp_int=False,
    )
    rgb = np.asarray(rgb, dtype=np.float64)
    if legal:
        rgb = rgb / _MAX_CODE
    return np.clip(rgb, 0.0, 1.0)


@register
class SmpteRp219_2_2016(Pattern):
    id = "smpte-rp219-2-2016"
    name = "SMPTE RP 219-2:2016 Color Bars"
    category = "Color Bars"
    description = (
        "Ultra High-Definition, 2048x1080 and 4096x2160 compatible color bar "
        "signal per SMPTE RP 219-2:2016, built from the normative 12-bit "
        "Y'CbCr code values."
    )
    parameters = [
        Parameter(
            "colorimetry",
            "Colorimetry",
            ParamType.CHOICE,
            default="conventional",
            choices=[Choice(k, v[0]) for k, v in _COLORIMETRIES.items()],
            description=(
                "Annex B conventional (mandatory for 2048x1080 and 4096x2160) "
                "or Annex A UHDTV (mandatory for 7680x4320). Selects both the "
                "code-value table and the output primaries."
            ),
        ),
        Parameter(
            "bar_widths",
            "Bar widths",
            ParamType.CHOICE,
            default="compatible",
            choices=[
                Choice("integer", "Ideal, rounded to integer"),
                Choice("compatible", "4:2:2 / 2-sample interleave compatible"),
                Choice("modified", "Modified 4:3 (SDTV down-convert safe)"),
            ],
            description="Annex C offers three recommended width sets per format.",
        ),
        Parameter(
            "sub_pattern_2",
            "Pattern 2 sub-pattern (*2)",
            ParamType.CHOICE,
            default="white100",
            choices=[Choice("white100", "100% White"), Choice("white75", "75% White")],
            description="Section 4.1 permits either 75% or 100% white here.",
        ),
        Parameter(
            "sub_black_super_white",
            "Sub-black valley / super-white peak (*5, *6)",
            ParamType.BOOL,
            default=False,
            description=(
                "Optional excursions in the middle third of pattern 4. Requires "
                "legal signal range to be representable."
            ),
        ),
        transfer_param(default="gamma-2.4"),
        range_param(SignalRange.LEGAL),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        _label, table, weights_name, color_space = _COLORIMETRIES[p["colorimetry"]]
        weights = colour.WEIGHTS_YCBCR[weights_name]
        signal_range = effective_range(p["transfer_function"], p["signal_range"])
        legal = signal_range is SignalRange.LEGAL

        cv = {name: _decode(code, weights, legal) for name, code in table.items()}

        geom = rp219_metrics(width, height, p["bar_widths"])
        d = geom["d"]
        bars = geom["bars"]
        k, g, h, i1, i2, i3, j1, j2, m = geom["pattern4"]
        right = width - d  # first column of the right-hand wing

        img = np.zeros((height, width, 3), dtype=np.float64)

        # Pattern heights, Table C.8: 7/12, 1/12, 1/12 and 1/4 of b, with
        # pattern 4 split into three equal thirds for sub-patterns *5 and *6.
        r1 = round(7 / 12 * height)
        r2 = round(8 / 12 * height)
        r3 = round(9 / 12 * height)
        r4 = round(10 / 12 * height)
        r5 = round(11 / 12 * height)

        # -- Pattern 1: 75% bars in the 4:3 area, 40% gray wings (*1). --------
        img[0:r1, 0:d] = cv["gray40"]
        img[0:r1, right:] = cv["gray40"]
        order = ["white75", "yellow75", "cyan75", "green75", "magenta75", "red75", "blue75"]
        x = d
        for name, bar_width in zip(order, bars, strict=True):
            img[0:r1, x : x + bar_width] = cv[name]
            x += bar_width

        # -- Pattern 2: chroma reference, 100% cyan / blue at the sides. ------
        img[r1:r2, 0:d] = cv["cyan100"]
        img[r1:r2, d : d + bars[0]] = cv[p["sub_pattern_2"]]
        img[r1:r2, d + bars[0] : right] = cv["white75"]
        img[r1:r2, right:] = cv["blue100"]

        # -- Pattern 3: 0% black (*3), Y ramp, 100% white; yellow/red sides. --
        img[r2:r3, 0:d] = cv["yellow100"]
        img[r2:r3, d : d + bars[0]] = cv["black"]
        ramp_start = d + bars[0]
        ramp_stop = right - bars[-1]
        ramp_len = max(ramp_stop - ramp_start, 1)
        # Linear slope of luminance code values from 0% to 100% white.
        ramp_y = np.linspace(table["black"][0], table["white100"][0], ramp_len)
        ramp_ycbcr = np.stack(
            [ramp_y, np.full(ramp_len, 2048.0), np.full(ramp_len, 2048.0)], axis=-1
        )
        img[r2:r3, ramp_start:ramp_stop] = _decode(ramp_ycbcr, weights, legal)
        img[r2:r3, ramp_stop:right] = cv["white100"]
        img[r2:r3, right:] = cv["red100"]

        # -- Pattern 4: PLUGE plus 0% black / 100% white; 15% gray wings (*4).
        img[r3:, 0:d] = cv["gray15"]
        img[r3:, right:] = cv["gray15"]
        segments = [
            ("black", k),
            ("white100", g),
            ("black", h),
            ("minus2", i1),
            ("black", i2),
            ("plus2", i3),
            ("black", j1),
            ("plus4", j2),
            ("black", m),
        ]
        x = d
        spans: dict[str, tuple[int, int]] = {}
        for index, (name, seg_width) in enumerate(segments):
            img[r3:, x : x + seg_width] = cv[name]
            if index == 0:
                spans["k"] = (x, x + seg_width)
            elif index == 1:
                spans["g"] = (x, x + seg_width)
            x += seg_width

        if p["sub_black_super_white"]:
            # Section 4.1: the middle third ramps from the 0% black / 100% white
            # level to the extreme at the mid-point and back again.
            for span_key, peak in (("k", "subblack"), ("g", "superwhite")):
                x0, x1 = spans[span_key]
                base = table["black"][0] if span_key == "k" else table["white100"][0]
                extreme = table[peak][0]
                n = x1 - x0
                t = 1.0 - np.abs(np.linspace(-1.0, 1.0, n))
                y = base + (extreme - base) * t
                excursion = np.stack([y, np.full(n, 2048.0), np.full(n, 2048.0)], axis=-1)
                img[r4:r5, x0:x1] = _decode(excursion, weights, legal)

        return PatternResult(
            image=np.ascontiguousarray(img, dtype=np.float32),
            signal_format=SignalFormat(
                color_space=color_space,
                transfer_function=p["transfer_function"],
                range=signal_range,
            ),
        )
