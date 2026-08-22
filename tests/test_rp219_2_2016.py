"""Conformance checks for the SMPTE RP 219-2:2016 color bar pattern."""

import colour
import numpy as np
import pytest

import open_test_patterns.patterns as P
from open_test_patterns.patterns.rp219_2_2016 import (
    _MAX_CODE,
    CONVENTIONAL_12BIT,
    UHDTV_12BIT,
    rp219_metrics,
)

STANDARD_FORMATS = [(2048, 1080), (3840, 2160), (4096, 2160), (7680, 4320)]
WIDTH_SETS = ["integer", "compatible", "modified"]

_WEIGHTS = {
    "conventional": colour.WEIGHTS_YCBCR["ITU-R BT.709"],
    "uhdtv": colour.WEIGHTS_YCBCR["ITU-R BT.2020"],
}
_TABLES = {"conventional": CONVENTIONAL_12BIT, "uhdtv": UHDTV_12BIT}


def _to_ycbcr(rgb, colorimetry):
    """Round a normalised legal-range R'G'B' sample back to 12-bit Y'CbCr."""
    return tuple(
        np.round(
            colour.RGB_to_YCbCr(
                np.asarray(rgb, dtype=np.float64),
                K=_WEIGHTS[colorimetry],
                in_bits=12,
                in_legal=True,
                in_int=True,
                out_bits=12,
                out_legal=True,
                out_int=True,
                clamp_int=False,
            )
        ).astype(int)
    )


@pytest.mark.parametrize("size", STANDARD_FORMATS, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("width_set", WIDTH_SETS)
def test_annex_c_widths_sum_to_image(size, width_set):
    width, height = size
    geom = rp219_metrics(width, height, width_set)
    assert 2 * geom["d"] + sum(geom["bars"]) == width
    # Pattern 4 spans exactly the central 4:3 region.
    assert sum(geom["pattern4"]) == sum(geom["bars"])
    assert len(geom["bars"]) == 7
    assert len(geom["pattern4"]) == 9


@pytest.mark.parametrize("size", STANDARD_FORMATS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_aspect_ratio_classification(size):
    expected = "256:135" if size in {(2048, 1080), (4096, 2160)} else "16:9"
    assert rp219_metrics(*size)["aspect"] == expected


@pytest.mark.parametrize("size", [(1920, 1080), (2560, 1440), (1024, 768), (3996, 2160)])
def test_derived_geometry_is_consistent(size):
    width, height = size
    geom = rp219_metrics(width, height)
    assert 2 * geom["d"] + sum(geom["bars"]) == width
    assert sum(geom["pattern4"]) == sum(geom["bars"])


@pytest.mark.parametrize("colorimetry", ["conventional", "uhdtv"])
def test_pattern1_code_values_match_annex(colorimetry):
    """Pattern 1 bars must reproduce the normative 12-bit Y'CbCr values."""
    width, height = 3840, 2160
    pattern = P.get_pattern("smpte-rp219-2-2016")
    result = pattern.render(
        width,
        height,
        {"colorimetry": colorimetry, "bar_widths": "compatible", "signal_range": "legal"},
    )
    image = result.image.astype(np.float64) * _MAX_CODE
    geom = rp219_metrics(width, height, "compatible")
    table = _TABLES[colorimetry]

    row = round(7 / 12 * height) // 2
    names = ["white75", "yellow75", "cyan75", "green75", "magenta75", "red75", "blue75"]
    x = geom["d"]
    for name, bar_width in zip(names, geom["bars"], strict=True):
        got = _to_ycbcr(image[row, x + bar_width // 2], colorimetry)
        assert got == table[name], f"{name}: {got} != {table[name]}"
        x += bar_width

    # 40% gray wings (*1).
    assert _to_ycbcr(image[row, geom["d"] // 2], colorimetry) == table["gray40"]


def test_pluge_steps_are_distinct():
    """-2%, 0%, +2% and +4% must be distinct; -2% must sit below black."""
    width, height = 3840, 2160
    pattern = P.get_pattern("smpte-rp219-2-2016")
    result = pattern.render(width, height, {"signal_range": "legal"})
    image = result.image.astype(np.float64) * _MAX_CODE
    geom = rp219_metrics(width, height, "compatible")

    row = round(9.5 / 12 * height)
    x = geom["d"]
    levels = []
    for seg_width in geom["pattern4"]:
        levels.append(_to_ycbcr(image[row, x + seg_width // 2], "conventional")[0])
        x += seg_width

    table = CONVENTIONAL_12BIT
    assert levels[3] == table["minus2"][0] == 186
    assert levels[4] == table["black"][0] == 256
    assert levels[5] == table["plus2"][0] == 326
    assert levels[7] == table["plus4"][0] == 396
    assert levels[3] < levels[4] < levels[5] < levels[7]


def test_sub_black_and_super_white_excursions():
    width, height = 3840, 2160
    pattern = P.get_pattern("smpte-rp219-2-2016")
    result = pattern.render(
        width, height, {"signal_range": "legal", "sub_black_super_white": True}
    )
    image = result.image.astype(np.float64) * _MAX_CODE
    geom = rp219_metrics(width, height, "compatible")
    k, g = geom["pattern4"][0], geom["pattern4"][1]
    d = geom["d"]

    mid_row = round(10.5 / 12 * height)
    valley = _to_ycbcr(image[mid_row, d + k // 2], "conventional")
    peak = _to_ycbcr(image[mid_row, d + k + g // 2], "conventional")
    assert valley == CONVENTIONAL_12BIT["subblack"]
    assert peak == CONVENTIONAL_12BIT["superwhite"]

    # Outer thirds stay at plain 0% black / 100% white.
    top_row = round(9.5 / 12 * height)
    assert _to_ycbcr(image[top_row, d + k // 2], "conventional") == CONVENTIONAL_12BIT["black"]
    assert (
        _to_ycbcr(image[top_row, d + k + g // 2], "conventional")
        == CONVENTIONAL_12BIT["white100"]
    )


def test_ramp_is_monotonic_black_to_white():
    width, height = 3840, 2160
    pattern = P.get_pattern("smpte-rp219-2-2016")
    result = pattern.render(width, height, {"signal_range": "legal"})
    image = result.image.astype(np.float64) * _MAX_CODE
    geom = rp219_metrics(width, height, "compatible")

    row = round(8.5 / 12 * height)
    start = geom["d"] + geom["bars"][0]
    stop = width - geom["d"] - geom["bars"][-1]
    luma = image[row, start:stop, 0]
    assert np.all(np.diff(luma) >= -1e-6)
    assert _to_ycbcr(image[row, start], "conventional")[0] == 256
    assert _to_ycbcr(image[row, stop - 1], "conventional")[0] == 3760


def test_colorimetry_selects_color_space():
    pattern = P.get_pattern("smpte-rp219-2-2016")
    conv = pattern.render(640, 360, {"colorimetry": "conventional"})
    uhd = pattern.render(640, 360, {"colorimetry": "uhdtv"})
    assert conv.signal_format.color_space == "rec709"
    assert uhd.signal_format.color_space == "rec2020"


def test_both_colorimetries_yield_same_rgb():
    """RP 219-2 s4.2: R'G'B' is identical; only the Y'CbCr encoding differs."""
    pattern = P.get_pattern("smpte-rp219-2-2016")
    conv = pattern.render(640, 360, {"colorimetry": "conventional"}).image
    uhd = pattern.render(640, 360, {"colorimetry": "uhdtv"}).image
    # The two Annex tables are independently rounded to integer code values, so
    # allow a couple of 12-bit steps of disagreement.
    assert np.max(np.abs(conv.astype(float) - uhd.astype(float))) < 4.0 / _MAX_CODE
