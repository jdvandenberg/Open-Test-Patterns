"""Bottom-left 12-bit RGB readout on measurement fields."""

import numpy as np
import pytest

import open_test_patterns.patterns as P
from open_test_patterns.patterns.annotate import (
    DISPLAY_SYSTEMS,
    LABEL_CODE,
    format_code_rgb,
    format_luminance_lines,
    format_nits,
    luminance_cdm2,
    measurement_lines,
)
from open_test_patterns.patterns.base import SignalRange
from open_test_patterns.patterns.util import CODE_12BIT_MAX, CODE_12BIT_MODE, VALUE_MODE

LABELED = ("centered-patch", "gray-field", "solid-color")


@pytest.mark.parametrize(
    "rgb, expected",
    [
        ([1.0, 1.0, 1.0], "R 4095  G 4095  B 4095"),
        ([0.0, 0.0, 0.0], "R 0  G 0  B 0"),
        (
            [2048 / CODE_12BIT_MAX, 1024 / CODE_12BIT_MAX, 512 / CODE_12BIT_MAX],
            "R 2048  G 1024  B 512",
        ),
        ([1.2, -0.3, 0.5], "R 4095  G 0  B 2048"),
    ],
)
def test_format_code_rgb_rounds_to_12_bit(rgb, expected):
    assert format_code_rgb(rgb) == expected


def test_white_has_reference_peaks():
    """Full-code white is 10000 nits PQ, 100 nits Rec.709, 48 nits DCI-P3."""
    white = np.array([1.0, 1.0, 1.0])
    pq, rec709, p3 = DISPLAY_SYSTEMS
    assert luminance_cdm2(white, pq) == pytest.approx(10000.0, abs=0.1)
    assert luminance_cdm2(white, rec709) == pytest.approx(100.0, abs=1e-6)
    assert luminance_cdm2(white, p3) == pytest.approx(48.0, abs=1e-6)


def test_eighteen_percent_grey_is_18_nits_on_rec709():
    """18% linear through γ2.4 is the studio mid-grey at a 100-nit peak."""
    code = np.full(3, 0.18 ** (1 / 2.4))
    assert luminance_cdm2(code, DISPLAY_SYSTEMS[1]) == pytest.approx(18.0, abs=1e-6)


def test_luminance_uses_each_system_not_the_files_tag():
    """The three readings interpret the same codes; they must not collapse."""
    mid = np.full(3, 2048 / CODE_12BIT_MAX)
    nits = [luminance_cdm2(mid, system) for system in DISPLAY_SYSTEMS]
    assert nits[0] == pytest.approx(92.359, abs=0.05)  # PQ of ~0.5
    assert nits[1] == pytest.approx(18.958, abs=0.05)  # 0.5**2.4 * 100
    assert nits[2] == pytest.approx(7.922, abs=0.05)  # 0.5**2.6 * 48
    assert nits[0] > nits[1] > nits[2]


def test_colored_patch_uses_luminance_weights():
    """A Rec.709 red is 21.26% of peak, not 100 nits."""
    red = np.array([1.0, 0.0, 0.0])
    assert luminance_cdm2(red, DISPLAY_SYSTEMS[1]) == pytest.approx(21.26, abs=0.02)


def test_legal_range_is_expanded_before_the_eotf():
    """Legal white (10-bit 940) must still read as reference-peak luminance."""
    legal_white = np.full(3, 940 / 1023)
    lines = measurement_lines(legal_white, SignalRange.LEGAL)
    assert lines[0] == format_code_rgb(legal_white)
    assert "10000 cd/m²" in lines[1]
    assert "100 cd/m²" in lines[2]
    assert "48.0 cd/m²" in lines[3]


def test_luminance_lines_name_the_three_systems():
    lines = format_luminance_lines(np.array([1.0, 1.0, 1.0]))
    assert lines[0].startswith("Rec.2020 PQ")
    assert lines[1].startswith("Rec.709 γ2.4")
    assert lines[2].startswith("DCI-P3 γ2.6")
    assert all("cd/m²" in line for line in lines)


@pytest.mark.parametrize(
    "value, expected",
    [(0.0012, "0.001"), (0.42, "0.42"), (18.96, "19.0"), (10000, "10000")],
)
def test_nits_formatting(value, expected):
    assert format_nits(value) == expected


@pytest.mark.parametrize("pattern_id", LABELED)
def test_label_is_mid_grey_at_bottom_left(pattern_id):
    """The readout is a fixed mid-grey code, not the field colour, and sits
    in the bottom-left — so a probe on the patch still sees the intended level."""
    pattern = P.get_pattern(pattern_id)
    result = pattern.render(640, 360, {})
    bottom_left = result.image[-90:, :280, 0]
    # Antialiased glyphs land on the label code; the rest of the corner is the field.
    assert np.any(np.abs(bottom_left - LABEL_CODE) < 0.02)
    # A probe at the centre still sees the field, not the label.
    assert abs(float(result.image[180, 320, 0]) - LABEL_CODE) > 0.05


def test_label_has_black_surround_on_a_white_field():
    """Without a dark halo the mid-grey glyphs vanish into a white solid."""
    pattern = P.get_pattern("solid-color")
    result = pattern.render(640, 360, {"color": [1.0, 1.0, 1.0], "transfer_function": "linear"})
    corner = result.image[-90:, :280, 0]
    assert np.any(corner < 0.05)
    assert np.any(np.abs(corner - LABEL_CODE) < 0.02)
    assert np.any(corner > 0.95)


def test_centered_patch_window_is_not_overwritten():
    pattern = P.get_pattern("centered-patch")
    result = pattern.render(
        640,
        360,
        {VALUE_MODE: CODE_12BIT_MODE, "color_code": [4095, 0, 0], "window_size": 0.2},
    )
    cy, cx = 180, 320
    assert np.allclose(result.image[cy, cx], [1.0, 0.0, 0.0], atol=1e-5)
    # Label lives on the black field, not in the window.
    assert result.image[-20:, :180].max() == pytest.approx(LABEL_CODE, abs=0.02)


def test_label_reports_the_encoded_code_value():
    """The numbers come from the finished image, after the transfer function."""
    pattern = P.get_pattern("solid-color")
    result = pattern.render(
        640,
        360,
        {"color": [0.5, 0.25, 0.0], "transfer_function": "gamma-2.4"},
    )
    encoded = result.image[180, 320]
    expected = np.array([0.5 ** (1 / 2.4), 0.25 ** (1 / 2.4), 0.0])
    assert np.allclose(encoded, expected, atol=1e-5)
    r, g, b = (int(round(c * CODE_12BIT_MAX)) for c in expected)
    assert format_code_rgb(encoded) == f"R {r}  G {g}  B {b}"


def test_twelve_bit_entry_matches_the_label_numbers():
    pattern = P.get_pattern("solid-color")
    result = pattern.render(
        640,
        360,
        {VALUE_MODE: CODE_12BIT_MODE, "color_code": [2048, 1024, 512]},
    )
    assert format_code_rgb(result.image[180, 320]) == "R 2048  G 1024  B 512"


def test_tiny_frames_are_not_annotated():
    """Existing pixel-exact tests render a handful of pixels; those stay clean."""
    pattern = P.get_pattern("solid-color")
    result = pattern.render(8, 4, {"color": [0.2, 0.2, 0.2], "transfer_function": "linear"})
    assert np.allclose(result.image, 0.2)


def test_gray_field_label_uses_the_encoded_level():
    pattern = P.get_pattern("gray-field")
    result = pattern.render(640, 360, {"level": 0.18, "transfer_function": "gamma-2.4"})
    encoded = float(result.image[180, 320, 0])
    expected = int(round(0.18 ** (1 / 2.4) * CODE_12BIT_MAX))
    assert format_code_rgb(result.image[180, 320]) == f"R {expected}  G {expected}  B {expected}"
    assert encoded == pytest.approx(0.18 ** (1 / 2.4), abs=1e-5)
