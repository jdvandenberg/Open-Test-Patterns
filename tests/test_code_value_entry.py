"""Authoring colors as 12-bit code values instead of linear light.

In code-value mode the entered number is the *final* code value: it is written
through unchanged and the transfer function only records how to read it. These
tests pin that down, since the whole point of the mode is hitting an exact level.
"""

import numpy as np
import pytest

import open_test_patterns.patterns as P
from open_test_patterns.patterns.util import (
    CODE_12BIT_MAX,
    CODE_12BIT_MODE,
    LINEAR_MODE,
    VALUE_MODE,
)

# Every pattern that offers a color input, and the color parameters it declares.
COLOR_PATTERNS = [p for p in P.all_patterns() if any(q.type == "color" for q in p.parameters)]


def color_names(pattern):
    """Base names of the color inputs, e.g. ``color`` or ``color_a``/``color_b``."""
    return [
        q.name for q in pattern.parameters if q.type == "color" and not q.name.endswith("_code")
    ]


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_code_mode_declares_both_inputs(pattern):
    names = {q.name for q in pattern.parameters}
    assert VALUE_MODE in names
    for base in color_names(pattern):
        assert f"{base}_code" in names


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_linear_mode_is_the_default(pattern):
    """The mode is additive: existing behaviour must be untouched by default."""
    assert pattern.resolve({})[VALUE_MODE] == LINEAR_MODE


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_code_value_range_is_12_bit(pattern):
    for q in pattern.parameters:
        if q.type == "color" and q.name.endswith("_code"):
            assert (q.minimum, q.maximum) == (0.0, CODE_12BIT_MAX)
            assert q.step == 1.0


@pytest.mark.parametrize("code", [0, 1, 940, 2048, 4095])
def test_entered_code_value_is_written_verbatim(code):
    """A patch authored at code N must come out at N/4095, whatever the curve."""
    pattern = P.get_pattern("solid-color")
    for tf_id in ("linear", "gamma-2.4", "srgb", "pq", "hlg"):
        result = pattern.render(
            8,
            4,
            {
                VALUE_MODE: CODE_12BIT_MODE,
                "color_code": [code, code, code],
                "transfer_function": tf_id,
            },
        )
        expected = code / CODE_12BIT_MAX
        assert np.allclose(result.image[0, 0], expected, atol=1e-6), tf_id


def test_code_mode_ignores_the_transfer_function():
    """Same code value under every curve yields the same pixels."""
    pattern = P.get_pattern("solid-color")

    def render(tf_id):
        return pattern.render(
            8,
            4,
            {
                VALUE_MODE: CODE_12BIT_MODE,
                "color_code": [2048, 512, 4095],
                "transfer_function": tf_id,
            },
        ).image

    baseline = render("linear")
    for tf_id in ("gamma-2.4", "srgb", "pq", "acescct"):
        assert np.array_equal(baseline, render(tf_id)), tf_id


def test_linear_mode_still_encodes():
    """Guard against the two modes collapsing into one."""
    pattern = P.get_pattern("solid-color")
    linear = pattern.render(
        8, 4, {VALUE_MODE: LINEAR_MODE, "color": [0.5, 0.5, 0.5], "transfer_function": "gamma-2.4"}
    )
    # Linear 0.5 through gamma 2.4 lands well above 0.5.
    assert linear.image[0, 0, 0] == pytest.approx(0.5 ** (1 / 2.4), abs=1e-6)

    code = pattern.render(
        8,
        4,
        {
            VALUE_MODE: CODE_12BIT_MODE,
            "color_code": [2048, 2048, 2048],
            "transfer_function": "gamma-2.4",
        },
    )
    assert code.image[0, 0, 0] == pytest.approx(2048 / CODE_12BIT_MAX, abs=1e-6)
    assert not np.allclose(linear.image, code.image)


def test_code_values_round_to_integers():
    """A 12-bit code value is an integer, so a typed fraction must not survive."""
    pattern = P.get_pattern("solid-color")
    result = pattern.render(
        4, 2, {VALUE_MODE: CODE_12BIT_MODE, "color_code": [2048.4, 2047.6, 0.2]}
    )
    assert np.allclose(
        result.image[0, 0], [2048 / CODE_12BIT_MAX, 2048 / CODE_12BIT_MAX, 0.0], atol=1e-6
    )


def test_code_values_clamp_to_full_scale():
    pattern = P.get_pattern("solid-color")
    assert pattern.resolve({"color_code": [9000, -5, 100]})["color_code"] == [4095.0, 0.0, 100.0]


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_each_mode_ignores_the_other_input(pattern):
    """The greyed-out control must genuinely have no effect."""
    bases = color_names(pattern)
    linear = {name: [0.25, 0.5, 0.75] for name in bases}
    codes = {f"{name}_code": [3000, 2000, 1000] for name in bases}

    in_linear = pattern.render(32, 8, {VALUE_MODE: LINEAR_MODE, **linear, **codes})
    linear_only = pattern.render(32, 8, {VALUE_MODE: LINEAR_MODE, **linear})
    assert np.array_equal(in_linear.image, linear_only.image)

    in_code = pattern.render(32, 8, {VALUE_MODE: CODE_12BIT_MODE, **linear, **codes})
    code_only = pattern.render(32, 8, {VALUE_MODE: CODE_12BIT_MODE, **codes})
    assert np.array_equal(in_code.image, code_only.image)


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_code_inputs_reach_the_image(pattern):
    """Changing the live control must change the output, for every pattern."""
    codes = {f"{name}_code": [4095, 4095, 4095] for name in color_names(pattern)}
    bright = pattern.render(64, 16, {VALUE_MODE: CODE_12BIT_MODE, **codes})
    dark = pattern.render(
        64, 16, {VALUE_MODE: CODE_12BIT_MODE, **{k: [512, 512, 512] for k in codes}}
    )
    assert not np.array_equal(bright.image, dark.image)
    assert bright.image.max() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_code_mode_output_stays_in_range(pattern):
    codes = {f"{name}_code": [4095, 0, 2048] for name in color_names(pattern)}
    img = pattern.render(64, 16, {VALUE_MODE: CODE_12BIT_MODE, **codes}).image
    assert np.all(np.isfinite(img))
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_legal_range_still_applies_to_code_values():
    """Legal range is a signal-level remap, so it applies in either mode."""
    pattern = P.get_pattern("solid-color")
    result = pattern.render(
        4,
        2,
        {VALUE_MODE: CODE_12BIT_MODE, "color_code": [4095, 4095, 4095], "signal_range": "legal"},
    )
    assert result.image[0, 0, 0] == pytest.approx(940 / 1023, abs=1e-4)


def test_peak_luminance_is_reported_only_for_absolute_curves_in_code_mode():
    """The preview needs the peak to normalise PQ; relative curves must not carry it."""
    pattern = P.get_pattern("solid-color")

    def fmt(tf_id):
        return pattern.render(
            4,
            2,
            {
                VALUE_MODE: CODE_12BIT_MODE,
                "color_code": [2048, 2048, 2048],
                "transfer_function": tf_id,
                "peak_luminance": 1000.0,
            },
        ).signal_format

    assert fmt("pq").peak_luminance == 1000.0
    assert fmt("gamma-2.4").peak_luminance is None
