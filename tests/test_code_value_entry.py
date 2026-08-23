"""Authoring colors as 10- or 12-bit code values instead of linear light.

In a code-value mode the entered number is the *final* code value: it is written
through unchanged. Encoding controls (color space, transfer, peak, range) are
greyed out and must not move a pixel.
"""

import numpy as np
import pytest

import open_test_patterns.patterns as P
from open_test_patterns.patterns.util import (
    CODE_10BIT_MAX,
    CODE_10BIT_MODE,
    CODE_12BIT_MAX,
    CODE_12BIT_MODE,
    CODE_MODES,
    LINEAR_MODE,
    VALUE_MODE,
)

# Every pattern that offers a color input, and the color parameters it declares.
COLOR_PATTERNS = [p for p in P.all_patterns() if any(q.type == "color" for q in p.parameters)]


def color_names(pattern):
    """Base names of the color inputs, e.g. ``color`` or ``color_a``/``color_b``."""
    return [q.name for q in pattern.parameters if q.type == "color" and "_code" not in q.name]


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_code_mode_declares_all_inputs(pattern):
    names = {q.name for q in pattern.parameters}
    assert VALUE_MODE in names
    for base in color_names(pattern):
        assert f"{base}_code" in names
        assert f"{base}_code_10" in names


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_linear_mode_is_the_default(pattern):
    """The mode is additive: existing behaviour must be untouched by default."""
    assert pattern.resolve({})[VALUE_MODE] == LINEAR_MODE


@pytest.mark.parametrize("pattern", COLOR_PATTERNS, ids=lambda p: p.id)
def test_code_value_ranges(pattern):
    for q in pattern.parameters:
        if q.type != "color":
            continue
        if q.name.endswith("_code_10"):
            assert (q.minimum, q.maximum) == (0.0, CODE_10BIT_MAX)
            assert q.step == 1.0
        elif q.name.endswith("_code"):
            assert (q.minimum, q.maximum) == (0.0, CODE_12BIT_MAX)
            assert q.step == 1.0


@pytest.mark.parametrize("code", [0, 1, 940, 2048, 4095])
def test_entered_12bit_code_is_written_verbatim(code):
    """A patch authored at 12-bit code N must come out at N/4095."""
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


@pytest.mark.parametrize("code", [0, 1, 64, 512, 940, 1023])
def test_entered_10bit_code_is_written_verbatim(code):
    """A patch authored at 10-bit code N must come out at N/1023."""
    pattern = P.get_pattern("solid-color")
    result = pattern.render(
        8,
        4,
        {VALUE_MODE: CODE_10BIT_MODE, "color_code_10": [code, code, code]},
    )
    assert np.allclose(result.image[0, 0], code / CODE_10BIT_MAX, atol=1e-6)


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
    """The greyed-out controls must genuinely have no effect."""
    bases = color_names(pattern)
    linear = {name: [0.25, 0.5, 0.75] for name in bases}
    codes12 = {f"{name}_code": [3000, 2000, 1000] for name in bases}
    codes10 = {f"{name}_code_10": [800, 400, 200] for name in bases}

    in_linear = pattern.render(32, 8, {VALUE_MODE: LINEAR_MODE, **linear, **codes12, **codes10})
    linear_only = pattern.render(32, 8, {VALUE_MODE: LINEAR_MODE, **linear})
    assert np.array_equal(in_linear.image, linear_only.image)

    in_12 = pattern.render(32, 8, {VALUE_MODE: CODE_12BIT_MODE, **linear, **codes12, **codes10})
    only_12 = pattern.render(32, 8, {VALUE_MODE: CODE_12BIT_MODE, **codes12})
    assert np.array_equal(in_12.image, only_12.image)

    in_10 = pattern.render(32, 8, {VALUE_MODE: CODE_10BIT_MODE, **linear, **codes12, **codes10})
    only_10 = pattern.render(32, 8, {VALUE_MODE: CODE_10BIT_MODE, **codes10})
    assert np.array_equal(in_10.image, only_10.image)


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


@pytest.mark.parametrize(
    "mode, params",
    [
        (CODE_12BIT_MODE, {"color_code": [4095, 4095, 4095]}),
        (CODE_10BIT_MODE, {"color_code_10": [1023, 1023, 1023]}),
    ],
)
def test_legal_range_still_applies_to_code_values(mode, params):
    """Legal range is a signal-level remap and stays available in code-value mode."""
    pattern = P.get_pattern("solid-color")
    result = pattern.render(4, 2, {VALUE_MODE: mode, **params, "signal_range": "legal"})
    assert result.image[0, 0, 0] == pytest.approx(940 / 1023, abs=1e-4)
    assert result.signal_format.range.value == "legal"


@pytest.mark.parametrize("mode", CODE_MODES)
def test_encoding_controls_are_declared_disabled_in_code_mode(mode):
    pattern = P.get_pattern("solid-color")
    names = {q.name: q for q in pattern.parameters}
    for name in ("color_space", "transfer_function", "peak_luminance"):
        conds = names[name].disabled_conditions()
        assert any(c.parameter == VALUE_MODE and mode in c.values for c in conds), name
    range_conds = names["signal_range"].disabled_conditions()
    assert not any(c.parameter == VALUE_MODE for c in range_conds)


def test_encoding_controls_are_ignored_in_code_mode():
    """Greyed-out encoding knobs must not change a pixel; range still can."""
    pattern = P.get_pattern("solid-color")
    base = {VALUE_MODE: CODE_12BIT_MODE, "color_code": [2048, 1024, 512]}
    a = pattern.render(
        8,
        4,
        {**base, "transfer_function": "pq", "peak_luminance": 1000.0, "color_space": "rec2020"},
    )
    b = pattern.render(
        8,
        4,
        {
            **base,
            "transfer_function": "gamma-2.4",
            "peak_luminance": 100.0,
            "color_space": "rec709",
        },
    )
    assert np.array_equal(a.image, b.image)
    assert a.signal_format.peak_luminance is None
    legal = pattern.render(8, 4, {**base, "signal_range": "legal"})
    assert not np.array_equal(a.image, legal.image)


def test_code_values_clamp_to_10bit_full_scale():
    pattern = P.get_pattern("solid-color")
    assert pattern.resolve({"color_code_10": [2000, -5, 100]})["color_code_10"] == [
        1023.0,
        0.0,
        100.0,
    ]
