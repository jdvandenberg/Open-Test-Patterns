import numpy as np
import pytest

import open_test_patterns.patterns as P
from open_test_patterns.patterns.base import SignalRange


@pytest.mark.parametrize("pattern", P.all_patterns(), ids=lambda p: p.id)
def test_pattern_generates_valid_image(pattern):
    result = pattern.render(160, 90, {})
    img = result.image
    assert img.shape == (90, 160, 3)
    assert img.dtype == np.float32
    assert np.all(np.isfinite(img))
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_defaults_are_coerced():
    pattern = P.get_pattern("gray-steps")
    resolved = pattern.resolve({})
    assert resolved["steps"] == 11


def test_color_components_are_clamped_to_unit_range():
    pattern = P.get_pattern("solid-color")
    assert pattern.resolve({"color": [2.5, -1.0, 0.5]})["color"] == [1.0, 0.0, 0.5]


def test_clamped_color_renders_the_reported_value():
    """The clamp must hold for a transfer function that would otherwise pass
    out-of-range values straight through, so the UI can echo what it renders."""
    pattern = P.get_pattern("solid-color")
    result = pattern.render(8, 4, {"color": [2.5, -1.0, 0.5], "transfer_function": "linear"})
    assert np.allclose(result.image[0, 0], [1.0, 0.0, 0.5])


@pytest.mark.parametrize(
    "pattern",
    [p for p in P.all_patterns() if any(q.type == "color" for q in p.parameters)],
    ids=lambda p: p.id,
)
def test_every_color_parameter_clamps(pattern):
    """Each color control clamps to its own declared range, which differs between
    the linear ([0, 1]) and 12-bit code value ([0, 4095]) inputs."""
    color_params = [q for q in pattern.parameters if q.type == "color"]
    over = 1e6
    resolved = pattern.resolve({q.name: [over, -over, 0.25] for q in color_params})
    for q in color_params:
        expected_max = 1.0 if q.maximum is None else q.maximum
        assert resolved[q.name] == [expected_max, 0.0, 0.25]


def test_legal_range_applies():
    pattern = P.get_pattern("color-bars")
    result = pattern.render(64, 36, {"signal_range": "legal"})
    assert result.signal_format.range is SignalRange.LEGAL
    assert result.image.min() >= 64 / 1023 - 1e-6
