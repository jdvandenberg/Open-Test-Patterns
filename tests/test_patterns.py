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


def test_legal_range_applies():
    pattern = P.get_pattern("color-bars")
    result = pattern.render(64, 36, {"signal_range": "legal"})
    assert result.signal_format.range is SignalRange.LEGAL
    assert result.image.min() >= 64 / 1023 - 1e-6
