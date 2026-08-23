"""Center Cross: one H and one V line, two pixels wide when the axis is even."""

import numpy as np
import pytest

import open_test_patterns.patterns as P


def _line_columns(img: np.ndarray) -> list[int]:
    """Columns that are entirely at the line level (the vertical stroke)."""
    # A vertical line paints a whole column; the horizontal stroke also paints
    # every column on its row(s), so look for columns that are *all* lit.
    return [c for c in range(img.shape[1]) if np.all(img[:, c, 0] > 0)]


def _line_rows(img: np.ndarray) -> list[int]:
    return [r for r in range(img.shape[0]) if np.all(img[r, :, 0] > 0)]


@pytest.mark.parametrize(
    "width, height, v_cols, h_rows",
    [
        (5, 5, [2], [2]),
        (6, 6, [2, 3], [2, 3]),
        (6, 5, [2, 3], [2]),
        (5, 6, [2], [2, 3]),
        (1920, 1080, [959, 960], [539, 540]),
        (1921, 1081, [960], [540]),
    ],
)
def test_cross_is_exactly_centered(width, height, v_cols, h_rows):
    pattern = P.get_pattern("center-cross")
    img = pattern.render(width, height, {"line_level": 1.0, "transfer_function": "linear"}).image
    assert _line_columns(img) == v_cols
    assert _line_rows(img) == h_rows
    # The rest of the field is black, except the two strokes.
    expected = np.zeros((height, width, 3), dtype=np.float32)
    expected[:, v_cols, :] = 1.0
    expected[h_rows, :, :] = 1.0
    assert np.array_equal(img, expected)


def test_center_cross_is_in_geometry():
    pattern = P.get_pattern("center-cross")
    assert pattern.name == "Center Cross"
    assert pattern.category == "Geometry"
