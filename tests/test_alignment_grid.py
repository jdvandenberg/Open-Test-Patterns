"""Alignment Grid: optional centred circle sized from frame height."""

import numpy as np

import open_test_patterns.patterns as P


def _render(width, height, **params):
    params.setdefault("line_level", 1.0)
    params.setdefault("transfer_function", "linear")
    return P.get_pattern("grid").render(width, height, params).image


def test_alignment_grid_is_in_geometry():
    p = P.get_pattern("grid")
    assert p.name == "Alignment Grid"
    assert p.category == "Geometry"
    circle = next(q for q in p.parameters if q.name == "circle")
    assert circle.default is True
    diameter = next(q for q in p.parameters if q.name == "circle_diameter")
    assert diameter.default == 75.0
    assert diameter.disabled_when.values == ("false",)


def test_circle_is_centered_at_three_quarters_height():
    width, height = 200, 100
    img = _render(width, height, columns=1, rows=1, circle=True, circle_diameter=75)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius = 0.75 * height / 2.0
    # A pixel on the right of the circle, away from the 1×1 grid border.
    x = int(round(cx + radius))
    y = int(round(cy))
    assert img[y, x, 0] == 1.0
    # Inside the circle, off the axes, should stay black.
    assert img[int(cy), int(cx), 0] == 0.0


def test_circle_can_be_turned_off():
    img = _render(64, 48, columns=1, rows=1, circle=False)
    # Only the 1×1 grid (the frame) is lit; the interior is black.
    interior = img[2:-2, 2:-2, 0]
    assert np.all(interior == 0.0)
