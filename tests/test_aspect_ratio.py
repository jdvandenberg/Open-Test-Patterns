"""Aspect Ratio geometry chart."""

import numpy as np

import open_test_patterns.patterns as P
from open_test_patterns.patterns.aspect_ratio import (
    CONTAINERS,
    FRAMES,
    RATIO_CHOICES,
    _arrow_height,
    _arrow_scale,
    aspect_frame,
)


def _render(width=None, height=None, **params):
    params.setdefault("transfer_function", "linear")
    params.setdefault("circle", False)
    container = params.get("container", "1080p")
    cw, ch = CONTAINERS[container]
    w = width if width is not None else cw
    h = height if height is not None else ch
    return P.get_pattern("aspect-ratio").render(w, h, params).image


def test_aspect_ratio_is_in_geometry():
    p = P.get_pattern("aspect-ratio")
    assert p.name == "Aspect Ratio"
    assert p.category == "Geometry"
    names = [q.name for q in p.parameters]
    assert names[:2] == ["container", "ratio"]
    circle = next(q for q in p.parameters if q.name == "circle")
    assert circle.default is True
    diameter = next(q for q in p.parameters if q.name == "circle_diameter")
    assert diameter.default == 75.0


def test_frame_table_covers_every_container_and_ratio():
    ratios = [c.value for c in RATIO_CHOICES]
    assert set(FRAMES) == set(CONTAINERS)
    for container, size in CONTAINERS.items():
        assert set(FRAMES[container]) == set(ratios)
        for ratio, (iw, ih) in FRAMES[container].items():
            cw, ch = size
            assert 1 <= iw <= cw
            assert 1 <= ih <= ch
            assert (iw == cw) or (ih == ch)
            assert aspect_frame(container, ratio) == (iw, ih)


def test_1080p_16x9_fills_the_container():
    img = _render(container="1080p", ratio="1.78")
    assert img.shape == (1080, 1920, 3)
    # Exact match: no pillar/letter box. Interior is mid-grey, not black.
    assert np.allclose(img[10, 10], [0.5, 0.5, 0.5])


def test_1080p_4x3_is_pillarboxed_black():
    img = _render(container="1080p", ratio="1.33", outside="black")
    x0 = (1920 - 1440) // 2
    assert np.allclose(img[540, 0], 0.0)
    assert np.allclose(img[540, x0 - 1], 0.0)
    # Interior of the 4:3 window, off the diagonals.
    assert np.allclose(img[100, x0 + 200], [0.5, 0.5, 0.5])


def test_1080p_240_is_letterboxed():
    img = _render(container="1080p", ratio="2.40", outside="black")
    y0 = (1080 - 800) // 2
    assert np.allclose(img[0, 960], 0.0)
    assert np.allclose(img[y0 - 1, 960], 0.0)
    assert np.allclose(img[y0 + 40, 200], [0.5, 0.5, 0.5])


def test_outside_can_be_red():
    img = _render(container="1080p", ratio="1.33", outside="red")
    assert np.allclose(img[540, 0], [1.0, 0.0, 0.0])


def test_dci_scope_exact_match():
    assert aspect_frame("dci-2k-scope", "2.39") == (2048, 858)
    img = _render(container="dci-2k-scope", ratio="2.39")
    assert img.shape == (858, 2048, 3)


def test_circle_uses_active_picture_height():
    img = _render(container="1080p", ratio="2.40", circle=True, circle_diameter=75, line_level=1)
    y0 = (1080 - 800) // 2
    cx = (1920 - 1) / 2.0
    cy = (y0 + (y0 + 800) - 1) / 2.0
    radius = 0.75 * 800 / 2.0
    # Integer pixel on the right of the 1px ring (round() overshoots the stroke).
    x = int(cx + radius)
    y = int(round(cy))
    assert img[y, x, 0] == 1.0
    # Inside the circle, off the diagonals, stays grey.
    assert np.allclose(img[y - 80, int(round(cx))], [0.5, 0.5, 0.5])


def test_inner_corners_are_on_the_diagonals():
    img = _render(container="1080p", ratio="1.33", line_level=1)
    x0 = (1920 - 1440) // 2
    x1 = x0 + 1440
    assert img[0, x0, 0] == 1.0
    assert img[0, x1 - 1, 0] == 1.0
    assert img[1079, x0, 0] == 1.0
    assert img[1079, x1 - 1, 0] == 1.0


def test_third_arrows_have_a_one_pixel_tip_on_the_border():
    img = _render(container="1080p", ratio="1.78", line_level=1)
    x_one, x_two = 640, 1280
    y_one, y_two = 360, 720
    # Tips sit in the 1px border at 1/3 and 2/3 of each inner edge.
    for x in (x_one, x_two):
        assert img[0, x, 0] == 1.0
        assert img[1079, x, 0] == 1.0
        # Altitude is 15px including the tip; the last row of the triangle
        # is still white, the next pixel is grey (off the diagonals).
        assert img[14, x, 0] == 1.0
        assert np.allclose(img[15, x], [0.5, 0.5, 0.5])
        assert img[1079 - 14, x, 0] == 1.0
        assert np.allclose(img[1079 - 15, x], [0.5, 0.5, 0.5])
    for y in (y_one, y_two):
        assert img[y, 0, 0] == 1.0
        assert img[y, 1919, 0] == 1.0
        assert img[y, 14, 0] == 1.0
        assert np.allclose(img[y, 15], [0.5, 0.5, 0.5])
        assert img[y, 1919 - 14, 0] == 1.0
        assert np.allclose(img[y, 1919 - 15], [0.5, 0.5, 0.5])
    # The tip is one pixel: just inside the border, neighbours of the tip
    # along the edge-normal are the triangle, but one pixel beside the tip
    # on the first inward row is still grey.
    assert np.allclose(img[1, x_one - 2], [0.5, 0.5, 0.5])
    assert np.allclose(img[y_one - 2, 1], [0.5, 0.5, 0.5])


def test_arrow_scale_follows_container_tier():
    assert _arrow_scale("1080p") == 1
    assert _arrow_scale("dci-2k-full") == 1
    assert _arrow_scale("dci-2k-flat") == 1
    assert _arrow_scale("dci-2k-scope") == 1
    assert _arrow_scale("uhd-4k") == 2
    assert _arrow_scale("dci-4k-full") == 2
    assert _arrow_scale("dci-4k-flat") == 2
    assert _arrow_scale("dci-4k-scope") == 2
    assert _arrow_scale("uhd-8k") == 4
    assert _arrow_scale("dci-8k") == 4
    assert _arrow_height(1920, 1920, scale=1) == 15
    assert _arrow_height(3840, 3840, scale=2) == 30
    assert _arrow_height(7680, 7680, scale=4) == 60
    # 8K preview is generated at 2048 on the long side.
    assert _arrow_height(2048, 8192, scale=4) == 15


def test_third_arrows_are_thirty_pixels_at_4k():
    img = _render(container="uhd-4k", ratio="1.78", line_level=1)
    x = round(3840 / 3)
    assert img[29, x, 0] == 1.0
    assert np.allclose(img[30, x], [0.5, 0.5, 0.5])


def test_third_arrows_stay_inside_the_active_picture():
    img = _render(container="1080p", ratio="1.33", outside="black", line_level=1)
    x0 = (1920 - 1440) // 2
    y_one = 360
    assert img[y_one, x0, 0] == 1.0
    assert np.allclose(img[y_one, x0 - 1], 0.0)


def test_catalog_lists_aspect_ratio_under_geometry():
    from fastapi.testclient import TestClient

    from open_test_patterns.api.app import app

    data = TestClient(app).get("/api/patterns").json()
    item = next(p for p in data if p["id"] == "aspect-ratio")
    assert item["name"] == "Aspect Ratio"
    assert item["category"] == "Geometry"
    names = [q["name"] for q in item["parameters"]]
    assert names[:2] == ["container", "ratio"]
