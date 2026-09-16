"""Pixel Grid: a lattice of single non-black pixels on a black field."""

import struct

import numpy as np
from fastapi.testclient import TestClient

import open_test_patterns.patterns as P
from open_test_patterns.api.app import app

client = TestClient(app)


def _render(width, height, **params):
    params.setdefault("transfer_function", "linear")
    return P.get_pattern("pixel-grid").render(width, height, params).image


def test_pixel_grid_is_in_frequency():
    p = P.get_pattern("pixel-grid")
    assert p.id == "pixel-grid"
    assert p.name == "Pixel Grid"
    assert p.category == "Frequency"


def test_spacing_two_lights_every_other_pixel():
    img = _render(8, 6, fill="solid", color=[1.0, 1.0, 1.0], spacing=2)
    lit = np.zeros((6, 8), dtype=bool)
    lit[0::2, 0::2] = True
    assert np.array_equal(img[..., 0] > 0, lit)
    assert np.all(img[lit] == 1.0)
    assert np.all(img[~lit] == 0.0)


def test_spacing_three_is_one_pixel_per_3x3_cell():
    img = _render(9, 9, fill="solid", color=[1.0, 1.0, 1.0], spacing=3)
    lit = np.argwhere(img[..., 0] > 0)
    assert {tuple(p) for p in lit} == {(r, c) for r in (0, 3, 6) for c in (0, 3, 6)}


def test_rgbw_quadrants_colour_the_pixels():
    img = _render(8, 8, fill="rgbw", spacing=2, color=[0.2, 0.3, 0.4])
    assert np.allclose(img[0, 0], [1, 0, 0])
    assert np.allclose(img[0, 4], [0, 1, 0])
    assert np.allclose(img[4, 0], [0, 0, 1])
    assert np.allclose(img[4, 4], [1, 1, 1])
    # The unused solid colour must not leak into RGBW mode.
    assert np.allclose(img[2, 2], [1, 0, 0])
    assert np.all(img[1, 0] == 0)


def test_solid_color_is_used_when_selected():
    img = _render(6, 6, fill="solid", color=[0.25, 0.5, 0.75], spacing=2)
    assert np.allclose(img[0, 0], [0.25, 0.5, 0.75])
    assert np.all(img[0, 1] == 0)


def test_preview_pixel_grid_is_not_filtered():
    """Adjacent black cells must stay black in the PNG, not smeared grey."""
    import os
    import tempfile

    from OpenImageIO import ImageBuf

    r = client.post(
        "/api/preview",
        json={
            "pattern_id": "pixel-grid",
            "width": 8,
            "height": 6,
            "params": {
                "fill": "solid",
                "color": [1.0, 1.0, 1.0],
                "spacing": 2,
                "transfer_function": "linear",
            },
        },
    )
    assert r.status_code == 200
    fd, path = tempfile.mkstemp(suffix=".png")
    os.write(fd, r.content)
    os.close(fd)
    try:
        buf = ImageBuf(path)
        lit = buf.getpixel(0, 0)
        dark = buf.getpixel(1, 0)
        assert lit[0] > 0.95
        assert dark[0] < 0.02
    finally:
        os.remove(path)
    """A filtered downscale would smear the lattice; the PNG must stay 1:1."""
    r = client.post(
        "/api/preview",
        json={
            "pattern_id": "pixel-grid",
            "width": 1920,
            "height": 1080,
            "params": {
                "fill": "solid",
                "color": [1.0, 1.0, 1.0],
                "spacing": 2,
                "transfer_function": "linear",
            },
        },
    )
    assert r.status_code == 200
    width, height = struct.unpack(">II", r.content[16:24])
    assert (width, height) == (1920, 1080)
