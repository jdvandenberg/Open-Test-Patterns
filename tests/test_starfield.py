"""Starfield: a lattice of single non-black pixels on a black field."""

import numpy as np
import pytest

import open_test_patterns.patterns as P


def _render(width, height, **params):
    params.setdefault("transfer_function", "linear")
    return P.get_pattern("starfield").render(width, height, params).image


def test_starfield_is_in_frequency():
    p = P.get_pattern("starfield")
    assert p.name == "Starfield"
    assert p.category == "Frequency"


def test_spacing_two_lights_every_other_pixel():
    img = _render(8, 6, fill="solid", color=[1.0, 1.0, 1.0], spacing=2)
    stars = np.zeros((6, 8), dtype=bool)
    stars[0::2, 0::2] = True
    assert np.array_equal(img[..., 0] > 0, stars)
    assert np.all(img[stars] == 1.0)
    assert np.all(img[~stars] == 0.0)


def test_spacing_three_is_one_pixel_per_3x3_cell():
    img = _render(9, 9, fill="solid", color=[1.0, 1.0, 1.0], spacing=3)
    lit = np.argwhere(img[..., 0] > 0)
    assert {tuple(p) for p in lit} == {(r, c) for r in (0, 3, 6) for c in (0, 3, 6)}


def test_rgbw_quadrants_colour_the_stars():
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
