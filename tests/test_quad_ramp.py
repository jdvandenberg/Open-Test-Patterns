"""Quad Ramp: four stacked or side-by-side color ramps."""

import numpy as np

import open_test_patterns.patterns as P


def _render(width, height, **params):
    params.setdefault("transfer_function", "linear")
    return P.get_pattern("quad-ramp").render(width, height, params).image


def test_quad_ramp_is_in_ramps():
    p = P.get_pattern("quad-ramp")
    assert p.id == "quad-ramp"
    assert p.name == "Quad Ramp"
    assert p.category == "Ramps & Steps"


def test_default_is_horizontal_rgbw():
    img = _render(8, 4)
    assert np.allclose(img[:, 0], 0.0)
    assert np.allclose(img[0, -1], [1.0, 0.0, 0.0])
    assert np.allclose(img[1, -1], [0.0, 1.0, 0.0])
    assert np.allclose(img[2, -1], [0.0, 0.0, 1.0])
    assert np.allclose(img[3, -1], [1.0, 1.0, 1.0])
    # Midpoint of the red row is half red.
    assert np.allclose(img[0, 4], [4 / 7, 0.0, 0.0], atol=1e-6)


def test_vertical_places_bands_left_to_right():
    img = _render(4, 8, direction="vertical")
    assert np.allclose(img[0, :], 0.0)
    assert np.allclose(img[-1, 0], [1.0, 0.0, 0.0])
    assert np.allclose(img[-1, 1], [0.0, 1.0, 0.0])
    assert np.allclose(img[-1, 2], [0.0, 0.0, 1.0])
    assert np.allclose(img[-1, 3], [1.0, 1.0, 1.0])


def test_custom_end_colors():
    img = _render(
        4,
        4,
        ramp1_end=[0.2, 0.0, 0.0],
        ramp2_end=[0.0, 0.4, 0.0],
        ramp3_end=[0.0, 0.0, 0.6],
        ramp4_end=[0.8, 0.8, 0.8],
    )
    assert np.allclose(img[0, -1], [0.2, 0.0, 0.0])
    assert np.allclose(img[1, -1], [0.0, 0.4, 0.0])
    assert np.allclose(img[2, -1], [0.0, 0.0, 0.6])
    assert np.allclose(img[3, -1], [0.8, 0.8, 0.8])
