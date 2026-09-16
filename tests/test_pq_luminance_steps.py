"""PQ Luminance Steps: nits labels at the foot of each column when they fit."""

import numpy as np
import pytest

import open_test_patterns.patterns as P
from open_test_patterns.color import transfer
from open_test_patterns.imageio.preview import to_display_srgb
from open_test_patterns.patterns.steps import LABEL_NITS, format_pq_step_label


def _render_result(width, height, **params):
    params.setdefault("transfer_function", "pq")
    return P.get_pattern("pq-luminance-steps").render(width, height, params)


def _render(width, height, **params):
    return _render_result(width, height, **params).image


def test_label_text_matches_the_nits_value():
    assert format_pq_step_label(5) == "5 cd/m²"
    assert format_pq_step_label(5.0) == "5 cd/m²"
    assert format_pq_step_label(100) == "100 cd/m²"
    assert format_pq_step_label(2.5) == "2.5 cd/m²"


def test_labels_are_drawn_when_columns_are_wide_enough():
    labeled = _render(1920, 1080, steps=11, first_luminance=5, increment=5)
    # 11 steps on 1920 is ~174 px/column; a black box + light type at the foot.
    # Sample the mid-frame so the top-left title does not interfere.
    mid = labeled[400:500, 80, 0]
    bottom = labeled[-40:, 40:140, 0]
    assert np.allclose(mid, mid[0], atol=1e-4)
    assert float(bottom.min()) < 0.05
    assert float(bottom.max()) > 0.4


def test_default_twenty_steps_still_carry_labels():
    labeled = _render(1920, 1080, steps=20, first_luminance=5, increment=5)
    bottom = labeled[-50:, 10:90, 0]
    assert float(bottom.min()) < 0.05
    assert float(bottom.max()) > 0.4


def test_labels_are_omitted_when_steps_make_columns_too_narrow():
    img = _render(1920, 1080, steps=64, first_luminance=5, increment=5)
    # Column labels must not appear; the title at the top-left still may.
    col = img[400:700, 20, 0]
    assert np.allclose(col, col[0], atol=1e-5)


def test_labels_are_omitted_when_the_frame_is_too_small():
    img = _render(160, 90, steps=20)
    col = img[:, 4, 0]
    assert np.allclose(col, col[0], atol=1e-5)


def test_preview_maps_the_brightest_step_near_white_without_clipping():
    result = _render_result(1920, 1080, steps=20, first_luminance=5, increment=5)
    assert result.signal_format.peak_luminance == 100.0
    disp = to_display_srgb(result)
    left = disp[100, 40]
    right = disp[100, 1880]
    mid = disp[100, 960]
    assert int(left.max()) < 90
    assert int(right.min()) > 200
    assert int(right.max()) < 255
    assert int(mid.mean()) > int(left.mean())
    assert int(right.mean()) > int(mid.mean())
    # Black boxes at the foot of both ends, with 50 cd/m² type inside.
    assert int(disp[-20:, 20:80].min()) < 20
    assert int(disp[-20:, 20:80].max()) > 150
    assert int(disp[-20:, 1840:1900].min()) < 20
    assert int(disp[-20:, 1840:1900].max()) > 150


def test_title_is_drawn_at_the_top_left():
    img = _render(1920, 1080, steps=20, first_luminance=5, increment=5)
    top_left = img[4:48, 8:420, 0]
    nits = transfer.decode(top_left, "pq")
    assert float(nits.min()) < 0.1
    assert float(nits.max()) == pytest.approx(LABEL_NITS, rel=0.08)


def test_label_type_is_fifty_nits_on_a_black_box():
    img = _render(1920, 1080, steps=11, first_luminance=5, increment=5)
    bottom = img[-50:, 40:140, 0]
    nits = transfer.decode(bottom, "pq")
    assert float(nits.min()) < 0.1
    assert float(nits.max()) == pytest.approx(LABEL_NITS, rel=0.08)
