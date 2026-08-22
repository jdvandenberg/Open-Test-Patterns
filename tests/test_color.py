import numpy as np
import pytest

from open_test_patterns.color import colorspaces, transfer


def test_pq_roundtrip():
    nits = np.array([0.0, 1.0, 100.0, 1000.0, 10000.0])
    code = transfer.encode(nits, "pq")
    assert np.all((code >= 0) & (code <= 1))
    back = transfer.decode(code, "pq")
    assert np.allclose(back, nits, atol=1e-3)


def test_legal_full_roundtrip():
    values = np.linspace(0, 1, 11)
    legal = transfer.full_to_legal(values)
    assert legal.min() >= 64 / 1023 - 1e-9
    assert legal.max() <= 940 / 1023 + 1e-9
    assert np.allclose(transfer.legal_to_full(legal), values, atol=1e-6)


def test_colorspaces_available():
    ids = {cs.id for cs in colorspaces.list_colorspaces()}
    assert {"rec709", "rec2020", "p3-d65"}.issubset(ids)


def test_aces_log_anchor_values():
    """ACEScc and ACEScct agree above the toe and both put linear 1.0 at 0.5548."""
    for tf_id in ("acescc", "acescct"):
        assert transfer.encode(0.18, tf_id) == pytest.approx(0.413588, abs=1e-6)
        assert transfer.encode(1.0, tf_id) == pytest.approx(0.554795, abs=1e-6)


def test_acescct_toe_preserves_black():
    """ACEScct's linear toe keeps black positive, so it survives the [0, 1] clamp."""
    assert transfer.encode(0.0, "acescct") == pytest.approx(0.072906, abs=1e-6)
    assert transfer.decode(0.072906, "acescct") == pytest.approx(0.0, abs=1e-6)


def test_acescc_black_clips_to_zero():
    """ACEScc puts linear 0 at -0.3584, which this pipeline clamps to 0.

    Documented rather than silently accepted: ACEScc's sub-zero black cannot be
    represented while pattern code values are constrained to [0, 1].
    """
    assert transfer.encode(0.0, "acescc") == 0.0
    # The unclamped curve really is negative there.
    import colour

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = float(colour.models.log_encoding_ACEScc(0.0))
    assert raw == pytest.approx(-0.358447, abs=1e-6)


def test_aces_log_round_trips_midtones():
    values = np.linspace(0.01, 1.0, 64)
    for tf_id in ("acescc", "acescct"):
        code = transfer.encode(values, tf_id)
        assert np.all((code >= 0.0) & (code <= 1.0))
        assert np.allclose(transfer.decode(code, tf_id), values, atol=1e-6)


def test_aces_log_differs_from_linear_and_each_other():
    assert transfer.encode(0.5, "acescc") != pytest.approx(0.5, abs=1e-3)
    # Identical above the toe, different below it.
    assert transfer.encode(0.5, "acescc") == pytest.approx(
        transfer.encode(0.5, "acescct"), abs=1e-9
    )
    assert transfer.encode(0.001, "acescc") != pytest.approx(
        transfer.encode(0.001, "acescct"), abs=1e-3
    )


def test_rgb_to_rgb_identity():
    rgb = np.array([0.2, 0.5, 0.8])
    out = colorspaces.RGB_to_RGB(rgb, "rec709", "rec709")
    assert np.allclose(out, rgb, atol=1e-6)
