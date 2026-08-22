import numpy as np

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


def test_rgb_to_rgb_identity():
    rgb = np.array([0.2, 0.5, 0.8])
    out = colorspaces.RGB_to_RGB(rgb, "rec709", "rec709")
    assert np.allclose(out, rgb, atol=1e-6)
