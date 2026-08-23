"""Exercise every transfer function against every pattern.

The HLG encoder was broken for a long time because the suite only ever rendered
patterns with their default transfer function, so no test touched HLG. These
tests walk the registry instead of naming curves, so a newly added transfer
function is covered automatically.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import open_test_patterns.patterns as P
from open_test_patterns.api.app import app
from open_test_patterns.color import transfer
from open_test_patterns.imageio import to_display_srgb

client = TestClient(app)

TRANSFER_IDS = [tf.id for tf in transfer.list_transfer_functions()]
PATTERNS_WITH_TRANSFER = [
    p for p in P.all_patterns() if any(q.name == "transfer_function" for q in p.parameters)
]


def test_registry_is_not_empty():
    assert len(TRANSFER_IDS) >= 8
    assert PATTERNS_WITH_TRANSFER


@pytest.mark.parametrize("tf_id", TRANSFER_IDS)
def test_encode_decode_are_callable(tf_id):
    """Guards against renamed colour-science functions, which fail at call time."""
    values = np.linspace(0.0, 1.0, 16)
    code = transfer.encode(values, tf_id)
    assert np.all(np.isfinite(code)), f"{tf_id} encode produced non-finite values"
    back = transfer.decode(code, tf_id)
    assert np.all(np.isfinite(back)), f"{tf_id} decode produced non-finite values"


@pytest.mark.parametrize("tf_id", TRANSFER_IDS)
def test_encode_is_monotonic(tf_id):
    """Every curve must preserve ordering, else it is not a transfer function."""
    if tf_id == "pq":
        pytest.skip("PQ input is absolute luminance, covered separately")
    values = np.linspace(0.0, 1.0, 64)
    code = transfer.encode(values, tf_id)
    assert np.all(np.diff(code) >= -1e-9), f"{tf_id} is not monotonic"


def test_hlg_round_trips():
    """Regression test for the BT.2100 HLG naming bug."""
    values = np.linspace(0.0, 1.0, 32)
    code = transfer.encode(values, "hlg")
    assert code.min() >= 0.0 and code.max() <= 1.0
    # HLG reaches peak code at scene linear 1.0.
    assert code[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(transfer.decode(code, "hlg"), values, atol=1e-6)


@pytest.mark.parametrize("tf_id", TRANSFER_IDS)
@pytest.mark.parametrize("pattern", PATTERNS_WITH_TRANSFER, ids=lambda p: p.id)
def test_pattern_renders_and_previews(pattern, tf_id):
    result = pattern.render(48, 32, {"transfer_function": tf_id})
    img = result.image
    assert img.shape == (32, 48, 3)
    assert np.all(np.isfinite(img)), f"{pattern.id} + {tf_id} produced non-finite values"
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert result.signal_format.transfer_function == tf_id
    assert to_display_srgb(result).shape == (32, 48, 3)


@pytest.mark.parametrize("tf_id", TRANSFER_IDS)
def test_preview_endpoint_accepts_every_transfer(tf_id):
    r = client.post(
        "/api/preview",
        json={
            "pattern_id": "gradient",
            "width": 64,
            "height": 32,
            "params": {"transfer_function": tf_id},
        },
    )
    assert r.status_code == 200, r.text
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_peak_luminance_is_disabled_for_relative_transfers():
    param = next(
        q for q in P.get_pattern("gradient").parameters if q.name == "peak_luminance"
    )
    conds = param.disabled_conditions()
    dep = next(c for c in conds if c.parameter == "transfer_function")
    # Disabled for every relative curve, enabled only for the absolute ones.
    absolute = {tf.id for tf in transfer.list_transfer_functions() if tf.is_absolute}
    relative = {tf.id for tf in transfer.list_transfer_functions() if not tf.is_absolute}
    assert set(dep.values) == relative
    assert absolute.isdisjoint(dep.values)
    assert "pq" in absolute


def test_peak_luminance_only_affects_absolute_transfers():
    """The greyed-out control must genuinely have no effect."""
    pattern = P.get_pattern("gradient")

    def render(tf_id, peak):
        return pattern.render(
            33, 4, {"transfer_function": tf_id, "peak_luminance": peak}
        )

    for tf_id in TRANSFER_IDS:
        low = render(tf_id, 100.0)
        high = render(tf_id, 1000.0)
        is_absolute = transfer.get_transfer_function(tf_id).is_absolute
        if is_absolute:
            assert not np.array_equal(low.image, high.image), f"{tf_id} ignored peak"
            assert high.signal_format.peak_luminance == 1000.0
        else:
            assert np.array_equal(low.image, high.image), f"{tf_id} was changed by peak"
            assert low.signal_format.peak_luminance is None
