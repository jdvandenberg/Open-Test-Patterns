import numpy as np
import pytest
from fastapi.testclient import TestClient

import open_test_patterns.patterns as P
from open_test_patterns.api.app import app
from open_test_patterns.color import transfer
from open_test_patterns.imageio import to_display_srgb
from open_test_patterns.patterns.base import PatternResult, SignalFormat, SignalRange

client = TestClient(app)


def test_bypass_is_registered_and_flagged():
    tf = transfer.get_transfer_function("bypass")
    assert tf.name == "Bypass (no processing)"
    assert tf.is_bypass
    assert not tf.is_absolute


def test_bypass_appears_in_pattern_choices():
    pattern = P.get_pattern("gray-steps")
    param = next(p for p in pattern.parameters if p.name == "transfer_function")
    assert "bypass" in [c.value for c in param.choices]


def test_bypass_encode_and_decode_are_identity():
    values = np.array([0.0, 0.02, 0.18, 0.5, 0.9, 1.0])
    assert np.array_equal(transfer.encode(values, "bypass"), values)
    assert np.array_equal(transfer.decode(values, "bypass"), values)


def test_bypass_writes_authored_values_verbatim():
    """A bypass pattern's code values must survive generation untouched."""
    pattern = P.get_pattern("gray-steps")
    steps = 11
    result = pattern.render(
        220, 32, {"steps": steps, "transfer_function": "bypass", "signal_range": "full"}
    )
    # Sample the centre of each step; expect an even 0..1 spread, not a curve.
    row = result.image[16]
    sampled = [float(row[int((i + 0.5) * 220 / steps), 0]) for i in range(steps)]
    assert np.allclose(sampled, np.linspace(0.0, 1.0, steps), atol=1e-3)


def _result(code, tf_id, color_space="rec709", signal_range=SignalRange.FULL):
    return PatternResult(
        image=np.asarray(code, dtype=np.float32),
        signal_format=SignalFormat(
            color_space=color_space, transfer_function=tf_id, range=signal_range
        ),
    )


def test_preview_shows_bypass_values_verbatim():
    code = np.full((4, 4, 3), 0.5, dtype=np.float32)
    display = to_display_srgb(_result(code, "bypass"))
    # 0.5 code value -> 128/255, with no sRGB encoding applied on the way out.
    assert np.all(display == 128)


def test_bypass_differs_from_linear_in_preview():
    """Bypass must not be a silent duplicate of Linear.

    Linear treats the values as light and sRGB-encodes them for display, so mid
    gray lifts well above 128; bypass leaves it alone.
    """
    code = np.full((4, 4, 3), 0.5, dtype=np.float32)
    bypassed = to_display_srgb(_result(code, "bypass"))
    linear = to_display_srgb(_result(code, "linear"))
    assert int(bypassed[0, 0, 0]) == 128
    assert int(linear[0, 0, 0]) > 180
    assert not np.array_equal(bypassed, linear)


def test_bypass_ignores_color_space():
    """Bypass skips gamut conversion, so the primaries make no difference.

    A mid, in-gamut color is used deliberately: a saturated one would clip on
    conversion and coincidentally match the bypassed value.
    """
    code = np.array([[[0.2, 0.5, 0.3]]], dtype=np.float32)
    as_2020 = to_display_srgb(_result(code, "bypass", color_space="rec2020"))
    as_709 = to_display_srgb(_result(code, "bypass", color_space="rec709"))
    assert np.array_equal(as_2020, as_709)
    assert np.array_equal(as_2020[0, 0], np.array([51, 128, 77], dtype=np.uint8))
    # Linear, by contrast, does matrix Rec.2020 into sRGB.
    assert not np.array_equal(
        to_display_srgb(_result(code, "linear", color_space="rec2020")),
        to_display_srgb(_result(code, "linear", color_space="rec709")),
    )


def test_bypass_ignores_signal_range_in_preview():
    """Bypass applies no range scaling, so legal metadata must not expand values."""
    code = np.full((2, 2, 3), 940 / 1023, dtype=np.float32)
    bypassed = to_display_srgb(_result(code, "bypass", signal_range=SignalRange.LEGAL))
    assert np.all(bypassed == 234)  # 940/1023 shown verbatim, not stretched to 255
    # Under a real curve the same metadata does expand to peak white.
    expanded = to_display_srgb(_result(code, "linear", signal_range=SignalRange.LEGAL))
    assert np.all(expanded == 255)


def test_bypass_forces_full_range_on_generation():
    """A requested legal range is dropped, not silently applied behind the UI."""
    pattern = P.get_pattern("gray-steps")
    result = pattern.render(
        220, 16, {"steps": 11, "transfer_function": "bypass", "signal_range": "legal"}
    )
    assert result.signal_format.range is SignalRange.FULL
    assert result.image.min() == pytest.approx(0.0, abs=1e-6)
    assert result.image.max() == pytest.approx(1.0, abs=1e-6)


def test_non_bypass_still_applies_legal_range():
    """Guard against the bypass rule leaking into the other transfer functions."""
    pattern = P.get_pattern("gray-steps")
    result = pattern.render(
        220, 16, {"steps": 11, "transfer_function": "gamma-2.4", "signal_range": "legal"}
    )
    assert result.signal_format.range is SignalRange.LEGAL
    assert result.image.min() >= 64 / 1023 - 1e-6
    assert result.image.max() <= 940 / 1023 + 1e-6


def test_effective_range_resolution():
    from open_test_patterns.patterns.util import effective_range

    assert effective_range("bypass", "legal") is SignalRange.FULL
    assert effective_range("bypass", SignalRange.LEGAL) is SignalRange.FULL
    assert effective_range("gamma-2.4", "legal") is SignalRange.LEGAL
    assert effective_range("gamma-2.4", "full") is SignalRange.FULL


def test_signal_range_declares_bypass_dependency():
    """The UI greys the control out from this metadata, so it must be present."""
    for pattern in P.all_patterns():
        params = {p.name: p for p in pattern.parameters}
        if "signal_range" not in params or "transfer_function" not in params:
            continue
        conds = params["signal_range"].disabled_conditions()
        assert conds, f"{pattern.id} signal_range has no disabled_when"
        bypass = next(c for c in conds if c.parameter == "transfer_function")
        assert "bypass" in bypass.values
        assert bypass.reason


def test_disabled_when_is_exposed_by_the_api():
    param = next(
        p
        for p in client.get("/api/patterns/gray-steps").json()["parameters"]
        if p["name"] == "signal_range"
    )
    bypass = next(d for d in param["disabled_when"] if d["parameter"] == "transfer_function")
    assert bypass["values"] == ["bypass"]
    assert bypass["reason"]


def test_rp219_bypass_drops_legal_range():
    """RP 219-2 resolves range itself, so it needs the same rule applied."""
    pattern = P.get_pattern("smpte-rp219-2-2016")
    result = pattern.render(256, 144, {"transfer_function": "bypass", "signal_range": "legal"})
    assert result.signal_format.range is SignalRange.FULL
    default = pattern.render(256, 144, {})
    assert default.signal_format.range is SignalRange.LEGAL


def test_transfers_endpoint_exposes_bypass():
    transfers = {t["id"]: t for t in client.get("/api/transfers").json()}
    assert transfers["bypass"]["is_bypass"] is True
    assert transfers["linear"]["is_bypass"] is False


@pytest.mark.parametrize("pattern", P.all_patterns(), ids=lambda p: p.id)
def test_every_pattern_renders_and_previews_with_bypass(pattern):
    names = {p.name for p in pattern.parameters}
    if "transfer_function" not in names:
        pytest.skip("pattern has no transfer function parameter")
    result = pattern.render(64, 36, {"transfer_function": "bypass"})
    assert result.signal_format.transfer_function == "bypass"
    assert np.all(np.isfinite(result.image))
    assert to_display_srgb(result).shape == (36, 64, 3)
