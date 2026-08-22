"""Shared helpers for pattern generators."""

from __future__ import annotations

import numpy as np

from ..color import colorspaces, transfer
from .base import (
    Choice,
    DisabledWhen,
    Parameter,
    ParamType,
    PatternResult,
    SignalFormat,
    SignalRange,
)

_BYPASS_DISABLES_RANGE = DisabledWhen(
    parameter="transfer_function",
    values=("bypass",),
    reason="Bypass passes code values through untouched, so no range scaling is applied.",
)


def canvas(width: int, height: int) -> np.ndarray:
    """Return a zeroed ``(height, width, 3)`` float image."""
    return np.zeros((height, width, 3), dtype=np.float64)


def colorspace_choices() -> list[Choice]:
    return [Choice(cs.id, cs.name) for cs in colorspaces.list_colorspaces()]


def transfer_choices(include_absolute: bool = True) -> list[Choice]:
    return [
        Choice(tf.id, tf.name)
        for tf in transfer.list_transfer_functions()
        if include_absolute or not tf.is_absolute
    ]


def colorspace_param(default: str = "rec709", name: str = "color_space") -> Parameter:
    return Parameter(
        name=name,
        label="Color space",
        type=ParamType.CHOICE,
        default=default,
        choices=colorspace_choices(),
        description="Working / output RGB color space.",
    )


def transfer_param(default: str = "linear", name: str = "transfer_function") -> Parameter:
    return Parameter(
        name=name,
        label="Transfer function",
        type=ParamType.CHOICE,
        default=default,
        choices=transfer_choices(),
        description="Encoding applied to the linear image.",
    )


def peak_luminance_param(default: float = 100.0) -> Parameter:
    return Parameter(
        name="peak_luminance",
        label="Peak luminance",
        type=ParamType.FLOAT,
        default=default,
        minimum=1.0,
        maximum=10000.0,
        step=1.0,
        unit="cd/m²",
        description="Absolute luminance mapped to 1.0 (used by PQ).",
    )


def range_param(default: SignalRange = SignalRange.FULL) -> Parameter:
    return Parameter(
        name="signal_range",
        label="Signal range",
        type=ParamType.CHOICE,
        default=default.value,
        choices=[Choice("full", "Full"), Choice("legal", "Legal / narrow")],
        description="Full-range or 10-bit legal-range code values.",
        disabled_when=_BYPASS_DISABLES_RANGE,
    )


def effective_range(transfer_function: str, signal_range: str | SignalRange) -> SignalRange:
    """Resolve the signal range that is actually applied.

    Bypass performs no scaling in either direction, so it always resolves to
    full range regardless of what was requested. Keeping this in one place means
    the greyed-out control in the UI and the generator agree by construction.
    """
    if isinstance(signal_range, str):
        signal_range = SignalRange(signal_range)
    if transfer.get_transfer_function(transfer_function).is_bypass:
        return SignalRange.FULL
    return signal_range


def finalize(
    linear_rgb: np.ndarray,
    *,
    color_space: str = "rec709",
    transfer_function: str = "linear",
    peak_luminance: float | None = None,
    signal_range: str | SignalRange = SignalRange.FULL,
) -> PatternResult:
    """Encode a relative-linear RGB image and wrap it with its SignalFormat.

    ``linear_rgb`` is expected to be relative linear light in ``[0, 1]`` in
    ``color_space``. For PQ the values are first scaled to absolute luminance by
    ``peak_luminance`` (nits) before encoding.
    """
    tf = transfer.get_transfer_function(transfer_function)
    signal_range = effective_range(transfer_function, signal_range)

    if tf.is_absolute:
        peak = peak_luminance if peak_luminance is not None else 100.0
        code = tf.encode(np.clip(linear_rgb, 0.0, None) * peak)
    else:
        code = tf.encode(linear_rgb)

    if signal_range is SignalRange.LEGAL:
        code = transfer.full_to_legal(code)

    return PatternResult(
        image=np.ascontiguousarray(code, dtype=np.float32),
        signal_format=SignalFormat(
            color_space=color_space,
            transfer_function=transfer_function,
            range=signal_range,
            peak_luminance=peak_luminance if tf.is_absolute else None,
        ),
    )


def finalize_signal(
    code_values: np.ndarray,
    *,
    color_space: str = "rec709",
    transfer_function: str = "gamma-2.4",
    peak_luminance: float | None = None,
    signal_range: str | SignalRange = SignalRange.FULL,
) -> PatternResult:
    """Wrap an already-encoded *code-value* image (e.g. color bars).

    Unlike :func:`finalize`, the values are treated as final code values and are
    not passed through a transfer function again. ``transfer_function`` here is
    documentation: it tells consumers (previews, writers) how to interpret the
    code values.
    """
    signal_range = effective_range(transfer_function, signal_range)
    code = np.clip(np.asarray(code_values, dtype=np.float64), 0.0, 1.0)
    if signal_range is SignalRange.LEGAL:
        code = transfer.full_to_legal(code)
    return PatternResult(
        image=np.ascontiguousarray(code, dtype=np.float32),
        signal_format=SignalFormat(
            color_space=color_space,
            transfer_function=transfer_function,
            range=signal_range,
            peak_luminance=peak_luminance,
        ),
    )
