"""Shared helpers for pattern generators."""

from __future__ import annotations

from typing import Any

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

# Colors can be authored either as relative linear light or as ready-made code
# values. The two are different quantities, not different units, so each gets its
# own control and only one is live at a time.
VALUE_MODE = "value_mode"
LINEAR_MODE = "linear"
CODE_12BIT_MODE = "code-12bit"
CODE_12BIT_MAX = 4095.0

_LINEAR_INAPPLICABLE = DisabledWhen(
    parameter=VALUE_MODE,
    values=(CODE_12BIT_MODE,),
    reason="Entering 12-bit code values, so the linear RGB input is unused.",
)
_CODE_INAPPLICABLE = DisabledWhen(
    parameter=VALUE_MODE,
    values=(LINEAR_MODE,),
    reason="Entering linear RGB, so the 12-bit code value input is unused.",
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
    # Only absolute transfer functions consume this, so it is disabled for every
    # relative one. Deriving the list from the registry means a newly added
    # absolute curve enables the control without touching this code.
    relative_transfers = tuple(
        tf.id for tf in transfer.list_transfer_functions() if not tf.is_absolute
    )
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
        disabled_when=DisabledWhen(
            parameter="transfer_function",
            values=relative_transfers,
            reason="Only absolute transfer functions such as PQ map code values to cd/m².",
        ),
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


def value_mode_param(default: str = LINEAR_MODE) -> Parameter:
    return Parameter(
        name=VALUE_MODE,
        label="Color entry",
        type=ParamType.CHOICE,
        default=default,
        choices=[
            Choice(LINEAR_MODE, "Linear RGB [0, 1]"),
            Choice(CODE_12BIT_MODE, "12-bit code value [0, 4095]"),
        ],
        description=(
            "Linear RGB is light and is passed through the transfer function. A "
            "12-bit code value is already encoded, so it is written out unchanged "
            "and the transfer function only records how to interpret it."
        ),
    )


def color_params(
    name: str = "color",
    label: str = "Color",
    default: list[float] | None = None,
) -> list[Parameter]:
    """Return the linear-light and 12-bit code-value controls for one color input.

    Both are always declared; ``value_mode`` decides which one the generator
    reads, and the other is greyed out.
    """
    linear_default = [1.0, 1.0, 1.0] if default is None else default
    return [
        Parameter(
            name,
            f"{label} (linear RGB)",
            ParamType.COLOR,
            default=linear_default,
            disabled_when=_LINEAR_INAPPLICABLE,
        ),
        Parameter(
            f"{name}_code",
            f"{label} (12-bit code value)",
            ParamType.COLOR,
            default=[float(round(c * CODE_12BIT_MAX)) for c in linear_default],
            minimum=0.0,
            maximum=CODE_12BIT_MAX,
            step=1.0,
            disabled_when=_CODE_INAPPLICABLE,
        ),
    ]


def authored_color(p: dict[str, Any], name: str = "color") -> np.ndarray:
    """Return one color input normalised to ``[0, 1]``.

    Code values are divided by full scale and rounded first, since a 12-bit code
    value is an integer. Normalising both modes to the same range lets a
    generator build its geometry once; :func:`finalize_authored` decides whether
    the numbers are treated as light or as code values.
    """
    if p.get(VALUE_MODE, LINEAR_MODE) == CODE_12BIT_MODE:
        code = np.asarray(p[f"{name}_code"], dtype=np.float64)
        return np.round(code) / CODE_12BIT_MAX
    return np.asarray(p[name], dtype=np.float64)


def finalize_authored(
    img: np.ndarray,
    *,
    value_mode: str,
    color_space: str = "rec709",
    transfer_function: str = "linear",
    peak_luminance: float | None = None,
    signal_range: str | SignalRange = SignalRange.FULL,
) -> PatternResult:
    """Finalize an image built from :func:`authored_color` values.

    In linear mode the transfer function is applied; in code-value mode the
    numbers are already code values and are written through untouched.
    """
    if value_mode == CODE_12BIT_MODE:
        is_absolute = transfer.get_transfer_function(transfer_function).is_absolute
        return finalize_signal(
            img,
            color_space=color_space,
            transfer_function=transfer_function,
            # Nothing is encoded here, but the preview still needs the peak to
            # normalise absolute curves. Mirror finalize() and drop it otherwise.
            peak_luminance=peak_luminance if is_absolute else None,
            signal_range=signal_range,
        )
    return finalize(
        img,
        color_space=color_space,
        transfer_function=transfer_function,
        peak_luminance=peak_luminance,
        signal_range=signal_range,
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
