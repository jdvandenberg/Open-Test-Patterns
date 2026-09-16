"""Audio test tones and noise."""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .patterns.base import Choice, DisabledWhen, Parameter, ParamType

SAMPLE_RATE = 48_000
FADE_SEC = 0.01
MAX_DURATION = 60.0

SINE_TONE_ID = "sine-tone"
SINE_TONE_NAME = "Test Tone"
SINE_TONE_CATEGORY = "Audio"
SINE_TONE_DESCRIPTION = (
    "Sine, triangle, sawtooth, and square tones, plus band-limited white and "
    "pink noise, at a chosen duration and peak level (dBFS)."
)

WAVEFORMS = (
    Choice("sine", "Sine"),
    Choice("triangle", "Triangle"),
    Choice("sawtooth", "Sawtooth"),
    Choice("square", "Square"),
    Choice("white", "White noise"),
    Choice("pink", "Pink noise"),
)
WAVEFORM_IDS = tuple(c.value for c in WAVEFORMS)
_NOISE = frozenset({"white", "pink"})
_TONES = tuple(w for w in WAVEFORM_IDS if w not in _NOISE)
_NOISE_DISABLES_FREQUENCY = DisabledWhen(
    parameter="waveform",
    values=("white", "pink"),
    reason="Noise uses a frequency range, not a single tone.",
)
_TONE_DISABLES_BAND = DisabledWhen(
    parameter="waveform",
    values=_TONES,
    reason="Periodic tones use a single frequency.",
)


def sine_tone_parameters() -> list[Parameter]:
    return [
        Parameter(
            "waveform",
            "Waveform",
            ParamType.CHOICE,
            default="sine",
            choices=list(WAVEFORMS),
        ),
        Parameter(
            "frequency",
            "Frequency",
            ParamType.FLOAT,
            default=440.0,
            minimum=1.0,
            maximum=20_000.0,
            step=1.0,
            unit="Hz",
            disabled_when=_NOISE_DISABLES_FREQUENCY,
        ),
        Parameter(
            "frequency_low",
            "Low frequency",
            ParamType.FLOAT,
            default=20.0,
            minimum=1.0,
            maximum=20_000.0,
            step=1.0,
            unit="Hz",
            disabled_when=_TONE_DISABLES_BAND,
            description="Lower edge of the noise band.",
        ),
        Parameter(
            "frequency_high",
            "High frequency",
            ParamType.FLOAT,
            default=20_000.0,
            minimum=1.0,
            maximum=20_000.0,
            step=1.0,
            unit="Hz",
            disabled_when=_TONE_DISABLES_BAND,
            description="Upper edge of the noise band.",
        ),
        Parameter(
            "duration",
            "Duration",
            ParamType.FLOAT,
            default=5.0,
            minimum=0.1,
            maximum=MAX_DURATION,
            step=0.1,
            unit="s",
        ),
        Parameter(
            "loudness",
            "Loudness",
            ParamType.FLOAT,
            default=-20.0,
            minimum=-60.0,
            maximum=0.0,
            step=0.1,
            unit="dBFS",
            description="Peak sample level relative to digital full scale.",
        ),
    ]


def sine_tone_catalog() -> dict[str, Any]:
    return {
        "id": SINE_TONE_ID,
        "name": SINE_TONE_NAME,
        "category": SINE_TONE_CATEGORY,
        "description": SINE_TONE_DESCRIPTION,
        "kind": "audio",
        "parameters": sine_tone_parameters(),
    }


def dbfs_to_peak(dbfs: float) -> float:
    """Return the amplitude whose sample peak is ``dbfs`` dBFS."""
    if dbfs > 0:
        raise ValueError("loudness must be <= 0 dBFS to avoid clipping")
    return 10.0 ** (dbfs / 20.0)


def _periodic(waveform: str, frequency: float, t: np.ndarray) -> np.ndarray:
    phase = 2 * math.pi * frequency * t
    if waveform == "sine":
        return np.sin(phase)
    if waveform == "triangle":
        return (2 / math.pi) * np.arcsin(np.clip(np.sin(phase), -1.0, 1.0))
    if waveform == "sawtooth":
        return 2.0 * np.mod(frequency * t, 1.0) - 1.0
    if waveform == "square":
        signed = np.sign(np.sin(phase))
        return np.where(signed == 0, 1.0, signed)
    raise ValueError(f"Unknown waveform {waveform!r}")


def _noise_band(frequency_low: float, frequency_high: float, sample_rate: int) -> tuple[float, float]:
    nyquist = sample_rate / 2.0
    lo = min(frequency_low, frequency_high)
    hi = max(frequency_low, frequency_high)
    lo = max(0.0, lo)
    hi = min(hi, nyquist)
    if hi <= lo:
        raise ValueError("noise high frequency must be greater than low frequency")
    return lo, hi


def _band_limited_noise(
    n: int,
    sample_rate: int,
    frequency_low: float,
    frequency_high: float,
    pink: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """White or 1/f noise, brick-wall filtered to ``[frequency_low, frequency_high]``."""
    lo, hi = _noise_band(frequency_low, frequency_high, sample_rate)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    if pink:
        scale = np.zeros_like(freqs)
        positive = freqs > 0
        scale[positive] = 1.0 / np.sqrt(freqs[positive])
        spec *= scale
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    spec[0] = 0.0
    return np.fft.irfft(spec, n=n)


def generate_tone(
    frequency: float = 440.0,
    duration: float = 5.0,
    loudness: float = -20.0,
    sample_rate: int = SAMPLE_RATE,
    waveform: str = "sine",
    frequency_low: float = 20.0,
    frequency_high: float = 20_000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return mono samples in ``[-1, 1]`` peaked at ``loudness`` dBFS."""
    if waveform not in WAVEFORM_IDS:
        raise ValueError(f"waveform must be one of {WAVEFORM_IDS}")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if duration > MAX_DURATION:
        raise ValueError(f"duration must be at most {MAX_DURATION:g} seconds")

    n = int(round(sample_rate * duration))
    peak = dbfs_to_peak(loudness)

    if waveform in _NOISE:
        rng = rng or np.random.default_rng()
        unit = _band_limited_noise(
            n,
            sample_rate,
            frequency_low,
            frequency_high,
            pink=waveform == "pink",
            rng=rng,
        )
    else:
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        nyquist = sample_rate / 2.0
        if frequency >= nyquist:
            raise ValueError(f"frequency must be below Nyquist ({nyquist:g} Hz)")
        t = np.arange(n, dtype=np.float64) / sample_rate
        unit = _periodic(waveform, frequency, t)

    unit = np.array(unit, dtype=np.float64, copy=True)
    fade = min(int(FADE_SEC * sample_rate), n // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
        unit[:fade] *= ramp
        unit[-fade:] *= ramp[::-1]
    peak_u = float(np.max(np.abs(unit)))
    if peak_u <= 0:
        raise ValueError("generated signal is silent")
    return peak * (unit / peak_u)


def wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    Path(path).write_bytes(wav_bytes(samples, sample_rate))


def tone_filename(
    frequency: float,
    duration: float,
    loudness: float,
    waveform: str = "sine",
    frequency_low: float = 20.0,
    frequency_high: float = 20_000.0,
) -> str:
    if waveform in _NOISE:
        lo, hi = _noise_band(frequency_low, frequency_high, SAMPLE_RATE)
        return (
            f"{waveform.capitalize()}_Noise_{lo:g}-{hi:g}Hz_"
            f"{duration:g}s_{loudness:g}dBFS.wav"
        )
    return f"{waveform.capitalize()}_{frequency:g}Hz_{duration:g}s_{loudness:g}dBFS.wav"
