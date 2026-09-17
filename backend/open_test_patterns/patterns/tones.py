"""Audio test tones and noise."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .base import Choice, DisabledWhen, Parameter, ParamType

SAMPLE_RATE = 44_100
SAMPLE_RATES = (44_100, 48_000, 96_000, 192_000)
BIT_DEPTHS = ("16", "24", "float32")
FADE_SEC = 0.01
MAX_DURATION = 60.0

SINE_TONE_ID = "sine-tone"
SINE_TONE_NAME = "Test Tone"
SINE_TONE_CATEGORY = "Audio"
SINE_TONE_DESCRIPTION = (
    "Sine, triangle, sawtooth, square, and linear frequency sweeps, plus "
    "band-limited white and pink noise, at a chosen duration and peak level (dBFS)."
)

WAVEFORMS = (
    Choice("sine", "Sine"),
    Choice("triangle", "Triangle"),
    Choice("sawtooth", "Sawtooth"),
    Choice("square", "Square"),
    Choice("sweep", "Sweep"),
    Choice("white", "White noise"),
    Choice("pink", "Pink noise"),
)
WAVEFORM_IDS = tuple(c.value for c in WAVEFORMS)
_NOISE = frozenset({"white", "pink"})
_RANGE = frozenset({"sweep", "white", "pink"})
_FIXED_TONES = tuple(w for w in WAVEFORM_IDS if w not in _RANGE)
_RANGE_DISABLES_FREQUENCY = DisabledWhen(
    parameter="waveform",
    values=("sweep", "white", "pink"),
    reason="This waveform uses a frequency range, not a single tone.",
)
_TONE_DISABLES_BAND = DisabledWhen(
    parameter="waveform",
    values=_FIXED_TONES,
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
            disabled_when=_RANGE_DISABLES_FREQUENCY,
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
            description="Sweep start, or the lower edge of the noise band.",
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
            description="Sweep end, or the upper edge of the noise band.",
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
        Parameter(
            "sample_rate",
            "Sample rate",
            ParamType.CHOICE,
            default="44100",
            choices=[
                Choice("44100", "44.1 kHz"),
                Choice("48000", "48 kHz"),
                Choice("96000", "96 kHz"),
                Choice("192000", "192 kHz"),
            ],
        ),
        Parameter(
            "bit_depth",
            "Bit depth",
            ParamType.CHOICE,
            default="16",
            choices=[
                Choice("16", "16-bit"),
                Choice("24", "24-bit"),
                Choice("float32", "32-bit float"),
            ],
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


def _linear_sweep(n: int, sample_rate: int, f0: float, f1: float) -> np.ndarray:
    """Sine whose instantaneous frequency ramps from ``f0`` to ``f1`` over the clip."""
    t = np.arange(n, dtype=np.float64) / sample_rate
    duration = n / sample_rate
    slope = (f1 - f0) / duration
    phase = 2 * math.pi * (f0 * t + 0.5 * slope * t * t)
    return np.sin(phase)


def _require_below_nyquist(freq: float, sample_rate: int, name: str) -> None:
    if freq <= 0:
        raise ValueError(f"{name} must be positive")
    nyquist = sample_rate / 2.0
    if freq >= nyquist:
        raise ValueError(f"{name} must be below Nyquist ({nyquist:g} Hz)")


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
    sample_rate = int(sample_rate)
    if sample_rate not in SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}")
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
    elif waveform == "sweep":
        _require_below_nyquist(frequency_low, sample_rate, "low frequency")
        _require_below_nyquist(frequency_high, sample_rate, "high frequency")
        unit = _linear_sweep(n, sample_rate, frequency_low, frequency_high)
    else:
        _require_below_nyquist(frequency, sample_rate, "frequency")
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


def _pcm_frames(samples: np.ndarray, bit_depth: str) -> tuple[bytes, int, int]:
    """Pack ``[-1, 1]`` samples. Returns ``(payload, bytes_per_sample, wav_format)``.

    ``wav_format`` is 1 for integer PCM and 3 for IEEE float.
    """
    if bit_depth not in BIT_DEPTHS:
        raise ValueError(f"bit_depth must be one of {BIT_DEPTHS}")
    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    if bit_depth == "16":
        pcm = np.clip(np.rint(x * 32767.0), -32768, 32767).astype("<i2")
        return pcm.tobytes(), 2, 1
    if bit_depth == "24":
        ints = np.clip(np.rint(x * 8388607.0), -8388608, 8388607).astype("<i4")
        packed = ints.view(np.uint8).reshape(-1, 4)[:, :3].copy()
        return packed.tobytes(), 3, 1
    return x.astype("<f4").tobytes(), 4, 3


def wav_bytes(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    bit_depth: str = "16",
) -> bytes:
    sample_rate = int(sample_rate)
    if sample_rate not in SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}")
    frames, sampwidth, audio_format = _pcm_frames(samples, bit_depth)
    nchannels = 1
    block_align = nchannels * sampwidth
    byte_rate = sample_rate * block_align
    bits = sampwidth * 8
    if audio_format == 1:
        fmt = struct.pack(
            "<HHIIHH", audio_format, nchannels, sample_rate, byte_rate, block_align, bits
        )
    else:
        # WAVE_FORMAT_IEEE_FLOAT: 18-byte fmt with cbSize=0.
        fmt = struct.pack(
            "<HHIIHHH",
            audio_format,
            nchannels,
            sample_rate,
            byte_rate,
            block_align,
            bits,
            0,
        )
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data_chunk = b"data" + struct.pack("<I", len(frames)) + frames
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk


def write_wav(
    path: str | Path,
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    bit_depth: str = "16",
) -> None:
    Path(path).write_bytes(wav_bytes(samples, sample_rate, bit_depth))


def _rate_tag(sample_rate: int) -> str:
    if sample_rate == 44_100:
        return "44.1kHz"
    return f"{sample_rate // 1000}kHz"


def _depth_tag(bit_depth: str) -> str:
    if bit_depth == "float32":
        return "32float"
    return f"{bit_depth}bit"


def tone_filename(
    frequency: float,
    duration: float,
    loudness: float,
    waveform: str = "sine",
    frequency_low: float = 20.0,
    frequency_high: float = 20_000.0,
    sample_rate: int = SAMPLE_RATE,
    bit_depth: str = "16",
) -> str:
    sample_rate = int(sample_rate)
    suffix = f"{duration:g}s_{loudness:g}dBFS_{_rate_tag(sample_rate)}_{_depth_tag(bit_depth)}.wav"
    if waveform in _NOISE:
        lo, hi = _noise_band(frequency_low, frequency_high, sample_rate)
        return f"{waveform.capitalize()}_Noise_{lo:g}-{hi:g}Hz_{suffix}"
    if waveform == "sweep":
        return f"Sweep_{frequency_low:g}-{frequency_high:g}Hz_{suffix}"
    return f"{waveform.capitalize()}_{frequency:g}Hz_{suffix}"


def main() -> int:
    """CLI entry: write a WAV to ``--output``."""
    parser = argparse.ArgumentParser(description="Generate a WAV test tone or noise burst.")
    parser.add_argument(
        "--frequency",
        type=float,
        default=440.0,
        help="Tone frequency in Hz (default: 440)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--loudness",
        type=float,
        default=-20.0,
        help="Peak level in dBFS (default: -20)",
    )
    parser.add_argument(
        "--waveform",
        default="sine",
        choices=list(WAVEFORM_IDS),
        help="Waveform, sweep, or noise type (default: sine)",
    )
    parser.add_argument(
        "--frequency-low",
        type=float,
        default=20.0,
        help="Sweep start or noise-band lower edge in Hz (default: 20)",
    )
    parser.add_argument(
        "--frequency-high",
        type=float,
        default=20_000.0,
        help="Sweep end or noise-band upper edge in Hz (default: 20000)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        choices=list(SAMPLE_RATES),
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--bit-depth",
        default="16",
        choices=list(BIT_DEPTHS),
        help="WAV sample format (default: 16)",
    )
    parser.add_argument("-o", "--output", default="tone.wav", help="Output WAV path")
    args = parser.parse_args()
    samples = generate_tone(
        args.frequency,
        args.duration,
        args.loudness,
        sample_rate=args.sample_rate,
        waveform=args.waveform,
        frequency_low=args.frequency_low,
        frequency_high=args.frequency_high,
    )
    write_wav(args.output, samples, args.sample_rate, args.bit_depth)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
