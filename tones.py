"""Generate a WAV test tone or noise burst at a given duration and peak level."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python tones.py` from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from open_test_patterns.audio import generate_tone, write_wav


def main() -> int:
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
        choices=["sine", "triangle", "sawtooth", "square", "sweep", "white", "pink"],
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
    parser.add_argument("-o", "--output", default="tone.wav", help="Output WAV path")
    args = parser.parse_args()
    samples = generate_tone(
        args.frequency,
        args.duration,
        args.loudness,
        waveform=args.waveform,
        frequency_low=args.frequency_low,
        frequency_high=args.frequency_high,
    )
    write_wav(args.output, samples)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
