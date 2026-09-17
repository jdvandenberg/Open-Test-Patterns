"""Test-tone generator and HTTP download."""

import math
import struct
import wave
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient

from open_test_patterns.api.app import app
from open_test_patterns.patterns.tones import generate_tone, tone_filename, wav_bytes

client = TestClient(app)

WAVEFORMS = ("sine", "triangle", "sawtooth", "square", "sweep", "white", "pink")


def _mid_peak_dbfs(samples: np.ndarray) -> float:
    mid = samples[len(samples) // 4 : 3 * len(samples) // 4]
    return 20 * math.log10(float(np.abs(mid).max()))


def test_default_tone_is_minus_20_dbfs_peak():
    samples = generate_tone()
    assert abs(_mid_peak_dbfs(samples) - (-20.0)) < 0.05
    assert len(samples) == 44_100 * 5


def test_waveforms_share_peak_loudness():
    for waveform in WAVEFORMS:
        samples = generate_tone(
            frequency=440,
            duration=0.2,
            loudness=-12,
            waveform=waveform,
            rng=np.random.default_rng(0),
        )
        peak_db = 20 * math.log10(float(np.abs(samples).max()))
        assert abs(peak_db - (-12.0)) < 0.05


def test_periodic_wave_shapes():
    sr = 48_000
    freq = 100
    start = int(0.1 * sr)
    period = sr // freq
    sl = slice(start, start + period)
    sine = generate_tone(freq, duration=0.2, loudness=0, waveform="sine", sample_rate=sr)
    tri = generate_tone(freq, duration=0.2, loudness=0, waveform="triangle", sample_rate=sr)
    saw = generate_tone(freq, duration=0.2, loudness=0, waveform="sawtooth", sample_rate=sr)
    sq = generate_tone(freq, duration=0.2, loudness=0, waveform="square", sample_rate=sr)
    t = np.arange(start, start + period) / sr
    assert np.allclose(sine[sl], np.sin(2 * math.pi * freq * t), atol=1e-9)
    assert float(np.abs(sq[sl]).min()) > 0.99
    assert saw[start] < -0.99
    assert saw[start + period // 4] > saw[start]
    assert abs(float(tri[start + period // 4]) - 1.0) < 0.02


def test_noise_band_is_stable_for_same_seed():
    kwargs = dict(
        duration=0.05,
        loudness=-20,
        waveform="white",
        frequency_low=20,
        frequency_high=20_000,
    )
    a = generate_tone(rng=np.random.default_rng(1), **kwargs)
    b = generate_tone(rng=np.random.default_rng(1), **kwargs)
    np.testing.assert_array_equal(a, b)


def test_noise_band_changes_the_signal():
    a = generate_tone(
        duration=0.2,
        waveform="white",
        frequency_low=100,
        frequency_high=300,
        rng=np.random.default_rng(2),
    )
    b = generate_tone(
        duration=0.2,
        waveform="white",
        frequency_low=2000,
        frequency_high=4000,
        rng=np.random.default_rng(2),
    )
    assert not np.allclose(a, b)


def test_noise_is_band_limited():
    samples = generate_tone(
        duration=0.5,
        loudness=-6,
        waveform="white",
        frequency_low=1000,
        frequency_high=2000,
        sample_rate=48_000,
        rng=np.random.default_rng(3),
    )
    spec = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / 48_000)
    in_band = float(spec[(freqs >= 1000) & (freqs <= 2000)].mean())
    out_band = float(
        spec[((freqs > 50) & (freqs < 500)) | ((freqs > 3000) & (freqs < 10_000))].mean()
    )
    assert in_band > 20 * out_band


def _zero_crossing_hz(samples: np.ndarray, sample_rate: int) -> float:
    crossings = int(np.sum(np.diff(np.signbit(samples))))
    return crossings / (2.0 * len(samples) / sample_rate)


def test_sweep_ramps_over_the_clip():
    sr = 48_000
    samples = generate_tone(
        duration=0.5,
        loudness=0,
        waveform="sweep",
        frequency_low=200,
        frequency_high=2000,
        sample_rate=sr,
    )
    fade = int(0.01 * sr)
    start = _zero_crossing_hz(samples[fade : fade + sr // 10], sr)
    end = _zero_crossing_hz(samples[-(fade + sr // 10) : -fade], sr)
    assert 150 < start < 500
    assert 1500 < end < 2200
    assert end > start * 3


def test_sweep_matches_linear_chirp():
    sr = 48_000
    duration = 0.2
    f0, f1 = 100.0, 400.0
    samples = generate_tone(
        duration=duration,
        loudness=0,
        waveform="sweep",
        frequency_low=f0,
        frequency_high=f1,
        sample_rate=sr,
    )
    n = int(round(sr * duration))
    t = np.arange(n, dtype=np.float64) / sr
    slope = (f1 - f0) / duration
    expected = np.sin(2 * math.pi * (f0 * t + 0.5 * slope * t * t))
    mid = slice(int(0.01 * sr), -int(0.01 * sr))
    assert np.allclose(samples[mid], expected[mid], atol=1e-9)


def test_tone_catalog_lists_waveforms():
    data = client.get("/api/patterns").json()
    tone = next(p for p in data if p["id"] == "sine-tone")
    assert tone["kind"] == "audio"
    assert tone["category"] == "Audio"
    assert tone["name"] == "Test Tone"
    names = [p["name"] for p in tone["parameters"]]
    assert names == [
        "waveform",
        "frequency",
        "frequency_low",
        "frequency_high",
        "duration",
        "loudness",
        "sample_rate",
        "bit_depth",
    ]
    rate = next(p for p in tone["parameters"] if p["name"] == "sample_rate")
    assert rate["default"] == "44100"
    assert [c["value"] for c in rate["choices"]] == ["44100", "48000", "96000", "192000"]
    depth = next(p for p in tone["parameters"] if p["name"] == "bit_depth")
    assert depth["default"] == "16"
    assert [c["value"] for c in depth["choices"]] == ["16", "24", "float32"]
    waveform = next(p for p in tone["parameters"] if p["name"] == "waveform")
    assert {c["value"] for c in waveform["choices"]} == set(WAVEFORMS)
    freq = next(p for p in tone["parameters"] if p["name"] == "frequency")
    assert set(freq["disabled_when"][0]["values"]) == {"sweep", "white", "pink"}
    low = next(p for p in tone["parameters"] if p["name"] == "frequency_low")
    high = next(p for p in tone["parameters"] if p["name"] == "frequency_high")
    assert low["default"] == 20
    assert high["default"] == 20_000
    assert set(low["disabled_when"][0]["values"]) == {"sine", "triangle", "sawtooth", "square"}
    assert high["disabled_when"][0]["values"] == low["disabled_when"][0]["values"]


def test_tone_download_is_wav():
    r = client.post(
        "/api/tone",
        json={"frequency": 1000, "duration": 0.2, "loudness": -20},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert "Sine_1000Hz_0.2s_-20dBFS_44.1kHz_16bit.wav" in r.headers["content-disposition"]
    with wave.open(BytesIO(r.content)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 44_100
        assert wav.getsampwidth() == 2


def test_tone_download_square_and_pink_filenames():
    square = client.post(
        "/api/tone",
        json={"waveform": "square", "frequency": 440, "duration": 0.1, "loudness": -18},
    )
    assert square.status_code == 200
    assert "Square_440Hz_0.1s_-18dBFS_44.1kHz_16bit.wav" in square.headers["content-disposition"]

    pink = client.post(
        "/api/tone",
        json={
            "waveform": "pink",
            "frequency": 440,
            "frequency_low": 20,
            "frequency_high": 20_000,
            "duration": 0.1,
            "loudness": -18,
        },
    )
    assert pink.status_code == 200
    assert "Pink_Noise_20-20000Hz_0.1s_-18dBFS_44.1kHz_16bit.wav" in pink.headers["content-disposition"]

    sweep = client.post(
        "/api/tone",
        json={
            "waveform": "sweep",
            "frequency_low": 20,
            "frequency_high": 20_000,
            "duration": 0.1,
            "loudness": -18,
        },
    )
    assert sweep.status_code == 200
    assert "Sweep_20-20000Hz_0.1s_-18dBFS_44.1kHz_16bit.wav" in sweep.headers["content-disposition"]


def test_unknown_waveform_rejected():
    r = client.post("/api/tone", json={"waveform": "chirp"})
    assert r.status_code == 422


def test_tone_filename():
    assert tone_filename(440, 5, -20) == "Sine_440Hz_5s_-20dBFS_44.1kHz_16bit.wav"
    assert tone_filename(440, 5, -20, "triangle") == "Triangle_440Hz_5s_-20dBFS_44.1kHz_16bit.wav"
    assert tone_filename(440, 5, -20, "white") == "White_Noise_20-20000Hz_5s_-20dBFS_44.1kHz_16bit.wav"
    assert (
        tone_filename(440, 5, -20, "pink", frequency_low=100, frequency_high=8000)
        == "Pink_Noise_100-8000Hz_5s_-20dBFS_44.1kHz_16bit.wav"
    )
    assert (
        tone_filename(440, 5, -20, "sweep", frequency_low=20, frequency_high=20_000)
        == "Sweep_20-20000Hz_5s_-20dBFS_44.1kHz_16bit.wav"
    )
    assert (
        tone_filename(1000, 1, -18, sample_rate=96_000, bit_depth="float32")
        == "Sine_1000Hz_1s_-18dBFS_96kHz_32float.wav"
    )


def test_wav_bytes_roundtrip():
    samples = generate_tone(frequency=440, duration=0.1, loudness=-20)
    data = wav_bytes(samples)
    assert data[:4] == b"RIFF"


def _wav_fmt(data: bytes) -> tuple[int, int, int, int]:
    """Return audio format, channels, sample rate, and bits per sample."""
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        if chunk_id == b"fmt ":
            fmt, channels, rate, _byte_rate, _align, bits = struct.unpack_from(
                "<HHIIHH", data, offset + 8
            )
            return fmt, channels, rate, bits
        offset += 8 + size
        if size % 2:
            offset += 1
    raise AssertionError("WAV has no fmt chunk")


def test_wav_24bit_pcm_and_float32():
    pcm_samples = generate_tone(frequency=440, duration=0.05, loudness=-20, sample_rate=48_000)
    pcm24 = wav_bytes(pcm_samples, sample_rate=48_000, bit_depth="24")
    with wave.open(BytesIO(pcm24)) as wav:
        assert wav.getframerate() == 48_000
        assert wav.getsampwidth() == 3
        assert wav.getnframes() == len(pcm_samples)

    float_samples = generate_tone(frequency=440, duration=0.05, loudness=-20, sample_rate=96_000)
    ieee = wav_bytes(float_samples, sample_rate=96_000, bit_depth="float32")
    fmt, channels, rate, bits = _wav_fmt(ieee)
    assert fmt == 3
    assert channels == 1
    assert rate == 96_000
    assert bits == 32
    assert len(float_samples) == int(round(96_000 * 0.05))


def test_tone_download_sample_rate_and_bit_depth():
    r = client.post(
        "/api/tone",
        json={
            "frequency": 1000,
            "duration": 0.05,
            "loudness": -20,
            "sample_rate": "192000",
            "bit_depth": "24",
        },
    )
    assert r.status_code == 200
    assert "Sine_1000Hz_0.05s_-20dBFS_192kHz_24bit.wav" in r.headers["content-disposition"]
    with wave.open(BytesIO(r.content)) as wav:
        assert wav.getframerate() == 192_000
        assert wav.getsampwidth() == 3

    bad = client.post("/api/tone", json={"sample_rate": 32000})
    assert bad.status_code == 422
