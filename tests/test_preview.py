"""sRGB previews must tone-map HDR instead of clipping it to 255."""

import numpy as np

from open_test_patterns.color import transfer
from open_test_patterns.imageio.preview import to_display_srgb
from open_test_patterns.patterns.base import PatternResult, SignalFormat, SignalRange


def _patch(code, *, tf, color_space="rec2020", peak=None):
    img = np.full((8, 8, 3), code, dtype=np.float32)
    return PatternResult(
        img,
        SignalFormat(
            color_space=color_space,
            transfer_function=tf,
            range=SignalRange.FULL,
            peak_luminance=peak,
        ),
    )


def _pq(nits: float, peak: float = 100.0) -> PatternResult:
    code = float(np.asarray(transfer.encode(nits, "pq")).reshape(-1)[0])
    return _patch(code, tf="pq", peak=peak)


def _mean8(result: PatternResult) -> int:
    return int(round(to_display_srgb(result).astype(np.float64).mean()))


def test_pq_hundred_nits_is_near_white_but_not_clipped():
    level = _mean8(_pq(100.0))
    assert 200 < level < 255


def test_pq_steps_stay_ordered_through_the_shoulder():
    five, hundred, thousand, tenk = (_mean8(_pq(n)) for n in (5.0, 100.0, 1000.0, 10000.0))
    assert five < hundred < thousand <= tenk <= 255


def test_sdr_peak_white_still_reaches_255():
    white = _patch(1.0, tf="gamma-2.4", color_space="srgb")
    assert _mean8(white) == 255
