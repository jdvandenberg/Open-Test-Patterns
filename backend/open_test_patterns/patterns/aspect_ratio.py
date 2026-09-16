"""Aspect-ratio framing chart.

Container sizes and inner crops are the pixel-exact values from the tables
below. They were cross-checked against:

* SMPTE ST 428-1, *D-Cinema Distribution Master — Image Characteristics* —
  normative 2K/4K DCI Full (2048×1080 / 4096×2160, 256:135), Flat (1.85:
  1998×1080 / 3996×2160), and Scope (2.39: 2048×858 / 4096×1716). The same
  numbers appear in Deluxe Digital Cinema, *Specifications for Digital Cinema
  Source and DCP Content Delivery* (v5.11) and the public DCI *Digital Cinema
  System Specification* (https://dcss.dcimovies.com/).
* ITU-R BT.709 / BT.2020 — 1080p (1920×1080), 4K UHD (3840×2160), 8K UHD /
  UHDTV2 (7680×4320). See also SMPTE RP 219-2 annex tables.
* 8K DCI (8192×4320) is not SDO-normative; it is the exact 2× scaling of
  4K DCI Full that preserves 256:135.
* Unravel Mastering Services, *Aspect Ratio Cheat Sheet* (PDF v3.0),
  https://www.unravel.com.au/aspect-ratio-cheat-sheet — aggregator used to
  compile the per-ratio crops. Non-standard ratios are stored as those
  verified even-pixel sizes rather than recomputed from a floating aspect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageBufAlgo

from .base import Choice, DisabledWhen, Parameter, ParamType, Pattern, PatternResult
from .registry import register
from .util import canvas, colorspace_param, finalize_signal, range_param, transfer_param

# Container id -> (width, height).
CONTAINERS: dict[str, tuple[int, int]] = {
    "dci-2k-full": (2048, 1080),
    "dci-2k-flat": (1998, 1080),
    "dci-2k-scope": (2048, 858),
    "dci-4k-full": (4096, 2160),
    "dci-4k-flat": (3996, 2160),
    "dci-4k-scope": (4096, 1716),
    "uhd-4k": (3840, 2160),
    "dci-8k": (8192, 4320),
    "uhd-8k": (7680, 4320),
    "1080p": (1920, 1080),
}

CONTAINER_CHOICES = [
    Choice("dci-2k-full", "2K DCI (Full) — 2048×1080"),
    Choice("dci-2k-flat", "2K DCI Flat — 1998×1080"),
    Choice("dci-2k-scope", "2K DCI Scope — 2048×858"),
    Choice("dci-4k-full", "4K DCI (Full) — 4096×2160"),
    Choice("dci-4k-flat", "4K DCI Flat — 3996×2160"),
    Choice("dci-4k-scope", "4K DCI Scope — 4096×1716"),
    Choice("uhd-4k", "4K UHD — 3840×2160"),
    Choice("dci-8k", "8K DCI — 8192×4320"),
    Choice("uhd-8k", "8K UHD — 7680×4320"),
    Choice("1080p", "1080p — 1920×1080"),
]

RATIO_CHOICES = [
    Choice("1.33", "1.33 (4:3)"),
    Choice("1.375", "1.375 (Academy)"),
    Choice("1.50", "1.50 (3:2)"),
    Choice("1.66", "1.66 (5:3)"),
    Choice("1.78", "1.78 (16:9)"),
    Choice("1.85", "1.85 (DCI Flat)"),
    Choice("1.90", "1.90"),
    Choice("2.00", "2.00"),
    Choice("2.35", "2.35"),
    Choice("2.37", "2.37"),
    Choice("2.39", "2.39 (DCI Scope)"),
    Choice("2.40", "2.40"),
    Choice("2.55", "2.55"),
    Choice("2.66", "2.66"),
]

# Inner active area (width, height) for each container × ratio.
# Pillarbox keeps container height; letterbox keeps container width.
FRAMES: dict[str, dict[str, tuple[int, int]]] = {
    "dci-2k-full": {
        "1.33": (1440, 1080),
        "1.375": (1484, 1080),
        "1.50": (1620, 1080),
        "1.66": (1800, 1080),
        "1.78": (1920, 1080),
        "1.85": (1998, 1080),
        "1.90": (2048, 1080),
        "2.00": (2048, 1024),
        "2.35": (2048, 870),
        "2.37": (2048, 864),
        "2.39": (2048, 858),
        "2.40": (2048, 852),
        "2.55": (2048, 802),
        "2.66": (2048, 768),
    },
    "dci-2k-flat": {
        "1.33": (1440, 1080),
        "1.375": (1484, 1080),
        "1.50": (1620, 1080),
        "1.66": (1800, 1080),
        "1.78": (1920, 1080),
        "1.85": (1998, 1080),
        "1.90": (1998, 1052),
        "2.00": (1998, 998),
        "2.35": (1998, 850),
        "2.37": (1998, 842),
        "2.39": (1998, 836),
        "2.40": (1998, 832),
        "2.55": (1998, 782),
        "2.66": (1998, 750),
    },
    "dci-2k-scope": {
        "1.33": (1144, 858),
        "1.375": (1178, 858),
        "1.50": (1286, 858),
        "1.66": (1430, 858),
        "1.78": (1524, 858),
        "1.85": (1586, 858),
        "1.90": (1626, 858),
        "2.00": (1716, 858),
        "2.35": (2016, 858),
        "2.37": (2032, 858),
        "2.39": (2048, 858),
        "2.40": (2048, 852),
        "2.55": (2048, 802),
        "2.66": (2048, 768),
    },
    "dci-4k-full": {
        "1.33": (2880, 2160),
        "1.375": (2970, 2160),
        "1.50": (3240, 2160),
        "1.66": (3600, 2160),
        "1.78": (3840, 2160),
        "1.85": (3996, 2160),
        "1.90": (4096, 2160),
        "2.00": (4096, 2048),
        "2.35": (4096, 1742),
        "2.37": (4096, 1728),
        "2.39": (4096, 1716),
        "2.40": (4096, 1706),
        "2.55": (4096, 1606),
        "2.66": (4096, 1538),
    },
    "dci-4k-flat": {
        "1.33": (2880, 2160),
        "1.375": (2970, 2160),
        "1.50": (3240, 2160),
        "1.66": (3600, 2160),
        "1.78": (3840, 2160),
        "1.85": (3996, 2160),
        "1.90": (3996, 2106),
        "2.00": (3996, 1998),
        "2.35": (3996, 1700),
        "2.37": (3996, 1684),
        "2.39": (3996, 1674),
        "2.40": (3996, 1664),
        "2.55": (3996, 1566),
        "2.66": (3996, 1502),
    },
    "dci-4k-scope": {
        "1.33": (2288, 1716),
        "1.375": (2358, 1716),
        "1.50": (2574, 1716),
        "1.66": (2860, 1716),
        "1.78": (3050, 1716),
        "1.85": (3174, 1716),
        "1.90": (3254, 1716),
        "2.00": (3432, 1716),
        "2.35": (4032, 1716),
        "2.37": (4066, 1716),
        "2.39": (4096, 1716),
        "2.40": (4096, 1706),
        "2.55": (4096, 1606),
        "2.66": (4096, 1538),
    },
    "uhd-4k": {
        "1.33": (2880, 2160),
        "1.375": (2970, 2160),
        "1.50": (3240, 2160),
        "1.66": (3600, 2160),
        "1.78": (3840, 2160),
        "1.85": (3840, 2074),
        "1.90": (3840, 2024),
        "2.00": (3840, 1920),
        "2.35": (3840, 1634),
        "2.37": (3840, 1620),
        "2.39": (3840, 1608),
        "2.40": (3840, 1600),
        "2.55": (3840, 1504),
        "2.66": (3840, 1442),
    },
    "dci-8k": {
        "1.33": (5760, 4320),
        "1.375": (5940, 4320),
        "1.50": (6480, 4320),
        "1.66": (7200, 4320),
        "1.78": (7680, 4320),
        "1.85": (7992, 4320),
        "1.90": (8192, 4320),
        "2.00": (8192, 4096),
        "2.35": (8192, 3484),
        "2.37": (8192, 3456),
        "2.39": (8192, 3432),
        "2.40": (8192, 3412),
        "2.55": (8192, 3212),
        "2.66": (8192, 3078),
    },
    "uhd-8k": {
        "1.33": (5760, 4320),
        "1.375": (5940, 4320),
        "1.50": (6480, 4320),
        "1.66": (7200, 4320),
        "1.78": (7680, 4320),
        "1.85": (7680, 4150),
        "1.90": (7680, 4050),
        "2.00": (7680, 3840),
        "2.35": (7680, 3268),
        "2.37": (7680, 3240),
        "2.39": (7680, 3216),
        "2.40": (7680, 3200),
        "2.55": (7680, 3010),
        "2.66": (7680, 2886),
    },
    "1080p": {
        "1.33": (1440, 1080),
        "1.375": (1484, 1080),
        "1.50": (1620, 1080),
        "1.66": (1800, 1080),
        "1.78": (1920, 1080),
        "1.85": (1920, 1036),
        "1.90": (1920, 1012),
        "2.00": (1920, 960),
        "2.35": (1920, 816),
        "2.37": (1920, 810),
        "2.39": (1920, 804),
        "2.40": (1920, 800),
        "2.55": (1920, 752),
        "2.66": (1920, 720),
    },
}

_INSIDE = np.array([0.5, 0.5, 0.5], dtype=np.float64)
_BLACK = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_RED = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_CIRCLE_OFF = DisabledWhen(
    parameter="circle",
    values=("false",),
    reason="Circle is off.",
)


def _choice_label(choices: list[Choice], value: str) -> str:
    for choice in choices:
        if choice.value == value:
            return choice.label
    return value


def framing_caption(container: str, ratio: str) -> str:
    """``1.85 (DCI Flat) within 4K UHD - 3840x2160``."""
    ratio_label = _choice_label(RATIO_CHOICES, ratio)
    container_name = _choice_label(CONTAINER_CHOICES, container).split(" — ", 1)[0]
    width, height = CONTAINERS[container]
    return f"{ratio_label} within {container_name} - {width}x{height}"


def aspect_frame(container: str, ratio: str) -> tuple[int, int]:
    """Return the inner (width, height) for ``container`` × ``ratio``."""
    try:
        return FRAMES[container][ratio]
    except KeyError as exc:
        raise ValueError(f"Unknown container/ratio {container!r}/{ratio!r}") from exc


def _mapped_rect(
    canvas_w: int,
    canvas_h: int,
    container_w: int,
    container_h: int,
    inner_w: int,
    inner_h: int,
) -> tuple[int, int, int, int]:
    """Centre the inner frame and scale it onto the output canvas.

    When ``canvas`` matches the container, coordinates are pixel-exact.
    """
    x0 = (container_w - inner_w) // 2
    y0 = (container_h - inner_h) // 2
    if canvas_w == container_w and canvas_h == container_h:
        return x0, y0, x0 + inner_w, y0 + inner_h

    def mx(x: int) -> int:
        return int(round(x * canvas_w / container_w))

    def my(y: int) -> int:
        return int(round(y * canvas_h / container_h))

    return mx(x0), my(y0), mx(x0 + inner_w), my(y0 + inner_h)


def _stroke_rect(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: int, color: np.ndarray
) -> None:
    t = max(int(thickness), 1)
    h, w = img.shape[:2]
    x0, x1 = max(x0, 0), min(x1, w)
    y0, y1 = max(y0, 0), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return
    t = min(t, x1 - x0, y1 - y0)
    img[y0 : y0 + t, x0:x1] = color
    img[y1 - t : y1, x0:x1] = color
    img[y0:y1, x0 : x0 + t] = color
    img[y0:y1, x1 - t : x1] = color


def _stroke_line(
    img: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    thickness: int,
    color: np.ndarray,
) -> None:
    h, w = img.shape[:2]
    n = max(int(np.hypot(x1 - x0, y1 - y0)) + 1, 1)
    xs = np.clip(np.rint(np.linspace(x0, x1, n)).astype(int), 0, w - 1)
    ys = np.clip(np.rint(np.linspace(y0, y1, n)).astype(int), 0, h - 1)
    img[ys, xs] = color
    if thickness > 1:
        half = thickness // 2
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                img[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] = color


def _stroke_circle(
    img: np.ndarray,
    *,
    cx: float,
    cy: float,
    radius: float,
    thickness: int,
    color: np.ndarray,
) -> None:
    height, width = img.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    half = thickness / 2.0
    mask = (dist >= max(radius - half, 0.0)) & (dist < radius + half)
    img[mask] = color


# Native altitude of the third-point arrows at 2K / 1080p, in container pixels,
# including the single tip pixel that sits in the aspect-ratio border. 4K
# containers use 2× this size; 8K containers use 4×.
_ARROW_HEIGHT = 15
_ARROW_SCALE_REF = 2048  # longest side of 2K DCI Full


def _arrow_scale(container: str) -> int:
    """1× at 2K/1080p, 2× at 4K, 4× at 8K."""
    cw, ch = CONTAINERS[container]
    return max(1, round(max(cw, ch) / _ARROW_SCALE_REF))


def _arrow_height(mapped: int, native: int, *, scale: int) -> int:
    """Scale the arrow when the canvas is not the container."""
    base = _ARROW_HEIGHT * scale
    if native <= 0:
        return base
    return max(1, int(round(base * mapped / native)))


def _thirds(start: int, end: int) -> tuple[int, int]:
    """Pixel positions at 1/3 and 2/3 of ``[start, end)``."""
    span = end - start
    return start + round(span / 3), start + round(2 * span / 3)


def _fill_arrow(
    img: np.ndarray,
    tip_x: int,
    tip_y: int,
    *,
    dx: int,
    dy: int,
    height: int,
    color: np.ndarray,
    clip: tuple[int, int, int, int],
) -> None:
    """Filled isosceles triangle with a 1px tip at ``(tip_x, tip_y)``.

    ``(dx, dy)`` is a unit step from the tip into the body — (0, 1) points
    down, (0, -1) up, (1, 0) right, (-1, 0) left. Altitude is ``height``
    pixels including the tip. ``clip`` is the inner picture ``(x0, y0, x1, y1)``
    with exclusive ends so the triangle never paints the pillar/letter box.
    """
    height = max(int(height), 1)
    max_half = (height - 1) // 2
    x0, y0, x1, y1 = clip
    ih, iw = img.shape[:2]

    def halfwidth(i: int) -> int:
        if i <= 0 or height <= 1:
            return 0
        # Round-half-up so row 0 is the single tip pixel.
        return (i * max_half * 2 + (height - 1)) // (2 * (height - 1))

    for i in range(height):
        hw = halfwidth(i)
        px = tip_x + dx * i
        py = tip_y + dy * i
        if dx == 0:
            if py < y0 or py >= y1 or py < 0 or py >= ih:
                continue
            xa = max(px - hw, x0, 0)
            xb = min(px + hw, x1 - 1, iw - 1)
            if xa <= xb:
                img[py, xa : xb + 1] = color
        else:
            if px < x0 or px >= x1 or px < 0 or px >= iw:
                continue
            ya = max(py - hw, y0, 0)
            yb = min(py + hw, y1 - 1, ih - 1)
            if ya <= yb:
                img[ya : yb + 1, px] = color


def _stroke_third_arrows(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    height_x: int,
    height_y: int,
    color: np.ndarray,
) -> None:
    """Arrows at 1/3 and 2/3 of each inner edge, tips in the border line."""
    clip = (x0, y0, x1, y1)
    tx0, tx1 = _thirds(x0, x1)
    ty0, ty1 = _thirds(y0, y1)
    for x in (tx0, tx1):
        _fill_arrow(img, x, y0, dx=0, dy=1, height=height_y, color=color, clip=clip)
        _fill_arrow(img, x, y1 - 1, dx=0, dy=-1, height=height_y, color=color, clip=clip)
    for y in (ty0, ty1):
        _fill_arrow(img, x0, y, dx=1, dy=0, height=height_x, color=color, clip=clip)
        _fill_arrow(img, x1 - 1, y, dx=-1, dy=0, height=height_x, color=color, clip=clip)


def _fit_caption_font(text: str, max_width: int, desired: int) -> tuple[int, Any]:
    """Shrink ``desired`` until ``text`` fits ``max_width``, returning (size, ROI)."""
    fontsize = max(int(desired), 12)
    roi = ImageBufAlgo.text_size(text, fontsize)
    while fontsize > 12 and (not roi.defined or roi.width > max_width):
        fontsize = max(12, int(fontsize * 0.9))
        roi = ImageBufAlgo.text_size(text, fontsize)
    return fontsize, roi


def _draw_caption(
    img: np.ndarray,
    text: str,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: np.ndarray,
) -> None:
    """Paint ``text`` in a black box, centred at 25% up from the inner bottom."""
    inner_w = x1 - x0
    inner_h = y1 - y0
    if inner_w < 64 or inner_h < 48 or not text:
        return
    desired = max(14, round(inner_h * 0.035))
    fontsize, roi = _fit_caption_font(text, max(8, round(inner_w * 0.92)), desired)
    if not roi.defined:
        return
    cx = (x0 + x1 - 1) / 2.0
    target_cy = y0 + inner_h * 0.75
    x = int(round(cx - roi.width / 2.0 - roi.xbegin))
    y = int(round(target_cy - (roi.ybegin + roi.yend) / 2.0))
    pad_x = max(8, round(fontsize * 0.45))
    pad_y = max(4, round(fontsize * 0.28))
    box_x0 = max(x0, x + roi.xbegin - pad_x)
    box_y0 = max(y0, y + roi.ybegin - pad_y)
    box_x1 = min(x1, x + roi.xend + pad_x)
    box_y1 = min(y1, y + roi.yend + pad_y)
    if box_x1 > box_x0 and box_y1 > box_y0:
        img[box_y0:box_y1, box_x0:box_x1] = _BLACK
    fill = (float(color[0]), float(color[1]), float(color[2]))
    buf = ImageBuf(np.ascontiguousarray(img, dtype=np.float32))
    if not ImageBufAlgo.render_text(buf, x, y, text, fontsize, "", fill, shadow=0):
        return
    img[:] = buf.get_pixels(oiio.FLOAT)


@register
class AspectRatio(Pattern):
    id = "aspect-ratio"
    name = "Framing / Aspect Ratio"
    category = "Geometry"
    description = (
        "A grey active picture in a locked container, with a border, third-point "
        "arrows, corner diagonals, and an optional centre circle. An optional "
        "caption names the container and aspect. Outside the chosen aspect is "
        "black or red."
    )
    parameters = [
        Parameter(
            "container",
            "Container",
            ParamType.CHOICE,
            default="1080p",
            choices=CONTAINER_CHOICES,
            description="Imager / canvas size. Output resolution follows this list.",
        ),
        Parameter(
            "ratio",
            "Aspect ratio",
            ParamType.CHOICE,
            default="1.78",
            choices=RATIO_CHOICES,
        ),
        Parameter(
            "outside",
            "Outside",
            ParamType.CHOICE,
            default="black",
            choices=[Choice("black", "Black"), Choice("red", "Red")],
            description="Colour of the pillarbox or letterbox.",
        ),
        Parameter(
            "line_level",
            "Line level",
            ParamType.FLOAT,
            default=1.0,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
        ),
        Parameter(
            "thickness",
            "Line thickness",
            ParamType.INT,
            default=1,
            minimum=1,
            maximum=32,
            unit="px",
        ),
        Parameter(
            "circle",
            "Circle",
            ParamType.BOOL,
            default=True,
            description="Circle centred on the active picture.",
        ),
        Parameter(
            "circle_diameter",
            "Circle diameter",
            ParamType.FLOAT,
            default=75.0,
            minimum=1.0,
            maximum=200.0,
            step=1.0,
            unit="%",
            description="Diameter as a percentage of the active-picture height.",
            disabled_when=_CIRCLE_OFF,
        ),
        Parameter(
            "caption",
            "Caption",
            ParamType.BOOL,
            default=True,
            description="Name the container and aspect ratio below the centre.",
        ),
        colorspace_param(),
        transfer_param(default="gamma-2.4"),
        range_param(),
    ]

    def generate(self, width: int, height: int, **p: Any) -> PatternResult:
        cw, ch = CONTAINERS[p["container"]]
        iw, ih = aspect_frame(p["container"], p["ratio"])
        img = canvas(width, height)
        outside = _RED if p["outside"] == "red" else _BLACK
        img[:, :] = outside
        x0, y0, x1, y1 = _mapped_rect(width, height, cw, ch, iw, ih)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = _INSIDE
            line = np.array([p["line_level"]] * 3, dtype=np.float64)
            t = p["thickness"]
            scale = _arrow_scale(p["container"])
            _stroke_rect(img, x0, y0, x1, y1, t, line)
            _stroke_line(img, x0, y0, x1 - 1, y1 - 1, t, line)
            _stroke_line(img, x0, y1 - 1, x1 - 1, y0, t, line)
            if p["circle"]:
                _stroke_circle(
                    img,
                    cx=(x0 + x1 - 1) / 2.0,
                    cy=(y0 + y1 - 1) / 2.0,
                    radius=(p["circle_diameter"] / 100.0) * (y1 - y0) / 2.0,
                    thickness=t,
                    color=line,
                )
            _stroke_third_arrows(
                img,
                x0,
                y0,
                x1,
                y1,
                height_x=_arrow_height(x1 - x0, iw, scale=scale),
                height_y=_arrow_height(y1 - y0, ih, scale=scale),
                color=line,
            )
            if p["caption"]:
                _draw_caption(
                    img,
                    framing_caption(p["container"], p["ratio"]),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    color=line,
                )
        return finalize_signal(
            img,
            color_space=p["color_space"],
            transfer_function=p["transfer_function"],
            signal_range=p["signal_range"],
        )
