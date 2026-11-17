"""Reference implementation of the lens falloff (vignetting) correction.

This numpy code is the definition the GStreamer element's CUDA kernel must
match. It is also what a dataset tool should call, so that training images and
the device get the same correction.

Conventions:
- Images are (H, W, 3) uint8, stored (gamma-encoded) values. Channel order does
  not matter: the maths uses the brightest channel and scales all three alike.
- Sizes are (height, width).
- Radius is normalised so r = 1 at the half diagonal of the reference frame.
"""
from __future__ import annotations

import numpy as np

MAX_RADIUS = 1.5  # normalised radius covered by the lookup table
TABLE_SIZE = 301  # samples over 0..MAX_RADIUS

HIGHLIGHT_MODES = ("knee", "clip")


def lens_table(image_height_mm, relative_illumination, pixel_um: float,
               size: tuple[int, int]) -> np.ndarray:
    """Falloff sampled evenly over normalised radius 0..MAX_RADIUS.

    image_height_mm, relative_illumination: the manufacturer's chart.
    pixel_um: sensor pixel size in micrometres, at the resolution of `size`.
    size: (height, width) of the frame whose half diagonal defines r = 1.
    Returns (TABLE_SIZE,) float64. Past the chart's last point the last value
    is held.
    """
    half_diagonal_mm = float(np.hypot(*size)) / 2.0 * pixel_um / 1000.0
    radii_mm = np.linspace(0.0, MAX_RADIUS, TABLE_SIZE) * half_diagonal_mm
    return np.interp(radii_mm, image_height_mm, relative_illumination)


def gain_map(table: np.ndarray, h: int, w: int, cx: float = 0.5, cy: float = 0.5,
             reference_size: tuple[int, int] | None = None) -> np.ndarray:
    """Per-pixel gain, (H, W) float32, always at least 1.

    cx, cy: lens centre as fractions of width and height.
    reference_size: (height, width) of the full frame when the image is a
    centred crop of it; defaults to the image itself.
    """
    ref_h, ref_w = reference_size if reference_size is not None else (h, w)
    half_diagonal = np.hypot(ref_h, ref_w) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)  # (H, W) each
    # Radius is measured from each pixel's centre, hence the + 0.5.
    radius = np.hypot(xx + 0.5 - cx * w, yy + 0.5 - cy * h) / half_diagonal  # (H, W)
    falloff = np.interp(radius, np.linspace(0.0, MAX_RADIUS, len(table)), table)
    return (1.0 / falloff).astype(np.float32)


def apply_gain(image: np.ndarray, gain: np.ndarray, knee: float = 200.0,
               highlight_mode: str = "knee") -> np.ndarray:
    """Correct a (H, W, 3) uint8 image with a (H, W) gain map. Returns uint8.

    highlight_mode "knee": each pixel gets full gain until its brightest
    channel would reach `knee`; brighter originals are compressed into
    knee..255, so the correction adds no clipping. One scale factor multiplies
    all three channels.
    highlight_mode "clip": clip(round(image * gain)) per channel, the pure lens
    map. Bright pixels can clip, and a clipped channel shifts the colour.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8, got {image.shape} {image.dtype}")
    if gain.shape != image.shape[:2]:
        raise ValueError(f"gain {gain.shape} does not match image {image.shape[:2]}")
    if highlight_mode not in HIGHLIGHT_MODES:
        raise ValueError(f"highlight_mode must be one of {HIGHLIGHT_MODES}")

    pixels = image.astype(np.float32)  # (H, W, 3)
    if highlight_mode == "clip":
        return np.clip(np.round(pixels * gain[:, :, None]), 0, 255).astype(np.uint8)

    brightest = image.max(axis=2).astype(np.float32)  # (H, W)
    start = knee / gain  # (H, W) original value that reaches the knee at full gain
    target = np.where(brightest > start,
                      knee + (255.0 - knee) * (brightest - start) / (255.0 - start),
                      brightest * gain)
    scale = target / np.maximum(brightest, 1.0)  # (H, W)
    return np.clip(np.round(pixels * scale[:, :, None]), 0, 255).astype(np.uint8)
