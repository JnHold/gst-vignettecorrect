"""Per-camera correction config, shared by the Python reference and the element.

The file uses GLib key-file syntax so the C++ element can read it with
GKeyFile and no extra dependency. Lists are separated by ';'.

    [vignette]
    image-height-mm=0;0.45;0.9
    relative-illumination=1;1;0.99
    pixel-size-um=2.4
    centre-x=0.5
    centre-y=0.5
    highlight-mode=knee
    knee=200
    # optional, for frames that are a centred crop of a larger sensor area:
    reference-width=3072
    reference-height=2048
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reference import HIGHLIGHT_MODES, gain_map, lens_table

GROUP = "vignette"


@dataclass(frozen=True)
class VignetteConfig:
    image_height_mm: tuple[float, ...]  # chart radii, strictly increasing from 0
    relative_illumination: tuple[float, ...]  # chart brightness, centre = 1
    pixel_size_um: float  # at the resolution of the frames being corrected
    centre_x: float = 0.5  # lens centre, fraction of width
    centre_y: float = 0.5  # lens centre, fraction of height
    highlight_mode: str = "knee"
    knee: float = 200.0
    reference_width: int | None = None
    reference_height: int | None = None

    def __post_init__(self) -> None:
        heights = np.asarray(self.image_height_mm, dtype=np.float64)
        illum = np.asarray(self.relative_illumination, dtype=np.float64)
        if heights.ndim != 1 or len(heights) < 2 or len(heights) != len(illum):
            raise ValueError("chart needs two or more points, and equal-length lists")
        if heights[0] != 0.0 or np.any(np.diff(heights) <= 0):
            raise ValueError("image-height-mm must start at 0 and strictly increase")
        if np.any(illum <= 0) or np.any(illum > 1):
            raise ValueError("relative-illumination must be in (0, 1]")
        if np.any(np.diff(illum) > 0):
            raise ValueError("relative-illumination must never rise with radius")
        if self.pixel_size_um <= 0:
            raise ValueError("pixel-size-um must be positive")
        if not (0.0 <= self.centre_x <= 1.0 and 0.0 <= self.centre_y <= 1.0):
            raise ValueError("centre-x and centre-y must be in [0, 1]")
        if self.highlight_mode not in HIGHLIGHT_MODES:
            raise ValueError(f"highlight-mode must be one of {HIGHLIGHT_MODES}")
        if not 0.0 < self.knee < 255.0:
            raise ValueError("knee must be between 0 and 255, exclusive")
        if (self.reference_width is None) != (self.reference_height is None):
            raise ValueError("set both reference-width and reference-height, or neither")
        if self.reference_width is not None and (
                self.reference_width <= 0 or self.reference_height <= 0):
            raise ValueError("reference size must be positive")

    def reference_size(self, h: int, w: int) -> tuple[int, int]:
        """(height, width) whose half diagonal defines r = 1 for an (h, w) frame."""
        if self.reference_width is None:
            return (h, w)
        return (self.reference_height, self.reference_width)

    def gain_map(self, h: int, w: int) -> np.ndarray:
        """Gain map for an (h, w) frame, (H, W) float32."""
        ref = self.reference_size(h, w)
        table = lens_table(self.image_height_mm, self.relative_illumination,
                           self.pixel_size_um, ref)
        return gain_map(table, h, w, self.centre_x, self.centre_y, ref)


def _join(values) -> str:
    return ";".join(repr(float(v)) for v in values)


def write_config(config: VignetteConfig, path: str | Path) -> None:
    lines = [
        f"[{GROUP}]",
        f"image-height-mm={_join(config.image_height_mm)}",
        f"relative-illumination={_join(config.relative_illumination)}",
        f"pixel-size-um={config.pixel_size_um!r}",
        f"centre-x={config.centre_x!r}",
        f"centre-y={config.centre_y!r}",
        f"highlight-mode={config.highlight_mode}",
        f"knee={config.knee!r}",
    ]
    if config.reference_width is not None:
        lines += [f"reference-width={config.reference_width}",
                  f"reference-height={config.reference_height}"]
    Path(path).write_text("\n".join(lines) + "\n")


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(";") if v.strip())


def read_config(path: str | Path) -> VignetteConfig:
    # GKeyFile is case-sensitive, has no interpolation, and uses '#' comments.
    parser = configparser.ConfigParser(interpolation=None, comment_prefixes=("#",),
                                       delimiters=("=",))
    parser.optionxform = str
    with open(path) as handle:
        parser.read_file(handle)
    if GROUP not in parser:
        raise ValueError(f"{path}: missing [{GROUP}] group")
    group = parser[GROUP]

    known = {"image-height-mm", "relative-illumination", "pixel-size-um", "centre-x",
             "centre-y", "highlight-mode", "knee", "reference-width", "reference-height"}
    unknown = set(group) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    for key in ("image-height-mm", "relative-illumination", "pixel-size-um"):
        if key not in group:
            raise ValueError(f"{path}: missing required key {key}")

    ref_w = group.get("reference-width")
    ref_h = group.get("reference-height")
    return VignetteConfig(
        image_height_mm=_floats(group["image-height-mm"]),
        relative_illumination=_floats(group["relative-illumination"]),
        pixel_size_um=float(group["pixel-size-um"]),
        centre_x=float(group.get("centre-x", "0.5")),
        centre_y=float(group.get("centre-y", "0.5")),
        highlight_mode=group.get("highlight-mode", "knee").strip(),
        knee=float(group.get("knee", "200")),
        reference_width=int(ref_w) if ref_w is not None else None,
        reference_height=int(ref_h) if ref_h is not None else None,
    )
