"""Minimal torchvision.transforms stub."""
from __future__ import annotations

from enum import Enum

from . import functional  # noqa: F401


class InterpolationMode(str, Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
