from __future__ import annotations

from enum import Enum


class InterpolationMode(str, Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"