"""Minimal torchvision.io stub for environments where torchvision is not installed.

Only used when importlib cannot find a real torchvision package.  Newer
versions of ``transformers`` import ``ImageReadMode`` and ``decode_image``
from ``torchvision.io`` at module scope, so this stub must at least expose
those names to avoid ``ImportError`` at import time.
"""

from enum import IntEnum


class ImageReadMode(IntEnum):
    """Mirrors torchvision.io.ImageReadMode with no-op values."""
    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


def decode_image(*args, **kwargs):
    raise NotImplementedError(
        "The local torchvision shim does not implement decode_image because the thesis harness is text-only."
    )


def read_video(*args, **kwargs):
    raise NotImplementedError(
        "The local torchvision shim does not implement video I/O because the thesis harness is text-only."
    )