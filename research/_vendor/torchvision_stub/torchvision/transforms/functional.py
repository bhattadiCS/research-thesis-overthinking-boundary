"""Minimal torchvision.transforms.functional stub.

Provides ``pil_to_tensor`` and ``to_pil_image`` which newer ``transformers``
imports at module scope from ``torchvision.transforms.functional``.
"""


def pil_to_tensor(pic):
    raise NotImplementedError(
        "The local torchvision shim does not implement pil_to_tensor because the thesis harness is text-only."
    )


def to_pil_image(pic, mode=None):
    raise NotImplementedError(
        "The local torchvision shim does not implement to_pil_image because the thesis harness is text-only."
    )
