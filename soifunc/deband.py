from __future__ import annotations

from vsdeband import f3k_deband
from vsmasktools import dre_edgemask
from vstools import (
    ConstantFormatVideoNode,
    InvalidVideoFormatError,
    VariableFormatError,
    check_variable,
    core,
    vs,
)

__all__ = [
    "retinex_deband",
]


def retinex_deband(
    clip: vs.VideoNode,
    threshold: int,
    showmask: bool = False,
) -> vs.VideoNode:
    """Debanding using contrast-adaptive edge masking.

    Args:
        clip: Input video (8-16bit YUV required).
        threshold: Debanding strength (0-255). Default ~16-48 recommended.
        showmask: If True, return edge mask instead of debanded clip.

    Returns:
        Debanded video clip or edge mask.

    Note:
        Does not add grain. Use vsdeband.AddNoise for post-denoising.
    """
    if threshold < 0 or threshold > 255:
        raise ValueError(f"threshold must be between 0-255, got {threshold}")

    if not check_variable(clip, retinex_deband):
        raise VariableFormatError("clip must have constant format and fps")

    if (
        clip.format.color_family != vs.YUV
        or clip.format.sample_type != vs.INTEGER
        or clip.format.bits_per_sample > 16
    ):
        raise InvalidVideoFormatError(
            retinex_deband,
            clip.format,
            "The format {format.name} is not supported! It must be an 8-16bit integer YUV bit format!",
        )

    mask: ConstantFormatVideoNode = dre_edgemask.CLAHE(clip)

    if showmask:
        return mask

    # The threshold value that `retinex_deband` takes is relative
    # to 8-bit videos, but `f3kdb` changed their threshold
    # values to be relative to 10-bit videos some time after this
    # function was created. To keep this function compatible,
    # we shift our threshold from 8-bit to 10-bit.
    deband = f3k_deband(clip, thr=(threshold << 2))
    return core.std.MaskedMerge(deband, clip, mask)
