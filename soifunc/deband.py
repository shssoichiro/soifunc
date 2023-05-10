from __future__ import annotations

from vsdeband import F3kdb
from vsmasktools import dre_edgemask
from vstools import InvalidVideoFormatError, check_variable, copy_signature, core, vs

__all__ = [
    "retinex_deband",
    "RetinexDeband",
]


def retinex_deband(
    clip: vs.VideoNode,
    threshold: int,
    showmask: bool = False,
) -> vs.VideoNode:
    """
    "medium" `threshold` in f3kdb is 48. I think that's a bit strong.
    16 might be a more sane starting point. Increase as needed.

    `mask_threshold` determines how sensitive the mask is.
    Lower values should preserve more detail.
    It does not need to be manually scaled for bit-depth,
    this function will do that automatically.

    This function does not add grain on its own. Use another function like
    `vsdeband.sized_grain` to do that.
    """
    assert check_variable(clip, retinex_deband)

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

    mask = dre_edgemask(clip)

    if showmask:
        return mask

    deband = F3kdb().deband(clip, thr=(threshold << 2))
    return core.std.MaskedMerge(deband, clip, mask)


# Aliases
@copy_signature(retinex_deband)
def RetinexDeband(**kwargs) -> vs.VideoNode:
    import warnings

    warnings.warn(
        "`RetinexDeband` has been deprecated in favor of `retinex_deband`!",
        DeprecationWarning,
    )

    return retinex_deband(**kwargs)
