import kagefunc
import vapoursynth as vs
import vsutil

from .internal import type_error, value_error

core = vs.core

__all__ = [
    "RetinexDeband",
]


def RetinexDeband(
    clip: vs.VideoNode,
    threshold: int,
    mask_threshold: int = 3000,
    showmask: bool = False,
) -> vs.VideoNode:
    """
    "medium" `threshold` in f3kdb is 48. I think that's a bit strong.
    16 might be a more sane starting point. Increase as needed.

    `mask_threshold` determines how sensitive the mask is, lower values
    should preserve more detail. It does not need to be manually scaled for bit-depth,
    this function will do that automatically.

    This function does not add grain on its own. Use another function like
    `kagefunc.adaptive_grain` to do that.
    """
    if (
        clip.format.color_family != vs.YUV
        or clip.format.sample_type != vs.INTEGER
        or clip.format.bits_per_sample > 16
    ):
        raise value_error("currently only supports 8-16 bit integer YUV input")
    bd_shift = 16 - clip.format.bits_per_sample
    mask_threshold = mask_threshold >> bd_shift
    mask = (
        kagefunc.retinex_edgemask(clip)
        .std.Expr(f"x {mask_threshold} > x 0 ?")
        .std.Inflate()
    )
    if showmask:
        return mask
    threshold = threshold << 2
    deband = clip.neo_f3kdb.Deband(
        y=threshold, cb=threshold, cr=threshold, grainy=0, grainc=0, scale=True
    )
    return core.std.MaskedMerge(deband, clip, mask)
