import kagefunc
import vapoursynth as vs
import vsdeband
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
    mask_threshold = mask_threshold >> (16 - clip.format.bits_per_sample)
    mask = (
        kagefunc.retinex_edgemask(clip)
        .std.Expr(f"x {mask_threshold} > x 0 ?")
        .std.Inflate()
    )
    if showmask:
        return mask
    debander = vsdeband.F3kdb()
    if debander.thr == 30:
        # Because the latest version of vs-deband CHANGED THE SCALING of the threshold parameter...
        raise type_error(
            "please update to the latest git version of vs-deband: https://github.com/Irrational-Encoding-Wizardry/vs-deband"
        )
    deband = debander.deband(clip, thr=(threshold << 8), grains=0)
    if clip.format.bits_per_sample != 16:
        deband = vsutil.depth(deband, clip.format.bits_per_sample)
    return core.std.MaskedMerge(deband, clip, mask)
