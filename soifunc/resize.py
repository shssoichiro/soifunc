from typing import List, Optional, Union

import muvsfunc
import vapoursynth as vs
import vsutil
from nnedi3_resample import nnedi3_resample

from .internal import value_error

core = vs.core

__all__ = ["GoodResize", "Descale", "DescaleMask"]


def GoodResize(
    clip: vs.VideoNode,
    width: int,
    height: int,
    gpu: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    """High quality resizing filter"""
    if clip.width == width and clip.height == height:
        return clip
    planes: List[vs.VideoNode] = vsutil.split(clip)
    upscale = width >= clip.width or height >= clip.height

    for i in range(len(planes)):
        if i == 0:
            if upscale:
                planes[0] = nnedi3_resample(
                    planes[0],
                    width,
                    height,
                    mode="nnedi3cl" if gpu else "znedi3",
                    nsize=4,
                    nns=4,
                    device=device,
                )
            else:
                planes[0] = muvsfunc.SSIM_downsample(
                    planes[0],
                    width,
                    height,
                    kernel="Lanczos",
                    smooth=0.5,
                    dither_type="error_diffusion",
                )
                planes[0] = vsutil.depth(planes[0], clip.format.bits_per_sample)
        else:
            planes[i] = planes[i].resize.Spline36(
                width >> clip.format.subsampling_w,
                height >> clip.format.subsampling_h,
                dither_type="error_diffusion",
            )

    if len(planes) == 1:
        return planes[0]

    return vsutil.join(planes, clip.format.color_family)


def Descale(
    clip,
    width: int,
    height: int,
    kernel: str = "Bicubic",
    b: float = 0.0,
    c: float = 0.5,
    taps: int = 3,
    mask_threshold: int = 3200,
    downscale_only: bool = False,
    show_mask: bool = False,
):
    """
    Descales a clip using best practices.

    Supposedly there were other functions to do this but they turned out to be buggy,
    so this exists. It:

    - Avoids touching chroma
    - Masks native resolution elements
    - And therefore: Returns a descaled clip with native resolution elements preserved

    It does not detect the proper resolution or kernel for you.

    Passing `downscale_only = True` will only do the downscaling portion, it will not mask or re-upscale.
    This may occasionally be useful. Using this argument will return *only* the downscaled luma plane.
    """
    y_src = vsutil.get_y(clip)
    kernel = kernel.lower()
    if kernel == "bilinear":
        y_desc = y_src.resize.Bilinear(width, height)
    elif kernel == "bicubic":
        y_desc = y_src.resize.Bicubic(width, height, filter_param_a=b, filter_param_b=c)
    elif kernel == "lanczos":
        y_desc = y_src.resize.Lanczos(width, height, filter_param_a=taps)
    elif kernel == "spline16":
        y_desc = y_src.resize.Spline16(width, height)
    elif kernel == "spline36":
        y_desc = y_src.resize.Spline36(width, height)
    elif kernel == "spline64":
        y_desc = y_src.resize.Spline64(width, height)
    else:
        raise value_error("Unsupported resize kernel specified")

    if downscale_only:
        return y_desc

    y_upsc = GoodResize(y_desc, y_src.width, y_src.height)
    mask = DescaleMask(mask_threshold, show_mask)
    if show_mask:
        return mask
    y = core.std.MaskedMerge(y_upsc, y_src, mask)
    return vsutil.join([y, vsutil.plane(clip, 1), vsutil.plane(clip, 2)])


def DescaleMask(
    src_clip,
    descaled_clip,
    threshold: int = 3200,
    show_mask: bool = False,
):
    """
    Generates a mask to preserve detail when downscaling.
    `src_clip` should be the clip prior to any descaling.
    `descaled_clip` should be the clip after descaling and rescaling.
    i.e. they should be the same resolution.

    It is generally easier to call `Descale` as a whole, but this may
    occasionally be useful on its own.
    """
    if (
        src_clip.format.bits_per_sample != descaled_clip.format.bits_per_sample
        or src_clip.format.width != descaled_clip.format.width
        or src_clip.format.height != descaled_clip.format.height
    ):
        raise value_error("Clips must have identical bit depth and resolution")

    bd_shift = 16 - src_clip.format.bits_per_sample
    threshold = threshold >> bd_shift
    mask = (
        core.std.Expr([src_clip, descaled_clip], "x y - abs")
        .std.Binarize(threshold)
        .std.Maximum()
        .std.Inflate()
        .std.Inflate()
    )
    if show_mask:
        return mask
