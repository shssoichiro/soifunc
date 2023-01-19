from typing import List

import muvsfunc
import vapoursynth as vs
import vsutil
from nnedi3_resample import nnedi3_resample

core = vs.core

__all__ = [
    "GoodResize",
]


def GoodResize(clip: vs.VideoNode, width: int, height: int) -> vs.VideoNode:
    """High quality resizing filter"""
    if clip.width == width and clip.height == height:
        return clip
    planes: List[vs.VideoNode] = vsutil.split(clip)
    upscale = width >= clip.width or height >= clip.height

    for i in range(len(planes)):
        if i == 0:
            if upscale:
                planes[0] = nnedi3_resample(
                    planes[0], width, height, mode="znedi3", nsize=4, nns=4
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
