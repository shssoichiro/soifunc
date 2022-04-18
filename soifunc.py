__all__ = ["GoodResize"]

from typing import List
import vapoursynth as vs

import muvsfunc
import vsutil


def GoodResize(clip: vs.VideoNode, width: int, height: int) -> vs.VideoNode:
    if clip.width == width and clip.height == height:
        return clip
    planes: List[vs.VideoNode] = vsutil.split(clip)
    upscale = width >= clip.width and height >= clip.height

    for i in range(len(planes)):
        if i == 0:
            if upscale:
                planes[0] = planes[0].jinc.JincResize(width, height)
            else:
                planes[0] = muvsfunc.SSIM_downsample(
                    planes[0],
                    width,
                    height,
                    kernel="Lanczos",
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
