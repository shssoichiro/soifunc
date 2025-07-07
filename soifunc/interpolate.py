from __future__ import annotations

import platform

import vstools
from vstools import vs

__all__ = ["rate_doubler", "decimation_fixer"]


def rate_doubler(clip: vs.VideoNode) -> vs.VideoNode:
    """
    A utility to double the framerate of a video via frame interpolation.

    Probably shouldn't just go spraying this everywhere,
    it's more for fun and science than anything.
    """
    import vsmlrt

    width = clip.width
    height = clip.height
    matrix = vstools.Matrix.from_video(clip)
    transfer = vstools.Transfer.from_video(clip)
    primaries = vstools.Primaries.from_video(clip)
    clip = clip.resize.Bicubic(
        format=vs.RGBS,
        width=next_multiple_of(64, width),
        height=next_multiple_of(64, height),
    )
    clip = vsmlrt.RIFE(
        clip,
        model=vsmlrt.RIFEModel.v4_25_heavy,
        # TODO: Make this handle more platforms other than just the machines I run on
        backend=(
            vsmlrt.Backend.ORT_DML()
            if platform.system() == "Windows"
            else vsmlrt.Backent.TRT_RTX()
        ),
    )
    # TODO: Handle other chroma samplings
    clip = clip.resize.Bicubic(
        format=vs.YUV420P16,
        width=width,
        height=height,
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
    )
    return clip


def decimation_fixer(clip: vs.VideoNode, cycle: int, offset: int = 0) -> vs.VideoNode:
    """
    Attempts to interpolate frames that were removed by bad decimation.
    Only works with static decimation cycles.
    `cycle` should be the output cycle, i.e. what did the idiot who decimated this
    pass into the decimation filter to achieve this monstrosity?

    Yeah, I know, "ThiS is bAd AND yOu shoUldn'T Do IT".
    Maybe people shouldn't decimate clips that don't need decimation.
    Sometimes you can't "just get a better source".
    """
    import vsmlrt

    if offset >= cycle:
        raise Exception("offset must be less than cycle")
    if cycle <= 0:
        raise Exception("cycle must be greater than zero")

    width = clip.width
    height = clip.height
    matrix = vstools.Matrix.from_video(clip)
    transfer = vstools.Transfer.from_video(clip)
    primaries = vstools.Primaries.from_video(clip)
    clip = clip.resize.Bicubic(
        format=vs.RGBS,
        width=next_multiple_of(64, width),
        height=next_multiple_of(64, height),
    )
    doubled = vsmlrt.RIFE(
        clip,
        model=vsmlrt.RIFEModel.v4_25_heavy,
        # TODO: Make this handle more platforms other than just the machines I run on
        backend=(
            vsmlrt.Backend.ORT_DML()
            if platform.system() == "Windows"
            else vsmlrt.Backent.TRT_RTX()
        ),
    )

    clip_len = clip.num_frames
    frame_to_restore = offset
    source_frame = offset - 1
    while frame_to_restore < clip_len:
        if frame_to_restore > 0:
            interp_frame = source_frame * 2 + 1
            interp = doubled[interp_frame]
            clip = clip[:frame_to_restore] + interp + clip[frame_to_restore:]
        frame_to_restore += cycle
        source_frame += cycle - 1
    clip = clip.std.AssumeFPS(
        fpsnum=clip.fps.numerator * cycle, fpsden=clip.fps.denominator * (cycle - 1)
    )

    # TODO: Handle other chroma samplings
    clip = clip.resize.Bicubic(
        format=vs.YUV420P16,
        width=width,
        height=height,
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
    )
    return clip


def next_multiple_of(multiple: int, param: int) -> int:
    rem = param % multiple
    if rem == 0:
        return param
    return param + (multiple - rem)
