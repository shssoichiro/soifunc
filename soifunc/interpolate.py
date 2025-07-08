from __future__ import annotations

import platform

import vstools
from vsmlrt import backendT
from vstools import vs

__all__ = ["rate_doubler", "decimation_fixer"]


def rate_doubler(
    clip: vs.VideoNode, multi: int = 2, backend: backendT | None = None
) -> vs.VideoNode:
    """
    A utility to scale the framerate of a video via frame interpolation.

    Probably shouldn't just go spraying this everywhere,
    it's more for fun and science than anything.
    """
    import vsmlrt

    width = clip.width
    height = clip.height
    matrix = vstools.Matrix.from_video(clip)
    transfer = vstools.Transfer.from_video(clip)
    primaries = vstools.Primaries.from_video(clip)
    clip = clip.misc.SCDetect(clip)
    clip = clip.resize.Bicubic(
        format=vs.RGBS,
        width=next_multiple_of(64, width),
        height=next_multiple_of(64, height),
    )
    clip = vsmlrt.RIFE(
        clip,
        multi=multi,
        model=vsmlrt.RIFEModel.v4_25_heavy,
        # Why these defaults? Because running ML stuff on AMD on Windows sucks hard.
        # Trial and error led me to finally find that ORT_DML works.
        backend=(
            backend
            if backend
            else (
                vsmlrt.Backend.ORT_DML()
                if platform.system() == "Windows"
                else vsmlrt.Backend.TRT_RTX()
            )
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


def decimation_fixer(
    clip: vs.VideoNode, cycle: int, offset: int = 0, backend: backendT | None = None
) -> vs.VideoNode:
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

    if offset >= cycle - 1:
        raise Exception("offset must be less than cycle - 1")
    if cycle <= 0:
        raise Exception("cycle must be greater than zero")

    width = clip.width
    height = clip.height
    fps = clip.fps
    input_cycle = cycle - 1
    matrix = vstools.Matrix.from_video(clip)
    transfer = vstools.Transfer.from_video(clip)
    primaries = vstools.Primaries.from_video(clip)
    clip = clip.misc.SCDetect(clip)
    clip = clip.resize.Bicubic(
        format=vs.RGBS,
        width=next_multiple_of(64, width),
        height=next_multiple_of(64, height),
    )
    doubled = vsmlrt.RIFE(
        clip,
        model=vsmlrt.RIFEModel.v4_25_heavy,
        backend=(
            backend
            if backend
            else (
                vsmlrt.Backend.ORT_DML()
                if platform.system() == "Windows"
                else vsmlrt.Backend.TRT_RTX()
            )
        ),
    )

    out_clip = None
    # This is the frame after our insertion point
    src_frame = offset
    last_src_frame = 0
    # This is the frame we want to grab from the doubled clip
    doub_frame = offset * 2 - 1
    while src_frame < clip.num_frames:
        if src_frame > 0:
            interp = doubled[doub_frame]
            if out_clip is None:
                out_clip = clip[last_src_frame:src_frame] + interp
            else:
                out_clip = out_clip + clip[last_src_frame:src_frame] + interp
        last_src_frame = src_frame
        src_frame += input_cycle
        doub_frame += input_cycle * 2
    out_clip += clip[last_src_frame:]
    out_clip = out_clip.std.AssumeFPS(
        fpsnum=fps.numerator * cycle // input_cycle, fpsden=fps.denominator
    )

    # TODO: Handle other chroma samplings
    out_clip = out_clip.resize.Bicubic(
        format=vs.YUV420P16,
        width=width,
        height=height,
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
    )
    return out_clip


def next_multiple_of(multiple: int, param: int) -> int:
    rem = param % multiple
    if rem == 0:
        return param
    return param + (multiple - rem)
