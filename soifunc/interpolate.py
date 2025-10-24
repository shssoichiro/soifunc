from __future__ import annotations

from typing import TYPE_CHECKING

import vstools
from vsscale import autoselect_backend
from vstools import vs

if TYPE_CHECKING:
    from vsmlrt import backendT

__all__ = ["rate_doubler", "decimation_fixer", "replace_dupes"]


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
    clip = clip.misc.SCDetect()
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
        backend=(backend if backend else autoselect_backend()),
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


def replace_dupes(
    clip: vs.VideoNode,
    max_length: int = 5,
    backend: backendT | None = None,
    threshold: float = 0.001,
) -> vs.VideoNode:
    """
    Detects strings of duplicate frames in a video and replaces them
    with interpolated frames from RIFE.

    Max number of continuous duplicates to detect is determined by the `max_length` parameter.
    `threshold` is the maximum average pixel difference (0-1 scale) to consider frames as duplicates.
    Lower values are stricter (frames must be more similar to be considered duplicates).
    """
    import vsmlrt

    # Store original properties
    width = clip.width
    height = clip.height
    matrix = vstools.Matrix.from_video(clip)
    transfer = vstools.Transfer.from_video(clip)
    primaries = vstools.Primaries.from_video(clip)

    # Compute frame differences using PlaneStats
    # This compares each frame with the previous one
    diff_clip = clip.std.PlaneStats(clip[0] + clip)

    # Prepare clip for RIFE (convert to RGBS and resize to multiple of 64)
    rife_clip = clip.resize.Bicubic(
        format=vs.RGBS,
        width=next_multiple_of(64, width),
        height=next_multiple_of(64, height),
    )

    # Create interpolated frames using RIFE (double the framerate)
    interpolated = vsmlrt.RIFE(
        rife_clip,
        multi=2,
        model=vsmlrt.RIFEModel.v4_25_heavy,
        backend=(backend if backend else autoselect_backend()),
    )

    # Convert interpolated frames back to original format
    interpolated = interpolated.resize.Bicubic(
        format=vs.YUV420P16,
        width=width,
        height=height,
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
    )

    # Track sequence state for lazy evaluation
    state = {"prev_len": 0}

    def select_frame(n):
        """
        Select interpolated frame if current frame is a duplicate,
        otherwise use original. Copies PlaneStatsDiff property to output
        to help users calibrate the threshold parameter.
        """
        if n == 0 or n == clip.num_frames - 1:
            state["prev_len"] = 0
            # Frame 0 and final frame are never duplicates
            # (no previous frame for 0, no next frame for final)
            output = clip[n : n + 1]
            diff_val = (
                0.0
                if n == 0
                else diff_clip.get_frame(n).props.get("PlaneStatsDiff", 1.0)
            )
            return output.std.SetFrameProp(prop="PlaneStatsDiff", floatval=diff_val)

        # Get difference from PlaneStats (lazy evaluation)
        f = diff_clip.get_frame(n)
        diff = f.props.get("PlaneStatsDiff", 1.0)

        # Determine if this is a duplicate
        if diff < threshold:
            new_len = state["prev_len"] + 1
            if new_len <= max_length:
                state["prev_len"] = new_len
                is_dupe = True
            else:
                state["prev_len"] = 0
                is_dupe = False
        else:
            state["prev_len"] = 0
            is_dupe = False

        if is_dupe:
            # Use interpolated frame between previous and current
            # If the original sequence is 0 1 2 where 0 and 1 are dupes,
            # the interpolated sequence will have 0 1 2 3 4 5
            # where 3 is the interpolated frame we want to fetch
            # to replace frame 1..

            output = interpolated[n * 2 + 1 : n * 2 + 2]
        else:
            output = clip[n : n + 1]

        # Attach PlaneStatsDiff property to output frame for threshold calibration
        return output.std.SetFrameProp(prop="PlaneStatsDiff", floatval=diff)

    # Apply frame selection with lazy evaluation
    result = clip.std.FrameEval(select_frame)

    return result


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
    clip = clip.misc.SCDetect()
    clip = clip.resize.Bicubic(
        format=vs.RGBS,
        width=next_multiple_of(64, width),
        height=next_multiple_of(64, height),
    )
    doubled = vsmlrt.RIFE(
        clip,
        model=vsmlrt.RIFEModel.v4_25_heavy,
        backend=(backend if backend else autoselect_backend()),
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
