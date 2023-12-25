from __future__ import annotations

from typing import Callable, Optional

from vsdenoise import DFTTest, FilterType
from vstools import core, vs

__all__ = [
    "MCDenoise",
    "magic_denoise",
]


def MCDenoise(
    clip: vs.VideoNode,
    denoiser: Callable[..., vs.VideoNode],
    prefilter: Optional[vs.VideoNode] = None,
) -> vs.VideoNode:
    """
    Applies motion compensation to a denoised clip to improve detail preservation.
    Credit to Clybius for creating this code.

    Params:
    - `denoiser`: A function defining how to denoise the motion-compensated frames.
    Params can be added using `functools.partial`.
    - `prefilter`: An optional prefiltered input clip to enable better searching for motion vectors

    Example usage:
    ```python
    import soifunc
    from vsdenoise import DFTTest
    import functools # functools is built in to python
    denoiser = functools.partial(DFTTest().denoise, sloc=1.5, tr=1)
    clip = soifunc.MCDenoise(clip, denoiser)
    ```
    """
    prefilter = prefilter or clip
    # one level (temporal radius) is enough for MRecalculate
    super = core.mv.Super(prefilter, hpad=16, vpad=16, levels=1)
    # all levels for MAnalyse
    superfilt = core.mv.Super(clip, hpad=16, vpad=16)

    # Generate motion vectors
    backward2 = core.mv.Analyse(
        superfilt, isb=True, blksize=16, overlap=8, delta=2, search=3, truemotion=True
    )
    backward = core.mv.Analyse(
        superfilt, isb=True, blksize=16, overlap=8, search=3, truemotion=True
    )
    forward = core.mv.Analyse(
        superfilt, isb=False, blksize=16, overlap=8, search=3, truemotion=True
    )
    forward2 = core.mv.Analyse(
        superfilt, isb=False, blksize=16, overlap=8, delta=2, search=3, truemotion=True
    )

    # Recalculate for higher consistency / quality
    backward_re2 = core.mv.Recalculate(
        super, backward2, blksize=8, overlap=4, search=3, truemotion=True
    )
    backward_re = core.mv.Recalculate(
        super, backward, blksize=8, overlap=4, search=3, truemotion=True
    )
    forward_re = core.mv.Recalculate(
        super, forward, blksize=8, overlap=4, search=3, truemotion=True
    )
    forward_re2 = core.mv.Recalculate(
        super, forward2, blksize=8, overlap=4, search=3, truemotion=True
    )

    # Pixel-based motion comp
    # Generate hierarchical frames from motion vector data
    backward_comp2 = core.mv.Flow(clip, super, backward_re2)
    backward_comp = core.mv.Flow(clip, super, backward_re)
    forward_comp = core.mv.Flow(clip, super, forward_re)
    forward_comp2 = core.mv.Flow(clip, super, forward_re2)

    # Interleave the mocomp'd frames
    interleave = core.std.Interleave(
        [forward_comp2, forward_comp, clip, backward_comp, backward_comp2]
    )

    clip = denoiser(interleave)

    # Every 5 frames, select the 3rd/middle frame (second digit counts from 0)
    return core.std.SelectEvery(clip, 5, 2)


def magic_denoise(clip: vs.VideoNode) -> vs.VideoNode:
    """
    Clybius's magic denoise function.

    Uses dark magic to denoise heavy grain from videos.
    Zero parameters, only magic.
    """
    super = core.mv.Super(clip, hpad=16, vpad=16, rfilter=4)
    superSharp = core.mv.Super(clip, hpad=16, vpad=16, rfilter=4)

    backward2 = core.mv.Analyse(
        super, isb=True, blksize=16, overlap=8, delta=2, search=3, dct=6
    )
    backward = core.mv.Analyse(super, isb=True, blksize=16, overlap=8, search=3, dct=6)
    forward = core.mv.Analyse(super, isb=False, blksize=16, overlap=8, search=3, dct=6)
    forward2 = core.mv.Analyse(
        super, isb=False, blksize=16, overlap=8, delta=2, search=3, dct=6
    )

    backward2 = core.mv.Recalculate(
        super, backward2, blksize=8, overlap=4, search=3, divide=2, dct=6
    )
    backward = core.mv.Recalculate(
        super, backward, blksize=8, overlap=4, search=3, divide=2, dct=6
    )
    forward = core.mv.Recalculate(
        super, forward, blksize=8, overlap=4, search=3, divide=2, dct=6
    )
    forward2 = core.mv.Recalculate(
        super, forward2, blksize=8, overlap=4, search=3, divide=2, dct=6
    )

    backward_re2 = core.mv.Finest(backward2)
    backward_re = core.mv.Finest(backward)
    forward_re = core.mv.Finest(forward)
    forward_re2 = core.mv.Finest(forward2)

    clip = core.mv.Degrain2(
        clip,
        superSharp,
        backward_re,
        forward_re,
        backward_re2,
        forward_re2,
        thsad=220,
        thscd1=300,
    )

    return DFTTest.denoise(
        clip,
        sloc=[(0.0, 0.8), (0.06, 1.1), (0.12, 1.0), (1.0, 1.0)],
        pmax=1000000,
        pmin=1.25,
        ftype=FilterType.MULT_RANGE,
        tbsize=3,
        ssystem=1,
    )
