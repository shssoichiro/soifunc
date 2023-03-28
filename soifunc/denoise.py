from __future__ import annotations

from typing import Any, Callable

from vsdenoise import DFTTest, FilterType
from vstools import CustomValueError, copy_signature, core, finalize_clip, vs

__all__ = [
    "ClipLimited",
    "MCDenoise",
    "BM3DCPU",
    "BM3DCuda",
    "BM3DCuda_RTC",
    "BM3DFast",
    "magic_denoise",
    "MagicDenoise",
]


def ClipLimited(clip: vs.VideoNode) -> vs.VideoNode:
    """
    DEPRECATED: Use `vstools.finalize_clip` instead!

    Compression introduces rounding errors and whatnot that can lead
    to some pixels in your source being outside the range of
    valid Limited range values. These are clamped to the valid
    range by the player on playback, but that means we can save
    a small amount of bitrate if we clamp them at encode time.
    This function does that.

    Recommended to use at the very end of your filter chain,
    in the final encode bit depth.
    """
    import warnings

    warnings.warn(
        "This function has been deprecated in favor of `vstools.finalize_clip`!",
        DeprecationWarning,
    )

    return finalize_clip(clip, None, func=ClipLimited)


def MCDenoise(
    clip: vs.VideoNode,
    denoiser: Callable[..., vs.VideoNode] | None = None,
    prefilter: vs.VideoNode | None = None,
) -> vs.VideoNode:
    """
    DEPRECATED: Use `vsdenoise.MVTools` instead!
    This function currently uses MVTools with sane presets internally!

    Applies motion compensation to a denoised clip to improve detail preservation.
    Credit to Clybius for creating this code.

    Params:
    - `denoiser`: A function defining how to denoise the motion-compensated frames.
    Params can be added using `functools.partial`.
    - `prefilter`: An optional prefiltered input clip to enable better searching for motion vectors

    Example usage:
    ```python
    import soifunc
    import dfttest2
    import functools #functools is built in to python
    denoiser = functools.partial(dfttest2.DFTTest, sigma=1.5, backend=dfttest2.Backend.CPU)
    clip = soifunc.MCDenoise(clip, denoiser)
    ```
    """
    import warnings

    from vsdenoise import MVTools, MVToolsPresets

    warnings.warn(
        "This function has been deprecated in favor of `vsdenoise.MVTools`!",
        DeprecationWarning,
    )

    if denoiser and prefilter is None:
        prefilter = denoiser(clip)

    mv = MVTools(prefilter or clip, **MVToolsPresets.SMDE)
    return mv.degrain(thSAD=75, ref=clip)


def BM3DCPU(clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
    """
    DEPRECATED: Use `vsdenoise.BM3DCPU.denoise` instead!

    BM3D wrapper, similar to mvsfunc, but using `bm3dcpu` which is about 50% faster.
    https://github.com/WolframRhodium/VapourSynth-BM3DCUDA

    See BM3DFast for usage.
    """
    import warnings

    warnings.warn(
        "Deprecated in favor of vsdenoise.BM3DCuda.denoise!", DeprecationWarning
    )

    from vsdenoise import BM3DCPU as BM3D_IEW

    return BM3D_IEW.denoise(clip, **kwargs)


def BM3DCuda(clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
    """
    DEPRECATED: Use `vsdenoise.BM3DCPU.denoise` instead!
    BM3D wrapper, similar to mvsfunc, but using `bm3dcuda`.
    https://github.com/WolframRhodium/VapourSynth-BM3DCUDA

    See BM3DFast for usage.
    """
    import warnings

    warnings.warn(
        "Deprecated in favor of vsdenoise.BM3DCuda.denoise!", DeprecationWarning
    )

    from vsdenoise import BM3DCuda as BM3D_IEW

    return BM3D_IEW.denoise(clip, **kwargs)


def BM3DCuda_RTC(clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
    """
    DEPRECATED: Use `vsdenoise.BM3DCPU.denoise` instead!
    BM3D wrapper, similar to mvsfunc, but using `bm3dcuda_rtc`.
    https://github.com/WolframRhodium/VapourSynth-BM3DCUDA

    See BM3DFast for usage.
    """
    import warnings

    warnings.warn(
        "Deprecated in favor of vsdenoise.BM3DCudaRTC.denoise!", DeprecationWarning
    )

    from vsdenoise import BM3DCudaRTC as BM3D_IEW

    return BM3D_IEW.denoise(clip, **kwargs)


def BM3DFast(
    clip: vs.VideoNode, algorithm: str = "bm3dcpu", **kwargs: Any
) -> vs.VideoNode:
    """
    Generic BM3DCUDA wrapper. Modified from the mvsfunc wrapper, with the arguments
    revised to match those supported by the BM3DCUDA functions.
    https://github.com/WolframRhodium/VapourSynth-BM3DCUDA
    """
    match algorithm.lower():
        case "bm3dcpu":
            return BM3DCPU(clip, **kwargs)
        case "bm3dcuda":
            return BM3DCuda(clip, **kwargs)
        case "bm3dcuda_rtc" | "bm3dcudartc":
            return BM3DCuda_RTC(clip, **kwargs)
        case _:
            raise CustomValueError(
                '"{algorithm}" is not a valid algorithm!',
                BM3DFast,
                reason='"{algorithm}" not in {algorithms}',
                algorithm=algorithm,
                algorithms=iter(["bm3dcpu", "bm3dcuda", "bm3dcuda_rtc", "bm3dcudartc"]),
            )


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


# Aliases
@copy_signature(magic_denoise)
def MagicDenoise(*args: Any, **kwargs: Any) -> vs.VideoNode:
    import warnings

    warnings.warn(
        "`MagicDenoise` has been deprecated in favor of `magic_denoise`!",
        DeprecationWarning,
    )

    return magic_denoise(*args, **kwargs)
