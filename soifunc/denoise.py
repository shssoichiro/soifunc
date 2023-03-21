from typing import Callable, Optional

import mvsfunc
import vapoursynth as vs

from .internal import type_error, value_error

core = vs.core

__all__ = [
    "ClipLimited",
    "MCDenoise",
    "BM3DCPU",
    "BM3DCuda",
    "BM3DCuda_RTC",
    "BM3DFast",
    "MagicDenoise",
]


def ClipLimited(clip: vs.VideoNode) -> vs.VideoNode:
    """
    DEPRECATED: Use `std.Limiter` instead

    Compression introduces rounding errors and whatnot that can lead
    to some pixels in your source being outside the range of
    valid Limited range values. These are clamped to the valid
    range by the player on playback, but that means we can save
    a small amount of bitrate if we clamp them at encode time.
    This function does that.

    Recommended to use at the very end of your filter chain,
    in the final encode bit depth.
    """
    return clip.std.Limiter()


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
    import dfttest2
    import functools #functools is built in to python
    denoiser = functools.partial(dfttest2.DFTTest, sigma=1.5, backend=dfttest2.Backend.CPU)
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

    clip = denoiser(clip=interleave)

    # Every 5 frames, select the 3rd/middle frame (second digit counts from 0)
    return core.std.SelectEvery(clip, 5, 2)


def BM3DCPU(
    input,
    **kwargs,
):
    """
    BM3D wrapper, similar to mvsfunc, but using `bm3dcpu` which is about 50% faster.
    https://github.com/WolframRhodium/VapourSynth-BM3DCUDA

    See BM3DFast for usage.
    """
    return BM3DFast(input, algorithm="bm3dcpu", **kwargs)


def BM3DCuda(
    input,
    **kwargs,
):
    """
    BM3D wrapper, similar to mvsfunc, but using `bm3dcuda`.
    https://github.com/WolframRhodium/VapourSynth-BM3DCUDA

    See BM3DFast for usage.
    """
    return BM3DFast(input, algorithm="bm3dcuda", **kwargs)


def BM3DCuda_RTC(
    input,
    **kwargs,
):
    """
    BM3D wrapper, similar to mvsfunc, but using `bm3dcuda_rtc`.
    https://github.com/WolframRhodium/VapourSynth-BM3DCUDA

    See BM3DFast for usage.
    """
    return BM3DFast(input, algorithm="bm3dcuda_rtc", **kwargs)


def BM3DFast(
    input,
    algorithm="bm3dcpu",
    sigma=None,
    radius1=None,
    radius2=None,
    profile1=None,
    profile2=None,
    refine=None,
    pre=None,
    ref=None,
    psample=None,
    matrix=None,
    full=None,
    output=None,
    css=None,
    depth=None,
    sample=None,
    cu_kernel=None,
    cu_taps=None,
    cu_a1=None,
    cu_a2=None,
    cu_cplace=None,
    cd_kernel=None,
    cd_taps=None,
    cd_a1=None,
    cd_a2=None,
    cd_cplace=None,
    block_step1=None,
    bm_range1=None,
    ps_num1=None,
    ps_range1=None,
    block_step2=None,
    bm_range2=None,
    ps_num2=None,
    ps_range2=None,
    **kwargs,
):
    """
    Generic BM3DCUDA wrapper. Modified from the mvsfunc wrapper, with the arguments
    revised to match those supported by the BM3DCUDA functions.
    https://github.com/WolframRhodium/VapourSynth-BM3DCUDA
    """
    if not (
        algorithm == "bm3dcpu" or algorithm == "bm3dcuda" or algorithm == "bm3dcuda_rtc"
    ):
        raise value_error(
            "algorithm must be a library from https://github.com/WolframRhodium/VapourSynth-BM3DCUDA"
        )
    alg_namespace = getattr(core, algorithm)

    # input clip
    clip = input

    if not isinstance(input, vs.VideoNode):
        raise type_error('"input" must be a clip!')

    # Get string format parameter "matrix"
    matrix = mvsfunc.GetMatrix(input, matrix, True)

    # Get properties of input clip
    sFormat = input.format

    sColorFamily = sFormat.color_family
    mvsfunc.CheckColorFamily(sColorFamily)
    sIsRGB = sColorFamily == vs.RGB
    sIsYUV = sColorFamily == vs.YUV
    sIsGRAY = sColorFamily == vs.GRAY

    sbitPS = sFormat.bits_per_sample
    sSType = sFormat.sample_type

    sHSubS = 1 << sFormat.subsampling_w
    sVSubS = 1 << sFormat.subsampling_h

    if full is None:
        # If not set, assume limited range for YUV and Gray input
        # Assume full range for YCgCo and OPP input
        if (sIsGRAY or sIsYUV) and (
            matrix == "RGB" or matrix == "YCgCo" or matrix == "OPP"
        ):
            fulls = True
        else:
            fulls = False if sIsYUV or sIsGRAY else True
    elif not isinstance(full, int):
        raise type_error('"full" must be a bool!')
    else:
        fulls = full

    # Get properties of internal processed clip
    if psample is None:
        psample = vs.FLOAT
    elif not isinstance(psample, int):
        raise type_error('"psample" must be an int!')
    elif psample != vs.INTEGER and psample != vs.FLOAT:
        raise value_error('"psample" must be either 0 (vs.INTEGER) or 1 (vs.FLOAT)!')
    pbitPS = 16 if psample == vs.INTEGER else 32
    pSType = psample

    # Chroma sub-sampling parameters
    if css is None:
        dHSubS = sHSubS
        dVSubS = sVSubS
        css = f"{dHSubS}{dVSubS}"
    elif not isinstance(css, str):
        raise type_error('"css" must be a str!')
    else:
        if css == "444" or css == "4:4:4":
            css = "11"
        elif css == "440" or css == "4:4:0":
            css = "12"
        elif css == "422" or css == "4:2:2":
            css = "21"
        elif css == "420" or css == "4:2:0":
            css = "22"
        elif css == "411" or css == "4:1:1":
            css = "41"
        elif css == "410" or css == "4:1:0":
            css = "42"
        dHSubS = int(css[0])
        dVSubS = int(css[1])

    if cu_cplace is not None and cd_cplace is None:
        cd_cplace = cu_cplace

    # Parameters processing
    if sigma is None:
        sigma = [5.0, 5.0, 5.0]
    else:
        if isinstance(sigma, int):
            sigma = float(sigma)
            sigma = [sigma, sigma, sigma]
        elif isinstance(sigma, float):
            sigma = [sigma, sigma, sigma]
        elif isinstance(sigma, list):
            while len(sigma) < 3:
                sigma.append(sigma[len(sigma) - 1])
        else:
            raise type_error("sigma must be a float[] or an int[]!")
    if sIsGRAY:
        sigma = [sigma[0], 0, 0]
    skip = sigma[0] <= 0 and sigma[1] <= 0 and sigma[2] <= 0

    if radius1 is None:
        radius1 = 0
    elif not isinstance(radius1, int):
        raise type_error('"radius1" must be an int!')
    elif radius1 < 0:
        raise value_error('valid range of "radius1" is [0, +inf)!')
    if radius2 is None:
        radius2 = radius1
    elif not isinstance(radius2, int):
        raise type_error('"radius2" must be an int!')
    elif radius2 < 0:
        raise value_error('valid range of "radius2" is [0, +inf)!')

    if profile1 is None:
        profile1 = "fast"
    elif not isinstance(profile1, str):
        raise type_error('"profile1" must be a str!')
    if profile2 is None:
        profile2 = profile1
    elif not isinstance(profile2, str):
        raise type_error('"profile2" must be a str!')

    if refine is None:
        refine = 1
    elif not isinstance(refine, int):
        raise type_error('"refine" must be an int!')
    elif refine < 0:
        raise value_error('valid range of "refine" is [0, +inf)!')

    if output is None:
        output = 0
    elif not isinstance(output, int):
        raise type_error('"output" must be an int!')
    elif output < 0 or output > 2:
        raise value_error('valid values of "output" are 0, 1 and 2!')

    if pre is not None:
        if not isinstance(pre, vs.VideoNode):
            raise type_error('"pre" must be a clip!')
        if pre.format.id != sFormat.id:
            raise value_error(
                'clip "pre" must be of the same format as the input clip!'
            )
        if pre.width != input.width or pre.height != input.height:
            raise value_error('clip "pre" must be of the same size as the input clip!')

    if ref is not None:
        if not isinstance(ref, vs.VideoNode):
            raise type_error('"ref" must be a clip!')
        if ref.format.id != sFormat.id:
            raise value_error(
                'clip "ref" must be of the same format as the input clip!'
            )
        if ref.width != input.width or ref.height != input.height:
            raise value_error('clip "ref" must be of the same size as the input clip!')

    # Get properties of output clip
    if depth is None:
        if output == 0:
            dbitPS = sbitPS
        else:
            dbitPS = pbitPS
    elif not isinstance(depth, int):
        raise type_error('"depth" must be an int!')
    else:
        dbitPS = depth
    if sample is None:
        if depth is None:
            if output == 0:
                dSType = sSType
            else:
                dSType = pSType
        else:
            dSType = vs.FLOAT if dbitPS >= 32 else vs.INTEGER
    elif not isinstance(sample, int):
        raise type_error('"sample" must be an int!')
    elif sample != vs.INTEGER and sample != vs.FLOAT:
        raise value_error('"sample" must be either 0 (vs.INTEGER) or 1 (vs.FLOAT)!')
    else:
        dSType = sample
    if depth is None and sSType != vs.FLOAT and sample == vs.FLOAT:
        dbitPS = 32
    elif depth is None and sSType != vs.INTEGER and sample == vs.INTEGER:
        dbitPS = 16
    if dSType == vs.INTEGER and (dbitPS < 1 or dbitPS > 16):
        raise value_error(f"{dbitPS}-bit integer output is not supported!")
    if dSType == vs.FLOAT and (dbitPS != 16 and dbitPS != 32):
        raise value_error(f"{dbitPS}-bit float output is not supported!")

    if output == 0:
        fulld = fulls
    else:
        # Always full range output when output=1|output=2 (full range RGB or full range OPP)
        fulld = True

    # Convert to processed format
    # YUV/RGB input is converted to opponent color space as full range YUV
    # Gray input is converted to full range Gray
    onlyY = False
    if sIsGRAY:
        onlyY = True
        # Convert Gray input to full range Gray in processed format
        clip = mvsfunc.Depth(clip, pbitPS, pSType, fulls, True, **kwargs)
        if pre is not None:
            pre = mvsfunc.Depth(pre, pbitPS, pSType, fulls, True, **kwargs)
        if ref is not None:
            ref = mvsfunc.Depth(ref, pbitPS, pSType, fulls, True, **kwargs)
    else:
        # Convert input to full range RGB
        clip = mvsfunc.ToRGB(
            clip,
            matrix,
            pbitPS,
            pSType,
            fulls,
            cu_kernel,
            cu_taps,
            cu_a1,
            cu_a2,
            cu_cplace,
            **kwargs,
        )
        if pre is not None:
            pre = mvsfunc.ToRGB(
                pre,
                matrix,
                pbitPS,
                pSType,
                fulls,
                cu_kernel,
                cu_taps,
                cu_a1,
                cu_a2,
                cu_cplace,
                **kwargs,
            )
        if ref is not None:
            ref = mvsfunc.ToRGB(
                ref,
                matrix,
                pbitPS,
                pSType,
                fulls,
                cu_kernel,
                cu_taps,
                cu_a1,
                cu_a2,
                cu_cplace,
                **kwargs,
            )
        # Convert full range RGB to full range OPP
        clip = mvsfunc.ToYUV(
            clip,
            "OPP",
            "444",
            pbitPS,
            pSType,
            True,
            cu_kernel,
            cu_taps,
            cu_a1,
            cu_a2,
            cu_cplace,
            **kwargs,
        )
        if pre is not None:
            pre = mvsfunc.ToYUV(
                pre,
                "OPP",
                "444",
                pbitPS,
                pSType,
                True,
                cu_kernel,
                cu_taps,
                cu_a1,
                cu_a2,
                cu_cplace,
                **kwargs,
            )
        if ref is not None:
            ref = mvsfunc.ToYUV(
                ref,
                "OPP",
                "444",
                pbitPS,
                pSType,
                True,
                cu_kernel,
                cu_taps,
                cu_a1,
                cu_a2,
                cu_cplace,
                **kwargs,
            )
        # Convert OPP to Gray if only Y is processed
        srcOPP = clip
        if sigma[1] <= 0 and sigma[2] <= 0:
            onlyY = True
            clip = core.std.ShufflePlanes([clip], [0], vs.GRAY)
            if pre is not None:
                pre = core.std.ShufflePlanes([pre], [0], vs.GRAY)
            if ref is not None:
                ref = core.std.ShufflePlanes([ref], [0], vs.GRAY)

    # Basic estimate
    if ref is not None:
        # Use custom basic estimate specified by clip "ref"
        flt = ref
    elif skip:
        flt = clip
    elif radius1 < 1:
        # Apply BM3D basic estimate
        # Optional pre-filtered clip for block-matching can be specified by "pre"
        flt = alg_namespace.BM3D(
            clip,
            ref=pre,
            sigma=sigma,
            block_step=block_step1,
            bm_range=bm_range1,
        )
    else:
        # Apply V-BM3D basic estimate
        # Optional pre-filtered clip for block-matching can be specified by "pre"
        flt = alg_namespace.BM3D(
            clip,
            ref=pre,
            sigma=sigma,
            radius=radius1,
            block_step=block_step1,
            bm_range=bm_range1,
            ps_num=ps_num1,
            ps_range=ps_range1,
        ).bm3d.VAggregate(radius=radius1, sample=pSType)
        # Shuffle Y plane back if not processed
        if not onlyY and sigma[0] <= 0:
            flt = core.std.ShufflePlanes([clip, flt, flt], [0, 1, 2], vs.YUV)

    # Final estimate
    for i in range(0, refine):
        if skip:
            flt = clip
        elif radius2 < 1:
            # Apply BM3D final estimate
            flt = alg_namespace.BM3D(
                clip,
                ref=flt,
                sigma=sigma,
                block_step=block_step2,
                bm_range=bm_range2,
            )
        else:
            # Apply V-BM3D final estimate
            flt = alg_namespace.BM3D(
                clip,
                ref=flt,
                sigma=sigma,
                radius=radius2,
                block_step=block_step2,
                bm_range=bm_range2,
                ps_num=ps_num2,
                ps_range=ps_range2,
            ).bm3d.VAggregate(radius=radius2, sample=pSType)
            # Shuffle Y plane back if not processed
            if not onlyY and sigma[0] <= 0:
                flt = core.std.ShufflePlanes([clip, flt, flt], [0, 1, 2], vs.YUV)

    # Convert to output format
    if sIsGRAY:
        clip = mvsfunc.Depth(flt, dbitPS, dSType, True, fulld, **kwargs)
    else:
        # Shuffle back to YUV if not all planes are processed
        if onlyY:
            clip = core.std.ShufflePlanes([flt, srcOPP, srcOPP], [0, 1, 2], vs.YUV)
        elif sigma[1] <= 0 or sigma[2] <= 0:
            clip = core.std.ShufflePlanes(
                [flt, clip if sigma[1] <= 0 else flt, clip if sigma[2] <= 0 else flt],
                [0, 1, 2],
                vs.YUV,
            )
        else:
            clip = flt
        # Convert to final output format
        if output <= 1:
            # Convert full range OPP to full range RGB
            clip = mvsfunc.ToRGB(
                clip,
                "OPP",
                pbitPS,
                pSType,
                True,
                cu_kernel,
                cu_taps,
                cu_a1,
                cu_a2,
                cu_cplace,
                **kwargs,
            )
        if output <= 0 and not sIsRGB:
            # Convert full range RGB to YUV
            clip = mvsfunc.ToYUV(
                clip,
                matrix,
                css,
                dbitPS,
                dSType,
                fulld,
                cd_kernel,
                cd_taps,
                cd_a1,
                cd_a2,
                cd_cplace,
                **kwargs,
            )
        else:
            # Depth conversion for RGB or OPP output
            clip = mvsfunc.Depth(clip, dbitPS, dSType, True, fulld, **kwargs)

    # Output
    return clip


# Clybius's magic denoise function.
#
# Uses dark magic to denoise heavy grain from videos.
# Zero parameters, only magic.
def MagicDenoise(
    clip: vs.VideoNode,
    gpu: bool = False,
) -> vs.VideoNode:
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
    return best_dfttest(
        clip,
        slocation=[(0.0, 0.8), (0.06, 1.1), (0.12, 1.0), (1.0, 1.0)],
        pmax=1000000,
        pmin=1.25,
        ftype=4,
        tbsize=3,
        ssystem=1,
        gpu=gpu,
    )


# Run dfttest using the best available plugin supported by the host system
def best_dfttest(clip: vs.VideoNode, gpu: bool = False, **kwargs) -> vs.VideoNode:
    try:
        import importlib

        dfttest2 = importlib.import_module("dfttest2")
    except ModuleNotFoundError:
        dfttest2 = None
    if dfttest2 is not None:
        backend = dfttest2.Backend.NVRTC if gpu else dfttest2.Backend.CPU
        return dfttest2.DFTTest(clip, backend=backend, **kwargs)
    else:
        return clip.dfttest.DFTTest(**kwargs)
