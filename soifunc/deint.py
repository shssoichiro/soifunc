from __future__ import annotations

import multiprocessing
from enum import StrEnum
from functools import partial
from typing import Any, Optional, Union

import havsfunc  # type:ignore[import]
from vsdenoise import DFTTest, fft3d
from vsrgtools import gauss_blur
from vstools import (
    CustomValueError,
    DitherType,
    FieldBased,
    FieldBasedT,
    UnsupportedFieldBasedError,
    check_variable,
    core,
    depth,
    fallback,
    get_depth,
    padder,
    scale_value,
    vs,
)

__all__ = [
    "SQTGMC",
    "Preset",
]


class Preset(StrEnum):
    """String Enum representing a SQTGMC preset."""

    _value_: str

    @classmethod
    def _missing_(cls: type[Preset], value: Any) -> Preset | None:
        if value is None:
            return cls.SLOW

        raise CustomValueError(
            f"\"{value}\" is not a valid value for {Preset}", Preset,
            f"\"{value}\" not in {[v.value for v in cls]}"
        )

    SLOWEST = "slowest"
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"
    FASTEST = "fastest"

    # TODO: define the presets in here as opposed to in SQTGMC


def SQTGMC(
    clip: vs.VideoNode,
    preset: Preset = Preset.SLOW,
    input_type: int = 0,
    tff: FieldBasedT | None = None,
    fps_divisor: int = 1,
    prog_sad_mask: float | None = None,
    sigma: float | None = None,
    show_noise: Union[bool, float] = 0.0,
    grain_restore: float | None = None,
    noise_restore: float | None = None,
    border: bool = False,
    gpu: bool = False,
    device: int | None = None,
) -> vs.VideoNode:
    """
    Stands for Simple Quality Temporal Global Motion Compensated (Deinterlacer).

    This is a modification of the QTGMC function from havsfunc, but simplified.
    QTGMC has 90 args, causing both its usability and maintainability to suffer.
    This version removes a majority of parameters, either baking them into a preset,
    auto-detecting them based on the video source, or removing their functionality entirely.

    The presets are also simplified into "slowest", "slow", "medium", "fast", and "fastest",
    and match/noise presets are no longer separate from the primary preset.

    Parameters:

    - clip: Clip to process.

    - preset: Sets a range of defaults for different encoding speeds.
      Select from "slowest", "slow", "medium", "fast", and "fastest".
      This can be either a Preset enum or a string representing the Preset.

    - input_type: Default = 0 for interlaced input.
      Settings 1 & 2 accept progressive input for deshimmer or repair.
      Frame rate of progressive source is not doubled.
      Mode 1 is for general progressive material.
      Mode 2 is designed for badly deinterlaced material.

    - tff: Since VapourSynth only has a weak notion of field order internally,
      `tff` may have to be set. Setting `tff` to `True` means top field first
      and `False` means bottom field first. Note that the `_FieldBased` frame property,
      if present, takes precedence over `tff`.

    - fps_divisor: 1 = Double-rate output, 2 = Single-rate output.
      Higher values can be used too (e.g. 60 fps & `fps_divisor=3` gives 20 fps output).

    - `prog_sad_mask`: Only applies to `input_type=2`.
      If `prog_sad_mask` > 0.0 then blend `input_type` modes 1 and 2 based on block motion SAD.
      Higher values help recover more detail, but repair fewer artifacts.
      Reasonable range about 2.0 to 20.0, or 0.0 for no blending.

    - sigma: Amount of noise known to be in the source,
      sensible values vary by source and denoiser, so experiment.
      Use `show_noise` to help.

    - show_noise: Display extracted and "deinterlaced" noise rather than normal output.
      Set to `True` or `False`, or set a value (around 4 to 16) to specify
      contrast for displayed noise. Visualising noise helps to determine suitable value
      for `sigma` - want to see noise and noisy detail,
      but not too much clean structure or edges - fairly subjective.

    - grain_restore: How much removed grain to restore before final temporal smooth.
      Retain "stable" grain and some detail.

    - noise_restore: How much removed noise to restore after final temporal smooth.
      Retains any kind of noise.

    - border: Pad a little vertically while processing (doesn't affect output size).
      Set `True` you see flickering on the very top or bottom line of the
      output. If you have wider edge effects than that, you should crop afterwards instead.

    - gpu: Whether to use the OpenCL version of supported dependencies.

    - device: Sets target OpenCL device.
    """
    assert clip.format
    assert check_variable(clip, SQTGMC)

    if input_type != 1 and tff is None:
        tff = FieldBased.from_video(clip, strict=True)

        if not tff.is_inter:
            raise UnsupportedFieldBasedError(
                "You must set `tff` or set the FieldBased property of your input clip!",
                SQTGMC, f"FieldBased={tff.pretty_string}"
            )

    is_gray = clip.format.color_family is vs.GRAY

    bits = get_depth(clip)
    neutral = 1 << (bits - 1)

    # ---------------------------------------
    # Presets

    # Select presets / tuning
    presets = ["slowest", "slow", "medium", "fast", "fastest"]
    preset = Preset(preset)
    # TODO: Update to fully use Preset instead of just having a defined enum as we do now

    try:
        preset_num = presets.index(preset.lower())
    except ValueError:
        raise CustomValueError("`preset` choice is invalid!", SQTGMC, f"{preset} not in {presets}")

    hd = clip.height >= 720

    # Tunings only affect block size in this version
    block_size = 32 if hd else 16
    block_size2 = 32

    # Preset groups, from slowest to fastest
    tr0 = [2, 2, 2, 1, 0][preset_num]
    tr1 = [2, 2, 1, 1, 1][preset_num]
    tr2_x = [3, 2, 1, 0, 0][preset_num]
    rep0 = [4, 4, 3, 0, 0][preset_num]
    rep2 = [4, 4, 4, 3, 0][preset_num]
    nn_size = [1, 1, 5, 4, 4][preset_num]
    n_neurons = [3, 2, 1, 0, 0][preset_num]
    edi_qual = [2, 1, 1, 1, 1][preset_num]
    s_mode = [2, 2, 2, 2, 0][preset_num]
    sl_mode_x = [2, 2, 2, 2, 0][preset_num]
    sl_rad = [3, 1, 1, 1, 1][preset_num]
    sbb = [3, 1, 0, 0, 0][preset_num]
    srch_clip_pp = [3, 3, 2, 1, 0][preset_num]
    sub_pel = [2, 2, 1, 1, 1][preset_num]
    block_size = [
        block_size,
        block_size,
        block_size,
        block_size2,
        block_size2,
    ][preset_num]
    overlap = [
        block_size // 2,
        block_size // 2,
        block_size // 2,
        block_size // 4,
        block_size // 4,
    ][preset_num]
    search = [5, 4, 4, 4, 0][preset_num]
    search_param = [24, 24, 16, 16, 1][preset_num]
    pel_search = [2, 2, 1, 1, 1][preset_num]
    true_motion = [True, True, False, False, False][preset_num]
    refine_motion = [True, False, False, False, False][preset_num]
    chroma_motion = [True, True, False, False, False][preset_num]
    precise = [True, False, False, False, False][preset_num]
    fast_ma = [False, False, False, False, True][preset_num]
    prog_sad_mask = fallback(
        prog_sad_mask,
        [10.0, 10.0, 10.0, 0.0, 0.0][preset_num],
    )

    # Noise presets                             Slower      Slow       Medium     Fast      Faster
    denoiser = ["dfttest", "dfttest", "dfttest", "fft3df", "fft3df"][preset_num]
    denoise_mc = [True, True, False, False, False][preset_num]
    noise_tr = [2, 1, 1, 1, 0][preset_num]
    noise_deint = ["generate", "bob", "", "", ""][preset_num]
    stabilize_noise = [True, True, True, False, False][preset_num]

    # The basic source-match step corrects and re-runs the interpolation of the input clip.
    # So it initially uses same interpolation settings as the main preset
    th_sad1: int = 640
    th_sad2: int = 256
    th_scd1: int = 180
    th_scd2: int = 98
    Str: float = 2.0
    Amp: float = 0.0625

    # ---------------------------------------
    # Settings

    # Core defaults
    tr2 = tr2_x

    # Sharpness defaults. Sharpness default is always 1.0 (0.2 with source-match),
    # but adjusted to give roughly same sharpness for all settings
    sl_mode = sl_mode_x
    if sl_rad <= 0:
        sl_mode = 0
    spatial_sl = sl_mode in [1, 3]
    temporal_sl = sl_mode in [2, 4]
    sharpness = 0.0 if s_mode <= 0 else 1.0
    # Adjust sharpness based on other settings
    sharp_mul = 2 if temporal_sl else 1.5 if spatial_sl else 1
    # [This needs a bit more refinement]
    sharpAdj = sharpness * (
        sharp_mul * (0.2 + tr1 * 0.15 + tr2 * 0.25) + (0.1 if s_mode == 1 else 0)
    )
    if s_mode <= 0:
        sbb = 0

    # Noise processing settings
    if preset == "slowest":
        noise_process = 2
    else:
        noise_process = 0
    if grain_restore is None:
        grain_restore = [0.0, 0.7, 0.3][noise_process]
    if noise_restore is None:
        noise_restore = [0.0, 0.3, 0.1][noise_process]
    if sigma is None:
        sigma = 2.0
    if isinstance(show_noise, bool):
        show_noise = show_noise * 10
    if show_noise > 0:
        noise_process = 2
        noise_restore = 1.0
    if noise_process <= 0:
        noise_tr = 0
        grain_restore = 0.0
        noise_restore = 0.0
    total_restore = grain_restore + noise_restore
    if total_restore <= 0:
        stabilize_noise = False
    noise_td = [1, 3, 5][noise_tr]
    noise_centre = scale_value(128.5, 8, bits) if denoiser == "fft3df" else neutral

    # MVTools settings
    Lambda = (1000 if true_motion else 100) * block_size * block_size // 64
    lsad = 1200 if true_motion else 400
    p_new = 50 if true_motion else 25
    p_level = 1 if true_motion else 0

    # Miscellaneous
    if input_type < 2:
        prog_sad_mask = 0.0

    # Get maximum temporal radius needed
    max_tr = max(sl_rad if temporal_sl else 0, 1, tr1, tr2, noise_tr)

    # ---------------------------------------
    # Pre-Processing

    w = clip.width
    h = clip.height

    # Pad vertically during processing (to prevent artefacts at top & bottom edges)
    if border:
        h += 8
        clip = padder(clip, top=h // 2, bottom=h // 2)

    hpad = vpad = block_size

    # ---------------------------------------
    # Motion Analysis

    # Bob the input as a starting point for motion search clip
    if input_type <= 0:
        bobbed = clip.resize.Bob(tff=tff, filter_param_a=0, filter_param_b=0.5)
    elif input_type != 1:
        bobbed = clip.std.Convolution(matrix=[1, 2, 1], mode="v")

    search_clip = (
        search_super
    ) = b_vec1 = f_vec1 = b_vec2 = f_vec2 = b_vec3 = f_vec3 = None

    cm_planes = [0, 1, 2] if chroma_motion and not is_gray else [0]

    # The bobbed clip will shimmer due to being derived from alternating fields.
    # Temporally smooth over the neighboring frames using a binomial kernel.
    # Binomial kernels give equal weight to even and odd frames and hence average away the shimmer.
    # The two kernels used are [1 2 1] and [1 4 6 4 1] for radius 1 and 2.
    # These kernels are true Gaussian kernels, which work well as a prefilter
    # before motion analysis (hence the original name for this script)
    # Create linear weightings of neighbors first: -2 -1 0 1 2
    if not isinstance(search_clip, vs.VideoNode):
        if tr0 > 0:
            ts1 = havsfunc.AverageFrames(
                bobbed, weights=[1] * 3, scenechange=28 / 255, planes=cm_planes
            )  # 0.00  0.33  0.33  0.33  0.00
        if tr0 > 1:
            ts2 = havsfunc.AverageFrames(
                bobbed, weights=[1] * 5, scenechange=28 / 255, planes=cm_planes
            )  # 0.20  0.20  0.20  0.20  0.20

    # Combine linear weightings to give binomial weightings - TR0=0: (1), TR0=1: (1:2:1), TR0=2: (1:4:6:4:1)
    if isinstance(search_clip, vs.VideoNode):
        binomial0 = None
    elif tr0 <= 0:
        binomial0 = bobbed
    elif tr0 == 1:
        binomial0 = core.std.Merge(
            ts1, bobbed, weight=0.25 if chroma_motion or is_gray else [0.25, 0]
        )
    else:
        binomial0 = core.std.Merge(
            core.std.Merge(
                ts1, ts2, weight=0.357 if chroma_motion or is_gray else [0.357, 0]
            ),
            bobbed,
            weight=0.125 if chroma_motion or is_gray else [0.125, 0],
        )

    # Remove areas of difference between temporal blurred motion search clip and bob
    # that are not due to bob-shimmer - removes general motion blur
    if isinstance(search_clip, vs.VideoNode) or rep0 <= 0:
        repair0 = binomial0
    else:
        repair0 = SQTGMC_KeepOnlyBobShimmerFixes(binomial0, bobbed, rep0, chroma_motion)

    matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

    # Blur image and soften edges to assist in motion matching of edge blocks.
    # Blocks are matched by SAD (sum of absolute differences between blocks), but even
    # a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
    if not isinstance(search_clip, vs.VideoNode):
        if srch_clip_pp == 1:
            spatial_blur = (
                core.resize.Bilinear(repair0, w // 2, h // 2)
                .std.Convolution(matrix=matrix, planes=cm_planes)
                .resize.Bilinear(w, h)
            )
        elif srch_clip_pp >= 2:
            spatial_blur = gauss_blur(
                core.std.Convolution(repair0, matrix=matrix, planes=cm_planes), sigma=0.5
            )
            spatial_blur = core.std.Merge(
                spatial_blur,
                repair0,
                weight=0.1 if chroma_motion or is_gray else [0.1, 0],
            )
        if srch_clip_pp <= 0:
            search_clip = repair0
        elif srch_clip_pp < 3:
            search_clip = spatial_blur
        else:
            expr = "x {i3} + y < x {i3} + x {i3} - y > x {i3} - y ? ?".format(
                i3=scale_value(3, 8, bits)
            )
            tweaked = core.std.Expr(
                [repair0, bobbed], expr=expr if chroma_motion or is_gray else [expr, ""]
            )
            expr = "x {i7} + y < x {i2} + x {i7} - y > x {i2} - x 51 * y 49 * + 100 / ? ?".format(
                i7=scale_value(7, 8, bits), i2=scale_value(2, 8, bits)
            )
            search_clip = core.std.Expr(
                [spatial_blur, tweaked],
                expr=expr if chroma_motion or is_gray else [expr, ""],
            )
        search_clip = havsfunc.DitherLumaRebuild(
            search_clip, s0=Str, c=Amp, chroma=chroma_motion
        )
        if bits > 8 and fast_ma:
            search_clip = depth(search_clip, 8, dither_type=DitherType.NONE)

    super_args = dict(pel=sub_pel, hpad=hpad, vpad=vpad)
    analyse_args = dict(
        blksize=block_size,
        overlap=overlap,
        search=search,
        searchparam=search_param,
        pelsearch=pel_search,
        truemotion=true_motion,
        lambda_=Lambda,
        lsad=lsad,
        pnew=p_new,
        plevel=p_level,
        global_=True,
        dct=0,
        chroma=chroma_motion,
    )
    recalculate_args = dict(
        thsad=th_sad1 // 2,
        blksize=max(block_size // 2, 4),
        search=search,
        searchparam=search_param,
        chroma=chroma_motion,
        truemotion=true_motion,
        pnew=p_new,
        overlap=max(overlap // 2, 2),
        dct=0,
    )

    # Calculate forward and backward motion vectors from motion search clip
    if max_tr > 0:
        if not isinstance(search_super, vs.VideoNode):
            search_super = search_clip.mv.Super(
                sharp=2, chroma=chroma_motion, **super_args
            )
        if not isinstance(b_vec1, vs.VideoNode):
            b_vec1 = search_super.mv.Analyse(isb=True, delta=1, **analyse_args)
            if refine_motion:
                b_vec1 = core.mv.Recalculate(search_super, b_vec1, **recalculate_args)
        if not isinstance(f_vec1, vs.VideoNode):
            f_vec1 = search_super.mv.Analyse(isb=False, delta=1, **analyse_args)
            if refine_motion:
                f_vec1 = core.mv.Recalculate(search_super, f_vec1, **recalculate_args)
    if max_tr > 1:
        if not isinstance(b_vec2, vs.VideoNode):
            b_vec2 = core.mv.Analyse(search_super, isb=True, delta=2, **analyse_args)
            if refine_motion:
                b_vec2 = core.mv.Recalculate(search_super, b_vec2, **recalculate_args)
        if not isinstance(f_vec2, vs.VideoNode):
            f_vec2 = core.mv.Analyse(search_super, isb=False, delta=2, **analyse_args)
            if refine_motion:
                f_vec2 = core.mv.Recalculate(search_super, f_vec2, **recalculate_args)
    if max_tr > 2:
        if not isinstance(b_vec3, vs.VideoNode):
            b_vec3 = core.mv.Analyse(search_super, isb=True, delta=3, **analyse_args)
            if refine_motion:
                b_vec3 = core.mv.Recalculate(search_super, b_vec3, **recalculate_args)
        if not isinstance(f_vec3, vs.VideoNode):
            f_vec3 = core.mv.Analyse(search_super, isb=False, delta=3, **analyse_args)
            if refine_motion:
                f_vec3 = core.mv.Recalculate(search_super, f_vec3, **recalculate_args)

    # ---------------------------------------
    # Noise Processing

    # Expand fields to full frame size before extracting noise (allows use of motion vectors which are frame-sized)
    if noise_process > 0:
        if input_type > 0:
            full_clip = clip
        else:
            full_clip = clip.resize.Bob(tff=tff, filter_param_a=0, filter_param_b=1)
    if noise_tr > 0:
        # TEST chroma OK?
        full_super = full_clip.mv.Super(levels=1, chroma=False, **super_args)

    cn_planes = [0]

    if noise_process > 0:
        # Create a motion compensated temporal window around current frame and use to guide denoisers
        if not denoise_mc or noise_tr <= 0:
            noise_window = full_clip
        elif noise_tr == 1:
            noise_window = core.std.Interleave(
                [
                    core.mv.Compensate(
                        full_clip, full_super, f_vec1, thscd1=th_scd1, thscd2=th_scd2
                    ),
                    full_clip,
                    core.mv.Compensate(
                        full_clip, full_super, b_vec1, thscd1=th_scd1, thscd2=th_scd2
                    ),
                ]
            )
        else:
            noise_window = core.std.Interleave(
                [
                    core.mv.Compensate(
                        full_clip, full_super, f_vec2, thscd1=th_scd1, thscd2=th_scd2
                    ),
                    core.mv.Compensate(
                        full_clip, full_super, f_vec1, thscd1=th_scd1, thscd2=th_scd2
                    ),
                    full_clip,
                    core.mv.Compensate(
                        full_clip, full_super, b_vec1, thscd1=th_scd1, thscd2=th_scd2
                    ),
                    core.mv.Compensate(
                        full_clip, full_super, b_vec2, thscd1=th_scd1, thscd2=th_scd2
                    ),
                ]
            )
        if denoiser == "dfttest":
            dn_window = DFTTest.denoise(
                noise_window,
                sigma=sigma * 4,
                tbsize=noise_td,
                planes=cn_planes,
            )
        else:
            dn_window = fft3d(
                noise_window,
                sigma=sigma,
                planes=cn_planes,
                bt=noise_td,
                ncpu=multiprocessing.cpu_count(),
                func=SQTGMC
            )

        # Rework denoised clip to match source format - various code paths here:
        # discard the motion compensation window,
        # discard doubled lines (from point resize)
        # Also reweave to get interlaced noise if source was interlaced
        # (could keep the full frame of noise, but it will be poor quality from the point resize)
        if not denoise_mc:
            if input_type > 0:
                denoised = dn_window
            else:
                denoised = havsfunc.Weave(
                    dn_window.std.SeparateFields(tff=tff).std.SelectEvery(
                        cycle=4, offsets=[0, 3]
                    ),
                    tff=tff,
                )
        elif input_type > 0:
            if noise_tr <= 0:
                denoised = dn_window
            else:
                denoised = dn_window.std.SelectEvery(cycle=noise_td, offsets=noise_tr)
        else:
            denoised = havsfunc.Weave(
                dn_window.std.SeparateFields(tff=tff).std.SelectEvery(
                    cycle=noise_td * 4, offsets=[noise_tr * 2, noise_tr * 6 + 3]
                ),
                tff=tff,
            )

        if total_restore > 0:
            # Get actual noise from difference. Then 'deinterlace' where we have weaved noise -
            # create the missing lines of noise in various ways
            noise = core.std.MakeDiff(clip, denoised, planes=cn_planes)
            if input_type > 0:
                deint_noise = noise
            elif noise_deint == "bob":
                deint_noise = noise.resize.Bob(
                    tff=tff, filter_param_a=0, filter_param_b=0.5
                )
            elif noise_deint == "generate":
                deint_noise = SQTGMC_Generate2ndFieldNoise(noise, denoised, False, tff)

            # Motion-compensated stabilization of generated noise
            if stabilize_noise:
                noise_super = deint_noise.mv.Super(
                    sharp=2, levels=1, chroma=False, **super_args
                )
                mc_noise = core.mv.Compensate(
                    deint_noise, noise_super, b_vec1, thscd1=th_scd1, thscd2=th_scd2
                )
                expr = (
                    f"x {neutral} - abs y {neutral} - abs > x y ? 0.6 * x y + 0.2 * +"
                )
                final_noise = core.std.Expr(
                    [deint_noise, mc_noise],
                    expr=expr if is_gray else [expr, ""],
                )
            else:
                final_noise = deint_noise

    # If NoiseProcess=1 denoise input clip.
    # If NoiseProcess=2 leave noise in the clip and let the temporal blurs "denoise" it for a stronger effect
    inner_clip = denoised if noise_process == 1 else clip

    # ---------------------------------------
    # Interpolation

    # Support badly deinterlaced progressive content - drop half the fields and reweave
    # to get 1/2fps interlaced stream appropriate for QTGMC processing
    if input_type > 1:
        eedi_input = havsfunc.Weave(
            inner_clip.std.SeparateFields(tff=tff).std.SelectEvery(
                cycle=4, offsets=[0, 3]
            ),
            tff=tff,
        )
    else:
        eedi_input = inner_clip

    # Create interpolated image as starting point for output
    edi1 = SQTGMC_Interpolate(
        eedi_input,
        input_type,
        nn_size,
        n_neurons,
        edi_qual,
        tff,
        gpu,
        device,
    )

    # InputType=2,3: use motion mask to blend luma between original clip & reweaved clip
    # based on ProgSADMask setting. Use chroma from original clip in any case
    if input_type < 2:
        edi = edi1
    elif prog_sad_mask is None or prog_sad_mask <= 0:
        # prog_sad_mask will never be None but mypy is stupid and garbage and why do I use it?
        if not is_gray:
            edi = core.std.ShufflePlanes(
                [edi1, inner_clip],
                planes=[0, 1, 2],
                colorfamily=clip.format.color_family,
            )
        else:
            edi = edi1
    else:
        input_type_blend = core.mv.Mask(search_clip, b_vec1, kind=1, ml=prog_sad_mask)
        edi = core.std.MaskedMerge(inner_clip, edi1, input_type_blend, planes=0)

    # Get the max/min value for each pixel over neighboring motion-compensated frames,
    # used for temporal sharpness limiting
    if tr1 > 0 or temporal_sl:
        edi_super = edi.mv.Super(sharp=2, levels=1, **super_args)
    if temporal_sl:
        b_comp1 = core.mv.Compensate(
            edi, edi_super, b_vec1, thscd1=th_scd1, thscd2=th_scd2
        )
        f_comp1 = core.mv.Compensate(
            edi, edi_super, f_vec1, thscd1=th_scd1, thscd2=th_scd2
        )
        t_max = core.std.Expr(
            [core.std.Expr([edi, f_comp1], expr="x y max"), b_comp1], expr="x y max"
        )
        t_min = core.std.Expr(
            [core.std.Expr([edi, f_comp1], expr="x y min"), b_comp1], expr="x y min"
        )
        if sl_rad > 1:
            b_comp3 = core.mv.Compensate(
                edi, edi_super, b_vec3, thscd1=th_scd1, thscd2=th_scd2
            )
            f_comp3 = core.mv.Compensate(
                edi, edi_super, f_vec3, thscd1=th_scd1, thscd2=th_scd2
            )
            t_max = core.std.Expr(
                [core.std.Expr([t_max, f_comp3], expr="x y max"), b_comp3],
                expr="x y max",
            )
            t_min = core.std.Expr(
                [core.std.Expr([t_min, f_comp3], expr="x y min"), b_comp3],
                expr="x y min",
            )

    # ---------------------------------------
    # Create basic output

    # Use motion vectors to blur interpolated image (edi) with motion-compensated previous and next frames.
    # As above, this is done to remove shimmer from alternate frames so the same binomial kernels are used.
    # However, by using motion-compensated smoothing this time we avoid motion blur. The use of
    # MDegrain1 (motion compensated) rather than TemporalSmooth makes the weightings *look* different,
    # but they evaluate to the same values
    #
    # Create linear weightings of neighbors first
    if tr1 > 0:
        degrain1 = core.mv.Degrain1(
            edi,
            edi_super,
            b_vec1,
            f_vec1,
            thsad=th_sad1,
            thscd1=th_scd1,
            thscd2=th_scd2,
        )  # 0.00  0.33  0.33  0.33  0.00
    if tr1 > 1:
        degrain2 = core.mv.Degrain1(
            edi,
            edi_super,
            b_vec2,
            f_vec2,
            thsad=th_sad1,
            thscd1=th_scd1,
            thscd2=th_scd2,
        )  # 0.33  0.00  0.33  0.00  0.33

    # Combine linear weightings to give binomial weightings - TR1=0: (1), TR1=1: (1:2:1), TR1=2: (1:4:6:4:1)
    if tr1 <= 0:
        binomial1 = edi
    elif tr1 == 1:
        binomial1 = core.std.Merge(degrain1, edi, weight=0.25)
    else:
        binomial1 = core.std.Merge(
            core.std.Merge(degrain1, degrain2, weight=0.2), edi, weight=0.0625
        )

    repair1 = binomial1
    match = repair1
    lossed1 = match

    # ---------------------------------------
    # Resharpen / retouch output

    # Resharpen to counteract temporal blurs.
    # Little sharpening needed for source-match mode since it has already recovered sharpness from source
    if s_mode <= 0:
        resharp = lossed1
    elif s_mode == 1:
        resharp = core.std.Expr(
            [lossed1, lossed1.std.Convolution(matrix=matrix)],
            expr=f"x x y - {sharpAdj} * +",
        )
    else:
        vresharp1 = core.std.Merge(
            lossed1.std.Maximum(coordinates=[0, 1, 0, 0, 0, 0, 1, 0]),
            lossed1.std.Minimum(coordinates=[0, 1, 0, 0, 0, 0, 1, 0]),
        )
        if precise:  # Precise mode: reduce tiny overshoot
            vresharp = core.std.Expr(
                [vresharp1, lossed1],
                expr="x y < x {i1} + x y > x {i1} - x ? ?".format(
                    i1=scale_value(1, 8, bits)
                ),
            )
        else:
            vresharp = vresharp1
        resharp = core.std.Expr(
            [lossed1, vresharp.std.Convolution(matrix=matrix)],
            expr=f"x x y - {sharpAdj} * +",
        )

    thin = resharp

    # Back blend the blurred difference between sharpened & unsharpened clip,
    # before (1st) sharpness limiting (Sbb == 1,3). A small fidelity improvement
    if sbb not in [1, 3]:
        back_blend1 = thin
    else:
        back_blend1 = core.std.MakeDiff(
            thin,
            gauss_blur(
                core.std.MakeDiff(thin, lossed1, planes=0).std.Convolution(
                    matrix=matrix, planes=0
                ),
                sigma=1.5,
            ),
            planes=0,
        )

    # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
    # Occurs here (before final temporal smooth) if SLMode == 1,2.
    # This location will restrict sharpness more, but any artefacts introduced will be smoothed
    if sl_mode == 1:
        if sl_rad <= 1:
            sharp_limit1 = core.rgvs.Repair(back_blend1, edi, mode=1)
        else:
            sharp_limit1 = core.rgvs.Repair(
                back_blend1, core.rgvs.Repair(back_blend1, edi, mode=12), mode=1
            )
    elif sl_mode == 2:
        sharp_limit1 = havsfunc.mt_clamp(back_blend1, t_max, t_min, 0, 0)
    else:
        sharp_limit1 = back_blend1

    # Back blend the blurred difference between sharpened & unsharpened clip,
    # after (1st) sharpness limiting (Sbb == 2,3). A small fidelity improvement
    if sbb < 2:
        back_blend2 = sharp_limit1
    else:
        back_blend2 = core.std.MakeDiff(
            sharp_limit1,
            gauss_blur(
                core.std.MakeDiff(sharp_limit1, lossed1, planes=0).std.Convolution(
                    matrix=matrix, planes=0
                ),
                sigma=1.5,
            ),
            planes=0,
        )

    # Add back any extracted noise, prior to final temporal smooth - this will restore detail
    # that was removed as "noise" without restoring the noise itself
    # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
    if grain_restore <= 0:
        add_noise1 = back_blend2
    else:
        expr = f"x {noise_centre} - {grain_restore} * {neutral} +"
        add_noise1 = core.std.MergeDiff(
            back_blend2,
            final_noise.std.Expr(expr=expr if is_gray else [expr, ""]),
            planes=cn_planes,
        )

    # Final light linear temporal smooth for denoising
    if tr2 > 0:
        stable_super = add_noise1.mv.Super(sharp=2, levels=1, **super_args)
    if tr2 <= 0:
        stable = add_noise1
    elif tr2 == 1:
        stable = core.mv.Degrain1(
            add_noise1,
            stable_super,
            b_vec1,
            f_vec1,
            thsad=th_sad2,
            thscd1=th_scd1,
            thscd2=th_scd2,
        )
    elif tr2 == 2:
        stable = core.mv.Degrain2(
            add_noise1,
            stable_super,
            b_vec1,
            f_vec1,
            b_vec2,
            f_vec2,
            thsad=th_sad2,
            thscd1=th_scd1,
            thscd2=th_scd2,
        )
    else:
        stable = core.mv.Degrain3(
            add_noise1,
            stable_super,
            b_vec1,
            f_vec1,
            b_vec2,
            f_vec2,
            b_vec3,
            f_vec3,
            thsad=th_sad2,
            thscd1=th_scd1,
            thscd2=th_scd2,
        )

    # Remove areas of difference between final output & basic interpolated image
    # that are not bob-shimmer fixes: repairs motion blur caused by temporal smooth
    if rep2 <= 0:
        repair2 = stable
    else:
        repair2 = SQTGMC_KeepOnlyBobShimmerFixes(stable, edi, rep2, True)

    # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
    # Occurs here (after final temporal smooth) if SLMode == 3,4.
    # Allows more sharpening here, but more prone to introducing minor artefacts
    if sl_mode == 3:
        if sl_rad <= 1:
            sharp_limit2 = core.rgvs.Repair(repair2, edi, mode=1)
        else:
            sharp_limit2 = core.rgvs.Repair(
                repair2, core.rgvs.Repair(repair2, edi, mode=12), mode=1
            )
    elif sl_mode >= 4:
        sharp_limit2 = havsfunc.mt_clamp(repair2, t_max, t_min, 0, 0)
    else:
        sharp_limit2 = repair2

    lossed2 = sharp_limit2

    # Add back any extracted noise, after final temporal smooth. This will appear as noise/grain in the output
    # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
    if noise_restore <= 0:
        add_noise2 = lossed2
    else:
        expr = f"x {noise_centre} - {noise_restore} * {neutral} +"
        add_noise2 = core.std.MergeDiff(
            lossed2,
            final_noise.std.Expr(expr=expr if is_gray else [expr, ""]),
            planes=cn_planes,
        )

    # ---------------------------------------
    # Post-Processing

    # Reduce frame rate
    if fps_divisor > 1:
        decimated = add_noise2.std.SelectEvery(cycle=fps_divisor, offsets=0)
    else:
        decimated = add_noise2

    # Crop off temporary vertical padding
    if border:
        cropped = decimated.std.Crop(top=4, bottom=4)
    else:
        cropped = decimated

    # Show output of choice + settings
    if show_noise <= 0:
        output = cropped
    else:
        expr = f"x {neutral} - {show_noise} * {neutral} +"
        output = final_noise.std.Expr(expr=expr if is_gray else [expr, repr(neutral)])
    output = output.std.SetFieldBased(value=0)

    return output


def SQTGMC_KeepOnlyBobShimmerFixes(
    clip: vs.VideoNode, ref: vs.VideoNode, repair: int = 1, chroma: bool = True
) -> vs.VideoNode:
    """
    Helper function: Compare processed clip with reference clip: only allow thin, horizontal areas of difference,
                     i.e. bob shimmer fixes
    Rough algorithm: Get difference, deflate vertically by a couple of pixels or so, then inflate again.
                     Thin regions will be removed by this process. Restore remaining areas of difference
                     back to as they were in reference clip
    """
    is_gray = clip.format.color_family == vs.GRAY
    planes = [0, 1, 2] if chroma and not is_gray else [0]

    bits = get_depth(clip)
    neutral = 1 << (bits - 1)

    # `ed` is the erosion distance - how much to deflate then reflate to remove thin areas of interest:
    #   0 = minimum to 6 = maximum
    # `od` is over-dilation level  - extra inflation to ensure areas to restore back are fully caught:
    #   0 = none to 3 = one full pixel
    # If `repair` < 10, then `ed` = `repair` and `od` = 0,
    #   otherwise ed = 10s digit and od = 1s digit
    #   (nasty method, but kept for compatibility with original TGMC)
    ed = repair if repair < 10 else repair // 10
    od = 0 if repair < 10 else repair % 10

    diff = core.std.MakeDiff(ref, clip)

    coordinates = [0, 1, 0, 0, 0, 0, 1, 0]

    # Areas of positive difference
    choke1 = diff.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 2:
        choke1 = choke1.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 5:
        choke1 = choke1.std.Minimum(planes=planes, coordinates=coordinates)
    if ed % 3 != 0:
        choke1 = choke1.std.Deflate(planes=planes)
    if ed in [2, 5]:
        choke1 = choke1.std.Median(planes=planes)
    choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 1:
        choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 4:
        choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)

    # Over-dilation - extra reflation up to about 1 pixel
    if od == 1:
        choke1 = choke1.std.Inflate(planes=planes)
    elif od == 2:
        choke1 = choke1.std.Inflate(planes=planes).std.Inflate(planes=planes)
    elif od >= 3:
        choke1 = choke1.std.Maximum(planes=planes)

    # Areas of negative difference (similar to above)
    choke2 = diff.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 2:
        choke2 = choke2.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 5:
        choke2 = choke2.std.Maximum(planes=planes, coordinates=coordinates)
    if ed % 3 != 0:
        choke2 = choke2.std.Inflate(planes=planes)
    if ed in [2, 5]:
        choke2 = choke2.std.Median(planes=planes)
    choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 1:
        choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 4:
        choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)

    if od == 1:
        choke2 = choke2.std.Deflate(planes=planes)
    elif od == 2:
        choke2 = choke2.std.Deflate(planes=planes).std.Deflate(planes=planes)
    elif od >= 3:
        choke2 = choke2.std.Minimum(planes=planes)

    # Combine above areas to find those areas of difference to restore
    expr1 = f"x {scale_value(129, 8, bits)} < x y {neutral} < {neutral} y ? ?"
    expr2 = f"x {scale_value(127, 8, bits)} > x y {neutral} > {neutral} y ? ?"
    restore = core.std.Expr(
        [
            core.std.Expr(
                [diff, choke1], expr=expr1 if chroma or is_gray else [expr1, ""]
            ),
            choke2,
        ],
        expr=expr2 if chroma or is_gray else [expr2, ""],
    )
    return core.std.MergeDiff(clip, restore, planes=planes)


def SQTGMC_Generate2ndFieldNoise(
    clip: vs.VideoNode,
    interleaved: vs.VideoNode,
    chroma_noise: bool = False,
    tff: Optional[bool] = None,
) -> vs.VideoNode:
    """
    Given noise extracted from an interlaced source (i.e. the noise is interlaced),
    generate "progressive" noise with a new "field" of noise injected.
    The new noise is centered on a weighted local average and uses the difference
    between local min & max as an estimate of local variance
    """
    is_gray = clip.format.color_family == vs.GRAY
    planes = [0, 1, 2] if chroma_noise and not is_gray else [0]

    bits = get_depth(clip)
    neutral = 1 << (bits - 1)

    orig_noise = clip.std.SeparateFields(tff=tff)
    noise_max = orig_noise.std.Maximum(planes=planes).std.Maximum(
        planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0]
    )
    noise_min = orig_noise.std.Minimum(planes=planes).std.Minimum(
        planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0]
    )
    random = (
        interleaved.std.SeparateFields(tff=tff)
        .std.BlankClip(color=[neutral] * clip.format.num_planes)
        .grain.Add(var=1800, uvar=1800 if chroma_noise else 0)
    )
    expr = f"x {neutral} - y * {scale_value(256, 8, bits)} / {neutral} +"
    var_random = core.std.Expr(
        [core.std.MakeDiff(noise_max, noise_min, planes=planes), random],
        expr=expr if chroma_noise or is_gray else [expr, ""],
    )
    new_noise = core.std.MergeDiff(noise_min, var_random, planes=planes)
    return havsfunc.Weave(core.std.Interleave([orig_noise, new_noise]), tff=tff)


def SQTGMC_Interpolate(
    clip: vs.VideoNode,
    input_type: int,
    nn_size: int,
    n_neurons: int,
    edi_qual: int,
    tff: Optional[bool] = None,
    gpu: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    """
    Interpolate input clip using method given in EdiMode. Use Fallback or Bob as result if mode not in list. If ChromaEdi string if set then interpolate chroma
    separately with that method (only really useful for EEDIx). The function is used as main algorithm starting point and for first two source-match stages
    """
    is_gray = clip.format.color_family == vs.GRAY
    field = 3 if tff else 2

    if gpu:
        nnedi3 = partial(core.nnedi3cl.NNEDI3CL, field=field, device=device)
    else:
        nnedi3 = partial(core.znedi3.nnedi3, field=field)

    if input_type == 1:
        return clip

    interp = nnedi3(clip, planes=[0], nsize=nn_size, nns=n_neurons, qual=edi_qual)
    if is_gray:
        return interp
    interp_uv = nnedi3(clip, planes=[1, 2], nsize=4, nns=0, qual=1)
    return core.std.ShufflePlanes(
        [interp, interp_uv], planes=[0, 1, 2], colorfamily=clip.format.color_family
    )
