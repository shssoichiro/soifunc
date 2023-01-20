## soifunc

Vapoursynth scripts that might be useful to someone

### Installation

#### Arch Linux

Install from [AUR](https://aur.archlinux.org/packages/vapoursynth-plugin-soifunc-git)

#### Other

First install the required plugins which are not available in pip:

- [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb)
- [kagefunc](https://github.com/Irrational-Encoding-Wizardry/kagefunc)
- [muvsfunc](https://github.com/WolframRhodium/muvsfunc)
- [havsfunc](https://github.com/WolframRhodium/havsfunc)
- [mvsfunc](https://github.com/HomeOfVapourSynthEvolution/mvsfunc)
- [znedi3](https://github.com/sekrit-twc/znedi3)
- [nnedi3_resample](https://github.com/HomeOfVapourSynthEvolution/nnedi3_resample)
- [BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA)

Install from pip:

```bash
pip install soifunc
```

Or the latest git version:

```bash
pip install git+https://github.com/shssoichiro/soifunc.git
```

### Usage

Any of the functions will require an `import soifunc` prior to where they are used.

#### GoodResize

`clip = soifunc.GoodResize(clip, 1920, 1080)`

Resizes a clip to the specified dimensions using a high quality method.

For upscaling, luma is resized using `nnedi3_resample`.

For downscaling, luma is resized using `SSIM_downsample`.

Chroma is always resized using `Spline36`.

**If this filter causes your video to produce a blank output**, see this issue: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny/issues/14

Additional Params:

- `gpu`: Whether to use the OpenCL version of supported dependencies (currently applies to upscaling).
- `device`: Sets target OpenCL device.

#### RetinexDeband

`clip = soifunc.RetinexDeband(clip, threshold = 16 [, showmask = False])`

High quality debanding using a retinex mask, designed to preserve details in both light and dark areas.

`threshold` controls debanding strength. `16` is a reasonable starting point. Increase as needed until satisfied.

`showmask` is an optional debugging parameter, setting this to `True` will output the mask that will be used to preserve edges.

Note that this debander does not automatically add grain.
If you need to add grain before encoding, use `kagefunc.adaptive_grain`.
If you're using AV1 grain synthesis, you _do not_ need to add grain before encoding.

#### ClipLimited

`clip = soifunc.ClipLimited(clip)`

Compression introduces rounding errors and whatnot that can lead
to some pixels in your source being outside the range of
valid Limited range values. These are clamped to the valid
range by the player on playback, but that means we can save
a small amount of bitrate if we clamp them at encode time.
This function does that.

Recommended to use at the very end of your filter chain,
in the final encode bit depth.

#### BM3DCUDA Wrappers

See [BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) for list of args.

`clip = soifunc.BM3DCPU(clip, ...args)`

`clip = soifunc.BM3DCuda(clip, ...args)`

`clip = soifunc.BM3DCuda_RTC(clip, ...args)`

Provides wrappers around the accelerated BM3D functions in BM3DCUDA, similar to the wrapper provided for the base BM3D plugin in mvsfunc.
These functions perform all necessary colorspace conversion, so they are considerably simpler to use than manually calling BM3DCuda.

#### MCDenoise

Applies motion compensation to a denoised clip to improve detail preservation.
Credit to Clybius for creating this code.

Example usage:

```python
import soifunc
import dfttest2
import functools    # functools is built in to python
denoiser = functools.partial(dfttest2.DFTTest, sigma=1.5, backend=dfttest2.Backend.CPU)
clip = soifunc.MCDenoise(clip, denoiser)
```

Params:

- `denoiser`: A function defining how to denoise the motion-compensated frames.
  Denoiser params can be added using `functools.partial`.
- `prefilter`: An optional prefiltered input clip to enable better searching for motion vectors

#### SQTGMC

This is a modification of the QTGMC function from havsfunc, but simplified.
QTGMC has 90 args and this causes both its usability and maintainability to suffer.
This version removes a majority of parameters, either baking them into a preset,
auto-detecting them based on the video source, or removing their functionality entirely.

The presets are also simplified into "slowest", "slow", "medium", "fast", and "fastest",
and match/noise presets are no longer separate from the primary preset.

Params:

- `clip`: The input video to apply deinterlacing to
- `preset`: Speed/quality tradeoff. One of "slowest", "slow", "medium", "fast", and "fastest"
  Default: "slow"
- `input_type`: Default = 0 for interlaced input.
  Settings 1 & 2 accept progressive input for deshimmer or repair.
  Frame rate of progressive source is not doubled.
  Mode 1 is for general progressive material.
  Mode 2 is designed for badly deinterlaced material.
- `tff`: Since VapourSynth only has a weak notion of field order internally,
  `tff` may have to be set. Setting `tff` to `True` means top field first
  and `False` means bottom field first. Note that the `_FieldBased` frame property,
  if present, takes precedence over `tff`.
- `fps_divisor`: 1 = Double-rate output, 2 = Single-rate output.
  Higher values can be used too (e.g. 60 fps & `fps_divisor=3` gives 20 fps output).
- `prog_sad_mask`: Only applies to `input_type=2`.
  If `prog_sad_mask` > 0.0 then blend `input_type` modes 1 and 2 based on block motion SAD.
  Higher values help recover more detail, but repair fewer artifacts.
  Reasonable range about 2.0 to 20.0, or 0.0 for no blending.
- `sigma`: Amount of noise known to be in the source,
  sensible values vary by source and denoiser, so experiment.
  Use `show_noise` to help.
- `show_noise`: Display extracted and "deinterlaced" noise rather than normal output.
  Set to `True` or `False`, or set a value (around 4 to 16) to specify
  contrast for displayed noise. Visualising noise helps to determine suitable value
  for `sigma` - want to see noise and noisy detail,
  but not too much clean structure or edges - fairly subjective.
- `grain_restore`: How much removed grain to restore before final temporal smooth.
  Retain "stable" grain and some detail.
- `noise_restore`: How much removed noise to restore after final temporal smooth.
  Retains any kind of noise.
- `border`: Pad a little vertically while processing (doesn't affect output size).
  Set `True` you see flickering on the very top or bottom line of the
  output. If you have wider edge effects than that, you should crop afterwards instead.
- `gpu`: Whether to use the OpenCL version of supported dependencies.
- `device`: Sets target OpenCL device.
