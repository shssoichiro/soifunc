## soifunc

Vapoursynth scripts that might be useful to someone

### Installation

#### Arch Linux

Install from [AUR](https://aur.archlinux.org/packages/vapoursynth-plugin-soifunc-git)

#### Other

First install the required plugins which are not available in pip:

- [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb)
- [znedi3](https://github.com/sekrit-twc/znedi3)
- [nnedi3_resample](https://github.com/HomeOfVapourSynthEvolution/nnedi3_resample)

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

#### good_resize

`clip = soifunc.good_resize(clip, 1920, 1080)`

Resizes a clip to the specified dimensions using a high quality method.

For upscaling, luma is resized using `nnedi3_resample`.

For downscaling, luma is resized using `SSIM_downsample`.

Chroma is always resized using `Spline36`.

**If this filter causes your video to produce a blank output**, see this issue: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny/issues/14

Additional Params:

- `gpu`: Whether to use the OpenCL version of supported dependencies. Defaults to auto-detect.

#### retinex_deband

`clip = soifunc.retinex_deband(clip, threshold = 16 [, showmask = False])`

High quality debanding using a retinex mask, designed to preserve details in both light and dark areas.

`threshold` controls debanding strength. `16` is a reasonable starting point. Increase as needed until satisfied.

`showmask` is an optional debugging parameter, setting this to `True` will output the mask that will be used to preserve edges.

Note that this debander does not automatically add grain.
If you need to add grain before encoding, use `vsdeband.AddNoise`.
If you're using AV1 grain synthesis, you _do not_ need to add grain before encoding.

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


#### MagicDenoise

Clybius's magic denoise function.

Uses dark magic to denoise heavy grain from videos.
Zero parameters, only magic.

Params:

- `clip`: The input video to apply deinterlacing to
- `gpu`: Whether to use the OpenCL version of supported dependencies.
