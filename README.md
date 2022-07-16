## soifunc

Vapoursynth scripts that might be useful to someone

### Installation

#### Arch Linux

Install from [AUR](https://aur.archlinux.org/packages/vapoursynth-plugin-soifunc-git)

#### Other

Copy the Python script to your Python site-packages folder. Ensure you have the required prerequisites installed:

- [debandshit](https://github.com/Irrational-Encoding-Wizardry/vs-debandshit)
- [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb)
- [kagefunc](https://github.com/Irrational-Encoding-Wizardry/kagefunc)
- [muvsfunc](https://github.com/WolframRhodium/muvsfunc)
- [mvsfunc](https://github.com/HomeOfVapourSynthEvolution/mvsfunc)
- [znedi3](https://github.com/sekrit-twc/znedi3)
- [nnedi3_resample](https://github.com/HomeOfVapourSynthEvolution/nnedi3_resample)
- [BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA)

### Usage

Any of the functions will require an `import soifunc` prior to where they are used.

#### GoodResize

`clip = soifunc.GoodResize(clip, 1920, 1080)`

Resizes a clip to the specified dimensions using a high quality method.

For upscaling, luma is resized using `nnedi3_resample`.

For downscaling, luma is resized using `SSIM_downsample`.

Chroma is always resized using `Spline36`.

#### RetinexDeband

`clip = soifunc.RetinexDeband(clip, threshold = 16 [, showmask = False])`

High quality debanding using a retinex mask, designed to preserve details in both light and dark areas.

`threshold` controls debanding strength. `16` is a reasonable starting point. Increase as needed until satisfied.

`showmask` is an optional debugging parameter, setting this to `True` will output the mask that will be used to preserve edges.

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
