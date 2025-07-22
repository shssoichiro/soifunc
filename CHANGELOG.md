# Changelog

## Version 0.11.3

- Use GPU backend selection logic from vs-jetpack

## Version 0.11.2

- Replace SSIM scaler with Hermite. SSIM scaler has had numerous bugs and it's not worth messing with anymore.

## Version 0.11.1

- Change the default ML runtime, since vs-mlrt on AUR doesn't provide TRT_RTX and I don't feel like figuring out how to add it
- Lazy import `BackendT` so that vs-mlrt is actually an optional dependency

## Version 0.11.0

- [Breaking] Update for vs-jetpack 0.5
- [Feature] Add some utilities for interpolating frames

## Version 0.10.1

- Use nnedi3 for upscaling by default as it is slightly sharper and has less ringing than EwaLanczos

## Version 0.10.0

- [Breaking] Remove the `use_waifu2x` parameter from `good_resize` (use the `gpu` parameter instead--this will auto-detect by default)
- Rework defaults for anime mode in `good_resize`
- Simplify `good_resize` code

## Version 0.9.2

- Refine block size selection

## Version 0.9.1

- Fix denoisers for resolutions below 1080p

## Version 0.9.0

- [Breaking] Upgrade denoise functions for the vs-jetpack 0.3 update

## Version 0.8.2

- Use CLAHE instead of Retinex for deband edge masking. Leaving function name unchanged for backwards compatibility.

## Version 0.8.1

- Fix an imports issue, whoops

## Version 0.8.0

- Remove `SQTGMC`
- Add `hqbm3d`
- Fix dependency versions, including Vapoursynth

## Version 0.7.0

- Add `anime` parameter to `good_resize`
- Update `good_resize` kernels
  - Live action will use EwaLanczos for luma upscaling, SSIM for luma downscaling, and Spline36 for chroma
  - Anime will use Waifu2x for upscaling and sigmoid Catrom for downscaling
- Fix `good_resize` `gpu` parameter to auto-detect by default
- Deprecate `SQTGMC`. Please return to using `havsfunc.QTGMC` instead.

## Version 0.6.0

- remove functions that were previously deprecated
- fixes for compat with most recent dependency versions
- expose SAD parameters in SQTGMC
- fix some presets in SQTGMC, mainly use a not-stupid nnsize
- remove dependency on havsfunc

## Version 0.5.0

- Deprecate functions that have (basically) equivalent functionality with existing functions in the interest of reducing code duplication and people trying to run the exact same functions from multiple \*funcs
- Update existing functions with updated tooling that should both run faster and be more resistant to unintended user input
- Improve some typing and make the package overall a bit more Pythonic, as well as make it a tad more typesafe in certain scenarios
- More useful and informative exceptions
- Expand some functions with functionality that can be built on in the future (i.e. presets)

## Version 0.4.2

- Fix Descale again, I need a better way to test these things before I push them

## Version 0.4.1

- Add `downscale_only` param to `Descale` function
- Expose `DescaleMask` as a separate function

## Version 0.4.0

- Deprecate `ClipLimited`, use `std.Limiter` instead
- Add `Descale` function

## Version 0.3.5

- Fix bug where SQGTMC may throw an error

## Version 0.3.4

- Fix threshold on Deband to scale correctly so that we have finer tuned debanding thresholds, which apparently never worked correctly in vs-deband if you were using neo_f3kdb.

## Version 0.3.3

- Things were still broken in the last version so I removed the vsdeband plugin entirely and now we just call neo_f3kdb directly
  - It turns out the bug in f3kdb that was the reason for the vsdeband script being needed had been fixed in neo_f3kdb a long time ago anyway.

## Version 0.3.2

- Fix compatibility with a breaking change that vs-deband made [here](https://github.com/Irrational-Encoding-Wizardry/vs-deband/commit/f9a9a9b3fed8319e0ec4c2237e6f9cd215b61619)

## Version 0.3.1

- Internal refactoring
- Fix potential syntax issue with `MagicDenoise`

## Version 0.3.0

- Add Clybius's `MagicDenoise` function

## Version 0.2.0

- Add gpu support for upscaling to `GoodResize` via `gpu` and `device` parameters.
- Add SQGTMC, a simplified version of QTGMC, also with gpu support.
- Add havsfunc as a dependency

## Version 0.1.0

- Initial release
