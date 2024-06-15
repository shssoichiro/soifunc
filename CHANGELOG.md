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
