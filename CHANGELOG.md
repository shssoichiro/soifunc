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
