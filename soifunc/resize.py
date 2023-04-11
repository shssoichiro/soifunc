from __future__ import annotations

from dataclasses import dataclass
from inspect import getfullargspec
from typing import Any, cast

from vskernels import Kernel, Scaler, ScalerT, Spline36
from vsscale import SSIM, GenericScaler
from vsscale import descale as descale_iew
from vsscale import descale_detail_mask
from vstools import check_variable_format, copy_signature, join, vs

__all__ = [
    "good_resize",
    "GoodResize",
    "descale",
    "Descale",
    "descale_mask",
    "DescaleMask",
    "HybridScaler",
]


def good_resize(
    clip: vs.VideoNode,
    width: int,
    height: int,
    gpu: bool = False,
) -> vs.VideoNode:
    """High quality resizing filter"""
    from vsaa import Nnedi3

    if (width, height) == (clip.width, clip.height):
        return clip

    return Nnedi3(opencl=gpu, scaler=HybridScaler(SSIM, Spline36)).scale(
        clip, width, height
    )


def descale(
    clip: vs.VideoNode,
    width: int | None = None,
    height: int = 720,
    kernel: str | Kernel = "Bicubic",
    b: float = 0.0,
    c: float = 0.5,
    taps: int = 3,
    mask_threshold: int = 3200,
    downscale_only: bool = False,
    show_mask: bool = False,
):
    """
    DEPRECATED: Use `vsscale.descale` instead!

    Descales a clip using best practices.

    Supposedly there were other functions to do this but they turned out to be buggy,
    so this exists. It:

    - Avoids touching chroma
    - Masks native resolution elements
    - And therefore: Returns a descaled clip with native resolution elements preserved

    It does not detect the proper resolution or kernel for you.

    Passing `downscale_only = True` will only do the downscaling portion, it will not mask or re-upscale.
    This may occasionally be useful. Using this argument will return *only* the downscaled luma plane.
    """
    import warnings

    warnings.warn("Deprecated in favor of `vsscale.descale`!", DeprecationWarning)

    kernel = _get_scaler(kernel, b=b, c=c, taps=taps)

    upscaler = HybridScaler(SSIM, Spline36) if not downscale_only else None

    rescaled = descale_iew(
        clip, width, height, kernels=[kernel], upscaler=upscaler, result=True
    )

    if show_mask:
        return rescaled.error_mask

    return cast(vs.VideoNode, rescaled.out)


def descale_mask(
    src_clip: vs.VideoNode,
    descaled_clip: vs.VideoNode,
    threshold: int = 3200,
    show_mask: bool = False,
):
    """
    DEPRECATED: Use `vsscale.descale_detail_mask` instead!

    Generates a mask to preserve detail when downscaling.
    `src_clip` should be the clip prior to any descaling.
    `descaled_clip` should be the clip after descaling and rescaling.
    i.e. they should be the same resolution.

    It is generally easier to call `Descale` as a whole, but this may
    occasionally be useful on its own.
    """
    import warnings

    warnings.warn(
        "Deprecated in favor of `vsscale.descale_detail_mask`!", DeprecationWarning
    )

    return descale_detail_mask(src_clip, descaled_clip, threshold)


@dataclass
class HybridScaler(GenericScaler):
    luma_scaler: ScalerT
    chroma_scaler: ScalerT

    def __post_init__(self) -> None:
        super().__post_init__()

        self._luma = Scaler.ensure_obj(self.luma_scaler)
        self._chroma = Scaler.ensure_obj(self.chroma_scaler)

    def scale(  # type:ignore
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        assert check_variable_format(clip, self.__class__)

        luma = self._luma.scale(clip, width, height, shift, **kwargs)

        if clip.format.num_planes == 1:
            return luma

        chroma = self._chroma.scale(clip, width, height, shift, **kwargs)

        return join(luma, chroma)


def _get_scaler(scaler: ScalerT, **kwargs: Any) -> Scaler:
    scaler_cls = Scaler.from_param(scaler, _get_scaler)

    args = getfullargspec(scaler_cls).args

    clean_kwargs = {key: value for key, value in kwargs.items() if key in args}

    return scaler_cls(**clean_kwargs)


# Aliases
@copy_signature(good_resize)
def GoodResize(*args: Any, **kwargs: Any) -> vs.VideoNode:
    import warnings

    warnings.warn(
        "`GoodResize` has been deprecated in favor of `good_resize`!",
        DeprecationWarning,
    )

    return good_resize(*args, **kwargs)


@copy_signature(descale)
def Descale(*args: Any, **kwargs: Any) -> vs.VideoNode:
    import warnings

    warnings.warn(
        "`Descale` has been deprecated in favor of `descale`!", DeprecationWarning
    )

    return descale(*args, **kwargs)


@copy_signature(descale_mask)
def DescaleMask(*args: Any, **kwargs: Any) -> vs.VideoNode:
    import warnings

    warnings.warn(
        "`DescaleMask` has been deprecated in favor of `descale_mask`!",
        DeprecationWarning,
    )

    return descale_mask(*args, **kwargs)
