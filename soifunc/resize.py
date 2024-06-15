from __future__ import annotations

from dataclasses import dataclass
from inspect import getfullargspec
from typing import Any

from vskernels import (
    Catrom,
    EwaLanczos,
    Hermite,
    KeepArScaler,
    Scaler,
    ScalerT,
    Spline36,
)
from vsscale import SSIM, GenericScaler, Waifu2x
from vstools import check_variable_format, join, vs

__all__ = [
    "good_resize",
    "HybridScaler",
]


@dataclass
class GoodScaler(KeepArScaler):
    """High quality resizing filter based on opinionated defaults"""

    def __init__(
        self,
        luma_scaler: ScalerT,
        chroma_scaler: ScalerT,
        **kwargs: Any,
    ) -> None:
        self.scaler = HybridScaler(luma_scaler, chroma_scaler)
        super().__init__(**kwargs)

    @property
    def kernel_radius(self) -> int:
        return self.scaler.kernel_radius

    def scale_function(  # type:ignore
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        if (width, height) == (clip.width, clip.height):
            return clip

        anime = kwargs.get("anime", False)
        gpu = kwargs.get("gpu", None)
        use_waifu2x = kwargs.get("use_waifu2x", None)

        if anime and use_waifu2x:
            return Waifu2x(cuda=gpu).scale(clip, width, height, shift)
        return self.scaler.scale(clip, width, height, shift)


def good_resize(
    clip: vs.VideoNode,
    width: int,
    height: int,
    shift: tuple[float, float] = (0, 0),
    gpu: bool | None = None,
    anime: bool = False,
    use_waifu2x: bool = False,
) -> vs.VideoNode:
    """High quality resizing filter

    Parameters
    ----------
    clip: VideoNode
        Video clip to apply resizing to.
    width: int
        Target width to resize to.
    height: int
        Target height to resize to.
    shift: tuple[float, float], optional
        Horizontal and vertical amount of shift to apply.
    gpu: bool, optional
        Whether to allow usage of GPU for Waifu2x.
        Defaults to None, which will auto-select based on available mlrt and hardware.
    anime: bool, optional
        Enables scalers that are better tuned toward anime.
        Defaults to False.
    use_waifu2x: bool, optional
        Enables Waifu2x. Will fall back to EwaLanczos if this is False.
        Defaults to False, since Waifu2x can be a pain to set up.
    """

    is_upscale = clip.width < width or clip.height < height
    if anime:
        if is_upscale and not use_waifu2x:
            luma_scaler = EwaLanczos()
            chroma_scaler = Spline36()
        else:
            # w2x handles upscaling differently, so we only need to specify the downscale kernel for it
            luma_scaler = Catrom(sigmoid=True)
            chroma_scaler = Catrom(sigmoid=True)
    elif is_upscale:
        luma_scaler = EwaLanczos()
        chroma_scaler = Spline36()
    else:
        luma_scaler = SSIM(scaler=Hermite(linear=True))
        chroma_scaler = Spline36()

    return GoodScaler(luma_scaler, chroma_scaler).scale(
        clip, width, height, shift=shift, gpu=gpu, anime=anime, use_waifu2x=use_waifu2x
    )


@dataclass
class HybridScaler(GenericScaler):
    luma_scaler: ScalerT
    chroma_scaler: ScalerT

    def __post_init__(self) -> None:
        super().__post_init__()

        self._luma = Scaler.ensure_obj(self.luma_scaler)
        self._chroma = Scaler.ensure_obj(self.chroma_scaler)

    @property
    def kernel_radius(self) -> int:
        return self._luma.kernel_radius

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
