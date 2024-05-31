from __future__ import annotations

from dataclasses import dataclass
from inspect import getfullargspec
from typing import Any

from vsaa import Nnedi3
from vskernels import KeepArScaler, Scaler, ScalerT, Spline36
from vsscale import SSIM, GenericScaler
from vstools import check_variable_format, join, vs

__all__ = [
    "good_resize",
    "GoodScaler",
    "HybridScaler",
]


@dataclass
class GoodScaler(KeepArScaler):
    """High quality resizing filter based on opinionated defaults"""

    def __init__(
        self,
        luma_scaler: ScalerT = SSIM,
        chroma_scaler: ScalerT = Spline36,
        **kwargs: Any,
    ) -> None:
        self.scaler = HybridScaler(
            SSIM.from_param(luma_scaler), Spline36.from_param(chroma_scaler)
        )
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

        return Nnedi3(opencl=kwargs.get("gpu", None), scaler=self.scaler).scale(
            clip, width, height, shift
        )


def good_resize(
    clip: vs.VideoNode,
    width: int,
    height: int,
    gpu: bool | None = None,
) -> vs.VideoNode:
    """High quality resizing filter"""
    """This is a convenience interface; use GoodScaler for more flexibility"""
    return GoodScaler().scale(clip, width, height, gpu=gpu)


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
