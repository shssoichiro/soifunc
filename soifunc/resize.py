from __future__ import annotations

from dataclasses import dataclass
from inspect import getfullargspec
from typing import Any

from vsaa.antialiasers.nnedi3 import Nnedi3SS
from vskernels import (
    Hermite,
    Scaler,
    ScalerT,
    Spline36,
)
from vsscale import SSIM, ArtCNN, GenericScaler
from vstools import check_variable_format, inject_self, is_gpu_available, join, vs

__all__ = [
    "good_resize",
    "HybridScaler",
]


def good_resize(
    clip: vs.VideoNode,
    width: int,
    height: int,
    shift: tuple[float, float] = (0, 0),
    gpu: bool | None = None,
    anime: bool = False,
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
        Whether to allow usage of GPU for ArtCNN.
        Defaults to None, which will auto-select based on available mlrt and hardware.
    anime: bool, optional
        Enables scalers that are better tuned toward anime.
        Defaults to False.
    """

    if gpu is None:
        gpu = is_gpu_available()

    is_upscale = clip.width < width or clip.height < height
    chroma_scaler = Spline36()
    if anime:
        if is_upscale:
            if gpu:
                luma_scaler = ArtCNN()
            else:
                luma_scaler = Nnedi3SS(scaler=Hermite(sigmoid=True))
        else:
            luma_scaler = Hermite(sigmoid=True)
    elif is_upscale:
        luma_scaler = Nnedi3SS(scaler=SSIM())
    else:
        luma_scaler = SSIM()

    return HybridScaler(luma_scaler, chroma_scaler).scale(
        clip, width, height, shift=shift
    )


@dataclass
class HybridScaler(GenericScaler):
    luma_scaler: ScalerT
    chroma_scaler: ScalerT

    def __post_init__(self) -> None:
        super().__post_init__()

        self._luma = Scaler.ensure_obj(self.luma_scaler)
        self._chroma = Scaler.ensure_obj(self.chroma_scaler)

    @inject_self.cached.property
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
