from __future__ import annotations

from vsaa.deinterlacers import NNEDI3
from vskernels import Hermite, LeftShift, Spline36, TopShift
from vsscale import ArtCNN
from vstools import (
    VariableFormatError,
    check_variable_format,
    is_gpu_available,
    join,
    vs,
)

__all__ = [
    "good_resize",
]


def good_resize(
    clip: vs.VideoNode,
    width: int,
    height: int,
    shift: tuple[TopShift | list[TopShift], LeftShift | list[LeftShift]] = (0, 0),
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

    # We've ended up where the only special case is anime + upscale + GPU enabled
    if anime and is_upscale and gpu:
        luma_scaler = ArtCNN(scaler=Hermite(sigmoid=True))
    elif is_upscale:
        luma_scaler = NNEDI3(scaler=Hermite(sigmoid=True))
    else:
        luma_scaler = Hermite(sigmoid=True)

    if not check_variable_format(clip, "good_resize"):
        raise VariableFormatError("Invalid clip format for good_resize")

    luma = luma_scaler.scale(clip, width, height, shift)

    # Grayscale doesn't need chroma processing
    if clip.format.num_planes == 1:
        return luma

    chroma = chroma_scaler.scale(clip, width, height, shift)

    return join(luma, chroma)
