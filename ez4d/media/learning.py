import torch
import numpy as np

from einops import rearrange
from typing import Union, List

from ..data import to_torch

DEFAULT_IMG_MEAN_RGB = [0.485, 0.456, 0.406]
DEFAULT_IMG_STD_RGB  = [0.229, 0.224, 0.225]

def imgs_to_x(
    imgs : Union[torch.Tensor, np.ndarray],
    mean : List[float] = DEFAULT_IMG_MEAN_RGB,
    std  : List[float] = DEFAULT_IMG_STD_RGB,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Prepare the RGB images to be the input of visual backbones.
    The standard procedure includes:
    1. Scale the images to [0, 1], while they are in [0, 255].
    2. Normalize the images with the given mean and std.
    3. Change the channel order from (..., H, W, C) to (..., C, H, W).

    ### Args
    - imgs: (...B, H, W, C), in range [0, 255]
    """
    imgs, recover_type_back = to_torch(imgs, temporary=True, device=None)
    imgs = (imgs / 255.0).float()

    mean = to_torch(mean, device=imgs.device).float().reshape(3)  # (3,)
    std  = to_torch(std,  device=imgs.device).float().reshape(3)  # (3,)

    x = (imgs - mean) / std  # (..., H, W, C)
    x = rearrange(x, '... H W C -> ... C H W')  # (..., C, H, W)
    return recover_type_back(x)


def x_to_imgs(
    x    : Union[torch.Tensor, np.ndarray],
    mean : List[float] = DEFAULT_IMG_MEAN_RGB,
    std  : List[float] = DEFAULT_IMG_STD_RGB,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert the input of visual backbones back to RGB images.
    The standard procedure includes:
    1. Change the channel order from (..., C, H, W) to (..., H, W, C).
    2. Denormalize the images with the given mean and std.
    3. Scale the images to [0, 255], while they are in [0, 1].

    ### Args
    - x: (...B, C, H, W), in range [0, 255]
    """
    x, recover_type_back = to_torch(x, temporary=True, device=None)

    mean = torch.tensor(mean, device=x.device).reshape(3)  # (3,)
    std  = torch.tensor(std,  device=x.device).reshape(3)  # (3,)

    imgs = rearrange(x, '... C H W -> ... H W C')  # (..., H, W, C)
    imgs = imgs * std + mean  # (..., H, W, C)

    imgs = torch.clamp(imgs, 0.0, 1.0)
    imgs = (imgs * 255).to(torch.uint8)  # (..., H, W, C)
    return recover_type_back(imgs)