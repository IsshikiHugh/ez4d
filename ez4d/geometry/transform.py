import torch
import numpy as np

from typing import Optional, Union


def T_to_Rt(
    T : Union[torch.Tensor, np.ndarray],
):
    """ Get (..., 3, 3) rotation matrix and (..., 3) translation vector from (..., 4, 4) transformation matrix. """
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T).float()
    assert T.shape[-2:] == (4, 4), f'T.shape[-2:] = {T.shape[-2:]}'

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    return R, t


def Rt_to_T(
    R : Union[torch.Tensor, np.ndarray],
    t : Union[torch.Tensor, np.ndarray],
):
    """ Get (..., 4, 4) transformation matrix from (..., 3, 3) rotation matrix and (..., 3) translation vector. """
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float()
    assert R.shape[-2:] == (3, 3), f'R should be a (..., 3, 3) matrix, but R.shape = {R.shape}'
    assert t.shape[-1] == 3, f't should be a (..., 3) vector, but t.shape = {t.shape}'
    assert R.shape[:-2] == t.shape[:-1], f'R and t should have the same shape prefix but {R.shape[:-2]} != {t.shape[:-1]}'

    T = torch.eye(4, device=R.device, dtype=R.dtype).repeat(R.shape[:-2] + (1, 1)) # (..., 4, 4)
    T[..., :3, :3] = R
    T[..., :3, 3] = t

    return T


def apply_Ts_on_pts(Ts:torch.Tensor, pts:torch.Tensor) -> torch.Tensor:
    """
    Apply transformation matrix `T` on the points `pts`.

    ### Args
    - Ts: torch.Tensor, (...B, 4, 4)
    - pts: torch.Tensor, (...B, N, 3)
    """
    assert len(pts.shape) >= 3 and pts.shape[-1] == 3, f'Shape of pts should be (...B, N, 3) but {pts.shape}'
    assert Ts.shape[-2:] == (4, 4), f'Shape of Ts should be (..., 4, 4) but {Ts.shape}'
    assert Ts.device == pts.device, f'Device of Ts and pts should be the same but {Ts.device} != {pts.device}'

    R = Ts[..., :3, :3]  # (...B, 3, 3)
    t = Ts[..., :3, 3]   # (...B, 3)
    ret_pts = torch.einsum('...ij,...nj->...ni', R, pts) + t[..., None, :]

    return ret_pts


def apply_T_on_pts(T:torch.Tensor, pts:torch.Tensor) -> torch.Tensor:
    """
    Apply transformation matrix `T` on the points `pts`.

    ### Args
    - T: torch.Tensor, (4, 4)
    - pts: torch.Tensor, (B, N, 3) or (N, 3)
    """
    assert len(T.shape) == 2 and T.shape[-2:] == (4, 4), f'Shape of T should be (4, 4) but {T.shape}'
    unbatched = len(pts.shape) == 2
    if unbatched:
        pts = pts[None]
    ret = apply_Ts_on_pts(T[None], pts)
    return ret.squeeze(0) if unbatched else ret