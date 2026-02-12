import torch
import numpy as np

from typing import Optional, Union


def T_to_Rt(
    T : Union[torch.Tensor, np.ndarray],
):
    """ 
    Get (..., 3, 3) rotation matrix and (..., 3) translation vector from (..., 4, 4) transformation matrix.

    ### Args
    - T: torch.Tensor, (..., 4, 4)

    ### Returns
    - R: torch.Tensor, (..., 3, 3)
    - t: torch.Tensor, (..., 3)
    """
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
    """ 
    Get (..., 4, 4) transformation matrix from (..., 3, 3) rotation matrix and (..., 3) translation vector.

    ### Args
    - R: torch.Tensor, (..., 3, 3)
    - t: torch.Tensor, (..., 3)

    ### Returns
    - T: torch.Tensor, (..., 4, 4)
    """
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

def R_to_T(
    R : Union[torch.Tensor, np.ndarray],
):
    """ 
    Get (..., 4, 4) transformation matrix from (..., 3, 3) rotation matrix.

    ### Args
    - R: torch.Tensor, (..., 3, 3)

    ### Returns
    - T: torch.Tensor, (..., 4, 4)
    """
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    assert R.shape[-2:] == (3, 3), f'R should be a (..., 3, 3) matrix, but R.shape = {R.shape}'
    t = torch.zeros(R.shape[:-2] + (3,), device=R.device, dtype=R.dtype) # (..., 3)
    return Rt_to_T(R, t)


def t_to_T(
    t : Union[torch.Tensor, np.ndarray],
):
    """ 
    Get (..., 4, 4) transformation matrix from (..., 3) translation vector.

    ### Args
    - t: torch.Tensor, (..., 3)

    ### Returns
    - T: torch.Tensor, (..., 4, 4)
    """
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float()
    assert t.shape[-1] == 3, f't should be a (..., 3) vector, but t.shape = {t.shape}'
    R = torch.eye(3, device=t.device, dtype=t.dtype).repeat(t.shape[:-1] + (1, 1)) # (..., 3, 3)
    return Rt_to_T(R, t)


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


def align_pcl(Y: torch.Tensor, X: torch.Tensor, weight=None, fixed_scale=False):
    """
    Align similarity transform to align X with Y using umeyama method. X' = s * R * X + t is aligned with Y.

    The code was adapted from [SLAHMR](https://github.com/vye16/slahmr/blob/58518fec991877bc4911e260776589185b828fe9/slahmr/geometry/pcl.py#L10-L60).

    ### Args
    - `Y`: torch.Tensor, shape (*, N, 3) first trajectory
    - `X`: torch.Tensor, shape (*, N, 3) second trajectory
    - `weight`: torch.Tensor, shape (*, N, 1) optional weight of valid correspondences
    - `fixed_scale`: bool, default = False

    ### Returns
    - `s` (*, 1)
    - `R` (*, 3, 3)
    - `t` (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1).to(Y.device) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1).to(Y.device)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
                torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)
                     .sum(dim=-1, keepdim=True)
                / var[..., 0]
            )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t


def similarity_align_to(
    S1 : torch.Tensor,
    S2 : torch.Tensor,
):
    """
    Computes a similarity transform (sR, t) that takes a set of 3D points S1 (3 x N)
    closest to a set of 3D points S2, where R is an 3x3 rotation matrix,
    t 3x1 translation, s scales. That is to solves the orthogonal Procrutes problem.

    The code was adapted from [WHAM](https://github.com/yohanshin/WHAM/blob/d1ade93ae83a91855902fdb8246c129c4b3b8a40/lib/eval/eval_utils.py#L201-L252).

    ### Args
    - `S1`: torch.Tensor, shape (...B, N, 3)
    - `S2`: torch.Tensor, shape (...B, N, 3)

    ### Returns
    - torch.Tensor, shape (...B, N, 3)
    """
    assert (S1.shape[-1] == 3 and S2.shape[-1] == 3), 'The last dimension of `S1` and `S2` must be 3.'
    assert (S1.shape[:-2] == S2.shape[:-2]), 'The batch size of `S1` and `S2` must be the same.'
    original_BN3 = S1.shape
    N = original_BN3[-2]
    S1 = S1.reshape(-1, N, 3) # (B', N, 3) <- (...B, N, 3)
    S2 = S2.reshape(-1, N, 3) # (B', N, 3) <- (...B, N, 3)
    B = S1.shape[0]

    S1 = S1.transpose(-1, -2) # (B', 3, N) <- (B', N, 3)
    S2 = S2.transpose(-1, -2) # (B', 3, N) <- (B', N, 3)
    _device = S2.device
    S1 = S1.to(_device)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True) # (B', 3, 1)
    mu2 = S2.mean(axis=-1, keepdims=True) # (B', 3, 1)
    X1 = S1 - mu1 # (B', 3, N)
    X2 = S2 - mu2 # (B', 3, N)

    # 2. Compute variance of X1 used for scales.
    var1 = torch.einsum('...BDN->...B', X1**2) # (B',)

    # 3. The outer product of X1 and X2.
    K = X1 @ X2.transpose(-1, -2) # (B', 3, 3)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K) # (B', 3, 3), (B', 3), (B', 3, 3)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(3, device=_device)[None].repeat(B, 1, 1) # (B', 3, 3)
    Z[:, -1, -1] *= (U @ V.transpose(-1, -2)).det().sign()

    # Construct R.
    R = V @ (Z @ U.transpose(-1, -2)) # (B', 3, 3)

    # 5. Recover scales.
    traces = [torch.trace(x)[None] for x in (R @ K)]
    scales = torch.cat(traces) / var1 # (B',)
    scales = scales[..., None, None] # (B', 1, 1)

    # 6. Recover translation.
    t = mu2 - (scales * (R @ mu1)) # (B', 3, 1)

    # 7. Error:
    S1_aligned = scales * (R @ S1) + t # (B', 3, N)

    S1_aligned = S1_aligned.transpose(-1, -2) # (B', N, 3) <- (B', 3, N)
    S1_aligned = S1_aligned.reshape(original_BN3) # (...B, N, 3)
    return S1_aligned # (...B, N, 3)