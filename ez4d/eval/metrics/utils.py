import torch
import torch.nn.functional as F
from ez4d.geometry.transform import align_pcl, similarity_align_to


m2mm = 1000.0


def L2_disttance(x:torch.Tensor, y:torch.Tensor):
    """
    Calculate the L2 error across the last dim of the input tensors.

    ### Args
    - `x`: torch.Tensor, shape (..., D)
    - `y`: torch.Tensor, shape (..., D)

    ### Returns
    - torch.Tensor, shape (...)
    """
    return (x - y).norm(dim=-1)


def first_k_frames_align_to(
    S1  : torch.Tensor,
    S2  : torch.Tensor,
    k_f : int,
    fixed_scale : bool = False,
):
    """
    Compute the transformation between the first trajectory segment of S1 and S2, and use
    the transformation to align S1 to S2.

    The code was modified from [SLAHMR](https://github.com/vye16/slahmr/blob/58518fec991877bc4911e260776589185b828fe9/slahmr/eval/tools.py#L68-L81).

    ### Args
    - `S1`: torch.Tensor, shape (..., L, N, 3)
    - `S2`: torch.Tensor, shape (..., L, N, 3)
    - `k_f`: int
        - The number of frames to use for alignment.
    - `fixed_scale`: bool, default = False
        - Whether to fix the scale of the transformation.
        
    ### Returns
    - `S1_aligned`: torch.Tensor, shape (..., L, N, 3)
        - The aligned S1.
    """
    assert (len(S1.shape) >= 3 and len(S2.shape) >= 3), 'The input tensors must have at least 3 dimensions.'
    original_shape = S1.shape  # (..., L, N, 3)
    L, N, _ = original_shape[-3:]
    S1 = S1.reshape(-1, L, N, 3)  # (B, L, N, 3)
    S2 = S2.reshape(-1, L, N, 3)  # (B, L, N, 3)
    B = S1.shape[0]

    # 1. Prepare the clouds to be aligned.
    S1_first = S1[:, :k_f, :, :].reshape(B, -1, 3)  # (B, 1, k_f * N, 3)
    S2_first = S2[:, :k_f, :, :].reshape(B, -1, 3)  # (B, 1, k_f * N, 3)

    # 2. Get the transformation to perform the alignment.
    s_first, R_first, t_first = align_pcl(
            X = S1_first,
            Y = S2_first,
            fixed_scale = fixed_scale,
        )  # (B, 1), (B, 3, 3), (B, 3)
    s_first = s_first.reshape(B, 1, 1, 1)  # (B, 1, 1, 1)
    t_first = t_first.reshape(B, 1, 1, 3)  # (B, 1, 1, 3)

    # 3. Perform the alignment on the whole sequence.
    S1_aligned = s_first * torch.einsum('Bij,BLNj->BLNi', R_first, S1) + t_first # (B, L, N, 3)
    S1_aligned = S1_aligned.reshape(original_shape)  # (..., L, N, 3)
    return S1_aligned  # (..., L, N, 3)