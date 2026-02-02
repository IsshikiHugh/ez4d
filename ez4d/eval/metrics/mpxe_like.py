from typing import Optional
import torch

from .utils import *


'''
All MPxE-like metrics will be implements here.

- Local Metrics: the inputs motion's translation should be removed (or may be automatically removed).
    - MPxE: call `eval_MPxE()`
    - PA-MPxE: cal `eval_PA_MPxE()`
- Global Metrics: the inputs motion's translation should be kept.
    - G-MPxE: call `eval_MPxE()`
    - W2-MPxE: call `eval_Wk_MPxE()`, and set k = 2
    - WA-MPxE: call `eval_WA_MPxE()`
'''


def eval_MPxE(
    pd      : torch.Tensor,
    gt      : torch.Tensor,
    scale   : float = m2mm,
    root_id : Optional[int] = None,
):
    '''
    Calculate the Mean Per <X> Error. <X> might be joints position (MPJPE), or vertices (MPVE).

    The results will be the sequence of MPxE of each multi-dim batch.

    ### Args
    - `pd`: torch.Tensor
        - (...B, N, 3), where B is the multi-dim batch size, N is points count in one batch
        - the predicted joints/vertices position data
    - `gt`: torch.Tensor
        - (...B, N, 3), where B is the multi-dim batch size, N is points count in one batch
        - the ground truth joints/vertices position data
    - `scale`: float, default = `m2mm`
    - `root_id`: Optional[int], default = None
        - The root joint index to remove the translation and rotation before measuring the error.
        - If None, no root alignment will be performed.
        - Common MPJPE/MPVE should be measured with root alignment.

    ### Returns
    - torch.Tensor, (...B)
    '''
    if root_id is not None:
        pd = pd.clone() - pd[..., root_id:root_id+1, :]
        gt = gt.clone() - gt[..., root_id:root_id+1, :]

    # Calculate the MPxE.
    ret = L2_disttance(pd, gt).mean(dim=-1) * scale # (...B,)
    return ret


def eval_PA_MPxE(
    pd    : torch.Tensor,
    gt    : torch.Tensor,
    scale : float = m2mm,
):
    '''
    Calculate the Procrustes-Aligned Mean Per <X> Error. <X> might be joints position (PA-MPJPE), or
    vertices (PA-MPVE). Targets will be Procrustes-aligned and then calculate the per frame MPxE.

    The results will be the sequence of MPxE of each batch.

    ### Args
    - `pd`: torch.Tensor, (...B, N, 3), where B is the multi-dim batch size, N is points count in one batch
        - The predicted joints/vertices position data
    - `gt`: torch.Tensor, (...B, N, 3), where B is the multi-dim batch size, N is points count in one batch
        - The ground truth joints/vertices position data.
    - `scale`: float, default = `m2mm`

    ### Returns
    - torch.Tensor, (...B)
    '''
    # Perform Procrustes alignment.
    pd_aligned = similarity_align_to(pd, gt) # (...B, N, 3)
    # Calculate the PA-MPxE
    return eval_MPxE(pd_aligned, gt, scale) # (...B,)


def eval_Wk_MPxE(
    pd    : torch.Tensor,
    gt    : torch.Tensor,
    scale : float = m2mm,
    k_f   : int   = 2,
):
    '''
    Calculate the first k frames aligned (World aligned) Mean Per <X> Error. <X> might be joints
    position (PA-MPJPE), or vertices (PA-MPVE). Targets will be aligned using the first k frames
    and then calculate the per frame MPxE.

    The results will be the sequence of MPxE of each batch.

    ### Args
    - `pd`: torch.Tensor, (..., L, N, 3), where L is the length of the sequence, N is points count in one batch
        - The predicted joints/vertices position data.
    - `gt`: torch.Tensor, (..., L, N, 3), where L is the length of the sequence, N is points count in one batch
        - The ground truth joints/vertices position data.
    - `scale`: float, default = `m2mm`
    - `k_f`: int, default = 2
        - The number of frames to use for alignment.

    ### Returns
    - torch.Tensor, (..., L)
    '''
    L = max(pd.shape[-3], gt.shape[-3])
    assert L >= 2, f'Length of the sequence should be at least 2, but got {L}.'
    # Perform first two alignment.
    pd_aligned = first_k_frames_align_to(pd, gt, k_f) # (..., L, N, 3)
    # Calculate the PA-MPxE
    return eval_MPxE(pd_aligned, gt, scale) # (..., L)


def eval_WA_MPxE(
    pd    : torch.Tensor,
    gt    : torch.Tensor,
    scale : float = m2mm,
):
    '''
    Calculate the all frames aligned (World All aligned) Mean Per <X> Error. <X> might be joints
    position (PA-MPJPE), or vertices (PA-MPVE). Targets will be aligned using the first k frames
    and then calculate the per frame MPxE.

    The results will be the sequence of MPxE of each batch.

    ### Args
    - `pd`: torch.Tensor, (..., L, N, 3), where L is the length of the sequence, N is points count in one batch
        - The predicted joints/vertices position data.
    - `gt`: torch.Tensor, (..., L, N, 3), where L is the length of the sequence, N is points count in one batch
        - The ground truth joints/vertices position data.
    - `scale`: float, default = `m2mm`

    ### Returns
    - torch.Tensor, (..., L)
    '''
    L_pd = pd.shape[-3]
    L_gt = gt.shape[-3]
    assert (L_pd == L_gt), f'Length of the sequence should be the same, but got {L_pd} and {L_gt}.'
    # WA_MPxE is just Wk_MPxE when k = L.
    return eval_Wk_MPxE(pd, gt, scale, L_gt)