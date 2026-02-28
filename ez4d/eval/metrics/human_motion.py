import torch

from ez4d.geometry.transform import apply_T_on_pts, Rt_to_T
from .utils import align_pcl, m2mm


def eval_RTE(pd_trans, gt_trans, scale=1e2):
    """
    Compute the root-translation-error (RTE) normalized by the displacement of the ground truth trajectory.
    ### Args
    - `gt_trans`: torch.Tensor, (..., L, 3), where L is the length of the sequence
        - The ground truth translation.
    - `pd_trans`: torch.Tensor, (..., L, 3), where L is the length of the sequence
        - The predicted translation.
    - `scale`: float, default = 1e2/
        - Default value ref: https://github.com/zju3dv/GVHMR/blob/088caff492aa38c2d82cea363b78a3c65a83118f/hmr4d/utils/eval/eval_utils.py#L123
    ### Returns
    - torch.Tensor, (..., L)
        - The normalized RTE.
    """
    # Compute the global alignment
    _, R_align, t_align = align_pcl(
            Y = gt_trans,
            X = pd_trans,
            fixed_scale = True,
        )
    T_align = Rt_to_T(R_align, t_align)
    pd_trans_aligned = apply_T_on_pts(T_align, pd_trans)

    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(gt_trans, gt_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(gt_trans - pd_trans_aligned, dim=-1)  # (..., L)

    # Normalize it to the displacement
    normalized_rte = rte / disp
    return normalized_rte * scale


def eval_Jitter(joints, fps=30, scale=0.1):
    """
    Compute the jitter of the motion. 
    The jitter refers to the delta of the linear acceleration of the joints.
    ### Args
    - `joints`: torch.Tensor, (..., L, J, 3), where L is the length of the sequence
        - The 3D joints positions in global coordinates.
    - `fps`: float, default = 30
        - The frame rate of the sequence.
    - `scale`: float, default = 0.1
        - Default value ref: https://github.com/zju3dv/GVHMR/blob/088caff492aa38c2d82cea363b78a3c65a83118f/hmr4d/utils/eval/eval_utils.py#L326
    ### Returns
    - torch.Tensor, (..., L-3)
        - The jitter.
    """
    jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return jitter * scale


def eval_FS_SMPL(gt, pd, thr=1e-2, scale=m2mm):
    """
    Compute the foot sliding error for SMPL vertices.
    The foot ground contact label is computed by the threshold of 1 cm/frame by default.
    ### Args
    - `gt`: torch.Tensor, (..., 6890, 3)
        - The target vertices.
    - `pd`: torch.Tensor, (..., 6890, 3)
        - The predicted vertices.
    - `thr`: float, default = 1e-2
        - The threshold of the foot static label.
        - Default value ref: https://github.com/zju3dv/GVHMR/blob/088caff492aa38c2d82cea363b78a3c65a83118f/hmr4d/utils/eval/eval_utils.py#L329
    - `scale`: float, default = `m2mm`
        - Default value ref: https://github.com/zju3dv/GVHMR/blob/088caff492aa38c2d82cea363b78a3c65a83118f/hmr4d/utils/eval/eval_utils.py#L125
    ### Returns
    - torch.Tensor, (..., L)
        - The foot sliding error.
    """
    assert gt.shape == pd.shape
    assert gt.shape[-2] == 6890, 'SMPL vertices excepted.'

    # Foot vertices idxs.
    foot_idxs = [3216, 3387, 6617, 6787]

    # Get the foot static label from GT.
    gt_feet_loc = gt[:, foot_idxs]
    gt_feet_disp = (gt_feet_loc[1:] - gt_feet_loc[:-1]).norm(2, dim=-1)
    static_label = gt_feet_disp[:] < thr

    # Get the unexcepted feet displacement from PD.
    pd_feet_loc = pd[:, foot_idxs]
    pd_feet_disp = (pd_feet_loc[1:] - pd_feet_loc[:-1]).norm(2, dim=-1)
    unexcepted_feet_disp = pd_feet_disp[static_label]  # feet disp of feet supposed to be static

    return unexcepted_feet_disp * scale