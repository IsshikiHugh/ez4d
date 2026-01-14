import ez4d.geometry.rotation as ezrot
import smplx
import torch

from typing import Dict, Optional
from ez4d.camera import Rt_to_T, T_to_Rt, apply_Ts_on_pts


def apply_T_on_params(
    global_orient : torch.Tensor,
    body_pose     : torch.Tensor,
    transl        : torch.Tensor,
    betas         : torch.Tensor,
    transform     : torch.Tensor,
    smpl_model    : smplx.body_models.SMPL,
):
    """
    Apply the transformation matrix `transform` on the SMPL parameters.
    And get the transformed SMPL parameters.

    ### Args
    - global_orient: torch.Tensor (B, 3)
    - body_pose: torch.Tensor (B, 23, 3)
    - transl: torch.Tensor (B, 3)
    - betas: torch.Tensor (B, 10)
    - transform: torch.Tensor (B, 4, 4) or (4, 4)
        - Transformation matrix to be applied to the SMPL parameters.
    - smpl_model: smplx.body_models.SMPL
        - The related SMPL model. Caller should be responsible for the 
          device, dtype, memory limits issues.

    ### Returns
    - new_params: Dict
        - The transformed SMPL parameters.
        - Keys: 'global_orient', 'body_pose', 'transl', 'betas'.
        - Values: torch.Tensor (B, 3), torch.Tensor (B, 23, 3), torch.Tensor (B, 3), torch.Tensor (B, 10).
    """
    # 0. Pre-check.
    if len(transform.shape) == 2:
        transform = transform[None]  # (1, 4, 4)
    B = transform.shape[0]
    assert global_orient.shape[0] == B, f'Shape of global_orient should be (B, 3) but {global_orient.shape}'
    assert body_pose.shape[0] == B, f'Shape of body_pose should be (B, 23, 3) but {body_pose.shape}'
    assert transl.shape[0] == B, f'Shape of transl should be (B, 3) but {transl.shape}'
    assert betas.shape[0] == B, f'Shape of betas should be (B, 10) but {betas.shape}'
    global_orient = global_orient.clone()
    body_pose = body_pose.clone()
    transl = transl.clone()
    betas = betas.clone()

    # 1. Apply rotation to the global orientation.
    R_mat, _ = T_to_Rt(transform)  # (B, 3, 3), (B, 3)
    orient_mat_old = ezrot.axis_angle_to_matrix(global_orient.reshape(-1, 3))  # (B, 3, 3)
    orient_mat_new = R_mat @ orient_mat_old  # (B, 3, 3)
    global_orient_new = ezrot.matrix_to_axis_angle(orient_mat_new).reshape(B, 1, 3)  # (B, 3)

    # 2. Calculate the root offset caused by the non-zero SMPL root position.
    #    Changing orientation won't affect this, so we need to manually transform
    #    that and add the delta offset to the translation.
    # TODO: make this free from full smpl forward.
    # TODO: the translation should only be affected by the betas.
    transl_free_smpl_outputs = smpl_model(
            global_orient = global_orient,
            body_pose     = body_pose,
            transl        = transl.new_zeros(B, 3),
            betas         = betas,
        )
    static_root_offset = transl_free_smpl_outputs.joints[:, 0, :].detach()  # (B, 3)
    del transl_free_smpl_outputs

    # 3. Calculate the new translation.
    #    The new translation should be influenced by both rotation and translation.
    #    And the annoying root offset issues from SMPL should be handled here.
    root_pos_old = static_root_offset + transl  # (B, 3)
    root_pos_new = apply_Ts_on_pts(Ts=transform, pts=root_pos_old[:, None, :])  # (B, 1, 3)
    transl_new = root_pos_new[:, 0, :] - static_root_offset  # (B, 3)

    new_params = {
            'global_orient' : global_orient_new,
            'body_pose'     : body_pose,
            'transl'        : transl_new,
            'betas'         : betas,
        }
    return new_params