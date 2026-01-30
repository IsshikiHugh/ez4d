import torch
import numpy as np

from typing import Optional, Union
from .data import to_torch, to_numpy

# Those functions are also useful for camera operations.
from .geometry.transform import (
    T_to_Rt,
    Rt_to_T,
    apply_Ts_on_pts,
    apply_T_on_pts,
)


def perspective_projection(
    points         : torch.Tensor,
    focal_length   : Optional[torch.Tensor] = None,
    camera_center  : Optional[torch.Tensor] = None,
    translation    : Optional[torch.Tensor] = None,
    rotation       : Optional[torch.Tensor] = None,
    K              : Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/utils/geometry.py#L64-L102

    ### Args
        - points: torch.Tensor, (B, N, 3)
            - The input 3D points.
        - focal_length: Optional[torch.Tensor], (B, 2)
            - The focal length in pixels. Can be none if K is provided.
        - camera_center: Optional[torch.Tensor], (B, 2)
            - The camera center in pixels. Can be none if K is provided.
        - translation: Optional[torch.Tensor], (B, 3)
            - The 3D camera translation.
        - rotation: Optional[torch.Tensor], (B, 3, 3)
            - The camera rotation.
        - K: Optional[torch.Tensor], (B, 3, 3)
            - The intrinsic camera matrix. Can be none if focal_length and camera_center are provided.

    ### Returns
        - torch.Tensor, (B, N, 2)
            - The projection of the input points.
    """
    B = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(B, -1, -1)
    if translation is None:
        translation = torch.zeros(B, 3, device=points.device, dtype=points.dtype)
    if camera_center is None:
        camera_center = torch.zeros(B, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    if K is None:
        K = torch.zeros([B, 3, 3], device=points.device, dtype=points.dtype)
        K[:,   0,  0] = focal_length[:, 0]
        K[:,   1,  1] = focal_length[:, 1]
        K[:,   2,  2] = 1.
        K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij, bkj -> bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij, bkj -> bki', K, projected_points)

    return projected_points[:, :, :-1]


def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000, img_size=224):
    """
    Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Copied from: https://github.com/nkolot/SPIN/blob/2476c436013055be5cb3905e4e4ecfa86966fac3/utils/geometry.py#L94-L132

    ### Args
        - S: shape = (25, 3)
            - 3D joint locations.
        - joints: shape = (25, 3)
            - 2D joint locations and confidence.
    ### Returns
        - shape = (3,)
            - Camera translation vector.
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_camera_trans(
    S            : torch.Tensor,
    joints_2d    : torch.Tensor,
    focal_length : float = 5000.,
    img_size     : float = 224.,
    conf_thre    : float = 4.,
):
    """
    Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Modified from: https://github.com/nkolot/SPIN/blob/2476c436013055be5cb3905e4e4ecfa86966fac3/utils/geometry.py#L135-L157

    ### Args
        - S: torch.Tensor, shape = (B, J, 3)
            - 3D joint locations.
        - joints: torch.Tensor, shape = (B, J, 3)
            - Ground truth 2D joint locations and confidence.
        - focal_length: float
        - img_size: float
        - conf_thre: float
            - Confidence threshold to judge whether we use gt_kp2d or that from OpenPose.

    ### Returns
        - torch.Tensor, shape = (B, 3)
            - Camera translation vectors.
    """
    device = S.device
    B = len(S)

    S = to_numpy(S)
    joints_2d = to_numpy(joints_2d)
    joints_conf = joints_2d[:, :, -1]   # (B, J)
    joints_2d   = joints_2d[:, :, :-1]  # (B, J, 2)
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(B):
        conf_i = joints_conf[i]
        # When the ground truth joints are not enough, use all the joints.
        if conf_i[25:].sum() < conf_thre:
            S_i = S[i]
            joints_i = joints_2d[i]
        else:
            S_i = S[i, 25:]
            conf_i = joints_conf[i, 25:]
            joints_i = joints_2d[i, 25:]


        trans[i] = estimate_translation_np(S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size)
    return torch.from_numpy(trans).to(device)