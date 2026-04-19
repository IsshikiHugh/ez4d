import os
import shutil
import numpy as np
import torch

from typing import Tuple


def find_blender() -> str:
    """Locate the Blender binary.

    Search order:
    1. ``$BLENDER_PATH`` environment variable
    2. ``shutil.which('blender')``

    Raises ``FileNotFoundError`` if Blender cannot be found.
    """
    path = os.environ.get('BLENDER_PATH')
    if path and os.path.isfile(path):
        return path
    path = shutil.which('blender')
    if path:
        return path
    raise FileNotFoundError(
        'Blender binary not found. Install Blender (>=3.0) and either '
        'add it to PATH or set the BLENDER_PATH environment variable.'
    )


# ---------------------------------------------------------------------------
# Camera convention conversion
# ---------------------------------------------------------------------------
#
# p3d_renderer stores ``self.R`` from ``look_at_rotation(pos, tgt).mT``, which
# (given PyTorch3D's row-vector convention ``x_cam = x_world @ R_pt3d + T``)
# makes ``self.R`` the column-vector world-to-camera rotation: that is,
# ``x_cam = R @ x_world + T``.  PyTorch3D camera-local axes are +X left,
# +Y up, +Z forward; Blender camera-local axes are +X right, +Y up,
# -Z forward — the axis flip is ``diag(-1, 1, -1)``.

_P3D_TO_BLENDER_FLIP = np.diag([-1.0, 1.0, -1.0])


def torch3d_Rt_to_blender_c2w(R, t) -> np.ndarray:
    """Convert p3d_renderer stored ``(R, t)`` to a Blender c2w 4x4.

    Parameters
    ----------
    R : (3, 3) array-like — the w2c rotation stored in ``p3d_renderer.Renderer.R``.
    t : (3,)   array-like — the w2c translation (``Renderer.T``).

    Returns
    -------
    c2w : (4, 4) ``np.ndarray`` suitable for ``cam_obj.matrix_world``.
    """
    R_np = _to_np(R).reshape(3, 3)
    t_np = _to_np(t).reshape(3)

    R_c2w_pt3d = R_np.T
    pos_world = -R_c2w_pt3d @ t_np
    R_c2w_blender = R_c2w_pt3d @ _P3D_TO_BLENDER_FLIP

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w_blender
    c2w[:3, 3] = pos_world
    return c2w


def blender_c2w_to_torch3d_Rt(c2w) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`torch3d_Rt_to_blender_c2w`.  Returns ``(R, t)``
    in the convention p3d_renderer expects.
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    R_c2w_blender = c2w[:3, :3]
    pos_world = c2w[:3, 3]

    R_c2w_pt3d = R_c2w_blender @ _P3D_TO_BLENDER_FLIP
    R_w2c = R_c2w_pt3d.T
    t = -R_w2c @ pos_world
    return R_w2c.astype(np.float32), t.astype(np.float32)


# ---------------------------------------------------------------------------
# Look-at in Blender convention
# ---------------------------------------------------------------------------

def look_at_blender_c2w(
    positions: torch.Tensor,
    targets: torch.Tensor,
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Compute Blender-convention c2w rotation matrices.

    The returned rotation's columns are the camera's ``(+X, +Y, +Z)`` axes
    expressed in world coordinates — suitable for ``matrix_world[:3, :3]``.

    Parameters
    ----------
    positions : (L, 3)
    targets   : (L, 3)
    up        : world up direction, default ``(0, 1, 0)``

    Returns
    -------
    R_c2w : (L, 3, 3)
    """
    up_vec = torch.tensor(up, dtype=positions.dtype, device=positions.device)
    up_vec = up_vec.unsqueeze(0).expand(len(positions), -1)

    forward = targets - positions
    forward = forward / forward.norm(dim=-1, keepdim=True)

    z_axis = -forward
    x_axis = torch.cross(up_vec, z_axis, dim=-1)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)

    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


# ---------------------------------------------------------------------------
# Camera trajectory generators
# ---------------------------------------------------------------------------

def get_cameras_following(
    verts: torch.Tensor,
    offsets: Tuple[float, float, float] = (-5.0, 1.0, 5.0),
    distance: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Camera trajectory that follows the mesh center.

    Ported from ``p3d_renderer/renderer.py:get_global_cameras_following``.

    Returns
    -------
    c2ws : (L, 4, 4)         — Blender camera-to-world matrices
    light_positions : (L, 3) — one light position per frame (world coords)
    """
    device = verts.device
    L = len(verts)
    offsets_t = torch.tensor([offsets], device=device, dtype=verts.dtype).repeat(L, 1)
    targets = verts.mean(1)

    directions = -offsets_t
    directions = directions / directions.norm(dim=-1, keepdim=True) * distance
    positions = targets - directions

    R_c2w = look_at_blender_c2w(positions, targets)
    c2ws = _c2w_from_R_and_pos(R_c2w, positions)
    light_positions = positions.detach().cpu().numpy()
    return c2ws, light_positions


def get_cameras_staring(
    verts: torch.Tensor,
    position: Tuple[float, float, float] = (-5.0, 5.0, 0.0),
    distance: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed camera position looking at the moving mesh.

    Ported from ``p3d_renderer/renderer.py:get_global_cameras_staring``.
    The light is fixed at the input ``position`` for every frame, matching
    the p3d reference — the camera itself tracks inward along the
    ``target - position`` direction to the given ``distance``.
    """
    device = verts.device
    L = len(verts)
    positions = torch.tensor([position], device=device, dtype=verts.dtype).repeat(L, 1)
    targets = verts.mean(1)

    directions = targets - positions
    directions = directions / directions.norm(dim=-1, keepdim=True) * distance
    positions = targets - directions

    R_c2w = look_at_blender_c2w(positions, targets)
    c2ws = _c2w_from_R_and_pos(R_c2w, positions)
    # Match p3d: light fixed at the original input position, not the
    # distance-adjusted camera position.
    light_positions = np.tile(np.asarray(position, dtype=np.float64)[None], (L, 1))
    return c2ws, light_positions


# ---------------------------------------------------------------------------
# Color conversion
# ---------------------------------------------------------------------------

def srgb_to_linear(c) -> list:
    """Convert sRGB components in [0, 1] to linear RGB.

    ColorPalette presets are sRGB (8-bit hex / 255), but Blender's
    Principled BSDF ``Base Color`` input is linear — feeding sRGB values
    directly washes out saturated colors relative to the p3d Phong output.
    """
    out = []
    for v in c:
        v = float(v)
        if v <= 0.04045:
            out.append(v / 12.92)
        else:
            out.append(((v + 0.055) / 1.055) ** 2.4)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _c2w_from_R_and_pos(R: torch.Tensor, pos: torch.Tensor) -> np.ndarray:
    """Assemble (L, 4, 4) c2w matrices from (L, 3, 3) rotation and (L, 3) position."""
    R_np = R.detach().cpu().numpy()
    pos_np = pos.detach().cpu().numpy()
    L = len(R_np)
    c2ws = np.zeros((L, 4, 4), dtype=np.float64)
    c2ws[:, :3, :3] = R_np
    c2ws[:, :3, 3] = pos_np
    c2ws[:, 3, 3] = 1.0
    return c2ws
