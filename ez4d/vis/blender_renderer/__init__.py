import cv2
import imageio
import torch
import numpy as np

from typing import List, Optional, Union, Tuple, Any
from pathlib import Path

from ez4d.vis.colors import ColorPalette
from ez4d.data import to_numpy, to_torch
from ez4d.media.io import save_img

from .renderer import BlenderRenderer
from .utils import (
    blender_c2w_to_torch3d_Rt,
    get_cameras_following,
    get_cameras_staring,
    srgb_to_linear,
    torch3d_Rt_to_blender_c2w,
)


def _resolve_color(color: Union[List[float], str]) -> List[float]:
    """Resolve a color spec to a linear-RGB list that Blender's Principled
    BSDF can consume directly."""
    if isinstance(color, str):
        color = ColorPalette.presets_float[color]
    return srgb_to_linear(list(color))


def _Rt_to_blender_c2ws(R_t, t_t, L) -> np.ndarray:
    """Tile/convert a user-supplied (R, t) (p3d convention) to (L, 4, 4)
    Blender c2w matrices."""
    R_t = to_numpy(R_t)
    t_t = to_numpy(t_t)
    if R_t.ndim == 2:
        R_t = np.tile(R_t[None], (L, 1, 1))
    if t_t.ndim == 1:
        t_t = np.tile(t_t[None], (L, 1))
    return np.stack([torch3d_Rt_to_blender_c2w(R_t[i], t_t[i]) for i in range(L)], axis=0)


def _c2ws_to_torch3d_Rt(camera_c2ws: np.ndarray, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-convert Blender c2ws back to p3d-convention (R, t)."""
    Rs, ts = [], []
    for i in range(len(camera_c2ws)):
        R_i, t_i = blender_c2w_to_torch3d_Rt(camera_c2ws[i])
        Rs.append(R_i)
        ts.append(t_i)
    R_out = torch.from_numpy(np.stack(Rs)).to(device)
    t_out = torch.from_numpy(np.stack(ts)).to(device)
    return R_out, t_out


def render_mesh_overlay_img(
    faces       : Union[torch.Tensor, np.ndarray],
    verts       : torch.Tensor,
    K4          : List,
    img         : np.ndarray,
    output_fn   : Optional[Union[str, Path]] = None,
    device      : str = 'cuda',
    resize      : float = 1.0,
    Rt          : Optional[Tuple[torch.Tensor]] = None,
    mesh_color  : Optional[Union[List[float], str]] = 'blue',
    **kwargs,
) -> Any:
    """Render the mesh overlay on a single image.

    API-compatible with ``p3d_renderer.render_mesh_overlay_img``.
    """
    frame = render_mesh_overlay_video(
        faces      = faces,
        verts      = verts[None],
        K4         = K4,
        frames     = img[None],
        device     = device,
        resize     = resize,
        Rt         = Rt,
        mesh_color = mesh_color,
        **kwargs,
    )[0]

    if output_fn is None:
        return frame
    save_img(frame, output_fn)


def render_mesh_overlay_video(
    faces      : Union[torch.Tensor, np.ndarray],
    verts      : torch.Tensor,
    K4         : List,
    frames     : np.ndarray,
    output_fn  : Optional[Union[str, Path]] = None,
    fps        : int = 30,
    device     : str = 'cuda',
    resize     : float = 1.0,
    Rt         : Tuple = None,
    mesh_color : Optional[Union[List[float], str]] = 'blue',
    **kwargs,
) -> Any:
    """Render the mesh overlay on video frames.

    API-compatible with ``p3d_renderer.render_mesh_overlay_video``.

    Additional keyword arguments
    ----------------------------
    engine : str
        ``'CYCLES'`` (default) or ``'BLENDER_EEVEE'``.
    samples : int
        Render samples (default 64).
    blender_exec : str or None
        Override path to the Blender binary.
    timeout : int
        Blender subprocess timeout in seconds (default 600).
    """
    faces = to_numpy(faces)
    verts = to_numpy(verts)
    assert len(K4) == 4, 'K4 must be a list of 4 elements.'
    assert frames.shape[0] == verts.shape[0], 'The length of frames and verts must be the same.'
    assert frames.shape[-1] == 3, 'The last dimension of frames must be 3.'

    L, H, W, _ = frames.shape

    # Overlay requires the image-space intrinsics to describe a camera whose
    # image is the same size as the background frames. Render resolution is
    # derived from K4 (matching p3d) — the two must agree.
    cx2, cy2 = int(round(K4[2] * 2)), int(round(K4[3] * 2))
    assert abs(cx2 - W) <= 1 and abs(cy2 - H) <= 1, (
        f'overlay requires K4 principal point near image center: '
        f'got K4={list(K4)}, frames shape=(H={H}, W={W}).'
    )

    mesh_color = _resolve_color(mesh_color)

    Rt_c2ws = _Rt_to_blender_c2ws(Rt[0], Rt[1], L) if Rt is not None else None

    renderer = BlenderRenderer(
        blender_exec=kwargs.get('blender_exec'),
        engine=kwargs.get('engine', 'CYCLES'),
        samples=kwargs.get('samples', 64),
        device=device,
        timeout=kwargs.get('timeout', 600),
    )

    result = renderer.render_overlay(
        faces=faces,
        verts=verts,
        K4=K4,
        frames=frames,
        Rt_c2ws=Rt_c2ws,
        mesh_color=mesh_color,
        resize=resize,
    )

    if output_fn is not None:
        writer = imageio.get_writer(output_fn, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
        for i in range(len(result)):
            writer.append_data(result[i])
        writer.close()
        return None

    return result


def render_mesh_with_ground(
    faces        : Union[torch.Tensor, np.ndarray],
    verts        : torch.Tensor,
    K4           : List,
    output_fn    : Optional[Union[str, Path]] = None,
    fps          : int = 30,
    device       : str = 'cuda',
    cam_style    : str = 'follow',
    return_cam   : bool = False,
    Rt           : Optional[Tuple[torch.Tensor]] = None,
    cam_distance : float = 50,
    mesh_color   : Optional[Union[List[float], str]] = 'blue',
    **kwargs,
) -> Any:
    """Render the mesh on the y=0 ground plane.

    API-compatible with ``p3d_renderer.render_mesh_with_ground``.
    """
    return render_meshes_with_ground(
        faces_list=[faces],
        verts_list=[verts],
        K4=K4,
        output_fn=output_fn,
        fps=fps,
        device=device,
        cam_style=cam_style,
        return_cam=return_cam,
        Rt=Rt,
        cam_distance=cam_distance,
        mesh_colors=[mesh_color] if mesh_color is not None else None,
        **kwargs,
    )


def render_meshes_with_ground(
    faces_list   : List[Union[torch.Tensor, np.ndarray]],
    verts_list   : List[torch.Tensor],
    K4           : List,
    output_fn    : Optional[Union[str, Path]] = None,
    fps          : int = 30,
    device       : str = 'cuda',
    cam_style    : str = 'follow',
    return_cam   : bool = False,
    Rt           : Optional[Tuple[torch.Tensor]] = None,
    cam_distance : float = 50,
    mesh_colors  : Optional[List[Union[List[float], str]]] = None,
    **kwargs,
) -> Any:
    """Render multiple meshes together on the y=0 ground plane.

    API-compatible with ``p3d_renderer.render_meshes_with_ground``.

    Additional keyword arguments
    ----------------------------
    engine : str
        ``'CYCLES'`` (default) or ``'BLENDER_EEVEE'``.
    samples : int
        Render samples (default 64).
    blender_exec : str or None
        Override path to the Blender binary.
    timeout : int
        Blender subprocess timeout in seconds (default 600).
    """
    N = len(verts_list)
    assert N == len(faces_list), 'faces_list and verts_list must have the same length.'
    L = verts_list[0].shape[0]
    for v in verts_list:
        assert v.shape[0] == L, 'All verts must have the same number of frames.'
    assert len(K4) == 4, 'K4 must be a list of 4 elements.'

    default_colors = ['blue', 'green', 'red', 'orange', 'purple', 'pink']
    if mesh_colors is None:
        mesh_colors = [default_colors[i % len(default_colors)] for i in range(N)]
    resolved_colors = [_resolve_color(c) for c in mesh_colors]

    np_faces_list = [to_numpy(f) for f in faces_list]
    np_verts_list = [to_numpy(v) for v in verts_list]

    verts_torch = to_torch(verts_list[0], device)
    all_verts_torch = torch.cat([to_torch(v, device) for v in verts_list], dim=1)

    motion_center = [
        (all_verts_torch[..., ax].max() + all_verts_torch[..., ax].min()).item() / 2
        for ax in [0, 1, 2]
    ]
    motion_radius = max([
        (all_verts_torch[..., ax] - motion_center[ax]).abs().max()
        for ax in [0, 2]
    ]).item()

    ground_params = {
        'length': motion_radius * 2 * 1.2 * 2,
        'center_x': motion_center[0],
        'center_z': motion_center[2],
    }

    if cam_style == 'given':
        assert Rt is not None, 'Rt must be provided when cam_style is "given".'
        R_t = to_numpy(Rt[0])
        t_t = to_numpy(Rt[1])
        if R_t.ndim == 2:
            R_t = np.tile(R_t[None], (L, 1, 1))
        if t_t.ndim == 1:
            t_t = np.tile(t_t[None], (L, 1))
        camera_c2ws = np.stack([
            torch3d_Rt_to_blender_c2w(R_t[i], t_t[i])
            for i in range(L)
        ], axis=0)
        # Light at camera position (camera position = -R^T @ t for p3d convention).
        light_positions = np.stack([
            -R_t[i].T @ t_t[i] for i in range(L)
        ], axis=0)
    elif cam_style == 'follow':
        camera_c2ws, light_positions = get_cameras_following(
            verts=verts_torch,
            offsets=(-5.0, 1.0, 5.0),
            distance=cam_distance,
        )
    elif cam_style == 'stare':
        camera_c2ws, light_positions = get_cameras_staring(
            verts=verts_torch,
            position=(motion_center[0] - 5.0, 1.0, motion_center[2] + 5.0),
            distance=cam_distance,
        )
    else:
        raise NotImplementedError(f'cam_style={cam_style!r} not supported.')

    if kwargs.get('bypass_render', False):
        return None, _c2ws_to_torch3d_Rt(camera_c2ws, device)

    cx2, cy2 = int(round(K4[2] * 2)), int(round(K4[3] * 2))
    resolution = (cx2, cy2)

    renderer = BlenderRenderer(
        blender_exec=kwargs.get('blender_exec'),
        engine=kwargs.get('engine', 'CYCLES'),
        samples=kwargs.get('samples', 64),
        device=device,
        timeout=kwargs.get('timeout', 600),
    )

    ret_frames = renderer.render_with_ground(
        faces_list=np_faces_list,
        verts_list=np_verts_list,
        K4=K4,
        ground_params=ground_params,
        camera_c2ws=camera_c2ws,
        light_positions=light_positions,
        mesh_colors=resolved_colors,
        resolution=resolution,
    )

    if output_fn is not None:
        writer = imageio.get_writer(output_fn, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
        for i in range(len(ret_frames)):
            writer.append_data(ret_frames[i])
        writer.close()
        ret_frames = None

    if return_cam:
        return ret_frames, _c2ws_to_torch3d_Rt(camera_c2ws, device)
    return ret_frames
