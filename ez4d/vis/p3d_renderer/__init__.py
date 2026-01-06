import cv2
import imageio
import torch
import numpy as np

from tqdm import tqdm
from typing import List, Optional, Union, Tuple, Any
from pathlib import Path

from ..colors import ColorPalette
from ...media.io import save_img

from .renderer import *


def render_mesh_overlay_img(
    faces       : Union[torch.Tensor, np.ndarray],
    verts       : torch.Tensor,
    K4          : List,
    img         : np.ndarray,
    output_fn   : Optional[Union[str, Path]] = None,
    device      : str = 'cuda',
    resize      : float = 1.0,
    Rt          : Optional[Tuple[torch.Tensor]] = None,
    mesh_color : Optional[Union[List[float], str]] = 'blue',
) -> Any:
    """
    Render the mesh overlay on the input video frames.

    ### Args
    - faces: Union[torch.Tensor, np.ndarray], (V, 3)
    - verts: torch.Tensor, (V, 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - img: np.ndarray, (H, W, 3)
    - output_fn: Union[str, Path] or None
        - The output file path, if None, return the rendered img.
    - fps: int, default 30
    - device: str, default 'cuda'
    - resize: float, default 1.0
        - The resize factor of the output video.
    - Rt: Tuple of Tensor, default None
        - The extrinsic camera matrix, in the form of (R, t).
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
        )[0]

    if output_fn is None:
        return frame
    else:
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
) -> Any:
    """
    Render the mesh overlay on the input video frames.

    ### Args
    - faces: Union[torch.Tensor, np.ndarray], (V, 3)
    - verts: torch.Tensor, (L, V, 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - frames: np.ndarray, (L, H, W, 3)
    - output_fn: Union[str, Path] or None
        - The output file path, if None, return the rendered frames.
    - fps: int, default 30
    - device: str, default 'cuda'
    - resize: float, default 1.0
        - The resize factor of the output video.
    - Rt: Tuple, default None
        - The extrinsic camera matrix, in the form of (R, t).
    """
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    assert len(K4) == 4, 'K4 must be a list of 4 elements.'
    assert frames.shape[0] == verts.shape[0], 'The length of frames and verts must be the same.'
    assert frames.shape[-1] == 3, 'The last dimension of frames must be 3.'
    if isinstance(mesh_color, str):
        mesh_color = ColorPalette.presets_float[mesh_color]

    # Prepare the data.
    L = frames.shape[0]
    focal_length = (K4[0] + K4[1]) / 2 # f = (fx + fy) / 2
    e, height = frames.shape[-2], frames.shape[-3]
    cx2, cy2 = int(K4[2] * 2), int(K4[3] * 2)
    # Prepare the renderer.
    renderer = Renderer(cx2, cy2, focal_length, device, faces)
    if Rt is not None:
        Rt = (to_torch(Rt[0], device), to_torch(Rt[1], device))
        renderer.create_camera(*Rt)

    if output_fn is None:
        result_frames = []
        for i in range(L):
            img = renderer.render_mesh(verts[i].to(device), frames[i], mesh_color)
            img = cv2.resize(img, (int(width * resize), int(height * resize)))
            result_frames.append(img)
        result_frames = np.stack(result_frames, axis=0)
        return result_frames
    else:
        writer = imageio.get_writer(output_fn, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
        # Render the video.
        output_seq_name = str(output_fn).split('/')[-1]
        for i in tqdm(range(L), desc=f'Rendering [{output_seq_name}]...'):
            img = renderer.render_mesh(verts[i].to(device), frames[i])
            writer.append_data(img)
            img = cv2.resize(img, (int(width * resize), int(height * resize)))
        writer.close()


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
    """
    Render the mesh on the y=0 ground plane.

    ### Args
    - faces: Union[torch.Tensor, np.ndarray], (V, 3)
    - verts: torch.Tensor, (L, V, 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - output_fn: Union[str, Path] or None
        - The output file path, if None, return the rendered frames.
    - fps: int, default 30
    - device: str, default 'cuda'
    - cam_style: str, default 'follow'
        - The camera style, 'stare', 'follow' or 'given'.
    - return_cam: bool, default False
        - Whether to return the camera parameters.
    - Rt: Tuple of Tensor, default None
        - Only used when cam_style is 'given'.
        - The extrinsic camera matrix, in the form of (R, t).
    - mesh_color: Union[List[float], str], default 'blue'
        - The color of the mesh.
    """
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    assert len(K4) == 4, 'K4 must be a list of 4 elements.'
    if isinstance(mesh_color, str):
        mesh_color = ColorPalette.presets_float[mesh_color]
    mesh_color = to_torch(mesh_color, device)

    # Prepare the data.
    L = verts.shape[0]
    focal_length = (K4[0] + K4[1]) / 2 # f = (fx + fy) / 2
    cx2, cy2 = int(K4[2] * 2), int(K4[3] * 2)

    # Prepare the renderer.
    renderer = Renderer(cx2, cy2, focal_length, device, faces)

    # Prepare the ground mesh.
    motion_center = [(verts[..., axis].max() + verts[..., axis].min()).item() / 2 for axis in [0, 1, 2]]  # (3,)
    motion_radius = max([(verts[..., axis] - motion_center[axis]).abs().max() for axis in [0, 2]]).item()  # (2,)

    renderer.set_ground(
        motion_radius * 2 * 1.2,
        motion_center[0],
        motion_center[2],
    )

    # Prepare the camera trajectory.
    if cam_style == 'given':
        # TODO: wrap the function to lights.
        assert Rt is not None, 'Rt must be provided when cam_style is "given".'
        R, t = to_torch(Rt[0], device), to_torch(Rt[1], device)
        if R.ndim == 2:
            R = R[None].repeat(L, 1, 1)
        if t.ndim == 1:
            t = t[None].repeat(L, 1)
        light_offsets = t - to_torch(motion_center, device=t.device)[None]
        light_offsets = 5 * light_offsets / torch.norm(light_offsets, dim=-1, keepdim=True) * 3.0
        light_locations = to_torch(motion_center, device=t.device)[None] + light_offsets
        lights = [PointLights(device=device, location=light_locations[i][None]) for i in range(L)]
    elif cam_style == 'follow':
        R, t, lights = get_global_cameras_following(
                verts    = verts,
                offsets  = (-5.0, 1.0, 5.0),
                distance = cam_distance,
                device   = device
            )
    elif cam_style == 'stare':
        R, t, lights = get_global_cameras_staring(
                verts    = verts,
                position = (motion_center[0]-5.0, 1.0, motion_center[2]+5.0),
                distance = cam_distance,
                device   = device
            )
    else:
        raise NotImplementedError

    if kwargs.get('bypass_render', False):
        return None, (R, t)

    if output_fn is None:
        result_frames = []
        for i in range(L):
            camera = renderer.create_camera(R[i], t[i])
            img = renderer.render_with_ground(
                    verts = verts[i][None].to(device),
                    colors = mesh_color[None].float(),
                    cameras = camera,
                    lights = lights[i],
            )
            result_frames.append(img)
        result_frames = np.stack(result_frames, axis=0)
        ret_frames = result_frames
    else:
        writer = imageio.get_writer(output_fn, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
        # Render the video.
        output_seq_name = str(output_fn).split('/')[-1]
        for i in tqdm(range(L), desc=f'Rendering [{output_seq_name}]...'):
            camera = renderer.create_camera(R[i].float(), t[i].float())
            img = renderer.render_with_ground(
                    verts = verts[i][None].to(device).float(),
                    colors = mesh_color[None].float(),
                    cameras = camera,
                    lights = lights[i],
                )
            writer.append_data(img)
        writer.close()
        ret_frames = None

    if return_cam:
        return ret_frames, (R, t)
    else:
        return ret_frames
