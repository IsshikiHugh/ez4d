"""BlenderRenderer — manages data serialization, Blender subprocess invocation,
and rendered-image collection."""

import cv2
import imageio
import json
import subprocess
import tempfile
import numpy as np

from pathlib import Path
from typing import List, Optional, Tuple

from .utils import find_blender


_WORKER_SCRIPT = str(Path(__file__).resolve().parent / 'blender_worker.py')


class BlenderRenderer:
    """Render meshes by driving Blender as a background subprocess.

    Parameters
    ----------
    blender_exec : str or None
        Path to the ``blender`` binary.  Auto-detected if *None*.
    engine : str
        ``'CYCLES'`` (path-traced, high quality) or ``'BLENDER_EEVEE'``
        (rasterised, fast).
    samples : int
        Render samples (Cycles) or TAA samples (EEVEE).
    device : str
        ``'cuda'`` selects GPU rendering in Cycles; ``'cpu'`` forces CPU.
    timeout : int
        Subprocess timeout in seconds.  Long sequences may need increasing.
    """

    def __init__(
        self,
        blender_exec: Optional[str] = None,
        engine: str = 'CYCLES',
        samples: int = 64,
        device: str = 'cuda',
        timeout: int = 600,
    ):
        self.blender_exec = blender_exec or find_blender()
        self.engine = engine
        self.samples = samples
        self.use_gpu = 'cuda' in str(device)
        self.timeout = int(timeout)

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_overlay(
        self,
        faces: np.ndarray,
        verts: np.ndarray,
        K4: List[float],
        frames: np.ndarray,
        Rt_c2ws: Optional[np.ndarray],
        mesh_color: List[float],
        resize: float,
    ) -> np.ndarray:
        """Render mesh overlaid on background frames.

        Parameters
        ----------
        faces : (F, 3)
        verts : (L, V, 3)
        K4 : [fx, fy, cx, cy]
        frames : (L, H, W, 3) uint8
        Rt_c2ws : (L, 4, 4) Blender c2w matrices, or None for identity
        mesh_color : [r, g, b] linear floats in [0, 1]
        resize : resize factor (applied after compositing)

        Returns
        -------
        result : np.ndarray, (L, H', W', 3) uint8
        """
        L, H, W, _ = frames.shape

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / 'output'
            output_dir.mkdir()

            mesh_npz = str(tmpdir / 'mesh_0.npz')
            _save_mesh(mesh_npz, faces, verts)

            camera_Rt = Rt_c2ws.tolist() if Rt_c2ws is not None else None

            params = {
                'mode': 'overlay',
                'engine': self.engine,
                'samples': self.samples,
                'use_gpu': self.use_gpu,
                'K4': list(map(float, K4)),
                'resolution': [int(W), int(H)],
                'num_frames': int(L),
                'mesh_npz_paths': [mesh_npz],
                'mesh_colors': [list(map(float, mesh_color))],
                'camera_Rt': camera_Rt,
            }

            params_path = str(tmpdir / 'params.json')
            with open(params_path, 'w') as f:
                json.dump(params, f)

            self._invoke_blender(params_path, str(output_dir))

            result = self._read_and_composite(output_dir, frames, L)

        if abs(resize - 1.0) > 1e-6:
            new_W, new_H = int(W * resize), int(H * resize)
            result = np.stack([
                cv2.resize(result[i], (new_W, new_H))
                for i in range(L)
            ], axis=0)

        return result

    def render_with_ground(
        self,
        faces_list: List[np.ndarray],
        verts_list: List[np.ndarray],
        K4: List[float],
        ground_params: dict,
        camera_c2ws: np.ndarray,
        light_positions: np.ndarray,
        mesh_colors: List[List[float]],
        resolution: Tuple[int, int],
    ) -> np.ndarray:
        """Render meshes on a ground plane.

        Parameters
        ----------
        faces_list : list of (F_i, 3)
        verts_list : list of (L, V_i, 3)
        K4 : [fx, fy, cx, cy]
        ground_params : dict with 'length', 'center_x', 'center_z'
        camera_c2ws : (L, 4, 4) Blender c2w matrices
        light_positions : (L, 3)
        mesh_colors : list of [r, g, b] linear floats
        resolution : (W, H)

        Returns
        -------
        result : np.ndarray, (L, H, W, 3) uint8
        """
        L = verts_list[0].shape[0]
        W, H = resolution

        assert light_positions.shape[0] == L, (
            f'light_positions length {light_positions.shape[0]} != num_frames {L}'
        )
        assert camera_c2ws.shape[0] == L, (
            f'camera_c2ws length {camera_c2ws.shape[0]} != num_frames {L}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / 'output'
            output_dir.mkdir()

            mesh_npz_paths = []
            for idx, (faces, verts) in enumerate(zip(faces_list, verts_list)):
                npz_path = str(tmpdir / f'mesh_{idx}.npz')
                _save_mesh(npz_path, faces, verts)
                mesh_npz_paths.append(npz_path)

            params = {
                'mode': 'ground',
                'engine': self.engine,
                'samples': self.samples,
                'use_gpu': self.use_gpu,
                'K4': list(map(float, K4)),
                'resolution': [int(W), int(H)],
                'num_frames': int(L),
                'mesh_npz_paths': mesh_npz_paths,
                'mesh_colors': [list(map(float, c)) for c in mesh_colors],
                'camera_c2ws': camera_c2ws.tolist(),
                'light_positions': light_positions.tolist(),
                'ground_params': {
                    'length': float(ground_params['length']),
                    'center_x': float(ground_params['center_x']),
                    'center_z': float(ground_params['center_z']),
                },
            }

            params_path = str(tmpdir / 'params.json')
            with open(params_path, 'w') as f:
                json.dump(params, f)

            self._invoke_blender(params_path, str(output_dir))

            result = self._read_frames(output_dir, L)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_blender(self, params_path: str, output_dir: str):
        """Launch Blender in background mode and run the worker script."""
        cmd = [
            self.blender_exec,
            '--background',
            '--python', _WORKER_SCRIPT,
            '--',
            params_path,
            output_dir,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            # Blender prints tracebacks to stdout, not stderr — include both.
            raise RuntimeError(
                f'Blender render failed (exit code {result.returncode}).\n'
                f'--- stdout (last 2k) ---\n{result.stdout[-2000:]}\n'
                f'--- stderr (last 2k) ---\n{result.stderr[-2000:]}'
            )

    @staticmethod
    def _read_and_composite(
        output_dir: Path,
        bg_frames: np.ndarray,
        num_frames: int,
    ) -> np.ndarray:
        """Read rendered RGBA PNGs and alpha-composite onto background frames."""
        results = []
        for i in range(num_frames):
            png_path = output_dir / f'frame_{i:06d}.png'
            rgba = imageio.v3.imread(str(png_path))            # (H, W, 4) uint8
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            fg = rgba[:, :, :3].astype(np.float32)
            bg = bg_frames[i].astype(np.float32)
            composite = fg * alpha + bg * (1.0 - alpha)
            results.append(composite.astype(np.uint8))
        return np.stack(results, axis=0)

    @staticmethod
    def _read_frames(output_dir: Path, num_frames: int) -> np.ndarray:
        """Read rendered RGB PNGs into a numpy array."""
        results = []
        for i in range(num_frames):
            png_path = output_dir / f'frame_{i:06d}.png'
            img = imageio.v3.imread(str(png_path))             # (H, W, 3) uint8
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            results.append(img)
        return np.stack(results, axis=0)


def _save_mesh(npz_path: str, faces: np.ndarray, verts: np.ndarray):
    """Save mesh with compact, predictable dtypes for the Blender worker."""
    np.savez(
        npz_path,
        faces=np.ascontiguousarray(faces, dtype=np.int32),
        verts=np.ascontiguousarray(verts, dtype=np.float32),
    )
