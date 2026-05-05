"""Quick visual test: render an SMPL body with the blender_renderer.

Reuses the same lighting / shading setup as test_blender_renderer.py,
but with a 6890-vertex SMPL T-pose instead of a cube — so we can see
how the world+headlamp combo handles a real human mesh.

Usage::

    python tests/test_blender_smpl.py [--output-dir DIR]
"""

import argparse
import numpy as np
import smplx
import torch
from pathlib import Path

from ez4d.vis.blender_renderer import (
    render_mesh_overlay_img,
    render_mesh_overlay_video,
    render_mesh_with_ground,
)


SMPL_DIR = '/Users/isshikih/my_/workspace/scientific_research/body_models/smpl/'


def make_smpl_tpose():
    """Return T-pose SMPL verts (Y-up, feet at y=0) and faces."""
    smpl = smplx.SMPL(SMPL_DIR, gender='neutral', batch_size=1)
    out = smpl()
    verts = out.vertices[0].detach().numpy().astype(np.float32)  # (6890, 3)
    faces = smpl.faces.astype(np.int64)                          # (13776, 3)

    # SMPL is natively Y-up (head at +Y) and faces -Z. Rotate 180° about Y
    # so the body faces +Z (toward the camera under the renderer's default Rt),
    # then place feet at y=0 to match the codebase's ground=Y=0 convention.
    verts[:, [0, 2]] *= -1                  # 180° about Y → body faces +Z
    verts[:, 1] -= verts[:, 1].min()        # feet at y=0
    return verts, faces


def make_moving_smpl(L=30, depth=4.0, x_amp=0.4, z_amp=0.7):
    """T-pose SMPL on an X-Z elliptical orbit in front of the camera.

    X uses sin(t), Z uses cos(t) so the body sweeps left↔right while also
    moving toward↔away from the camera, tracing an ellipse viewed from above.
    """
    verts0, faces = make_smpl_tpose()
    verts = np.tile(verts0[None], (L, 1, 1))             # (L, V, 3)
    verts[..., 2] += depth                                # baseline depth
    # Lower body so the figure is roughly centered vertically in frame.
    verts[..., 1] -= verts[..., 1].mean()
    t = np.linspace(0, 2 * np.pi, L, dtype=np.float32)
    verts[..., 0] += x_amp * np.sin(t)[:, None]
    verts[..., 2] += z_amp * np.cos(t)[:, None]
    return verts, faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/test_blender_renderer')
    parser.add_argument('--engine', default='BLENDER_EEVEE')
    parser.add_argument('--samples', type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(engine=args.engine, samples=args.samples, device='cpu')

    W, H = 1280, 960
    K4 = [float(W), float(W), W / 2.0, H / 2.0]   # focal scales with W to keep FOV

    # Single overlay image (T-pose at depth 3)
    verts, faces = make_smpl_tpose()
    verts[..., 2] += 3.0
    verts[..., 1] -= verts[..., 1].mean()
    img = np.full((H, W, 3), 200, dtype=np.uint8)

    out_img = out_dir / 'smpl_overlay_img.png'
    render_mesh_overlay_img(
        faces=faces, verts=torch.from_numpy(verts),
        K4=K4, img=img, output_fn=str(out_img),
        mesh_color='blue', **kw,
    )
    print(f'[ok] {out_img}')

    # Overlay video (sway)
    L = 20
    verts_seq, faces = make_moving_smpl(L=L)
    frames = np.full((L, H, W, 3), 200, dtype=np.uint8)

    out_vid = out_dir / 'smpl_overlay_video.mp4'
    render_mesh_overlay_video(
        faces=faces, verts=torch.from_numpy(verts_seq),
        K4=K4, frames=frames, output_fn=str(out_vid),
        fps=15, mesh_color='blue', **kw,
    )
    print(f'[ok] {out_vid}')

    # Ground render
    out_ground = out_dir / 'smpl_ground.mp4'
    render_mesh_with_ground(
        faces=faces, verts=torch.from_numpy(verts_seq),
        K4=K4, output_fn=str(out_ground),
        fps=15, cam_style='follow', mesh_color='blue', **kw,
    )
    print(f'[ok] {out_ground}')


if __name__ == '__main__':
    main()
