"""Compare blender_renderer vs p3d_renderer outputs.

Usage::

    # Step 1: Generate p3d reference (works now)
    python tests/test_blender_vs_p3d.py --p3d-only

    # Step 2: Generate blender outputs (requires Blender)
    python tests/test_blender_vs_p3d.py --blender-only [--engine BLENDER_EEVEE] [--samples 16]

    # Step 3: Both (requires Blender)
    python tests/test_blender_vs_p3d.py

Results are saved to /tmp/test_renderer_compare/
"""

import argparse
import numpy as np
import torch
from pathlib import Path


OUTPUT_DIR = Path('/tmp/test_renderer_compare')


def make_cube(center=(0, 0, 0), size=1.0):
    s = size / 2.0
    cx, cy, cz = center
    verts = np.array([
        [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s],
        [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s],
        [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s],
    ], dtype=np.float32)
    faces = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,4,5],[0,5,1],[2,6,7],[2,7,3],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ], dtype=np.int64)
    return verts, faces


def make_moving_cube(L=30, center=(0, 1, 0), size=1.0, amplitude=2.0):
    verts0, faces = make_cube(center=center, size=size)
    t = np.linspace(0, 2 * np.pi, L, dtype=np.float32)
    offsets = np.zeros((L, 1, 3), dtype=np.float32)
    offsets[:, 0, 0] = amplitude * np.sin(t)
    return verts0[None] + offsets, faces


K4 = [640.0, 640.0, 320.0, 240.0]


def run_p3d(out_dir: Path):
    from ez4d.vis.p3d_renderer import (
        render_mesh_overlay_img,
        render_mesh_with_ground,
        render_meshes_with_ground,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    L = 10

    # --- overlay img ---
    verts, faces = make_cube(center=(0, 0, 5), size=1.0)
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    result = render_mesh_overlay_img(
        faces=faces, verts=torch.from_numpy(verts),
        K4=K4, img=img, mesh_color='blue',
    )
    from ez4d.media.io import save_img
    save_img(result, str(out_dir / 'p3d_overlay_img.png'))
    print(f'  p3d overlay_img -> {out_dir / "p3d_overlay_img.png"}')

    # --- single mesh with ground ---
    verts_seq, faces = make_moving_cube(L=L, center=(0, 1, 0))
    out_path = out_dir / 'p3d_ground_single.mp4'
    render_mesh_with_ground(
        faces=faces, verts=torch.from_numpy(verts_seq),
        K4=K4, output_fn=str(out_path), fps=10,
        cam_style='follow', mesh_color='blue',
    )
    print(f'  p3d ground_single -> {out_path}')

    # --- two meshes with ground ---
    verts1, faces1 = make_moving_cube(L=L, center=(-1.5, 1, 0), amplitude=1.0)
    verts2, faces2 = make_moving_cube(L=L, center=(1.5, 1, 0), amplitude=1.5)
    out_path = out_dir / 'p3d_ground_multi.mp4'
    render_meshes_with_ground(
        faces_list=[faces1, faces2],
        verts_list=[torch.from_numpy(verts1), torch.from_numpy(verts2)],
        K4=K4, output_fn=str(out_path), fps=10,
        cam_style='follow', mesh_colors=['blue', 'red'],
    )
    print(f'  p3d ground_multi -> {out_path}')


def run_blender(out_dir: Path, engine='BLENDER_EEVEE', samples=16):
    from ez4d.vis.blender_renderer import (
        render_mesh_overlay_img,
        render_mesh_with_ground,
        render_meshes_with_ground,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    L = 10
    kw = dict(engine=engine, samples=samples)

    # --- overlay img ---
    verts, faces = make_cube(center=(0, 0, 5), size=1.0)
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    render_mesh_overlay_img(
        faces=faces, verts=torch.from_numpy(verts),
        K4=K4, img=img,
        output_fn=str(out_dir / 'blender_overlay_img.png'),
        mesh_color='blue', **kw,
    )
    print(f'  blender overlay_img -> {out_dir / "blender_overlay_img.png"}')

    # --- single mesh with ground ---
    verts_seq, faces = make_moving_cube(L=L, center=(0, 1, 0))
    out_path = out_dir / 'blender_ground_single.mp4'
    render_mesh_with_ground(
        faces=faces, verts=torch.from_numpy(verts_seq),
        K4=K4, output_fn=str(out_path), fps=10,
        cam_style='follow', mesh_color='blue', **kw,
    )
    print(f'  blender ground_single -> {out_path}')

    # --- two meshes with ground ---
    verts1, faces1 = make_moving_cube(L=L, center=(-1.5, 1, 0), amplitude=1.0)
    verts2, faces2 = make_moving_cube(L=L, center=(1.5, 1, 0), amplitude=1.5)
    out_path = out_dir / 'blender_ground_multi.mp4'
    render_meshes_with_ground(
        faces_list=[faces1, faces2],
        verts_list=[torch.from_numpy(verts1), torch.from_numpy(verts2)],
        K4=K4, output_fn=str(out_path), fps=10,
        cam_style='follow', mesh_colors=['blue', 'red'], **kw,
    )
    print(f'  blender ground_multi -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p3d-only', action='store_true')
    parser.add_argument('--blender-only', action='store_true')
    parser.add_argument('--engine', default='BLENDER_EEVEE')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if not args.blender_only:
        print('=== P3D Renderer ===')
        run_p3d(out_dir)
        print()

    if not args.p3d_only:
        print('=== Blender Renderer ===')
        run_blender(out_dir, engine=args.engine, samples=args.samples)
        print()

    print(f'Results saved to: {out_dir}')
    print('Compare side-by-side:')
    print(f'  Overlay:       p3d_overlay_img.png  vs  blender_overlay_img.png')
    print(f'  Ground single: p3d_ground_single.mp4  vs  blender_ground_single.mp4')
    print(f'  Ground multi:  p3d_ground_multi.mp4  vs  blender_ground_multi.mp4')


if __name__ == '__main__':
    main()
