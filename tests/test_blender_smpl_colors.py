"""Render the SMPL T-pose under every ColorPalette preset and tile the
results into a single comparison grid.

Usage::
    python tests/test_blender_smpl_colors.py
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path

import cv2
import imageio.v3 as iio

sys.path.insert(0, str(Path(__file__).parent))

from ez4d.vis.blender_renderer import render_mesh_overlay_img
from ez4d.vis.colors import ColorPalette
from test_blender_smpl import make_smpl_tpose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/test_blender_renderer/colors')
    parser.add_argument('--engine', default='BLENDER_EEVEE')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--res', type=int, default=512,
                        help='per-render width (height = 3/4 * width)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W = args.res
    H = (W * 3) // 4
    K4 = [float(W), float(W), W / 2.0, H / 2.0]

    verts, faces = make_smpl_tpose()
    verts[..., 2] += 3.0
    verts[..., 1] -= verts[..., 1].mean()
    img = np.full((H, W, 3), 200, dtype=np.uint8)

    color_names = list(ColorPalette.presets.keys())
    rendered = {}
    for name in color_names:
        out_path = out_dir / f'smpl_{name}.png'
        render_mesh_overlay_img(
            faces=faces, verts=torch.from_numpy(verts),
            K4=K4, img=img, output_fn=str(out_path),
            mesh_color=name, device='cpu',
            engine=args.engine, samples=args.samples,
        )
        rendered[name] = iio.imread(str(out_path))[..., :3]
        print(f'  [ok] {name}')

    # Tile into a grid: 4 cols × ceil(N/4) rows. Annotate each cell with
    # the color name.
    cols = 4
    rows = (len(color_names) + cols - 1) // cols
    cell_h, cell_w = H, W
    grid = np.full((rows * cell_h, cols * cell_w, 3), 230, dtype=np.uint8)
    for i, name in enumerate(color_names):
        r, c = divmod(i, cols)
        cell = rendered[name].copy()
        cv2.putText(cell, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)
        grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = cell

    grid_path = out_dir / 'smpl_colors_grid.png'
    iio.imwrite(str(grid_path), grid)
    print(f'\nGrid: {grid_path}  ({grid.shape[1]}x{grid.shape[0]})')


if __name__ == '__main__':
    main()
