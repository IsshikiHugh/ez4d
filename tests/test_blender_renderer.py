"""Test script for ez4d.vis.blender_renderer.

Usage::

    python tests/test_blender_renderer.py [--engine CYCLES|BLENDER_EEVEE] [--samples N] [--output-dir DIR]

Requires Blender >= 3.0 on PATH or via $BLENDER_PATH.
"""

import argparse
import numpy as np
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_cube(center=(0, 0, 0), size=1.0):
    """Return (verts, faces) for a simple cube.

    verts: (8, 3) float32
    faces: (12, 3) int64
    """
    s = size / 2.0
    cx, cy, cz = center
    verts = np.array([
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 6, 5], [4, 7, 6],  # back
        [0, 4, 5], [0, 5, 1],  # bottom
        [2, 6, 7], [2, 7, 3],  # top
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.int64)
    return verts, faces


def make_moving_cube(L=30, center=(0, 1, 0), size=1.0, amplitude=2.0):
    """A cube that sways left-right over L frames.

    Returns verts: (L, 8, 3), faces: (12, 3)
    """
    verts0, faces = make_cube(center=center, size=size)
    t = np.linspace(0, 2 * np.pi, L, dtype=np.float32)
    offsets = np.zeros((L, 1, 3), dtype=np.float32)
    offsets[:, 0, 0] = amplitude * np.sin(t)
    verts = verts0[None] + offsets  # (L, 8, 3)
    return verts, faces


def make_background(L, H=480, W=640):
    """Solid gray background frames."""
    return np.full((L, H, W, 3), 200, dtype=np.uint8)


def default_K4(W=640, H=480):
    """Reasonable intrinsics for the given resolution."""
    fx = fy = max(W, H)
    cx, cy = W / 2.0, H / 2.0
    return [fx, fy, cx, cy]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_overlay_img(out_dir: Path, **kw):
    """render_mesh_overlay_img: single cube on a gray image."""
    from ez4d.vis.blender_renderer import render_mesh_overlay_img

    verts, faces = make_cube(center=(0, 0, 5), size=1.0)
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    K4 = default_K4()

    out_path = out_dir / 'overlay_img.png'
    render_mesh_overlay_img(
        faces=faces,
        verts=torch.from_numpy(verts),
        K4=K4,
        img=img,
        output_fn=str(out_path),
        mesh_color='blue',
        **kw,
    )
    assert out_path.exists(), f'Output not created: {out_path}'
    print(f'  [PASS] overlay_img -> {out_path}')


def test_overlay_video(out_dir: Path, **kw):
    """render_mesh_overlay_video: moving cube on gray frames."""
    from ez4d.vis.blender_renderer import render_mesh_overlay_video

    L = 10
    verts, faces = make_moving_cube(L=L, center=(0, 0, 5))
    frames = make_background(L)
    K4 = default_K4()

    # Test return-frames mode
    result = render_mesh_overlay_video(
        faces=faces,
        verts=torch.from_numpy(verts),
        K4=K4,
        frames=frames,
        mesh_color='green',
        **kw,
    )
    assert result.shape == (L, 480, 640, 3), f'Unexpected shape: {result.shape}'
    print(f'  [PASS] overlay_video (return frames): shape={result.shape}')

    # Test write-to-file mode
    out_path = out_dir / 'overlay_video.mp4'
    render_mesh_overlay_video(
        faces=faces,
        verts=torch.from_numpy(verts),
        K4=K4,
        frames=frames,
        output_fn=str(out_path),
        fps=10,
        mesh_color='green',
        **kw,
    )
    assert out_path.exists(), f'Output not created: {out_path}'
    print(f'  [PASS] overlay_video (write file) -> {out_path}')


def test_mesh_with_ground(out_dir: Path, **kw):
    """render_mesh_with_ground: cube on ground plane, follow camera."""
    from ez4d.vis.blender_renderer import render_mesh_with_ground

    L = 10
    verts, faces = make_moving_cube(L=L, center=(0, 1, 0))
    K4 = default_K4()

    # Test return-frames + return_cam
    result, (R, t) = render_mesh_with_ground(
        faces=faces,
        verts=torch.from_numpy(verts),
        K4=K4,
        cam_style='follow',
        return_cam=True,
        mesh_color='blue',
        **kw,
    )
    assert result.shape[0] == L, f'Expected {L} frames, got {result.shape[0]}'
    assert R.shape == (L, 3, 3), f'Unexpected R shape: {R.shape}'
    assert t.shape == (L, 3), f'Unexpected t shape: {t.shape}'
    print(f'  [PASS] mesh_with_ground (follow, return_cam): frames={result.shape}, R={R.shape}, t={t.shape}')

    # Test write-to-file
    out_path = out_dir / 'ground_single.mp4'
    render_mesh_with_ground(
        faces=faces,
        verts=torch.from_numpy(verts),
        K4=K4,
        output_fn=str(out_path),
        fps=10,
        cam_style='follow',
        mesh_color='red',
        **kw,
    )
    assert out_path.exists(), f'Output not created: {out_path}'
    print(f'  [PASS] mesh_with_ground (write file) -> {out_path}')


def test_meshes_with_ground(out_dir: Path, **kw):
    """render_meshes_with_ground: two cubes, different colors."""
    from ez4d.vis.blender_renderer import render_meshes_with_ground

    L = 10
    verts1, faces1 = make_moving_cube(L=L, center=(-1.5, 1, 0), amplitude=1.0)
    verts2, faces2 = make_moving_cube(L=L, center=(1.5, 1, 0), amplitude=1.5)
    K4 = default_K4()

    out_path = out_dir / 'ground_multi.mp4'
    result = render_meshes_with_ground(
        faces_list=[faces1, faces2],
        verts_list=[torch.from_numpy(verts1), torch.from_numpy(verts2)],
        K4=K4,
        output_fn=str(out_path),
        fps=10,
        cam_style='follow',
        mesh_colors=['blue', 'red'],
        **kw,
    )
    assert out_path.exists(), f'Output not created: {out_path}'
    print(f'  [PASS] meshes_with_ground (2 meshes) -> {out_path}')

    # Test stare cam_style (return frames)
    result = render_meshes_with_ground(
        faces_list=[faces1, faces2],
        verts_list=[torch.from_numpy(verts1), torch.from_numpy(verts2)],
        K4=K4,
        cam_style='stare',
        **kw,
    )
    assert result.shape[0] == L
    print(f'  [PASS] meshes_with_ground (stare): shape={result.shape}')


def test_bypass_render(**kw):
    """bypass_render kwarg should skip Blender invocation entirely."""
    from ez4d.vis.blender_renderer import render_mesh_with_ground

    L = 5
    verts, faces = make_moving_cube(L=L, center=(0, 1, 0))
    K4 = default_K4()

    result, (R, t) = render_mesh_with_ground(
        faces=faces,
        verts=torch.from_numpy(verts),
        K4=K4,
        cam_style='follow',
        return_cam=True,
        bypass_render=True,
        **kw,
    )
    assert result is None, 'bypass_render should return None frames'
    assert R.shape == (L, 3, 3)
    assert t.shape == (L, 3)
    print(f'  [PASS] bypass_render: R={R.shape}, t={t.shape}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Test blender_renderer')
    parser.add_argument('--engine', default='BLENDER_EEVEE',
                        help='Render engine: CYCLES or BLENDER_EEVEE (default: BLENDER_EEVEE)')
    parser.add_argument('--samples', type=int, default=16,
                        help='Render samples (default: 16 for fast testing)')
    parser.add_argument('--output-dir', default='/tmp/test_blender_renderer',
                        help='Directory for test outputs')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(engine=args.engine, samples=args.samples)

    print(f'Output dir: {out_dir}')
    print(f'Engine: {args.engine}, Samples: {args.samples}')
    print()

    tests = [
        ('bypass_render', lambda: test_bypass_render(**kw)),
        ('overlay_img', lambda: test_overlay_img(out_dir, **kw)),
        ('overlay_video', lambda: test_overlay_video(out_dir, **kw)),
        ('mesh_with_ground', lambda: test_mesh_with_ground(out_dir, **kw)),
        ('meshes_with_ground', lambda: test_meshes_with_ground(out_dir, **kw)),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        print(f'Running: {name}')
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f'  [FAIL] {name}: {e}')
            failed += 1
        print()

    print(f'Results: {passed} passed, {failed} failed out of {passed + failed}')


if __name__ == '__main__':
    main()
