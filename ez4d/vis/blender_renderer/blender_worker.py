"""Blender-side render worker.

Invoked as::

    blender --background --python blender_worker.py -- <params.json> <output_dir>

This script runs **inside Blender's Python interpreter**.  It depends only on
``bpy``, ``mathutils``, the standard library, and ``numpy`` (bundled with
Blender).
"""

import sys
import json
import numpy as np
from pathlib import Path
from mathutils import Matrix

import bpy


# ---- Scene helpers --------------------------------------------------------

def clear_scene():
    """Remove every object from the default scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_engine(engine: str, samples: int, use_gpu: bool):
    scene = bpy.context.scene
    scene.render.engine = engine

    if engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        if use_gpu:
            scene.cycles.device = 'GPU'
            prefs = bpy.context.preferences.addons['cycles'].preferences
            # Try OPTIX first, then CUDA, then HIP
            for compute_type in ('OPTIX', 'CUDA', 'HIP'):
                try:
                    prefs.compute_device_type = compute_type
                    prefs.get_devices()
                    for d in prefs.devices:
                        d.use = True
                    break
                except Exception:
                    continue
        else:
            scene.cycles.device = 'CPU'

    elif engine in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
        # EEVEE settings (Blender 3.x uses BLENDER_EEVEE, 4.x uses BLENDER_EEVEE_NEXT)
        if hasattr(scene, 'eevee'):
            scene.eevee.taa_render_samples = samples


def setup_camera(K4, resolution):
    """Create a camera matching pixel-space intrinsics ``[fx, fy, cx, cy]``."""
    fx, fy, cx, cy = K4
    W, H = resolution

    cam_data = bpy.data.cameras.new('Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Sensor & focal length
    sensor_width = 36.0  # mm (arbitrary reference)
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = sensor_width
    cam_data.lens = fx * sensor_width / W

    # Principal point offset (shift is in fraction of sensor dimension)
    cam_data.shift_x = (W / 2.0 - cx) / W
    cam_data.shift_y = (cy - H / 2.0) / H

    # Handle non-square pixels (fx != fy)
    scene = bpy.context.scene
    if abs(fy - fx) > 1e-6:
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = fx / fy
    else:
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = 1.0

    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100

    return cam_obj


def setup_render_settings(mode: str):
    """Configure output format and transparency."""
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'

    # Use the Standard view transform so Principled-BSDF linear colors
    # round-trip to sRGB output without Filmic tone-mapping.
    try:
        scene.view_settings.view_transform = 'Standard'
    except (TypeError, AttributeError):
        pass

    if mode == 'overlay':
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = 'RGBA'
    else:
        scene.render.film_transparent = False
        scene.render.image_settings.color_mode = 'RGB'
        scene.world = bpy.data.worlds.new('World')
        scene.world.use_nodes = True
        bg_node = scene.world.node_tree.nodes['Background']
        bg_node.inputs['Color'].default_value = (0.95, 0.95, 0.95, 1.0)
        bg_node.inputs['Strength'].default_value = 1.0


# ---- Lighting -------------------------------------------------------------

def setup_lights_overlay():
    """Three-point lighting for overlay mode (similar to Raymond lights)."""
    key_data = bpy.data.lights.new('KeyLight', type='SUN')
    key_data.energy = 3.0
    key_obj = bpy.data.objects.new('KeyLight', key_data)
    bpy.context.scene.collection.objects.link(key_obj)
    key_obj.rotation_euler = (0.8, 0.0, -0.5)

    fill_data = bpy.data.lights.new('FillLight', type='SUN')
    fill_data.energy = 1.5
    fill_obj = bpy.data.objects.new('FillLight', fill_data)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (0.8, 0.0, 2.6)

    rim_data = bpy.data.lights.new('RimLight', type='SUN')
    rim_data.energy = 1.0
    rim_obj = bpy.data.objects.new('RimLight', rim_data)
    bpy.context.scene.collection.objects.link(rim_obj)
    rim_obj.rotation_euler = (-0.3, 0.0, 1.5)


def setup_lights_ground(light_position):
    """Sun lamp + point light for ground-plane mode."""
    sun_data = bpy.data.lights.new('SunLight', type='SUN')
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new('SunLight', sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (0.8, 0.0, -0.5)

    point_data = bpy.data.lights.new('PointLight', type='POINT')
    point_data.energy = 500.0
    point_obj = bpy.data.objects.new('PointLight', point_data)
    bpy.context.scene.collection.objects.link(point_obj)
    point_obj.location = tuple(light_position)

    return point_obj


# ---- Mesh creation --------------------------------------------------------

def create_mesh_object(name, faces, verts_frame0, color):
    """Create a Blender mesh object with a Principled BSDF material.

    Parameters
    ----------
    name : str
    faces : np.ndarray, (F, 3)
    verts_frame0 : np.ndarray, (V, 3)
    color : list of 3 floats in [0, 1]

    Returns
    -------
    obj : bpy.types.Object
    """
    mesh = bpy.data.meshes.new(f'{name}_mesh')
    verts_list = verts_frame0.tolist()
    faces_list = faces.tolist()
    mesh.from_pydata(verts_list, [], faces_list)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # Material
    mat = bpy.data.materials.new(f'{name}_mat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0
    obj.data.materials.append(mat)

    # Enable smooth shading
    for poly in mesh.polygons:
        poly.use_smooth = True

    return obj


def update_mesh_vertices(obj, verts):
    """Update vertex positions of a Blender mesh object.

    Parameters
    ----------
    obj : bpy.types.Object
    verts : np.ndarray, (V, 3)
    """
    mesh = obj.data
    flat = verts.astype(np.float64).ravel()
    mesh.vertices.foreach_set('co', flat)
    mesh.update()


# ---- Ground plane ---------------------------------------------------------

def create_ground_plane(ground_params):
    """Create a checkerboard ground plane.

    Parameters
    ----------
    ground_params : dict with keys 'length', 'center_x', 'center_z'
    """
    length = ground_params['length']
    c1 = ground_params['center_x']
    c2 = ground_params['center_z']
    color0 = ground_params.get('color0', [0.8, 0.9, 0.9])
    color1 = ground_params.get('color1', [0.6, 0.7, 0.7])
    tile_width = ground_params.get('tile_width', 0.5)

    radius = length / 2.0
    # Cap per-side tile count so pathological motion extents don't blow up
    # geometry (num_tiles**2 tiles, 4 verts + 2 tris each).
    max_tiles_per_side = 200
    num_tiles = max(2, int(length / tile_width))
    if num_tiles > max_tiles_per_side:
        num_tiles = max_tiles_per_side
        tile_width = length / num_tiles

    # Create materials
    mat0 = bpy.data.materials.new('GroundTile0')
    mat0.use_nodes = True
    mat0.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (*color0, 1.0)
    mat0.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.8

    mat1 = bpy.data.materials.new('GroundTile1')
    mat1.use_nodes = True
    mat1.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (*color1, 1.0)
    mat1.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.8

    all_verts = []
    all_faces = []
    face_mat_indices = []
    vert_offset = 0

    for i in range(num_tiles):
        for j in range(num_tiles):
            u0 = j * tile_width - radius + c1
            v0 = i * tile_width - radius + c2

            tile_verts = [
                (u0, 0.0, v0),
                (u0, 0.0, v0 + tile_width),
                (u0 + tile_width, 0.0, v0 + tile_width),
                (u0 + tile_width, 0.0, v0),
            ]
            all_verts.extend(tile_verts)

            # Two triangles per tile (double-sided via two opposing winding orders)
            all_faces.append((vert_offset, vert_offset + 1, vert_offset + 3))
            all_faces.append((vert_offset + 1, vert_offset + 2, vert_offset + 3))

            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            mat_idx = 0 if use_color0 else 1
            face_mat_indices.extend([mat_idx, mat_idx])

            vert_offset += 4

    mesh = bpy.data.meshes.new('GroundMesh')
    mesh.from_pydata(all_verts, [], all_faces)
    mesh.update()

    obj = bpy.data.objects.new('Ground', mesh)
    bpy.context.scene.collection.objects.link(obj)

    obj.data.materials.append(mat0)
    obj.data.materials.append(mat1)
    for fi, mat_idx in enumerate(face_mat_indices):
        mesh.polygons[fi].material_index = mat_idx

    return obj


# ---- Main render loop -----------------------------------------------------

def main():
    argv = sys.argv
    sep_idx = argv.index('--')
    params_path = argv[sep_idx + 1]
    output_dir = argv[sep_idx + 2]

    with open(params_path, 'r') as f:
        params = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = params['mode']                   # 'overlay' or 'ground'
    engine = params.get('engine', 'CYCLES')
    samples = params.get('samples', 64)
    use_gpu = params.get('use_gpu', True)
    K4 = params['K4']
    resolution = params['resolution']       # [W, H]
    num_frames = params['num_frames']

    # -- Scene setup --
    clear_scene()
    setup_engine(engine, samples, use_gpu)
    cam_obj = setup_camera(K4, resolution)
    setup_render_settings(mode)

    # -- Lighting --
    point_light_obj = None
    light_positions = None
    if mode == 'overlay':
        setup_lights_overlay()
    else:
        light_positions = np.asarray(params['light_positions'], dtype=np.float64)  # (L, 3)
        assert light_positions.shape[0] == num_frames, (
            f'light_positions length {light_positions.shape[0]} != num_frames {num_frames}'
        )
        point_light_obj = setup_lights_ground(light_positions[0])

    # -- Ground plane --
    if mode == 'ground' and 'ground_params' in params:
        create_ground_plane(params['ground_params'])

    # -- Load meshes --
    mesh_npz_paths = params['mesh_npz_paths']
    mesh_colors = params['mesh_colors']
    mesh_objects = []
    mesh_verts_arrays = []

    for idx, npz_path in enumerate(mesh_npz_paths):
        data = np.load(npz_path)
        faces = data['faces']
        verts = data['verts']       # (L, V, 3) or (V, 3)
        if verts.ndim == 2:
            verts = verts[np.newaxis]  # (1, V, 3)
        mesh_verts_arrays.append(verts)
        color = mesh_colors[idx]
        obj = create_mesh_object(f'Mesh_{idx}', faces, verts[0], color)
        mesh_objects.append(obj)

    # -- Camera trajectory --
    camera_c2ws = None
    if 'camera_c2ws' in params:
        camera_c2ws = np.array(params['camera_c2ws'])  # (L, 4, 4)

    # -- Camera from Rt (overlay mode) --
    camera_Rt = None
    if params.get('camera_Rt') is not None:
        camera_Rt = params['camera_Rt']  # list of (4, 4) Blender c2w matrices

    # -- Render each frame --
    for frame_idx in range(num_frames):
        # Update mesh vertices
        for obj, verts_arr in zip(mesh_objects, mesh_verts_arrays):
            fi = min(frame_idx, len(verts_arr) - 1)
            update_mesh_vertices(obj, verts_arr[fi])

        # Update camera
        if camera_c2ws is not None:
            c2w = camera_c2ws[frame_idx]
            cam_obj.matrix_world = Matrix(c2w.tolist())
        elif camera_Rt is not None:
            c2w = camera_Rt[frame_idx]
            cam_obj.matrix_world = Matrix(c2w)
        else:
            # Identity camera (looking down -Z)
            cam_obj.matrix_world = Matrix.Identity(4)

        # Update point light position (ground mode)
        if point_light_obj is not None:
            point_light_obj.location = tuple(light_positions[frame_idx])

        # Render
        filepath = str(output_dir / f'frame_{frame_idx:06d}.png')
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
