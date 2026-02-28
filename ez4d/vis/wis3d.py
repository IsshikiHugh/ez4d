import torch
import numpy as np
import trimesh

from typing import Union, List, overload, Tuple, Optional
from wis3d import Wis3D

from .colors import ColorPalette
from ..data import to_numpy
from ..geometry.rotation import axis_angle_to_matrix


class HWis3D(Wis3D):
    """ Abstraction of Wis3D for human motion. """

    def __init__(
        self,
        out_path    : str   = 'wis3d',
        seq_name    : str   = 'debug',
        xyz_pattern : tuple = ('x', 'y', 'z'),
    ):
        seq_name = seq_name.replace('/', '-')
        super().__init__(out_path, seq_name, xyz_pattern)


    def add_text(self, text:str):
        """
        Add an item of vertices whose name is used to put the text message. *Dirty use!*

        ### Args
        - text: str
        """
        fake_verts = np.array([[0, 0, 0]])
        self.add_point_cloud(
            vertices = fake_verts,
            colors   = None,
            name     = text,
        )


    def add_text_seq(self, texts:List[str], offset:int=0):
        """
        Add an item of vertices whose name is used to put the text message. *Dirty use!*

        ### Args
        - texts: List[str]
            - The list of text messages.
        - offset: int, default = 0
            - The offset for the sequence index.
        """
        fake_verts = np.array([[0, 0, 0]])
        for i, text in enumerate(texts):
            self.set_scene_id(i + offset)
            self.add_point_cloud(
                vertices = fake_verts,
                colors   = None,
                name     = text,
            )

    def add_image_seq(self, imgs:List[np.ndarray], name:str, offset:int=0):
        """
        Add an item of vertices whose name is used to put the image. *Dirty use!*

        ### Args
        - imgs: List[np.ndarray]
            - The list of images.
        - offset: int, default = 0
            - The offset for the sequence index.
        """
        for i, img in enumerate(imgs):
            self.set_scene_id(i + offset)
            self.add_image(
                image = img,
                name  = name,
            )

    def add_motion_mesh(
        self,
        verts  : Union[torch.Tensor, np.ndarray],
        faces  : Union[torch.Tensor, np.ndarray],
        name   : str,
        colors : Union[str, torch.Tensor, np.ndarray] = 'gray',
        offset : int = 0,
    ):
        """
        Add sequence of vertices and face(s) to the wis3d viewer.

        ### Args
        - verts: torch.Tensor or np.ndarray, (L, V, 3), L ~ sequence length, V ~ number of vertices
        - faces: torch.Tensor or np.ndarray, (F, 3) or (L, F, 3), F ~ number of faces, L ~ sequence length
        - colors: str or torch.Tensor or np.ndarray, (L, V, 3), L ~ sequence length, V ~ number of vertices, default = None
            - If it's a string, all vertices will be colored with that according to the ColorPalette.
            - The colors of the vertices, values in [0, 255].
        - name: str
            - The name of the point cloud.
        - offset: int, default = 0
            - The offset for the sequence index.
        """
        assert (len(verts.shape) == 3), 'The input `verts` should have 3 dimensions: (L, V, 3).'
        assert (verts.shape[-1] == 3), 'The last dimension of `verts` should be 3.'
        if isinstance(verts, np.ndarray):
            verts = torch.from_numpy(verts)
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        if len(faces.shape) == 2:
            faces = faces[None].repeat(verts.shape[0], 0)
        assert (len(faces.shape) == 3), 'The input `faces` should have 2 or 3 dimensions: (F, 3) or (L, F, 3).'
        assert (faces.shape[-1] == 3), 'The last dimension of `faces` should be 3.'
        assert (verts.shape[0] == faces.shape[0]), 'The first dimension of `verts` and `faces` should be the same.'

        L, _, _ = verts.shape
        verts = verts.detach().cpu()

        # Add vertices frame by frame.
        for i in range(L):
            self.set_scene_id(i + offset)
            if isinstance(colors, str):
                v_colors = to_numpy(ColorPalette.presets_float[colors])
            elif isinstance(colors, torch.Tensor) or isinstance(colors, np.ndarray):
                v_colors = colors[i] / 255.0
            else:
                raise ValueError(f'The input `colors` should be a string or a torch.Tensor or a np.ndarray, but got {type(colors)}.')
            self.add_mesh(
                vertices      = verts[i],
                faces         = faces[i],
                vertex_colors = v_colors,
                name          = name,
            )  # type: ignore

        # Reset Wis3D scene id.
        self.set_scene_id(0)


    def add_motion_verts(
        self,
        verts : Union[torch.Tensor, np.ndarray],
        name  : str,
        offset: int = 0,
    ):
        """
        Add sequence of vertices to the wis3d viewer.

        ### Args
        - verts: torch.Tensor or np.ndarray, (L, V, 3), L ~ sequence length, V ~ number of vertices
        - name: str
            - The name of the point cloud.
        - offset: int, default = 0
            - The offset for the sequence index.
        """
        assert (len(verts.shape) == 3), 'The input `verts` should have 3 dimensions: (L, V, 3).'
        assert (verts.shape[-1] == 3), 'The last dimension of `verts` should be 3.'
        if isinstance(verts, np.ndarray):
            verts = torch.from_numpy(verts)

        L, _, _ = verts.shape
        verts = verts.detach().cpu()

        # Add vertices frame by frame.
        for i in range(L):
            self.set_scene_id(i + offset)
            self.add_point_cloud(
                vertices = verts[i],
                colors   = None,
                name     = name,
            )

        # Reset Wis3D scene id.
        self.set_scene_id(0)


    def add_motion_skel(
        self,
        joints    : Union[torch.Tensor, np.ndarray],
        bones     : Union[list, torch.Tensor],
        colors    : Union[list, torch.Tensor],
        name      : str,
        offset    : int = 0,
        threshold : float = 0.5,
    ):
        """
        Add sequence of joints with specific skeleton to the wis3d viewer.

        ### Args
        - joints: torch.Tensor or np.ndarray, shape = (L, J, 3) or (L, J, 4), L ~ sequence length, J ~ number of joints
        - bones: list
            - A list of bones of the skeleton, i.e. the edge in the kinematic trees.
        - colors: list
            - The color of the skeleton, values in [0, 255].
        - name: str
            - The name of the point cloud.
        - offset: int, default = 0
            - The offset for the sequence index.
        - threshold: float, default = 0.5
            - Threshold to filter the confidence of the joints. It's useless when no confidence provided.
        """
        assert (len(joints.shape) == 3), 'The input `joints` should have 3 dimensions: (L, J, 3).'
        assert (joints.shape[-1] == 3 or joints.shape[-1] == 4), 'The last dimension of `joints` should be 3 or 4.'
        if isinstance(joints, np.ndarray):
            joints = torch.from_numpy(joints)
        if isinstance(bones, List):
            bones = torch.tensor(bones)
        if isinstance(colors, List):
            colors = torch.tensor(colors)

        # Get the sequence length.
        joints = joints.detach().cpu() # (L, J, 3) or (L, J, 4)
        L, J, D = joints.shape
        if D == 4:
            conf = joints[:, :, 3]
            joints = joints[:, :, :3]
        else:
            conf = None

        # Add vertices frame by frame.
        for i in range(L):
            self.set_scene_id(i + offset)
            bones_s = joints[i][bones[:, 0]]
            bones_e = joints[i][bones[:, 1]]
            if conf is not None:
                mask = torch.logical_and(conf[i][bones[:, 0]] > threshold, conf[i][bones[:, 1]] > threshold)
                bones_s, bones_e = bones_s[mask], bones_e[mask]
            if len(bones_s) > 0:
                self.add_lines(
                    start_points = bones_s,
                    end_points   = bones_e,
                    colors       = colors,
                    name         = name,
                )

        # Reset Wis3D scene id.
        self.set_scene_id(0)


    def add_vec_seq(
        self,
        vecs      : torch.Tensor,
        name      : str,
        offset    : int = 0,
        seg_num   : int = 16,
        thickness : float = 0.01,
    ):
        """
        Add directional line sequence to the wis3d viewer.

        The line will be gradient colored, and the direction of the vector is visualized as from dark to light.

        ### Args
        - vecs: torch.Tensor, (L, 2, 3) or (L, N, 2, 3), L ~ sequence length, N ~ vectors counts in one frame,
              then give the start 3D point and end 3D point.
        - name: str
            - The name of the vector.
        - offset: int, default = 0
            - The offset for the sequence index.
        - seg_num: int, default = 16
            - The number of segments for gradient color, will just change the visualization effect.
        - thickness: float, default = 0.01
            - The thickness of the line.
        """
        if len(vecs.shape) == 3:
            vecs = vecs[:, None, :, :] # (L, 2, 3) -> (L, 1, 2, 3)
        assert (len(vecs.shape) == 4), 'The input `vecs` should have 3 or 4 dimensions: (L, 2, 3) or (L, N, 2, 3).'
        assert (vecs.shape[-2:] == (2, 3)), f'The last two dimension of `vecs` should be (2, 3), but got vecs.shape = {vecs.shape}.'

        # Get the sequence length.
        L, N, _, _ = vecs.shape
        vecs = vecs.detach().cpu()

        # Cut the line into segments.
        steps_delta = (vecs[:, :, [1]] - vecs[:, :, [0]]) / (seg_num + 1) # (L, N, 1, 3)
        steps_cnt   = torch.arange(seg_num + 1).reshape((1, 1, seg_num + 1, 1)) # (1, 1, seg_num+1, 1)
        segs = steps_delta * steps_cnt + vecs[:, :, [0]] # (L, N, seg_num+1, 3)
        start_pts = segs[:, :, :-1] # (L, N, seg_num, 3)
        end_pts   = segs[:, :, 1:] # (L, N, seg_num, 3)

        # Prepare the gradient colors.
        grad_colors = torch.linspace(0, 255, seg_num).reshape((1, seg_num, 1)).repeat(N, 1, 3) # (N, seg_num, 3)

        # Add vertices frame by frame.
        for i in range(L):
            self.set_scene_id(i + offset)
            self.add_lines(
                start_points = start_pts[i].reshape(-1, 3),
                end_points   = end_pts[i].reshape(-1, 3),
                colors       = grad_colors.reshape(-1, 3),
                thickness    = thickness,
                name         = name,
            )

        # Reset Wis3D scene id.
        self.set_scene_id(0)


    def add_traj(
        self,
        positions : torch.Tensor,
        name      : str,
        offset    : int = 0,
    ):
        """
        Visualize the the positions change across the time as trajectory. The newer position will be brighter.

        ### Args
        - positions: torch.Tensor, (L, 3), L ~ sequence length
        - name: str
            - The name of the trajectory.
        - offset: int, default = 0
            - The offset for the sequence index.
        """
        assert (len(positions.shape) == 2), 'The input `positions` should have 2 dimensions: (L, 3).'
        assert (positions.shape[-1] == 3), 'The last dimension of `positions` should be 3.'

        # Get the sequence length.
        L, _ = positions.shape
        positions = positions.detach().cpu()
        traj = positions[[0]] # (1, 3)

        # Prepare the gradient colors.
        grad_colors = torch.linspace(208, 48, L).reshape((L, 1)).repeat(1, 3) # (L, 3)

        for i in range(L):
            traj = torch.cat((traj, positions[[i]]), dim=0) # (i+2, 3)
            self.set_scene_id(i + offset)
            self.add_lines(
                start_points = traj[:-1],
                end_points   = traj[1:],
                colors       = grad_colors[-(i+1):],
                name         = name,
            )

        # Reset Wis3D scene id.
        self.set_scene_id(0)


    def add_sphere_sensors(
        self,
        positions  : torch.Tensor,
        radius     : Union[torch.Tensor, float],
        activities : torch.Tensor,
        name       : str,
    ):
        """
        Draw the sphere sensors with different colors to represent the activities. The color is from white to red.

        ### Args
        - positions: torch.Tensor, (N, 3), N ~ number of sensors
        - radius: torch.Tensor or float, (N,), N ~ number of sensors
        - activities: torch.Tensor, (N)
            - The activities of the sensors, from 0 to 1.
        - name: str
            - The name of the spheres.
        """
        assert (len(positions.shape) == 2), 'The input `positions` should have 2 dimensions: (N, 3).'
        assert (positions.shape[-1] == 3), 'The last dimension of `positions` should be 3.'
        N, _ = positions.shape
        if isinstance(radius, float):
            radius = torch.Tensor(radius).reshape(1).repeat(N) # (N)
        elif len(radius.shape) == 0:
            radius = radius.reshape(1).repeat(N)

        colors = torch.ones(size=(N, 3)).float()
        colors[:, 0] = 255
        colors[:, 1] = (1 - activities) ** 2 * 255
        colors[:, 2] = (1 - activities) ** 2 * 255
        self.add_spheres(
            centers = positions,
            radius  = radius,
            colors  = colors,
            name    = name,
        )


    def add_sphere_sensors_seq(
        self,
        positions  : torch.Tensor,
        radius     : Union[torch.Tensor, float],
        activities : torch.Tensor,
        name       : str,
        offset     : int = 0,
    ):
        """
        Draw the sphere sensors with different colors to represent the activities. The color is from white to red.

        ### Args
        - positions: torch.Tensor, (L, N, 3), N ~ number of sensors
        - radius: torch.Tensor or float, (L, N,), N ~ number of sensors
        - activities: torch.Tensor, (L, N)
            - The activities of the sensors, from 0 to 1.
        - name: str
            - The name of the spheres.
        - offset: int, default = 0
            - The offset for the sequence index.
        """
        assert (len(positions.shape) == 3), 'The input `positions` should have 3 dimensions: (L, N, 3).'
        assert (positions.shape[-1] == 3), 'The last dimension of `positions` should be 3.'
        L, N, _ = positions.shape

        for i in range(L):
            self.set_scene_id(i + offset)
            self.add_sphere_sensors(
                positions  = positions[i],
                radius     = radius,
                activities = activities[i],
                name       = name,
            )

    def add_motion_floor(
        self,
        center       : Union[torch.Tensor, np.ndarray] = np.zeros((3,)),
        seq_length   : int   = 1,
        thickness    : float = 0.1,
        n_grids_half : int   = 5,
        grid_size    : float = 1.0,
        up_dir       : str   = 'y+',      # {'x', 'y', 'z'} * {'+', '-'}
        name         : str   = 'floor',
        offset       : int   = 0,
    ):
        """
        Add a checkerboard floor plane to the wis3d viewer.
        The grid is always aligned with the center, i.e., the center should be the shared corner of the
        four neighboring grids.

        ### Args
            - center (torch.Tensor or np.ndarray): (3,)
                - The center of the floor plane.
            - seq_length: int,
                - The sequence length.
            - thickness: float, default = 0.1
                - The thickness of the floor plane. It will only expand along the downward direction.
            - n_grad_radian: int, default = 5
                - The number of grid squares per side.
            - grid_size: float, default = 1.0
                - The size of one grid square.
            - up_dir: str, default = 'y+'
                - The direction of the up direction. It can be 'x+', 'x-', 'y+', 'y-', 'z+', 'z-'.
                - For example, 'y+' means the up direction is along the positive y-axis.
            - name: str
                - The name of the floor plane.
            - offset: int, default = 0
                - The offset for the sequence index.
        """
        for i in range(seq_length):
            self.set_scene_id(i + offset)
            self.add_floor(
                center       = center,
                thickness    = thickness,
                n_grids_half = n_grids_half,
                grid_size    = grid_size,
                up_dir       = up_dir,
                name         = name,
            )
        # Reset Wis3D scene id.
        self.set_scene_id(0)

    def _make_aabox(
        self,
        min_xyz : np.ndarray,
        max_xyz : np.ndarray,
        colors  : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        <<< Vibe Coding Marked >>>
        Generate a single axis-aligned box from min/max corners in world xyz.

        ### Args
            - min_xyz: (3,) world coords of the min corner
            - max_xyz: (3,) world coords of the max corner
            - colors: (3,) uint8 single color, or (8, 3) uint8 per-vertex

        ### Returns
            - vertices: (8, 3)
                - The vertices of the box.
            - faces: (12, 3)
                - The faces of the box.
            - vertex_colors: (8, 3) uint8
                - The colors of the vertices, from 0 to 255.
        """
        min_xyz = np.asarray(min_xyz).flatten()
        max_xyz = np.asarray(max_xyz).flatten()
        assert min_xyz.shape == (3,) and max_xyz.shape == (3,)
        verts = np.zeros((8, 3))
        for iz in range(2):
            for iy in range(2):
                for ix in range(2):
                    idx = 2 * ix + iy + 4 * iz
                    verts[idx, 0] = min_xyz[0] if ix == 0 else max_xyz[0]
                    verts[idx, 1] = min_xyz[1] if iy == 0 else max_xyz[1]
                    verts[idx, 2] = min_xyz[2] if iz == 0 else max_xyz[2]

        faces = np.array([
                [3, 2, 0], [1, 3, 0],  # z_min
                [7, 5, 4], [6, 7, 4],  # z_max
                [5, 1, 0], [4, 5, 0],  # x_min
                [7, 6, 2], [3, 7, 2],  # x_max
                [6, 4, 0], [2, 6, 0],  # y_min
                [7, 3, 1], [5, 7, 1],  # y_max
            ], dtype=np.int64)
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape == (3,):
            vertex_colors = np.broadcast_to(colors, (8, 3)).copy()
        else:
            assert colors.shape == (8, 3), "colors must be (3,) or (8, 3)"
            vertex_colors = colors.copy()
        return verts, faces, vertex_colors

    def add_floor(
        self,
        center       : Union[torch.Tensor, np.ndarray] = np.zeros((3,)),
        thickness    : float = 0.1,
        n_grids_half : int   = 5,
        grid_size    : float = 1.0,
        up_dir       : str   = 'y+',      # {'x', 'y', 'z'} * {'+', '-'}
        name         : str   = 'floor',
    ):
        """
        <<< Vibe Coding Marked >>>
        Add a checkerboard floor plane to the wis3d viewer.
        The checkerboard is built as n*n pure-color boxes via add_box.

        ### Args
            - center (torch.Tensor or np.ndarray): (3,)
                - The center of the floor plane.
            - thickness: float, default = 0.1
                - The thickness of the floor plane. It will only expand along the downward direction.
            - n_grids_half: int, default = 5
                - Half of the number of grid squares per side.
            - grid_size: float, default = 1.0
                - The size of one grid square.
            - up_dir: str, default = 'y+'
                - The direction of the up direction. It can be 'x+', 'x-', 'y+', 'y-', 'z+', 'z-'.
            - name: str
                - The name of the floor plane.
        """
        assert len(up_dir) == 2 and up_dir[0] in ['x', 'y', 'z'] and up_dir[1] in ['+', '-'], \
            f"Invalid up_dir: {up_dir}. Expected format: 'x+', 'x-', 'y+', 'y-', 'z+', or 'z-'."
        up_axis = up_dir[0]
        up_sign = 1.0 if up_dir[1] == '+' else -1.0
        if up_axis == 'x':
            up_idx, axis1_idx, axis2_idx = 0, 1, 2
        elif up_axis == 'y':
            up_idx, axis1_idx, axis2_idx = 1, 0, 2
        else:
            up_idx, axis1_idx, axis2_idx = 2, 0, 1

        if isinstance(center, torch.Tensor):
            center = center.detach().cpu().numpy()
        center = np.array(center).flatten()
        assert len(center) == 3, f"center should have shape (3,), but got {center.shape}"

        num_grids_per_side = n_grids_half * 2
        length = num_grids_per_side * grid_size
        half_length = length / 2.0
        color_white = np.array([255, 255, 255], dtype=np.uint8)
        color_dark = np.array([64, 64, 64], dtype=np.uint8)

        # Plane extent in world: axis1/axis2 from center ± half_length; up: top = center, bottom = center - thickness*up_sign
        bottom_up = center[up_idx] - thickness * up_sign
        top_up = center[up_idx]
        min_up = min(bottom_up, top_up)
        max_up = max(bottom_up, top_up)

        verts_list, faces_list, colors_list = [], [], []
        vertex_offset = 0
        for i in range(num_grids_per_side):
            for j in range(num_grids_per_side):
                o1 = -half_length + i * grid_size
                o2 = -half_length + j * grid_size
                min_xyz = np.array([
                    center[0], center[1], center[2],
                ], dtype=np.float64)
                max_xyz = min_xyz.copy()
                min_xyz[axis1_idx] = center[axis1_idx] + o1
                max_xyz[axis1_idx] = center[axis1_idx] + o1 + grid_size
                min_xyz[axis2_idx] = center[axis2_idx] + o2
                max_xyz[axis2_idx] = center[axis2_idx] + o2 + grid_size
                min_xyz[up_idx] = min_up
                max_xyz[up_idx] = max_up

                is_white = (i + j) % 2 == 0
                color = color_white if is_white else color_dark
                v, f, c = self._make_aabox(min_xyz, max_xyz, color)
                verts_list.append(v)
                faces_list.append(f + vertex_offset)
                colors_list.append(c)
                vertex_offset += 8

        vertices = np.concatenate(verts_list, axis=0)
        faces = np.concatenate(faces_list, axis=0)
        vertex_colors = np.concatenate(colors_list, axis=0)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = np.hstack([vertex_colors, np.full((len(vertices), 1), 255, dtype=np.uint8)])

        self.add_mesh(
            vertices = mesh,
            name     = name,
        )


    # ===== Overriding methods from original Wis3D. =====


    def add_lines(
        self,
        start_points: torch.Tensor,
        end_points  : torch.Tensor,
        colors      : Optional[Union[list, torch.Tensor]] = None,
        name        : str   = None,
        thickness   : float = 0.01,
        resolution  : int   = 4,
    ):
        """
        Add lines by points. Overriding the original `add_lines` method to use mesh to provide browser from crash.

        ### Args
        - start_points: torch.Tensor, (N, 3), N ~ number of lines
        - end_points: torch.Tensor, (N, 3), N ~ number of lines
        - colors: list or torch.Tensor, (N, 3)
            - The color of the lines, from 0 to 255.
        - name: str
            - The name of the vector.
        - thickness: float, default = 0.01
            - The thickness of the lines.
        - resolution: int, default = 3
            - The 'line' was actually a poly-cylinder, and the resolution how it looks like a cylinder.
        """
        if isinstance(colors, List):
            colors = torch.tensor(colors)

        assert (len(start_points.shape) == 2), 'The input `start_points` should have 2 dimensions: (N, 3).'
        assert (len(end_points.shape) == 2), 'The input `end_points` should have 2 dimensions: (N, 3).'
        assert (start_points.shape == end_points.shape), 'The input `start_points` and `end_points` should have the same shape.'

        # ===== Prepare the data. =====
        N, _ = start_points.shape
        device = start_points.device
        dir = end_points - start_points # (N, 3)
        dis = torch.norm(dir, dim=-1, keepdim=True) # (N, 1)
        dir = dir / dis # (N, 3)
        K = resolution + 1 # the first & the last point share the position
        # Find out directions that are negative to the y-axis.
        vec_y = torch.Tensor([[0, 1, 0]]).float().to(device) # (1, 3)
        neg_mask = (dir @ vec_y.transpose(-1, -2) < 0).squeeze() # (N,)

        # ===== Get the ending surface vertices of the cylinder. =====
        # 1. Get the surface vertices template in x-z plain.
        radius = torch.linspace(0, 2*torch.pi, K) # (K,)
        v_ending_temp = \
            torch.stack(
                [torch.cos(radius), torch.zeros_like(radius), torch.sin(radius)],
                dim = -1
            ) # (K, 3)
        v_ending_temp *= thickness # (K, 3)
        v_ending_temp = v_ending_temp[None].repeat(N, 1, 1) # (N, K, 3)

        # 2. Rotate the template plane to the direction of the line.
        rot_axis = torch.linalg.cross(vec_y, dir) # (N, 3)
        rot_axis[neg_mask] *= -1
        rot_mat = axis_angle_to_matrix(rot_axis) # (N, 3, 3)
        v_ending_temp = v_ending_temp @ rot_mat.transpose(-1, -2)
        v_ending_temp = v_ending_temp.to(device)

        # 3. Move the template plane to the start and end points and get the cylinder vertices.
        v_cylinder_start = v_ending_temp + start_points[:, None] # (N, K, 3)
        v_cylinder_end = v_ending_temp + end_points[:, None] # (N, K, 3)
        #    Swap the start and end points for the negative direction to adjust the normal direction.
        v_cylinder_start[neg_mask], v_cylinder_end[neg_mask] = v_cylinder_end[neg_mask], v_cylinder_start[neg_mask]
        v_cylinder = torch.cat([v_cylinder_start, v_cylinder_end], dim=1) # (N, 2*K, 3)

        # ===== Calculate the face index. =====
        idx = torch.arange(0, 2*K, device=device).to(device) # (2*K,)
        idx_s, idx_e = idx[:K], idx[K:]
        f_cylinder = torch.cat([
            # Two ending surface.
            torch.stack([idx_s[0].repeat(K-2), idx_s[1:-1], idx_s[2:]], dim=-1),
            torch.stack([idx_e[0].repeat(K-2), idx_e[2:], idx_e[1:-1]], dim=-1),
            # The side surface.
            torch.stack([idx_e[:-1], idx_s[1:], idx_s[:-1]], dim=-1),
            torch.stack([idx_e[:-1], idx_e[1:], idx_s[1:]], dim=-1),
        ], dim=0) # (4*K-4, 3)
        f_cylinder = f_cylinder[None].repeat(N, 1, 1) # (N, 4*K-4, 3)

        # ===== Calculate the face index. =====
        if colors is not None:
            c_cylinder = colors / 255.0 # (N, 3)
            c_cylinder = c_cylinder[:, None].repeat(1, 2*K, 1) # (N, 2*K, 3)
        else:
            c_cylinder = None

        N, V = v_cylinder.shape[:2]
        v_cylinder = v_cylinder.reshape(-1, 3) # (N*(2*K), 3)

        # ===== Manually match the points index before flatten. =====
        f_cylinder = f_cylinder + torch.arange(0, N, device=device).unsqueeze(1).unsqueeze(1) * V
        f_cylinder = f_cylinder.reshape(-1, 3) # (N*(4*K-4), 3)
        if c_cylinder is not None:
            c_cylinder = c_cylinder.reshape(-1, 3) # (N*(2*K), 3)

        self.add_mesh(
            vertices      = v_cylinder,
            vertex_colors = c_cylinder,
            faces         = f_cylinder,
            name          = name,
        )