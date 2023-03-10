import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Union
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes


Tensor = Union[torch.Tensor, np.ndarray]


def apply_tf(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert points.shape[1] == 3
    assert tf.shape == (4, 4)
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    out = tf @ points_homo.T
    return out[:3, :].T


def find_boxes_corners(boxes: Tensor) -> List:
    # box convention:
    # forward: 0 - 1 - 2 - 3, backward: 4 - 5 - 6 - 7, up: 0 - 1 - 5 - 4

    xs = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float) / 2.0
    ys = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float) / 2.0
    zs = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float) / 2.0
    out = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        dx, dy, dz = box[3: 6].tolist()
        vers = np.concatenate([xs.reshape(-1, 1) * dx, ys.reshape(-1, 1) * dy, zs.reshape(-1, 1) * dz], axis=1)  # (8, 3)
        ref_from_box = np.eye(4)
        yaw = box[6]
        cy, sy = np.cos(yaw), np.sin(yaw)
        ref_from_box[:3, :3] = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])
        ref_from_box[:3, 3] = box[:3]
        vers = apply_tf(ref_from_box, vers)
        out.append(vers)

    return out


def show_point_cloud(xyz: Tensor, boxes: Tensor = None, xyz_color: Tensor = None, boxes_color: Tensor = None, poses: Tensor = None, 
                     show_box_velo=False,
                     points_lane_direction: Tensor = None,
                     mask_on_drivable_area: Tensor = None,
                     mask_foreground: Tensor = None,
                     show_lidar_frame: bool = False) -> None:
    """
    Visualize pointcloud & annotations
    Args:
        xyz: (N, 3)
        boxes: (N_b, 7[+2][+1])
        xyz_color: (N, 3) - r, g, b
        boxes_color: (N_b, 3) - r, g, b
        poses: (Np, 3) - x, y, yaw
    """
    def create_cube(vers, box_color):
        # vers: (8, 3)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # front
            [4, 5], [5, 6], [6, 7], [7, 4],  # back
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting front & back
            [0, 2], [1, 3]  # denote forward face
        ]
        if box_color is None:
            colors = [[1, 0, 0] for _ in range(len(lines))]  # red
        else:
            colors = [box_color for _ in range(len(lines))]
        cube = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vers),
            lines=o3d.utility.Vector2iVector(lines),
        )
        cube.colors = o3d.utility.Vector3dVector(colors)
        return cube

    def rot_z(yaw):
        cos, sin = np.cos(yaw), np.sin(yaw)
        rot = np.array([
            cos, -sin, 0,
            sin, cos, 0,
            0, 0, 1
        ]).reshape(3, 3)
        return rot

    def xyyaw2pose(x, y, yaw):
        out = np.eye(4)
        out[:3, :3] = rot_z(yaw)
        out[:2, -1] = [x, y]
        return out

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if xyz_color is not None:
        pcd.colors = o3d.utility.Vector3dVector(xyz_color)

    obj_to_display = [pcd]

    if boxes is not None:
        assert len(boxes.shape) == 2 and boxes.shape[1] >= 7, f"expect (?, 7), get {boxes.shape}"
        boxes_corners = find_boxes_corners(boxes)  # list of (8, 3)
        o3d_cubes = [create_cube(box_corners, boxes_color[b_idx] if boxes_color is not None else None)
                     for b_idx, box_corners in enumerate(boxes_corners)]
        obj_to_display += o3d_cubes

    if poses is not None:
        for pidx in range(poses.shape[0]):
            pose = xyyaw2pose(*poses[pidx].tolist())
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            frame = frame.transform(pose)
            obj_to_display.append(frame)
    
    if show_box_velo:
        assert boxes.shape[1] > 8, f"expect boxes.shape[1] > 8, get {boxes.shape[1]}"
        # for each box having non-zero velo, draw a line @ its center corresponding to its displacement in the next 0.5 sec,
        for b_idx in range(boxes.shape[0]):
            velo = boxes[b_idx, [-3, -2]]
            if torch.linalg.norm(velo) < 0.5:
                continue
            vers = torch.stack([boxes[b_idx, :3], boxes[b_idx, :3]], dim=0)
            vers[1, :2] = vers[1, :2] + velo * 0.5  # velocity * 0.5 sec
            o3d_displace = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vers),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            obj_to_display.append(o3d_displace)

    if points_lane_direction is not None:
        assert xyz.shape[0] == points_lane_direction.shape[0],\
              f"expect xyz.shape[0] == points_lane_direction.shape[0]; get {xyz.shape[0]} != {points_lane_direction.shape[0]}"
        assert mask_on_drivable_area is not None
        assert mask_foreground is not None
        mask_viz_lane_dir = torch.logical_and(mask_on_drivable_area, mask_foreground)
        pts, lane_dir = xyz[mask_viz_lane_dir], points_lane_direction[mask_viz_lane_dir]
        for pidx in range(0, pts.shape[0], 10):
            direction = torch.stack([pts[pidx], pts[pidx]], dim=0)
            direction[1, 0] = direction[1, 0] + np.cos(lane_dir[pidx]) * 1.0  # length of vector representing lane_dir = 1.0 (m)
            direction[1, 1] = direction[1, 1] + np.sin(lane_dir[pidx]) * 1.0  # length of vector 
            o3d_dir = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(direction),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            o3d_dir.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
            obj_to_display.append(o3d_dir)
    
    if show_lidar_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=np.zeros(3))
        obj_to_display.append(mesh_frame)

    o3d.visualization.draw_geometries(obj_to_display)


def show_bird_eye_view(axe: Axes, 
                       xyz: np.ndarray, boxes: np.ndarray, 
                       point_cloud_range: np.ndarray,
                       resolution_xy: float,
                       boxes_color: np.ndarray = None, boxes_id: np.ndarray = None):
    
    # Show Points:
    # -----------------
    bev_size_xy = np.ceil((point_cloud_range[3: 5] - point_cloud_range[:2]) / resolution_xy).astype(int)
    bev_img = np.zeros((bev_size_xy[1], bev_size_xy[0]))  # (H, W)
    
    # find occupied pillars
    mask_in_range = np.logical_and(xyz[:, :2] > point_cloud_range[:2], xyz[:, :2] < point_cloud_range[3: 5]).all(axis=1)
    bev_xy = np.floor((xyz[mask_in_range, :2] - point_cloud_range[:2]) / resolution_xy).astype(int)  # (N_pts, 2)
    bev_flat_coord = bev_xy[:, 1] * bev_size_xy[0] + bev_xy[:, 0]  # (N_pts)
    unq_coord = np.unique(bev_flat_coord)
    occ_pillar_y = unq_coord // bev_size_xy[0]
    occ_pillar_x = unq_coord % bev_size_xy[0]
    bev_img[occ_pillar_y, occ_pillar_x] = 1.0
    
    axe.imshow(bev_img, origin='lower')

    # Show Boxes:
    # -----------------
    boxes_vers = find_boxes_corners(boxes)  # List[(8, 3)]
    for bidx, vers in enumerate(boxes_vers):
        top = vers[[0, 1, 5, 4], :2]  # (4, 2)
        # to bev
        top = (top - point_cloud_range[:2]) / resolution_xy  # (N_pts, 2)
        axe.plot(top[[0, 1, 2, 3, 0], 0], top[[0, 1, 2, 3, 0], 1], 'r-')
        
        top_center = np.mean(top[[0, 2]], axis=0)
        mid_forward_edge = np.mean(top[[0, 1]], axis=0)
        axe.plot([top_center[0], mid_forward_edge[0]], [top_center[1], mid_forward_edge[1]], 'b-')

        if boxes_id is not None:
            this_id = boxes_id[bidx]
            axe.annotate(str(this_id), 
                         xy=(top_center[0], top_center[1]), xycoords='data',
                         xytext=(top_center[0], top_center[1]), textcoords='data',
                         size=15, color='white',
                         va="center", ha="center")
            
    axe.set_xlim([0, bev_size_xy[0]])
    axe.set_ylim([0, bev_size_xy[1]])
