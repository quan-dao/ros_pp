from nuscenes.nuscenes import NuScenes
from typing import Tuple

from geometry_tools import *


def get_nuscenes_sensor_pose_in_ego_vehicle(nusc: NuScenes, curr_sd_token: str):
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_cs_rec = nusc.get('calibrated_sensor', curr_rec['calibrated_sensor_token'])
    ego_from_curr = tf(curr_cs_rec['translation'], curr_cs_rec['rotation'])
    return ego_from_curr


def get_nuscenes_sensor_pose_in_global(nusc: NuScenes, curr_sd_token: str):
    ego_from_curr = get_nuscenes_sensor_pose_in_ego_vehicle(nusc, curr_sd_token)
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_ego_rec = nusc.get('ego_pose', curr_rec['ego_pose_token'])
    glob_from_ego = tf(curr_ego_rec['translation'], curr_ego_rec['rotation'])
    glob_from_curr = glob_from_ego @ ego_from_curr
    return glob_from_curr


def get_nuscenes_point_cloud(nusc: NuScenes, scene_idx: int, target_sample_idx: int, 
                             pad_points_with_batch_idx: True, 
                             point_cloud_range: np.ndarray = None) -> Tuple[np.ndarray, str]:
    """
    Get point cloud in LiDAR frame
    """
    # get sample token
    scene = nusc.scene[scene_idx]
    sample_tk = scene['first_sample_token']
    for _ in range(target_sample_idx):
        sample = nusc.get('sample', sample_tk)
        sample_tk = sample['next']

    sample = nusc.get('sample', sample_tk)

    # get points
    pcfile = nusc.get_sample_data_path(sample['data']['LIDAR_TOP'])
    pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (N, 4) - (x, y, z, intensity)
    pc = np.pad(pc, pad_width=[(0, 0), (0, 1)], mode='constant', constant_values=0.0)  # (N, 5)

    if pad_points_with_batch_idx:
        pc = np.pad(pc, pad_width=[(0, 0), (1, 0)], constant_values=0.0)

    if point_cloud_range is not None:
        mask_in_range = np.all(np.logical_and(pc[:, 1: 4] > point_cloud_range[:3], pc[:, 1: 4] < point_cloud_range[3:] - 1e-3), axis=1)
        pc = pc[mask_in_range]

    return pc, sample_tk

