import numpy as np
from pathlib import Path
from nuscenes import NuScenes

from nuscenes_tools import get_nuscenes_point_cloud, get_nuscenes_sensor_pose_in_global
from geometry_tools import apply_tf
from visualization_tools import show_point_cloud


POINT_CLOUD_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])


def main(scene_idx, init_sample_idx):
    detection_root = Path(f'artifacts/nuscenes_scene{scene_idx}_initSampleIdx{init_sample_idx}')
    dets_file = list(detection_root.glob('*.npy'))
    dets_file_idx = np.array([int(file.parts[-1].split('_')[-1].split('.')[0]) for file in dets_file])
    sorted_idx = np.argsort(dets_file_idx)
    dets_file_idx = dets_file_idx[sorted_idx].tolist()
    dets_file = [dets_file[idx] for idx in sorted_idx.tolist()]

    nusc = NuScenes(dataroot='/home/user/dataset/nuscenes', version='v1.0-mini')
    target_SE3_glob = None

    for sample_idx, filename in zip(dets_file_idx, dets_file):
        print(f'{sample_idx} | {filename}') 
        # get points & boxes
        points, sample_tk = get_nuscenes_point_cloud(nusc, scene_idx, sample_idx, True, POINT_CLOUD_RANGE)
        boxes = np.load(filename)  # (N, 10) - x, y, z, dx, dy, dz, yaw | score, cls_idx, batch_idx

        # map points & boxes to global frame (1st ego vehicle)
        sample = nusc.get('sample', sample_tk)
        glob_SE3_lidar = get_nuscenes_sensor_pose_in_global(nusc, sample['data']['LIDAR_TOP'])
        if target_SE3_glob is None:
            target_SE3_glob = np.linalg.inv(glob_SE3_lidar)
        
        target_SE3_lidar = target_SE3_glob @ glob_SE3_lidar

        points[:, 1: 4] = apply_tf(target_SE3_lidar, points[:, 1: 4])

        cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
        zeros, ones = np.zeros(boxes.shape[0]), np.ones(boxes.shape[0])
        boxes_ori = np.stack([
            cos, -sin, zeros,
            sin, cos, zeros,
            zeros, zeros, ones
        ], axis=1).reshape(boxes.shape[0], 3, 3)  # (N, 3, 3)
        lidar_SE3_boxes = np.pad(boxes_ori, pad_width=[(0, 0), (0, 1), (0, 1)], constant_values=0.0)  # (N, 4, 4)
        lidar_SE3_boxes[:, :, -1] = np.pad(boxes[:, :3], pad_width=[(0, 0), (0, 1)], constant_values=1.0)

        target_SE3_boxes = np.einsum('ij, bjk -> bik', target_SE3_lidar, lidar_SE3_boxes)  # (N, 4, 4)
        boxes[:, :3] = target_SE3_boxes[:, :3, -1]
        boxes[:, 6] = np.arctan2(target_SE3_boxes[:, 1, 0], target_SE3_boxes[:, 0, 0])

        # display
        show_point_cloud(points[:, 1: 4], boxes, show_lidar_frame=True)


if __name__ == '__main__':
    main(scene_idx=2, init_sample_idx=5)

