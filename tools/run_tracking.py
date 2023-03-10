import numpy as np
from pathlib import Path
from nuscenes import NuScenes
import matplotlib.pyplot as plt
import time

from nuscenes_tools import get_nuscenes_point_cloud, get_nuscenes_sensor_pose_in_global
from geometry_tools import apply_tf
from visualization_tools import show_point_cloud, show_bird_eye_view
from mot.ab3dmot import track_1step


POINT_CLOUD_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])


def main(scene_idx, init_sample_idx):
    detection_root = Path(f'artifacts/nuscenes_scene{scene_idx}_initSampleIdx{init_sample_idx}')
    dets_file = list(detection_root.glob('*.npy'))
    dets_file_idx = np.array([int(file.parts[-1].split('_')[-1].split('.')[0]) for file in dets_file])
    sorted_idx = np.argsort(dets_file_idx)
    dets_file_idx = dets_file_idx[sorted_idx].tolist()
    dets_file = [dets_file[idx] for idx in sorted_idx.tolist()]

    viz_root = Path(f'artifacts/viz_nuscenes_scene{scene_idx}_initSampleIdx{init_sample_idx}')
    viz_root.mkdir(exist_ok=True)

    nusc = NuScenes(dataroot='/home/user/dataset/nuscenes', version='v1.0-mini', verbose=False)
    target_SE3_glob = None

    # init tracks
    num_state, num_info = 11, 3
    tracks = np.zeros((0, num_state + num_info))
    tracks_P = np.zeros((0, num_state, num_state))
    track_counter = 0

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

        # tracking
        # =============================================================
        
        # organize boxes for tracking
        boxes_cls = boxes[:, -2].astype(int)
        boxes = boxes[:, [0, 1, 2, 6, 3, 4, 5]]  # (N, 7) - x, y, z, YAW, length, width, height
        chosen_cls_idx = 0  # car

        # invoke track
        tic = time.time()
        tracks, tracks_P, track_counter = track_1step(tracks, tracks_P, track_counter, boxes[boxes_cls == chosen_cls_idx], chosen_cls_idx)
        print(f"{sample_idx} | track time: {time.time() - tic}")

        # format output
        mask_report_tracks = tracks[:, -3] > 1
        tracked_boxes = tracks[mask_report_tracks]
        tracked_boxes = tracked_boxes[:, [0, 1, 2, 4, 5, 6, 3]]
        tracked_id = tracks[mask_report_tracks, -1].astype(int)

        # display
        # ============================================================
        print(f"{sample_idx} | track_counter: {track_counter}")
        print('-------------------------')
        boxes = boxes[:, [0, 1, 2, 4, 5, 6, 3]]
        # show_point_cloud(points[:, 1: 4], boxes, show_lidar_frame=True)
        
        fig, axe = plt.subplots()
        show_bird_eye_view(axe, points[:, 1: 4], tracked_boxes, boxes_id=tracked_id, point_cloud_range=POINT_CLOUD_RANGE, resolution_xy=0.2)
        fig.savefig(viz_root / f'bev_sample_{sample_idx}.png')

        # break


if __name__ == '__main__':
    main(scene_idx=2, init_sample_idx=5)

