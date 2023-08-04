import numpy as np
import torch
import onnx
import onnxruntime as ort
import time
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

from .cfgs.nuscenes_models.cbgs_dyn_pp_centerpoint import data_cfg
from .export_pointpillar import CenterPointPart0
from mot.ab3dmot import track_1step
from .visualization_tools import show_bird_eye_view


POINT_CLOUD_RANGE = data_cfg.POINT_CLOUD_RANGE
VOXEL_SIZE = data_cfg.VOXEL_SIZE


class Detector(object):
    def __init__(self, score_threshold: float = 0.3):
        self._score_threshold = score_threshold
        self.part0 = CenterPointPart0()
        self.part0.load_params_from_file(filename='/home/host/Desktop/quan_ws/ros_pp/tools/pretrained_models/cbgs_pp_centerpoint_nds6070.pth', to_cpu=True)
        self.part0.eval()
        self.part0.cuda()

        onnx_file = "/home/host/Desktop/quan_ws/ros_pp/tools/pointpillar_onnx/pointpillar_part1_no_nms.onnx"
        onnx_part1 = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_part1)
        print('check complete')
        self.ort_session = ort.InferenceSession(onnx_file, providers=["CUDAExecutionProvider"])

    def detect(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: (N, 6) - batch_idx, x, y, z, intensity, time
        
        Returns:
            boxes: (M, 10) - x, y, z, dx, dy, dz, yaw | score, cls_idx, batch_idx
        """
        assert points.shape[1] == 6, f"expect (batch_idx, x, y, z, intensity, time), get {points.shape}"
        points = torch.from_numpy(points).float().cuda()

        with torch.no_grad():
            voxel_coords, features = self.part0(points)
        
        voxel_coords = voxel_coords.cpu().numpy()
        features = features.cpu().numpy()

        boxes = self.ort_session.run(None, {'voxel_coords': voxel_coords, 'features': features})[0]
        mask_valid = boxes[:, -3] > self._score_threshold
        return boxes[mask_valid]


class Tracktor(object):
    def __init__(self,
                 chosen_class_index: int, 
                 cost_threshold: float = 11.0,
                 min_hit_to_report: int = 2,
                 indices_to_format_input: List[int] = [0, 1, 2, 6, 3, 4, 5], 
                 indices_to_format_output: List[int] = [0, 1, 2, 4, 5, 6, 3],
                 track_couters_init: int = 0,
                 num_miss_to_kill: int = 3):
        """
        State: (x, y, z, yaw, l, w, h, dx, dy, dz, dyaw)
        
        Track = [state || num_hit, num_miss, id]
        
        Observations = (x, y, z, yaw, l, w, h)

        """
        self._num_state = 11
        self._num_info = 3
        self._min_hit_to_report = min_hit_to_report
        self._chosen_class_index = chosen_class_index
        self._cost_threshold = cost_threshold
        
        self._indices_to_format_input = indices_to_format_input
        self._indices_to_format_output = indices_to_format_output

        # init tracks' state & their covariance
        self.tracks = np.zeros((0, self._num_state + self._num_info))
        self.tracks_P = np.zeros((0, self._num_state, self._num_state))
        self.track_counter = track_couters_init
        self.num_miss_to_kill = num_miss_to_kill

    def update_(self, detections: np.ndarray) -> None:
        """
        Inplace function -> result is written on self.tracks & self.tracks_P

        Args:
            detections: (N, 10) - x, y, z, dx, dy, dz, yaw | score, cls_idx, batch_idx
        """
        assert detections.shape[1] == 10, f"expect x, y, z, dx, dy, dz, yaw | score, cls_idx, batch_idx; get {detections.shape}"

        # organize boxes for tracking
        boxes_cls = detections[:, -2].astype(int)
        boxes = detections[:, self._indices_to_format_input]  # (N, 7) - x, y, z, YAW, length, width, height
        
        self.tracks, self.tracks_P, self.track_counter = track_1step(self.tracks, 
                                                                     self.tracks_P, 
                                                                     self.track_counter, 
                                                                     boxes[boxes_cls == self._chosen_class_index], 
                                                                     self._chosen_class_index,
                                                                     self._cost_threshold, self.num_miss_to_kill)
    
    def report_tracks(self) -> Tuple[np.ndarray]:
        """
        Returns:
            track_boxes: (N, 7) - x, y, z, dx, dy, dz, yaw
            track_id: (N,) 
        """
        mask_report_tracks = self.tracks[:, -3] > self._min_hit_to_report
        tracked_boxes = self.tracks[mask_report_tracks]
        tracked_boxes = tracked_boxes[:, self._indices_to_format_output]
        tracked_id = self.tracks[mask_report_tracks, -1].astype(int)
        return tracked_boxes, tracked_id


class DataGenerator(object):
    """
    Only support RSU -> static LiDAR
    """
    def __init__(self, dataroot: Path, num_sweeps: int = 1, remove_points_outside: bool = True, pad_points_with_batch_idx: bool = True):
        self._pc_range = np.array(POINT_CLOUD_RANGE)
        self._root = dataroot
        self._num_sweeps = num_sweeps
        self._remove_points_outside = remove_points_outside
        self._pad_points_with_batch_idx = pad_points_with_batch_idx

        pc_files = list(self._root.glob('*.npy'))
        pc_indices = np.array([int(pcfile.parts[-1].split('pointcloud')[1].split('.')[0]) for pcfile in pc_files])
        sorted_idx = np.argsort(pc_indices)
        self._pc_indices = pc_indices[sorted_idx].tolist()
        self._pc_files = [pc_files[_i] for _i in sorted_idx.tolist()]

    def __len__(self):
        return len(self._pc_indices)
    
    def __getitem__(self, index: int) -> np.ndarray:
        """
        Returns:
            points: (N, 6) - batch_idx, x, y, z, intensity, time_lag (== 0.0)
        """
        assert 0 <= index < len(self), f"index {index} is out of range, must in [0, {len(self)})"
        init_idx = max(index - self._num_sweeps, 0)
        points = []
        for _idx in range(init_idx, init_idx + self._num_sweeps):
            pc = np.load(self._pc_files[_idx])  # (N, 5) - x, y, z, intensity, timestamp_ns
            pc[:, -1] = 0.0  # replace timestamp with time lag -> here, pretend there is no time lag
            points.append(pc)
        
        points = np.concatenate(points)

        if self._remove_points_outside:
            mask_inside = np.logical_and(points[:, :3] > self._pc_range[:3], points[:, :3] < self._pc_range[3:] - 1e-3).all(axis=1)
            points = points[mask_inside]

        if self._pad_points_with_batch_idx:
            points = np.pad(points, pad_width=[(0, 0), (1, 0)], constant_values=0.0)  # (N, 6)        

        return points


def main():
    data_gen = DataGenerator(dataroot=Path('../data/Clouds_02_21-15_02_38_0'), num_sweeps=1)
    detector = Detector(score_threshold=0.2)
    tracktor_ped = Tracktor(chosen_class_index=8, cost_threshold=2.5)

    viz_root = Path('./artifacts/viz_Clouds_02_21-15_02_38_0')
    viz_root.mkdir(exist_ok=True)
    point_cloud_range = np.array(POINT_CLOUD_RANGE)

    track_root = Path('./artifacts/track_Clouds_02_21-15_02_38_0')
    track_root.mkdir(exist_ok=True)

    for d_idx in range(len(data_gen)):
        if d_idx > 300:
            break
        # get data
        points = data_gen[d_idx]
        
        tic = time.time()
        # detection
        boxes = detector.detect(points)
        
        # tracking
        tracktor_ped.update_(boxes)
        tracked_boxes, tracked_id = tracktor_ped.report_tracks()

        tac = time.time()

        print(f'{d_idx} boxes: {boxes.shape}')
        print(f'{d_idx} track_counter: {tracktor_ped.track_counter}')
        print(f'{d_idx} time: {tac - tic}')
        print('------------------------------------------')

        fig, axe = plt.subplots()
        show_bird_eye_view(axe, points[:, 1: 4], tracked_boxes, boxes_id=tracked_id, point_cloud_range=point_cloud_range, 
                           resolution_xy=VOXEL_SIZE[0])
        fig.savefig(viz_root / f'bev_pointcloud_{d_idx}.png')

        track_result = np.concatenate([tracked_boxes, tracked_id.reshape(-1, 1)], axis=1)
        np.save(track_root / f'track_pointcloud_{d_idx}.npy', track_result)


if __name__ == '__main__':
    main()
