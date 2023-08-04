import numpy as np
import torch
import pickle
import argparse
import time

from ros_pp.models.detectors import Detector3DTemplate
from cfgs.nuscenes_models.cbgs_dyn_pp_centerpoint import model_cfg, data_cfg
from ros_pp.models.backbones_3d.vfe.dynamic_pillar_vfe import DynamicPillarVFE, PseudoDynamicPillarVFE
from ros_pp.models.backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter
from ros_pp.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
from ros_pp.models.dense_heads.center_head import CenterHead

from common_utils import create_logger
from second_trt import generate_predicted_boxes


trt_filename = 'pointpillar_part2d_trt.pth'
path_pretrained_weights = './pretrained_models/cbgs_pp_centerpoint_nds6070.pth'

logger = create_logger('artifacts/blah.txt')
voxel_size = np.array(data_cfg.VOXEL_SIZE)  # size_x, size_y, size_z
point_cloud_range = np.array(data_cfg.POINT_CLOUD_RANGE)  # x_min, y_min, z_min, x_max, y_max, z_max
grid_size = np.floor((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).astype(int)  # 


class PointPillar_Part3D(Detector3DTemplate):
    def __init__(self):
        super().__init__()
        self.vfe = PseudoDynamicPillarVFE(model_cfg.VFE, data_cfg.NUM_POINT_FEATURES)
        self.map_to_bev_module = PointPillarScatter(model_cfg.MAP_TO_BEV, grid_size)

    def forward(self, points: torch.Tensor):
        voxel_coords, features = self.vfe(points)
        spatial_features = self.map_to_bev_module(voxel_coords, features)
        return spatial_features


class PointPillar_Part2D(Detector3DTemplate):
    def __init__(self):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(model_cfg.BACKBONE_2D, 
                                           model_cfg.MAP_TO_BEV.NUM_BEV_FEATURES)
        self.dense_head = CenterHead(model_cfg.DENSE_HEAD, 
                                     self.backbone_2d.get_output_feature_dim(),
                                     len(data_cfg.CLASSES), 
                                     data_cfg.CLASSES, 
                                     grid_size, point_cloud_range, voxel_size)
    
    def forward(self, spatial_features: torch.Tensor):
        """
        Args:
            points: (N, 3 + C)
        
        Returns:
            pred_boxes: (N, 10)
        """
        spatial_features_2d = self.backbone_2d(spatial_features)
        pred_boxes = self.dense_head(spatial_features_2d)
        return pred_boxes


def make_trt():
    from torch2trt import torch2trt


    part3d = PointPillar_Part3D()
    part3d.load_params_from_file(path_pretrained_weights, logger=logger)
    part3d.eval().cuda()

    with open('artifacts/one_nuscenes_point_cloud.pkl', 'rb') as f:
        data = pickle.load(f)
        # pad points with batch_idx & time
        points = torch.from_numpy(np.pad(data['points'], pad_width=[(0, 0), (1, 1)], constant_values=0)).float().cuda()

    part2d = PointPillar_Part2D()
    part2d.load_params_from_file(path_pretrained_weights, logger=logger)
    part2d.eval().cuda()

    # first run
    spatial_features = part3d(points)

    # convert to TensorRT feeding sample data as input
    part2d_trt = torch2trt(part2d, [spatial_features])

    torch.save(part2d_trt.state_dict(), f'artifacts/{trt_filename}') 
    print(f'artifacts/{trt_filename}')


def inference():
    from torch2trt import TRTModule

    
    part3d = PointPillar_Part3D()
    part3d.load_params_from_file(path_pretrained_weights, logger=logger)
    part3d.eval().cuda()

    part2d_trt = TRTModule()
    part2d_trt.load_state_dict(torch.load(f'artifacts/{trt_filename}'))

    # ------ inference param
    heads_cls_idx = [
        torch.tensor([0]).long().cuda(),
        torch.tensor([1, 2]).long().cuda(),
        torch.tensor([3, 4]).long().cuda(),
        torch.tensor([5]).long().cuda(),
        torch.tensor([6, 7]).long().cuda(),
        torch.tensor([8, 9]).long().cuda(),
    ]

    for idx_frame in range(10):
        with open(f'artifacts/frame{idx_frame}_nuscenes_point_cloud.pkl', 'rb') as f:
            data = pickle.load(f)
            # pad points with batch_idx & time
            points = torch.from_numpy(np.pad(data['points'], pad_width=[(0, 0), (1, 1)], constant_values=0)).float().cuda()

        start = time.time()
        # ----- forward
        spatial_features = part3d(points)
        heads_hm, heads_center, heads_center_z, heads_dim, heads_rot = part2d_trt(spatial_features)
        batch_boxes = generate_predicted_boxes(heads_hm, heads_center, heads_center_z, heads_dim, heads_rot,
                                            feature_map_stride=model_cfg.DENSE_HEAD.FEATURE_MAP_STRIDE,
                                            voxel_size=data_cfg.VOXEL_SIZE,
                                            point_cloud_range=data_cfg.POINT_CLOUD_RANGE,
                                            post_proc_cfg=model_cfg.DENSE_HEAD.POST_PROCESSING, 
                                            heads_cls_idx=heads_cls_idx)
        end = time.time()
        print(f'exec time frame {idx_frame}: ', end - start)
    
        data_out = {'points': points, 'pred_boxes': batch_boxes}
        torch.save(data_out, f'artifacts/pillar_data_out_trt_frame{idx_frame}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--make_trt', type=int, default=0)
    parser.add_argument('--inference', type=int, default=0)
    args = parser.parse_args()
    
    if args.make_trt == 1:
        make_trt()
    elif args.inference == 1:
        inference()