import numpy as np
import torch
import pickle
from torch2trt import torch2trt

from common_utils import create_logger

from cfgs.nuscenes_models.cbgs_voxel01_res3d_centerpoint import data_cfg, model_cfg
from ros_pp.models.backbones_3d.vfe import MeanVFE
from ros_pp.models.backbones_3d import VoxelResBackBone8x
from ros_pp.models.backbones_2d import HeightCompression, BaseBEVBackbone
from ros_pp.models.dense_heads import CenterHead
from ros_pp.models.detectors import Detector3DTemplate
from o3d_visualization import PointsPainter


class CenterPoint_Part3D(Detector3DTemplate):
    def __init__(self):
        super().__init__()
        point_cloud_range = data_cfg.POINT_CLOUD_RANGE
        voxel_size = data_cfg.VOXEL_SIZE
        grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).astype(int)

        self.vfe = MeanVFE(model_cfg.VFE, data_cfg.NUM_POINT_FEATURES)
        self.backbone_3d = VoxelResBackBone8x(model_cfg.BACKBONE_3D, 
                                              self.vfe.get_output_feature_dim(), 
                                              grid_size)
        self.map_to_bev_module = HeightCompression(model_cfg.MAP_TO_BEV)
        
    def forward(self, points: torch.Tensor):
        """
        Args:
            points: (N, 3 + C)
        
        Returns:
            pred_boxes: (N, 10)
        """
        points_mean, voxel_coords = self.vfe(points)
        encoded_spconv_tensor = self.backbone_3d(points_mean, voxel_coords)
        spatial_features = self.map_to_bev_module(encoded_spconv_tensor)
        return spatial_features


class CenterPoint_Part2D(Detector3DTemplate):
    def __init__(self):
        super().__init__()
        point_cloud_range = data_cfg.POINT_CLOUD_RANGE
        voxel_size = data_cfg.VOXEL_SIZE
        grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size).astype(int)

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


def main():
    logger = create_logger('artifact/blah.txt')
    part3d = CenterPoint_Part3D()
    part3d.load_params_from_file('./pretrained_models/cbgs_voxel01_centerpoint_nds_6454.pth', logger=logger)
    part3d.eval()
    part3d.cuda()

    with open('artifact/one_nuscenes_point_cloud.pkl', 'rb') as f:
        data = pickle.load(f)

    # pad points with time
    points = torch.from_numpy(np.pad(data['points'], pad_width=[(0, 0), (0, 1)], constant_values=0)).float().cuda()
    print(f"poitns: {points.shape}")

    part2d = CenterPoint_Part2D()
    part2d.load_params_from_file('./pretrained_models/cbgs_voxel01_centerpoint_nds_6454.pth', logger=logger)
    part2d.eval()
    part2d.cuda()

    # first run
    spatial_features = part3d(points)

    # convert to TensorRT feeding sample data as input
    part2d_trt = torch2trt(part2d, [spatial_features])

    torch.save(part2d_trt.state_dict(), 'artifacts/second_part2d_trt.pth')
    print('artifacts/second_part2d_trt.pth')


if __name__ == '__main__':
    main()



