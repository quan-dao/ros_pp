import numpy as np
import torch
import pickle

from common_utils import create_logger

from cfgs.nuscenes_models.cbgs_voxel01_res3d_centerpoint import data_cfg, model_cfg
from ros_pp.models.backbones_3d.vfe import MeanVFE
from ros_pp.models.backbones_3d import VoxelResBackBone8x
from ros_pp.models.backbones_2d import HeightCompression, BaseBEVBackbone
from ros_pp.models.dense_heads import CenterHead
from ros_pp.models.detectors import Detector3DTemplate
from o3d_visualization import PointsPainter


class CenterPoint(Detector3DTemplate):
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
        self.backbone_2d = BaseBEVBackbone(model_cfg.BACKBONE_2D, 
                                           self.map_to_bev_module.get_output_feature_dim())
        self.dense_head = CenterHead(model_cfg.DENSE_HEAD, 
                                     self.backbone_2d.get_output_feature_dim(),
                                     len(data_cfg.CLASSES), 
                                     data_cfg.CLASSES, 
                                     grid_size, point_cloud_range, voxel_size)
    
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
        spatial_features_2d = self.backbone_2d(spatial_features)
        pred_boxes = self.dense_head(spatial_features_2d)
        return pred_boxes


def make_test_batch():
    from nuscenes import NuScenes
    from nuscenes_utils import get_one_pointcloud
    from o3d_visualization import PointsPainter
    import pickle

    nusc = NuScenes(dataroot='/home/user/dataset/nuscenes', verbose=True)
    scene = nusc.scene[0]
    sample_tk = scene['first_sample_token']
    for idx in range(10):
        sample = nusc.get('sample', sample_tk)
        points = get_one_pointcloud(nusc, sample['data']['LIDAR_TOP'])

        # painter = PointsPainter(points[:, :3])
        # painter.show()

        data = {'points': points}
        with open(f'./artifacts/frame{idx}_nuscenes_point_cloud.pkl', 'wb') as f:
            pickle.dump(data, f)

        # move on
        sample_tk = sample['next']

    print('data is saved to: artifacts/one_nuscenes_point_cloud.pkl')


def main():
    logger = create_logger('artifact/blah.txt')
    second = CenterPoint()

    print('---')
    print(second)
    print('---')

    second.load_params_from_file('./pretrained_models/cbgs_voxel01_centerpoint_nds_6454.pth', logger=logger)
    second.eval()
    second.cuda()

    with open('artifact/one_nuscenes_point_cloud.pkl', 'rb') as f:
        data = pickle.load(f)

    # pad points with time
    points = torch.from_numpy(np.pad(data['points'], pad_width=[(0, 0), (0, 1)], constant_values=0)).float().cuda()
    print(f"poitns: {points.shape}")

    pred_boxes = second(points)
    data_out = {'points': points, 'pred_boxes': pred_boxes}
    torch.save(data_out, 'artifact/data_out.pth')


def viz_prediction():
    for idx_frame in range(10):
        data_out = torch.load(f'artifacts/data_out_trt_half0_frame{idx_frame}.pth', map_location=torch.device('cpu'))
        for k, v in data_out.items():
            print(f"{k} | {v.shape}")

        points = data_out['points'].detach()
        pred_boxes = data_out['pred_boxes'].detach()

        painter = PointsPainter(points[:, :3], pred_boxes[:, :7])
        painter.show()


if __name__ == '__main__':
    viz_prediction()

