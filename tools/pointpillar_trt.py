import numpy as np
import torch
import pickle
import argparse
import time

from spconv.pytorch.utils import PointToVoxel

from ros_pp.models.detectors import Detector3DTemplate
from cfgs.nuscenes_models.cbgs_pseudo_dyn_pp_centerpoint import model_cfg, data_cfg
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
        self.patch_generator = PatchGenerator(model_cfg.PATCH_GENERATOR)
        self.vfe = PseudoDynamicPillarVFE(model_cfg.VFE, data_cfg.NUM_POINT_FEATURES, voxel_size, self.patch_generator.patch_point_cloud_range)
        self.map_to_bev_module = PointPillarScatter(model_cfg.MAP_TO_BEV, [self.patch_generator.patch_grid_size, self.patch_generator.patch_grid_size, 1])

    def forward(self, points: torch.Tensor):
        voxel_features, coords, voxel_num_points = self.patch_generator(points)
        voxel_coords, features = self.vfe(voxel_features, coords, voxel_num_points)
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


@torch.no_grad()
def make_trt():
    from torch2trt import torch2trt


    part3d = PointPillar_Part3D()
    part3d.load_params_from_file(path_pretrained_weights, logger=logger)
    part3d.eval().cuda()

    with open('artifacts/one_nuscenes_point_cloud.pkl', 'rb') as f:
        data = pickle.load(f)
        # pad points with batch_idx & time
        points = torch.from_numpy(np.pad(data['points'], pad_width=[(0, 0), (0, 1)], constant_values=0)).float().cuda()

    part2d = PointPillar_Part2D()
    part2d.load_params_from_file(path_pretrained_weights, logger=logger)
    part2d.eval().cuda()

    # first run
    spatial_features = part3d(points)

    # convert to TensorRT feeding sample data as input
    part2d_trt = torch2trt(part2d, [spatial_features])

    torch.save(part2d_trt.state_dict(), f'artifacts/{trt_filename}') 
    print(f'artifacts/{trt_filename}')


@torch.no_grad()
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
        torch.tensor([1]).long().cuda(),
    ]

    for idx_frame in range(10):
        with open(f'artifacts/frame{idx_frame}_nuscenes_point_cloud.pkl', 'rb') as f:
            data = pickle.load(f)
            # pad points with time
            points = torch.from_numpy(np.pad(data['points'], pad_width=[(0, 0), (0, 1)], constant_values=0)).float().cuda()

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


class PatchGenerator(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.point_cloud_range = np.array(model_cfg.POINT_CLOUD_RANGE)
        self.patch_stride = model_cfg.PATCH_STRIDE
        self.patch_radius = model_cfg.PATCH_RADIUS
        self.patch_grid_size = np.floor(2.0 * model_cfg.PATCH_RADIUS / model_cfg.VOXEL_SIZE).astype(int).item()
        self.patch_num_min_points = model_cfg.PATCH_NUM_MIN_POINTS
        self.max_num_patches = model_cfg.MAX_NUM_PATCHES
        self.patch_point_cloud_range = np.array([-model_cfg.PATCH_RADIUS, -model_cfg.PATCH_RADIUS, -5.0, model_cfg.PATCH_RADIUS, model_cfg.PATCH_RADIUS, 3.0])
        
        self._voxel_generator = PointToVoxel(
            vsize_xyz=model_cfg.VOXEL_SIZE,
            coors_range_xyz=self.patch_point_cloud_range,
            num_point_features=model_cfg.NUM_POINT_FEATURES,
            max_num_points_per_voxel=model_cfg.MAX_NUM_POINTS_PER_VOXEL,
            max_num_voxels=model_cfg.MAX_NUM_VOXELS,
            device=torch.device("cuda:0")
        )
        # ---- cache repetitive computation
        grid_dxdy = np.floor((self.point_cloud_range[3: 5] - self.point_cloud_range[:2]) / self.patch_stride).astype(int)
        xx, yy = np.meshgrid(np.arange(grid_dxdy[0]), np.arange(grid_dxdy[1]))
        self.patch_center_grid_coord = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        self.patch_center_3d_coord = (self.patch_center_grid_coord.astype(float) + 0.5) * self.patch_stride + self.point_cloud_range[:2]  # (P, 2)
        # to torch.Tensor
        self.patch_center_3d_coord = torch.from_numpy(self.patch_center_3d_coord).float().cuda()
        self.patch_ori = torch.atan2(self.patch_center_3d_coord[:, 1], self.patch_center_3d_coord[:, 0])
        
    def forward(self, points: torch.Tensor):
        """
        Args:
            points: (N, 3+C) - x, y, z, C-channel
        """
        # ---- point-to-patch correspondance
        
        # remove points outside of range
        mask_in_range = torch.logical_and(points[:, :3] > self.point_cloud_range[:3], points[:, :3] < self.point_cloud_range[3:] - 1e-3).all(dim=1)
        points = points[mask_in_range]  # (N, 3 + C)

        # find points in each patch
        dist_points2patch = torch.norm(points[:, :2].unsqueeze(1) - patch_center_3d_coord.unsqueeze(0), dim=2)
        mask_points_in_patch = dist_points2patch < self.patch_radius  # (N, P)
        
        # filter patches having small #points
        mask_empty_patch = mask_points_in_patch.sum(dim=0) < self.patch_num_min_points  # (P,)
        mask_valid_patch = torch.logical_not(mask_empty_patch)
        
        # apply mask
        mask_points_in_patch = mask_points_in_patch[:, mask_valid_patch]  # (N, P)
        patch_center_3d_coord = self.patch_center_3d_coord[mask_valid_patch]  # (P, 2)
        patch_ori = self.patch_ori[mask_valid_patch]  # (P,)
        # -----

        if patch_center_3d_coord.shape[0] > self.max_num_patches:
            dist_patch2lidar = torch.norm(patch_center_3d_coord, dim=1)
            _, sorted_indices = torch.sort(dist_patch2lidar)
            sorted_indices = sorted_indices[:self.max_num_patches]

            mask_points_in_patch[:, sorted_indices]
            patch_center_3d_coord = patch_center_3d_coord[sorted_indices]
            patch_ori = patch_ori[sorted_indices]

        # ---- main loop: map points to patch canonical coord & voxelize
        all_voxel_feat, all_voxel_coord, all_num_points = list(), list(), list()
        for idx_patch in range(patch_center_3d_coord.shape[0]):
            # extract
            points_in_patch = points[mask_points_in_patch[:, idx_patch]]  # (N, 3 + C)
            
            # map to canonical
            # translate
            points_in_patch[:, :2] -= patch_center_3d_coord[idx_patch, :2]
            # rotate
            cos, sin = torch.cos(patch_ori[idx_patch]), torch.sin(patch_ori[idx_patch])
            rot = torch.stack([
                cos,  -sin,
                sin,  cos 
            ], dim=1).reshape(2, 2)
            points_in_patch[:, :2] = torch.matmul(points_in_patch[:, :2], rot)

            # voxelize
            voxel_features, _coords, voxel_num_points = self._voxel_generator(points)
            # add batch_idx to voxel
            voxel_coords = _coords.new_zeros(_coords.shape[0], 4)
            voxel_coords[:, 0] = idx_patch
            voxel_coords[:, 1:] = _coords

            # store
            all_voxel_feat.append(voxel_features)
            all_voxel_coord.append(voxel_coords)
            all_num_points.append(voxel_num_points)
        # ---

        all_voxel_feat = torch.cat(all_voxel_feat)
        all_voxel_coord = torch.cat(all_voxel_coord)
        all_num_points = torch.cat(all_num_points)

        return all_voxel_feat, all_voxel_coord, all_num_points, patch_center_3d_coord, patch_ori


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--make_trt', type=int, default=0)
    parser.add_argument('--inference', type=int, default=0)
    args = parser.parse_args()
    
    if args.make_trt == 1:
        make_trt()
    elif args.inference == 1:
        inference()
