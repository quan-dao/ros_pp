import numpy as np
import torch
import torch.nn.functional as F
import pickle
from typing import List
from easydict import EasyDict 
import argparse
import time

from common_utils import create_logger

from cfgs.nuscenes_models.cbgs_voxel01_res3d_centerpoint import data_cfg, model_cfg
from ros_pp.models.backbones_3d.vfe import MeanVFE
from ros_pp.models.backbones_3d import VoxelResBackBone8x
from ros_pp.models.backbones_2d import HeightCompression, BaseBEVBackbone
from ros_pp.models.dense_heads import CenterHead
from ros_pp.models.detectors import Detector3DTemplate
from ros_pp.model_utils.centernet_utils import _transpose_and_gather_feat, _topk, nms_axis_aligned


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


def decode_bbox_from_heatmap(heatmap: torch.Tensor, 
                             center: torch.Tensor, 
                             center_z: torch.Tensor, 
                             dim: torch.Tensor,
                             rot: torch.Tensor,
                             feature_map_stride: int,
                             voxel_size: List[float],
                             point_cloud_range: List[float],
                             post_center_limit_range: torch.Tensor,
                             max_obj_per_sample: int,
                             score_thresh: float,
                             batch_size: int = 1) -> List[List[torch.Tensor]]:
        K = max_obj_per_sample

        scores, inds, class_ids, ys, xs = _topk(heatmap, K=K)
        center = _transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
        rot = _transpose_and_gather_feat(rot, inds).view(batch_size, K, 2)
        center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
        dim = _transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

        angle = torch.atan(rot[..., 1] / rot[..., 0]).unsqueeze(-1)  # (batch_size, K, 1)

        # get center coord on feature map (pixel coord)
        xs = xs.view(batch_size, K, 1) + center[..., [0]]  # (batch_size, K, 1)
        ys = ys.view(batch_size, K, 1) + center[..., [1]]  # (batch_size, K, 1)

        # get center coord in 3D
        xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        # get size in 3D
        dim = torch.exp(dim)

        final_box_preds = torch.cat(([xs, ys, center_z, dim, angle]), dim=-1)  # (batch_size, K, 7)
        final_scores = scores.view(batch_size, K)
        final_class_ids = class_ids.view(batch_size, K)

        # filter boxes outside range
        mask = torch.logical_and((final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2), 
                                 (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2))  # (batch_size, K)
        # filter low confident boxes
        mask = torch.logical_and(mask, final_scores > score_thresh)  # (batch_size, K)

        batch_pred = list()
        for b_idx in range(batch_size):
            current_pred = [
                final_box_preds[b_idx, mask[b_idx]],  # (N, 7)
                final_scores[b_idx, mask[b_idx]],  # (N,)
                final_class_ids[b_idx, mask[b_idx]]  # (N,)
            ]
            batch_pred.append(current_pred)

        return batch_pred


def generate_predicted_boxes(heads_hm: List[torch.Tensor], 
                             heads_center: List[torch.Tensor], 
                             heads_center_z: List[torch.Tensor],
                             heads_dim: List[torch.Tensor],
                             heads_rot: List[torch.Tensor],
                             feature_map_stride: int,
                             voxel_size: List[float],
                             point_cloud_range: List[float],
                             post_proc_cfg: EasyDict,
                             heads_cls_idx: List[torch.Tensor],
                             batch_size: int = 1) -> List[List[torch.Tensor]]:
    """
    Args:
        heads_hm: List[(B, n_cls_of_this_head, H, W)] 
        heads_center: List[(B, 2, H, W)]
        heads_center_z: List[(B, 1, H, W)]
        heads_dim: List[(B, 3, H, W)]
        heads_rot: List[(B, 2, H, W)]

    Returns:
        batch_boxes: (N, 10) - x, y, z, dx, dy, dz, yaw || score, labels || batch_idx
    """
    def sigmoid(x: torch.Tensor):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y
    
    num_heads = len(heads_cls_idx)

    batch_boxes = []   # (N, 10) - x, y, z, dx, dy, dz, yaw || score, labels || batch_idx

    for head_idx in range(num_heads):
        batch_hm = heads_hm[head_idx]  # (B, N_cls, H, W)
        batch_center = heads_center[head_idx]
        batch_center_z = heads_center_z[head_idx]
        batch_dim = heads_dim[head_idx]
        batch_rot = heads_rot[head_idx]

        this_head_pred = decode_bbox_from_heatmap(batch_hm, batch_center, batch_center_z, batch_dim, batch_rot,
                                                  feature_map_stride, voxel_size, point_cloud_range, 
                                                  torch.tensor(post_proc_cfg.POST_CENTER_LIMIT_RANGE).cuda().float(),
                                                  post_proc_cfg.MAX_OBJ_PER_SAMPLE,
                                                  post_proc_cfg.SCORE_THRESH,
                                                  batch_size)
        # this_head_pred: List[List[torch.Tensor, torch.Tensor, torch.Tensor]] - len == batch_size

        # axis-aligned NMS instead of BEV NMS for speed
        for batch_idx in range(batch_size):
            boxes, scores, labels = nms_axis_aligned(*this_head_pred[batch_idx], 
                                                     post_proc_cfg.NMS_IOU_THRESH, 
                                                     post_proc_cfg.NMS_PRE_MAXSIZE, 
                                                     post_proc_cfg.NMS_POST_MAXSIZE)
            
            pred_boxes = torch.cat([boxes,  # (N, 7) - x, y, z, dx, dy, dz, yaw 
                                    scores.unsqueeze(1),  # (N, 1) 
                                    heads_cls_idx[head_idx][labels.long()].unsqueeze(1),  # (N, 1) - class index
                                    boxes.new_zeros(boxes.shape[0], 1) + batch_idx],  # (N, 1) - batch_idx 
                                    dim=1)

            # store this_head 's prediction to final output
            batch_boxes.append(pred_boxes)
    
    batch_boxes = torch.cat(batch_boxes)
    
    return batch_boxes


def make_trt(half_precision: bool = False):
    from torch2trt import torch2trt


    logger = create_logger('artifacts/blah.txt')
    part3d = CenterPoint_Part3D()
    part3d.load_params_from_file('./pretrained_models/cbgs_voxel01_centerpoint_nds_6454.pth', logger=logger)
    part3d.eval()
    part3d.cuda()

    with open('artifacts/one_nuscenes_point_cloud.pkl', 'rb') as f:
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
    if not half_precision:
        part2d_trt = torch2trt(part2d, [spatial_features])
    else:
        part2d_trt = torch2trt(part2d, [spatial_features.half()], fp16_mode=True)

    if not half_precision:
        torch.save(part2d_trt.state_dict(), 'artifacts/second_part2d_trt.pth')
    else:
        torch.save(part2d_trt.state_dict(), 'artifacts/second_part2d_trt_fp16.pth')
    print('artifacts/second_part2d_trt.pth')


def inference(half_precision: bool = False):
    from torch2trt import TRTModule


    logger = create_logger('artifacts/blah.txt')
    part3d = CenterPoint_Part3D()
    part3d.load_params_from_file('./pretrained_models/cbgs_voxel01_centerpoint_nds_6454.pth', logger=logger)
    part3d.eval()
    part3d.cuda()

    part2d_trt = TRTModule()
    if not half_precision:
        part2d_trt.load_state_dict(torch.load('artifacts/second_part2d_trt.pth'))
    else:
        part2d_trt.load_state_dict(torch.load('artifacts/second_part2d_trt_fp16.pth'))

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
            # pad points with time
            points = torch.from_numpy(np.pad(data['points'], pad_width=[(0, 0), (0, 1)], constant_values=0)).float().cuda()

        start = time.time()
        # ----- forward
        spatial_features = part3d(points)
        if half_precision:
            spatial_features = spatial_features.half()
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
        torch.save(data_out, f'artifacts/data_out_trt_half{int(half_precision)}_frame{idx_frame}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--make_trt', type=int, default=0)
    parser.add_argument('--half_precision', type=int, default=0)
    parser.add_argument('--inference', type=int, default=0)
    args = parser.parse_args()
    
    if args.make_trt == 1:
        make_trt(args.half_precision==1)
    elif args.inference == 1:
        inference(args.half_precision==1)

