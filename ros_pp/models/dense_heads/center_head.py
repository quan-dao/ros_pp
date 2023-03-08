import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from typing import List, Tuple

from ...model_utils.centernet_utils import _transpose_and_gather_feat, _topk, nms_axis_aligned


class SeparateHead(nn.Module):
    def __init__(self, input_channels, num_classes_in_this_head, num_conv: int = 2, init_bias=-2.19, use_bias=False):
        super().__init__()

        self.sub_heads_name = ['hm', 'center', 'center_z', 'dim', 'rot']
        self.head_out_channels = [num_classes_in_this_head, 2, 1, 3, 2]

        for cur_name, cur_out_channels in zip(self.sub_heads_name, self.head_out_channels):
            fc_list = []
            for _ in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU(inplace=True)
                ))
            fc_list.append(nn.Conv2d(input_channels, cur_out_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        out = list()
        for cur_name in self.sub_heads_name:
            out.append(self.__getattr__(cur_name)(x))

        return out


class CenterHead(nn.Module):
    def __init__(self, model_cfg, 
                 input_channels: int, 
                 num_class: int, 
                 class_names: List[str], 
                 grid_size: Tuple[int, int, int], 
                 point_cloud_range: Tuple[float, float, float, float, float, float], 
                 voxel_size: Tuple[float, float, float]):
        """
        NOTE: not support predicting velocity

        Args:
            point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)
            voxel_size: (size_x, size_y, size_z)
        """
        super().__init__()
        self.num_class = num_class
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = model_cfg.get('FEATURE_MAP_STRIDE', 4)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.tensor([self.class_names.index(x) for x in cur_class_names if x in class_names]).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(inplace=True),
        )

        self.heads_list = nn.ModuleList()
        for cur_class_names in self.class_names_each_head:
            self.heads_list.append(
                SeparateHead(
                    input_channels=model_cfg.SHARED_CONV_CHANNEL,
                    num_classes_in_this_head=len(cur_class_names),
                    num_conv=model_cfg.get('NUM_CONV_IN_ONE_HEAD', 2),
                    init_bias=-2.19,
                    use_bias=model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.num_heads = len(self.heads_list)

        self.batch_size = model_cfg.get('BATCH_SIZE', 1)
        self.post_center_limit_range = torch.tensor(model_cfg.POST_PROCESSING.POST_CENTER_LIMIT_RANGE).cuda().float()
        self.max_obj_per_sample = model_cfg.POST_PROCESSING.MAX_OBJ_PER_SAMPLE
        self.score_thresh = model_cfg.POST_PROCESSING.SCORE_THRESH
        self.nms_iou_thresh = model_cfg.POST_PROCESSING.NMS_IOU_THRESH
        self.nms_pre_max_size = model_cfg.POST_PROCESSING.NMS_PRE_MAXSIZE 
        self.nms_post_max_size = model_cfg.POST_PROCESSING.NMS_POST_MAXSIZE

        self.heads_cls_idx = [
            torch.tensor([0]).long().cuda(),
            torch.tensor([1, 2]).long().cuda(),
            torch.tensor([3, 4]).long().cuda(),
            torch.tensor([5]).long().cuda(),
            torch.tensor([6, 7]).long().cuda(),
            torch.tensor([8, 9]).long().cuda(),
        ]
    
    def sigmoid(self, x: torch.Tensor):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y
    
    def decode_bbox_from_heatmap(self, 
                                 heatmap: torch.Tensor, 
                                 center: torch.Tensor, 
                                 center_z: torch.Tensor, 
                                 dim: torch.Tensor,
                                 rot: torch.Tensor) -> List[List[torch.Tensor]]:
        K = self.max_obj_per_sample
        batch_size = self.batch_size

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
        xs = xs * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = ys * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        # get size in 3D
        dim = torch.exp(dim)

        final_box_preds = torch.cat(([xs, ys, center_z, dim, angle]), dim=-1)  # (batch_size, K, 7)
        final_scores = scores.view(batch_size, K)
        final_class_ids = class_ids.view(batch_size, K)

        # filter boxes outside range
        mask = torch.logical_and((final_box_preds[..., :3] >= self.post_center_limit_range[:3]).all(2), 
                                 (final_box_preds[..., :3] <= self.post_center_limit_range[3:]).all(2))  # (batch_size, K)
        # filter low confident boxes
        mask = torch.logical_and(mask, final_scores > self.score_thresh)  # (batch_size, K)

        batch_pred = list()
        for b_idx in range(self.batch_size):
            current_pred = [
                final_box_preds[b_idx, mask[b_idx]],  # (N, 7)
                final_scores[b_idx, mask[b_idx]],  # (N,)
                final_class_ids[b_idx, mask[b_idx]]  # (N,)
            ]
            batch_pred.append(current_pred)

        return batch_pred

    def generate_predicted_boxes(self, 
                                 heads_hm: List[torch.Tensor], 
                                 heads_center: List[torch.Tensor], 
                                 heads_center_z: List[torch.Tensor],
                                 heads_dim: List[torch.Tensor],
                                 heads_rot: List[torch.Tensor]) \
          -> List[List[torch.Tensor]]:
        """
        Args:
            heads_hm: List[(B, n_cls_of_this_head, H, W)] 
            heads_center: List[(B, 2, H, W)]
            heads_center_z: List[(B, 1, H, W)]
            heads_dim: List[(B, 3, H, W)]
            heads_rot: List[(B, 2, H, W)]

        Returns:
            List[pred_per_sample]
                pred_persample = (pred_boxes, pred_scores, pred_labels)
        """
        batch_boxes = []   # (N, 10) - x, y, z, dx, dy, dz, yaw || score, labels || batch_idx

        for head_idx in range(self.num_heads):
            batch_hm = self.sigmoid(heads_hm[head_idx])  # (B, N_cls, H, W)
            
            # TODO: find local peak in 3x3 region to replace NMS
            batch_hm_peak = nn.functional.max_pool2d(batch_hm, kernel_size=3, stride=1, padding=1)
            batch_hm_peak_mask = torch.absolute(batch_hm - batch_hm_peak) < 1e-8
            batch_hm = batch_hm * batch_hm_peak_mask.float()

            batch_center = heads_center[head_idx]
            batch_center_z = heads_center_z[head_idx]
            batch_dim = heads_dim[head_idx]
            batch_rot = heads_rot[head_idx]

            this_head_pred = self.decode_bbox_from_heatmap(batch_hm, batch_center, batch_center_z, batch_dim, batch_rot)
            # this_head_pred: List[List[torch.Tensor, torch.Tensor, torch.Tensor]] - len == batch_size

            # axis-aligned NMS instead of BEV NMS for speed
            for batch_idx in range(self.batch_size):
                boxes, scores, labels = nms_axis_aligned(*this_head_pred[batch_idx], self.nms_iou_thresh, 
                                                         self.nms_pre_max_size, 
                                                         self.nms_post_max_size)
                
                pred_boxes = torch.cat([boxes,  # (N, 7) - x, y, z, dx, dy, dz, yaw 
                                        scores.unsqueeze(1),  # (N, 1) 
                                        self.heads_cls_idx[head_idx][labels.long()].unsqueeze(1),  # (N, 1) - class index
                                        boxes.new_zeros(boxes.shape[0], 1) + batch_idx],  # (N, 1) - batch_idx 
                                        dim=1)

                # store this_head 's prediction to final output
                batch_boxes.append(pred_boxes)
        
        batch_boxes = torch.cat(batch_boxes)
        
        return batch_boxes
    
    def forward(self, spatial_features_2d: torch.Tensor):
        """
        Returns:
            batch_boxes: (N, 10) - x, y, z, dx, dy, dz, yaw || score, labels || batch_idx
        """
        x = self.shared_conv(spatial_features_2d)

        heads_hm, heads_center, heads_center_z, heads_dim, heads_rot = list(), list(), list(), list(), list()

        for head_idx, head in enumerate(self.heads_list):
            hm, center, center_z, dim, rot = head(x)
            # ---
            heads_hm.append(hm)
            heads_center.append(center)
            heads_center_z.append(center_z)
            heads_dim.append(dim)
            heads_rot.append(rot)
            
        batch_boxes = self.generate_predicted_boxes(heads_hm, heads_center, heads_center_z, heads_dim, heads_rot)  # (N, 10)

        return batch_boxes
