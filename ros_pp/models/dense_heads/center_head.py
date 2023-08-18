import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from typing import List, Tuple


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

        # self.heads_cls_idx = [
        #     torch.tensor([0]).long().cuda(),
        #     torch.tensor([1, 2]).long().cuda(),
        #     torch.tensor([3, 4]).long().cuda(),
        #     torch.tensor([5]).long().cuda(),
        #     torch.tensor([6, 7]).long().cuda(),
        #     torch.tensor([8, 9]).long().cuda(),
        # ]
    
    def forward(self, spatial_features_2d: torch.Tensor):
        """
        Returns:
            heads_hm: List[(B, n_cls_of_this_head, H, W)] 
            heads_center: List[(B, 2, H, W)]
            heads_center_z: List[(B, 1, H, W)]
            heads_dim: List[(B, 3, H, W)]
            heads_rot: List[(B, 2, H, W)]
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
            
        # batch_boxes = self.generate_predicted_boxes(heads_hm, heads_center, heads_center_z, heads_dim, heads_rot)  # (N, 10)

        return heads_hm, heads_center, heads_center_z, heads_dim, heads_rot
