import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel


class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self._voxel_generator = PointToVoxel(
        #     vsize_xyz=model_cfg.VOXEL_SIZE,
        #     coors_range_xyz=model_cfg.POINT_CLOUD_RANGE,
        #     num_point_features=model_cfg.NUM_POINT_FEATURES,
        #     max_num_points_per_voxel=model_cfg.MAX_NUM_POINTS_PER_VOXEL,
        #     max_num_voxels=model_cfg.MAX_NUM_VOXELS,
        #     device=torch.device("cuda:0")
        # )
    
    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError

