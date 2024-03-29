import torch
import numpy as np
from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg)
        self.num_point_features = num_point_features
    
    def get_output_feature_dim(self):
        return self.num_point_features
    
    def forward(self, points: np.ndarray):
        """
        Args:
            points: (N, 3 + C)

        Returns:
            voxel_features: (V, C)
            voxel_coords: (V, 3)
        """
        voxel_features, voxel_coords, voxel_num_points = self._voxel_generator(points)
        assert len(voxel_features.shape) == 3, f"{voxel_features.shape} != (num_vox, num_points_per_vox, C)"
        
        assert voxel_coords.shape[1] == 3, f"{voxel_coords.shape[1]} != 3"
        # add batch_idx to voxel
        padded_voxel_coords = voxel_coords.new_zeros(voxel_coords.shape[0], 4)
        padded_voxel_coords[:, 1:] = voxel_coords

        points_mean = voxel_features.sum(dim=1)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        return points_mean, padded_voxel_coords

