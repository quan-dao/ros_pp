import torch
import torch.nn as nn
from typing import Tuple


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size: Tuple[int, int, int], **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        self.batch_size = model_cfg.get('BATCH_SIZE', 1)
        assert self.nz == 1

    def forward(self, coords: torch.Tensor, pillar_features: torch.Tensor) -> torch.Tensor:
        batch_spatial_features = []
        for batch_idx in range(self.batch_size):
            spatial_feature = pillar_features.new_zeros(self.num_bev_features, self.nz * self.nx * self.ny)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(self.batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        return batch_spatial_features
