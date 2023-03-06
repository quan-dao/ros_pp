import torch
import torch.nn as nn
from math import ceil


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        model_cfg = model_cfg

        if model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(model_cfg.LAYER_NUMS) == len(model_cfg.LAYER_STRIDES) == len(model_cfg.NUM_FILTERS)
            layer_nums = model_cfg.LAYER_NUMS
            layer_strides = model_cfg.LAYER_STRIDES
            num_filters = model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(model_cfg.UPSAMPLE_STRIDES) == len(model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    stride = ceil(1 / stride)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True)
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ))

        self.num_bev_features = c_in

    def forward(self, spatial_features: torch.Tensor):
        """
        Args:
            spatial_features: (B, C, H, W)
        Returns:
        """
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        return x  # a.k.a spatial_features_2d
    