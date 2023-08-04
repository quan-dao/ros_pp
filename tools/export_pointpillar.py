import numpy as np
import torch
import torch.nn as nn
from ros_pp.models.backbones_3d import vfe
from ros_pp.models import backbones_2d
from ros_pp.models.backbones_2d import map_to_bev
from ros_pp.models import dense_heads
from cfgs.nuscenes_models.cbgs_dyn_pp_centerpoint import data_cfg, model_cfg
from typing import Tuple


class ModelTemplate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        raise NotImplementedError
    
    def _load_state_dict(self, loaded_model_state: dict, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        update_model_state = {}
        for key, val in loaded_model_state.items():
            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)  # overwrite current `state_dict` with `updated value` in `update_model_state`
            # => layer doesn't have pretrained weight are not updated
            self.load_state_dict(state_dict)
        return state_dict, update_model_state
    
    def load_params_from_file(self, filename, to_cpu=False):
        import os

        if not os.path.isfile(filename):
            raise FileNotFoundError

        print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        loaded_model_state = checkpoint['model_state']
        
            
        version = checkpoint.get("version", None)
        if version is not None:
            print('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(loaded_model_state, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        print('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))


class CenterPointPart0(ModelTemplate):
    def __init__(self):
        super().__init__()
        voxel_size = np.array(data_cfg.VOXEL_SIZE)  # size_x, size_y, size_z
        point_cloud_range = np.array(data_cfg.POINT_CLOUD_RANGE)  # x_min, y_min, z_min, x_max, y_max, z_max
        grid_size = np.floor((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).astype(int)  # nx, ny, nz

        vfe_cfg = model_cfg.VFE
        self.add_module('vfe', 
                        vfe.__all__[vfe_cfg.NAME](vfe_cfg, vfe_cfg.NUM_RAW_POINT_FEATURES, voxel_size, grid_size, point_cloud_range))

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor]:
        voxel_coords, features = self.vfe(points)
        return voxel_coords, features


class CenterPointPart1(ModelTemplate):
    def __init__(self):
        super().__init__()
        voxel_size = np.array(data_cfg.VOXEL_SIZE)  # size_x, size_y, size_z
        point_cloud_range = np.array(data_cfg.POINT_CLOUD_RANGE)  # x_min, y_min, z_min, x_max, y_max, z_max
        grid_size = np.floor((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).astype(int)  # nx, ny, nz

        to_bev_cfg = model_cfg.MAP_TO_BEV
        self.add_module('map_to_bev_module', 
                        map_to_bev.__all__[to_bev_cfg.NAME](to_bev_cfg, grid_size))
        
        bbone_2d_cfg = model_cfg.BACKBONE_2D
        self.add_module('backbone_2d',
                        backbones_2d.__all__[bbone_2d_cfg.NAME](bbone_2d_cfg, self.map_to_bev_module.num_bev_features))
        
        dhead_cfg = model_cfg.DENSE_HEAD
        self.add_module('dense_head',
                        dense_heads.__all__[dhead_cfg.NAME](dhead_cfg, self.backbone_2d.num_bev_features, len(data_cfg.CLASSES), data_cfg.CLASSES,
                                                            grid_size, point_cloud_range, voxel_size))
    
    def forward(self, voxel_coords: torch.Tensor, features: torch.Tensor):
        batch_spatial_features = self.map_to_bev_module(voxel_coords, features)  # onnx
        spatial_features_2d = self.backbone_2d(batch_spatial_features)  # onnx
        batch_boxes = self.dense_head(spatial_features_2d)  # onnx
        return batch_boxes


def make_dummy_input(scene_idx=1, target_sample_idx=10, pad_points_with_batch_idx=True) -> np.ndarray:
    """
    Returns:
        pc: (N, 6) - batch_idx, x, y, z, intensity, time
    """ 
    from nuscenes.nuscenes import NuScenes

    def get_sample_data_point_cloud(nusc: NuScenes, sample_data_token: str, time_lag: float) \
        -> np.ndarray:
        """
        Returns:
            pc: (N, 5) - (x, y, z, intensity, time)
        """
        pcfile = nusc.get_sample_data_path(sample_data_token)
        pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (N, 4) - (x, y, z, intensity)
        pc = np.pad(pc, pad_width=[(0, 0), (0, 1)], mode='constant', constant_values=time_lag)  # (N, 5)
        return pc

    nusc = NuScenes(dataroot='../data/nuscenes/v1.0-mini', version='v1.0-mini', verbose=False)
    scene = nusc.scene[scene_idx]
    sample_tk = scene['first_sample_token']
    for _ in range(target_sample_idx):
        sample = nusc.get('sample', sample_tk)
        sample_tk = sample['next']

    sample = nusc.get('sample', sample_tk)
    pc = get_sample_data_point_cloud(nusc, sample['data']['LIDAR_TOP'], time_lag=0.0)
    if pad_points_with_batch_idx:
        pc = np.pad(pc, pad_width=[(0, 0), (1, 0)], constant_values=0.0)
    return pc


def remove_points_outside_range(points: np.ndarray, point_cloud_range: np.ndarray):
    mask_in = np.logical_and(points[:, 1: 4] > point_cloud_range[:3], points[:, 1: 4] < point_cloud_range[3:] - 1e-3).all(axis=1)
    return points[mask_in]


if __name__ == '__main__':
    import lovely_tensors as lt
    
    
    lt.monkey_patch()

    part0 = CenterPointPart0()
    part0.load_params_from_file(filename='./pretrained_models/cbgs_pp_centerpoint_nds6070.pth', to_cpu=True)
    part0.eval()
    part0.cuda()

    part1 = CenterPointPart1()
    part1.load_params_from_file(filename='./pretrained_models/cbgs_pp_centerpoint_nds6070.pth', to_cpu=True)
    part1.eval()
    part1.cuda()

    points = make_dummy_input()
    points = remove_points_outside_range(points, np.array(data_cfg.POINT_CLOUD_RANGE))
    points = torch.from_numpy(points).float().cuda()

    # ---------------------
    # invoking CenterPoint, module-by-module
    with torch.no_grad():
        voxel_coords, features = part0(points)
        batch_boxes = part1(voxel_coords, features)  # onnx
    
    print('batch_boxes: ', batch_boxes)

    # ---------------------
    # exporting
    onnx_dir = 'pointpillar_onnx'
    torch.onnx.export(part1, 
                    (voxel_coords, features), 
                    f"{onnx_dir}/pointpillar_part1_no_nms.onnx", 
                    export_params=True, 
                    opset_version=11, 
                    do_constant_folding=True,
                    input_names=['voxel_coords', 'features'],
                    output_names=['batch_boxes'],
                    dynamic_axes={'voxel_coords': {0: 'num_pillars'}, 'features': {0: 'num_pillars'},
                                    'batch_boxes': {0: 'num_boxes'}}
                    )

    print('-----------------------\n',
          'finish export')

