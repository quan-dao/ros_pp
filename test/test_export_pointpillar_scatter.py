import numpy as np
import torch
import lovely_tensors as lt
from easydict import EasyDict as edict
import onnx
import onnxruntime as ort
from typing import Tuple
from ros_pp.models.backbones_2d.map_to_bev import PointPillarScatter


cfg = edict({
    'NAME': PointPillarScatter,
    'NUM_BEV_FEATURES': 64,
    'BATCH_SIZE': 1
})
grid_size = (512, 512, 1)  # nx, ny, nz


def make_dummy_input(num_pillars: int, batch_size: int, grid_size: Tuple[int, int, int], num_features=64, to_cuda=False) \
    -> Tuple[torch.Tensor, torch.Tensor]:
    batch_idx = torch.randint(batch_size, (num_pillars,))
    coord_z = torch.zeros(num_pillars)
    coord_y = torch.randint(grid_size[1], (num_pillars,))
    coord_x = torch.randint(grid_size[0], (num_pillars,))
    voxel_coord = torch.stack([batch_idx, coord_z, coord_y, coord_x], dim=1).long()
    voxel_features = torch.rand(num_pillars, num_features).float()
    
    if to_cuda:
        return voxel_coord.cuda(), voxel_features.cuda()
    
    return voxel_coord, voxel_features


def export():
    net = PointPillarScatter(cfg, grid_size=grid_size)
    print('------------------\n', 
          net, 
          '\n---------------------------')
    
    net.cuda()
    net.eval()

    num_pillars = 10000
    voxel_coord, voxel_features = make_dummy_input(num_pillars, cfg.BATCH_SIZE, grid_size, cfg.NUM_BEV_FEATURES, to_cuda=True)

    with torch.no_grad():
        out = net(voxel_coord, voxel_features)

    torch.onnx.export(net, 
                      (voxel_coord, voxel_features), 
                      "pointpillar_scatter.onnx", 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True,
                      input_names=['voxel_coord', 'voxel_features'],
                      output_names=['output'],
                      dynamic_axes={
                        'voxel_coord': {0: 'num_pillars'},
                        'voxel_features': {0: 'num_pillars'}, 
                        },
                      verbose=True
                    )
    

def check():
    onnx_model = onnx.load("pointpillar_scatter.onnx")
    onnx.checker.check_model(onnx_model)
    print('check complete')


def infer():
    ort_session = ort.InferenceSession("pointpillar_scatter.onnx")
    num_pillars = 50000
    voxel_coord, voxel_features = make_dummy_input(num_pillars, cfg.BATCH_SIZE, grid_size, cfg.NUM_BEV_FEATURES, to_cuda=False)
    outputs = ort_session.run(
        None,
        {"voxel_coord": voxel_coord.numpy(), 'voxel_features': voxel_features.numpy()},
    )
    print(outputs[0].shape)


if __name__ == '__main__':
    lt.monkey_patch()
    export()
    check()
    infer()
