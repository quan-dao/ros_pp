import numpy as np
import torch
import lovely_tensors as lt
from easydict import EasyDict as edict
import onnx
import onnxruntime as ort
from typing import Tuple
from ros_pp.models.dense_heads import CenterHead


cfg = edict({
    'NAME': 'CenterHead',
    'BATCH_SIZE': 1,
    'CLASS_NAMES_EACH_HEAD': [
        ['car'], 
        ['truck', 'construction_vehicle'],
        ['bus', 'trailer'],
        ['barrier'],
        ['motorcycle', 'bicycle'],
        ['pedestrian', 'traffic_cone'],
    ],
    'SHARED_CONV_CHANNEL': 64,
    'USE_BIAS_BEFORE_NORM': True,
    'NUM_CONV_IN_ONE_HEAD': 2,
    'FEATURE_MAP_STRIDE': 4,
    'POST_PROCESSING':{
        'SCORE_THRESH': -0.1,
        'NMS_IOU_THRESH': 0.3,
        'NMS_PRE_MAXSIZE': 1000, 
        'NMS_POST_MAXSIZE': 83,
        'POST_CENTER_LIMIT_RANGE': [-61000.2, -61000.2, -10000.0, 61000.2, 61000.2, 10000.0],
        'MAX_OBJ_PER_SAMPLE': 500,
    }
})
in_channels = 384
class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
grid_size = (512, 512, 1)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8.0]


def export():
    net = CenterHead(cfg, input_channels=in_channels, num_class=len(class_names), class_names=class_names, 
                     grid_size=grid_size, 
                     point_cloud_range=point_cloud_range,
                     voxel_size=voxel_size)
    print('------------------\n', 
          net, 
          '\n---------------------------')
    
    net.cuda()
    net.eval()

    dummy_input = torch.rand(cfg.BATCH_SIZE, in_channels, int(grid_size[1] / cfg.FEATURE_MAP_STRIDE), int(grid_size[0] / cfg.FEATURE_MAP_STRIDE),
                              device='cuda')
    
    with torch.no_grad():
        out = net(dummy_input)

    torch.onnx.export(net, 
                      dummy_input, 
                      "center_head.onnx", 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      verbose=False)
    

def check():
    onnx_model = onnx.load("center_head.onnx")
    onnx.checker.check_model(onnx_model)
    print('check complete')


def infer():
    ort_session = ort.InferenceSession("center_head.onnx")
    outputs = ort_session.run(
        None,
        {"input": np.random.randn(cfg.BATCH_SIZE, 
                                  in_channels, 
                                  int(grid_size[1] / cfg.FEATURE_MAP_STRIDE), 
                                  int(grid_size[0] / cfg.FEATURE_MAP_STRIDE)).astype(np.float32)},
    )
    print(outputs)


if __name__ == '__main__':
    export()
    check()
    infer()
