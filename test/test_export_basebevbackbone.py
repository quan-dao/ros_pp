import torch
from easydict import EasyDict as edict
import onnx
import onnxruntime as ort
import numpy as np

from ros_pp.models.backbones_2d import BaseBEVBackbone


def export():
    cfg = edict({
        'NAME': 'BaseBEVBackbone',
        'LAYER_NUMS': [3, 5, 5],
        'LAYER_STRIDES': [2, 2, 2],
        'NUM_FILTERS': [64, 128, 256],
        'UPSAMPLE_STRIDES': [0.5, 1, 2],
        'NUM_UPSAMPLE_FILTERS': [128, 128, 128]
    })
    net = BaseBEVBackbone(cfg, 64)
    print('------------------\n', 
          net, 
          '\n---------------------------')
    
    net.cuda()
    net.eval()

    dummy_input = torch.rand(1, 64, 512, 512, device='cuda')
    
    with torch.no_grad():
        out = net(dummy_input)

    torch.onnx.export(net, 
                      dummy_input, 
                      "base_bev_backbone.onnx", 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      verbose=True)
    

def check():
    onnx_model = onnx.load("base_bev_backbone.onnx")
    onnx.checker.check_model(onnx_model)
    print('check complete')


def infer():
    ort_session = ort.InferenceSession("base_bev_backbone.onnx")
    outputs = ort_session.run(
        None,
        {"input": np.random.randn(1, 64, 512, 512).astype(np.float32)},
    )
    print(outputs[0].shape)


if __name__ == '__main__':
    export()
    check()
    infer()

