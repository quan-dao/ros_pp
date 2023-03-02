import torch
from easydict import EasyDict as edict

from ros_pp.models.backbones_2d import BaseBEVBackbone


def main():
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
    

if __name__ == '__main__':
    main()

