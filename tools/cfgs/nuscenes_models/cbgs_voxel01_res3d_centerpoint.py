import numpy as np
from easydict import EasyDict as edict


BATCH_SIZE = 1
POINT_FEATURES = ['x', 'y', 'z', 'intensity', 'timestamp']

data_cfg = edict({
    'CLASSES': ['car','truck', 'construction_vehicle', 'bus', 'trailer', 
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],

    'POINT_CLOUD_RANGE': np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    'VOXEL_SIZE': np.array([0.1, 0.1, 0.2]),
    'POINT_FEATURES': POINT_FEATURES,
    'NUM_POINT_FEATURES': len(POINT_FEATURES)
})

model_cfg = edict({
    'NAME': 'CenterPoint',

    'VFE':{
        'NAME': 'MeanVFE',
        'VOXEL_SIZE': data_cfg.VOXEL_SIZE,
        'POINT_CLOUD_RANGE': data_cfg.POINT_CLOUD_RANGE,
        'NUM_POINT_FEATURES': data_cfg.NUM_POINT_FEATURES,
        'MAX_NUM_POINTS_PER_VOXEL': 10,
        'MAX_NUM_VOXELS': 60000,
    },

    'BACKBONE_3D': {
        'NAME': 'VoxelResBackBone8x'
    },
        
    'MAP_TO_BEV':{
        'NAME': 'HeightCompression',
        'NUM_BEV_FEATURES': 256,  # 2 * 128
    },

    'BACKBONE_2D':{
        'NAME': 'BaseBEVBackbone',
        'LAYER_NUMS': [5, 5],
        'LAYER_STRIDES': [1, 2],
        'NUM_FILTERS': [128, 256],
        'UPSAMPLE_STRIDES': [1, 2],
        'NUM_UPSAMPLE_FILTERS': [256, 256],
    },

    'DENSE_HEAD':{
        'NAME': 'CenterHead',
        'BATCH_SIZE': BATCH_SIZE,
        'CLASS_NAMES_EACH_HEAD': [
            ['car'], # 0
            ['truck', 'construction_vehicle'],  # 1, 2
            ['bus', 'trailer'],  # 3, 4
            ['barrier'],  # 5
            ['motorcycle', 'bicycle'],  # 6, 7
            ['pedestrian', 'traffic_cone'],  # 8, 9
        ],
        'SHARED_CONV_CHANNEL': 64,
        'USE_BIAS_BEFORE_NORM': True,
        'NUM_CONV_IN_ONE_HEAD': 2,
        'FEATURE_MAP_STRIDE': 8,

        'POST_PROCESSING':{
            'SCORE_THRESH': 0.3,
            'NMS_IOU_THRESH': 0.3,
            'NMS_PRE_MAXSIZE': 200, 
            'NMS_POST_MAXSIZE': 30,
            'POST_CENTER_LIMIT_RANGE': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            'MAX_OBJ_PER_SAMPLE': 500,
        }
    }

})

