from easydict import EasyDict as edict


BATCH_SIZE = 20
POINT_FEATURES = ['x', 'y', 'z', 'intensity', 'timestamp']

data_cfg = edict({
    'POINT_CLOUD_RANGE': [-43.2, -43.2, -5.0, 43.2, 43.2, 3.0],
    'VOXEL_SIZE': [0.2, 0.2, 8.0],
    'CLASSES': ['car', 'pedestrian'],
    'NUM_POINT_FEATURES': len(POINT_FEATURES)
})


model_cfg = edict({
    'NAME': 'CenterPoint',
    'BATCH_SIZE': BATCH_SIZE,

    'PATCH_GENERATOR':
        {
            'VOXEL_SIZE': data_cfg.VOXEL_SIZE,
            'POINT_CLOUD_RANGE': data_cfg.POINT_CLOUD_RANGE,
            'NUM_POINT_FEATURES': data_cfg.NUM_POINT_FEATURES,
            'MAX_NUM_POINTS_PER_VOXEL': 8,
            'MAX_NUM_VOXELS': 30000,
            'PATCH_STRIDE': 20.,
            'PATCH_RADIUS': 19.2,
            'PATCH_NUM_MIN_POINTS': 10,
            'MAX_NUM_PATCHES': BATCH_SIZE,
        },

    'VFE': 
        {
            'NAME': 'PseudoDynamicPillarVFE',
            'WITH_DISTANCE': False,
            'USE_ABSOLUTE_XYZ': True,
            'USE_NORM': True,
            'NUM_FILTERS': [64,],
            'NUM_RAW_POINT_FEATURES': 5,
        },
    
    'MAP_TO_BEV': 
        {
            'NAME': 'PointPillarScatter',
            'NUM_BEV_FEATURES': 64,
            'BATCH_SIZE': BATCH_SIZE
        },
    
    'BACKBONE_2D':
        {
            'NAME': 'BaseBEVBackbone',
            'LAYER_NUMS': [3, 5, 5],
            'LAYER_STRIDES': [2, 2, 2],
            'NUM_FILTERS': [64, 128, 256],
            'UPSAMPLE_STRIDES': [0.5, 1, 2],
            'NUM_UPSAMPLE_FILTERS': [128, 128, 128]
        },

    'DENSE_HEAD':
        {
            'NAME': 'CenterHead',
            'BATCH_SIZE': BATCH_SIZE,
            'CLASS_NAMES_EACH_HEAD': [
                ['car'], # 0
                ['pedestrian'],  # 8, 9
            ],
            'SHARED_CONV_CHANNEL': 64,
            'USE_BIAS_BEFORE_NORM': True,
            'NUM_CONV_IN_ONE_HEAD': 2,
            'FEATURE_MAP_STRIDE': 4,
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
