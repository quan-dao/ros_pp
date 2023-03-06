import numpy as np
import torch
import onnx
import onnxruntime as ort
import time

from cfgs.nuscenes_models.cfg_cbgs_dyn_pp_centerpoint import data_cfg, model_cfg
from export_pointpillar import make_dummy_input, remove_points_outside_range, CenterPointPart0


def main():
    part0 = CenterPointPart0()
    part0.load_params_from_file(filename='./pretrained_models/cbgs_pp_centerpoint_nds6070.pth', to_cpu=True)
    part0.eval()
    part0.cuda()

    onnx_part1 = onnx.load("pointpillar_onnx/pointpillar_part1.onnx")
    onnx.checker.check_model(onnx_part1)
    print('check complete')
    ort_session = ort.InferenceSession("pointpillar_onnx/pointpillar_part1.onnx")

    points = make_dummy_input()

    time_tic = time.time()

    points = remove_points_outside_range(points, np.array(data_cfg.POINT_CLOUD_RANGE))
    points = torch.from_numpy(points).float().cuda()

    with torch.no_grad():
        voxel_coords, features = part0(points)
    
    voxel_coords = voxel_coords.cpu().numpy()
    features = features.cpu().numpy()
    outputs = ort_session.run(None, {'voxel_coords': voxel_coords, 'features': features})

    print('infer time: ', time.time() - time_tic)

    batch_boxes = outputs[0]
    print('batch_boxes: ', batch_boxes.shape)

    np.save('dummy_output.npy', batch_boxes)



if __name__ == '__main__':
    main()

