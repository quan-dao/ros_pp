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

    onnx_file = "pointpillar_onnx/pointpillar_part1_no_nms.onnx"

    onnx_part1 = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_part1)
    print('check complete')
    ort_session = ort.InferenceSession(onnx_file, providers=["CUDAExecutionProvider"])

    points = make_dummy_input()
    points_ = remove_points_outside_range(points, np.array(data_cfg.POINT_CLOUD_RANGE))

    for i in range(10):
        time_tic = time.time()

        points = torch.from_numpy(np.copy(points_)).float().cuda()

        with torch.no_grad():
            voxel_coords, features = part0(points)
        time_voxelize = time.time() - time_tic


        voxel_coords = voxel_coords.cpu().numpy()
        features = features.cpu().numpy()

        time_to_gpu = time.time() - time_tic

        outputs = ort_session.run(None, {'voxel_coords': voxel_coords, 'features': features})
        time_infer = time.time() - time_tic

        print(f"{i} time_voxelize: ", time_voxelize)
        print(f"{i} time_to_gpu: ", time_to_gpu)
        print(f'{i} infer time: ', time_infer)

    batch_boxes = outputs[0]
    print('batch_boxes: ', batch_boxes.shape)

    np.save('dummy_output_no_nms.npy', batch_boxes)


if __name__ == '__main__':
    main()

