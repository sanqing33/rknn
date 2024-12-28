import os
import sys
import numpy as np
from rknn.api import RKNN

# 导出rknn模型路径
RKNN_MODEL = './model/ppocrv4_det_rk3588.rknn'

# ONNX模型路径
MODEL_PATH = './ch_PP-OCRv4_det_infer/ppocrv4_det.onnx'

# 是否量化
DEFAULT_QUANT = True

DATASET_PATH = './imgs/dataset_20.txt'

# 默认是rk3588平台
platform = "rk3588"

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 57.12, 57.375]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=MODEL_PATH)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=DEFAULT_QUANT, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
