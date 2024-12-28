import os
import sys
import numpy as np
from rknn.api import RKNN

# 导出rknn模型路径
RKNN_MODEL = './model/yolov5s_seg.rknn'

# pt模型路径
MODEL_PATH = './best.pt'

DATASET_PATH = './dataset.txt'
DEFAULT_QUANT = True

# 默认是rk3588平台
platform = "rk3588"

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    #ret = rknn.load_onnx(model=MODEL_PATH)
    ret = rknn.load_pytorch(model=MODEL_PATH, input_size_list=[[1, 3, 640, 640]])
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

    # 精度分析,,输出目录./snapshot
    #print('--> Accuracy analysis')
    #ret = rknn.accuracy_analysis(inputs=['./subset/000000052891.jpg'])
    #if ret != 0:
    #    print('Accuracy analysis failed!')
    #    exit(ret)
    #print('done')

    # Release
    rknn.release()