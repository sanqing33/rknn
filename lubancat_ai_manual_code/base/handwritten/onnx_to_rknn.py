# -*- coding: utf-8 -*-

# 2023 embedfire zgwinli555@163.com

import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = './handwritten.onnx'
RKNN_MODEL = './model/RK3588/handwritten.rknn'
IMG_PATH = './0.jpg'
IMG_SIZE = 28
QUANTIZE_ON = False

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # 这里测试lubancat-4指定平台是rk3588，如果是lubancat0/1/2指定平台rk3566、rk3568
    print('--> Config model')
    rknn.config(mean_values=[[127.5]], std_values=[[127.5]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    #ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    ret = rknn.build(do_quantization=QUANTIZE_ON)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img], data_format='nchw')
    print('done')

    # output
    print("outputs: ", outputs)
    print("本次预测的数字是:", np.argmax(outputs))

    rknn.release()
