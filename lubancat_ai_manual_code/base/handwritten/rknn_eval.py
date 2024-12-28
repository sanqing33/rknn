# -*- coding: utf-8 -*-

# 2023 embedfire zgwinli555@163.com

import numpy as np
import cv2
from rknn.api import RKNN

RKNN_MODEL = 'handwritten.rknn'
IMG_PATH = './0.jpg'
IMG_SIZE = 28

if __name__ == '__main__':
    # 创建RKNN
    # 开启verbose=True，查看调试信息。
    rknn = RKNN(verbose=True)
    #rknn = RKNN()

    # 导入RKNN模型，path参数指定模型路径
    print('--> Loading model')
    ret = rknn.load_rknn(path=RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 初始化运行时环境，指定连接的板卡NPU平台,
    # perf_debug开启进行性能评估时开启debug模式，eval_mem进入内存评估模式
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588', device_id='192.168.103.121:5555')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 模型性能进行评估，默认is_print是true，打印内存使用情况
    #print('--> eval_perf')
    #rknn.eval_perf()
    #print('done')

    # 调试，模型性能进行评估,默认is_print是true，打印内存使用情况
    #print('--> eval_memory')
    #rknn.eval_memory()
    #print('done')
    
    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)
    #print(img)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    # output
    print("outputs: ", outputs)
    print("本次预测的数字是:", np.argmax(outputs))
    
    rknn.release()

