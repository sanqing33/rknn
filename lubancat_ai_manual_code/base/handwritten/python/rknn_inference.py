# 2023 embedfire zgwinli555@163.com

import numpy as np
import cv2
import sys
import platform
from rknnlite.api import RKNNLite

IMG_PATH = './9.jpg'
RK3588_RKNN_MODEL = '../model/RK3588/handwritten.rknn'
RK3566_RK3568_RKNN_MODEL = '../model/RK356X/handwritten.rknn'
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

def get_host():
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                elif 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host
        
if __name__ == '__main__':
  
  # 获取对应平台的模型
  host_name = get_host()
  if host_name == 'RK3566_RK3568':
    rknn_model = RK3566_RK3568_RKNN_MODEL
  elif host_name == 'RK3588':
    rknn_model = RK3588_RKNN_MODEL
  else:
    print("This demo cannot run on the current platform: {}".format(host_name))
    exit(-1)

  # 创建RKNNLite对象
  # rknn_lite = RKNNLite(verbose=True)
  rknn_lite = RKNNLite()
  
  # 调用load_rknn接口导入RKNN模型，需要对应平台（rk356x/rk3588）的模型
  print('--> Load RKNN model')
  ret = rknn_lite.load_rknn(rknn_model)
  if ret != 0:
      print('Load RKNN model failed')
      exit(ret)
  print('done')
  
  # 调用init_runtime接口初始化运行时环境
  print('--> Init runtime environment')
  ret = rknn_lite.init_runtime()
  if ret != 0:
      print('Init runtime environment failed!')
      exit(ret)
  print('done')
  
  # 读取图像，对图像数据预处理
  img = cv2.imread(IMG_PATH)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = cv2.resize(img,(28,28))

  # 推理
  print('--> Running model')
  outputs = rknn_lite.inference(inputs=[img])
  
  # 输出结果
  print("outputs: ", outputs)
  print("板端本次预测的数字是:", np.argmax(outputs))
  
  # 调用release接口释放RKNNLite对象
  rknn_lite.release()


