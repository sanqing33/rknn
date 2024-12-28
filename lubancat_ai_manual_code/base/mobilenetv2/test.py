import cv2
import numpy as np
from rknnlite.api import RKNNLite

INPUT_SIZE = 224

RK3566_RK3568_RKNN_MODEL = 'MobileNetV2.rknn'
RK3588_RKNN_MODEL = 'MobileNetV2.rknn'

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n Class    Prob\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        topi = '{}:    {:.3}% \n'.format(class_names[(index[0][0])], value*100)
        top5_str += topi
    print(top5_str)

if __name__ == '__main__':

    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(RK3588_RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    ori_img = cv2.imread('./tulips.jpg')
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    
    # init runtime environment
    print('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    #ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[img])
    print(outputs[0][0])
    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn_lite.release()
