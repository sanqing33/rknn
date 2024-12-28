import numpy as np
import cv2
from rknn.api import RKNN

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n class    prob\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        topi = '{}:    {:.3}% \n'.format(class_names[(index[0][0])], value*100)
        top5_str += topi
    print(top5_str)

def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

if __name__ == '__main__':

    model = './MobileNetV2.pt'

    input_size_list = [[1, 3, 224, 224]]

    # Create RKNN object
    rknn = RKNN()

    # Pre-process config, 默认设置rk3588
    print('--> Config model')
    rknn.config(mean_values=[[128, 128, 128]], std_values=[[128, 128, 128]], target_platform='rk3588')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('./MobileNetV2.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    #Set inputs
    img = cv2.imread('./sun.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, 0)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    # np.save('./MobileNetV2.npy', outputs[0])
    print(outputs[0][0])
    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()
