import platform
import random
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time

# decice tree for RK356x/RK3576/RK3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

# model path
RK3566_RK3568_RKNN_MODEL = 'detr_800_for_rk3566_rk3568.rknn'
RK3588_RKNN_MODEL = 'detr_800_for_rk3588.rknn'
RK3562_RKNN_MODEL = 'detr_800_for_rk3562.rknn'
RK3576_RKNN_MODEL = 'detr_800_for_rk3576.rknn'

IMG_PATH = "./model/000000039769.jpg"
IMG_SIZE = 800
OBJ_THRESH = 0.9

CLASSES = (
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
)

CLASS_COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASSES))]

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                elif 'rk3576' in device_compatible_str:
                    host = 'RK3576'
                elif 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

def cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1)

def softmax(x):
    """ softmax function """
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    x -= np.max(x, axis = -1, keepdims = True)
    x = np.exp(x) / np.sum(np.exp(x), axis = -1, keepdims = True)
    return x

def post_process(outputs, target_sizes, threshold):
    # outputs[0] : (1, 100, 92)
    # outputs[1] : (1, 100, 4)
    out_logits,out_bbox = outputs[0], outputs[1]
  
    print(out_logits)
    prob = softmax(out_logits)
    scores = np.max(prob[..., :-1], axis=-1)
    labels = np.argmax(prob[..., :-1], axis=-1)

    # convert to [x0, y0, x1, y1]
    boxes = cxcywh_to_xyxy(out_bbox)

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = np.split(target_sizes, target_sizes.shape[1], axis=1)[0], np.split(target_sizes, target_sizes.shape[1], axis=1)[1]
    img_h = img_h.astype(float)
    img_w = img_w.astype(float)
    scale_fct = np.hstack([img_w, img_h, img_w, img_h])
    boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b in zip(scores, labels, boxes):
        score = s[s > threshold]
        label = l[s > threshold]
        box = b[s > threshold]
        results.append({"scores": score, "labels": label, "boxes": box})

    return results

def letterbox(im, new_shape, color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

if __name__ == '__main__':

    # Get device information
    host_name = get_host()
    if host_name == 'RK3566_RK3568':
        rknn_model = RK3566_RK3568_RKNN_MODEL
    elif host_name == 'RK3562':
        rknn_model = RK3562_RKNN_MODEL
    elif host_name == 'RK3576':
        rknn_model = RK3576_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite = RKNNLite()

    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # Set inputs
    image = cv2.imread(IMG_PATH)
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # letterbox
    img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)
    img_shape = np.array([(IMG_SIZE, IMG_SIZE)])

    # Init runtime environment
    print('--> Init runtime environment')
    # Run on RK356x / RK3576 / RK3588 with Debian OS, do not need specify target.
    if host_name in ['RK3576', 'RK3588']:
        # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[img])
    print('done')

    # post_process
    results = post_process(outputs, img_shape, OBJ_THRESH)
    
    # results
    _results = results[0]
    for score, label, (xmin, ymin, xmax, ymax) in zip( _results['scores'].tolist(), _results['labels'].tolist(), _results['boxes'].tolist()):
        # unletterbox result
        xmin = np.clip((xmin - dw)/ratio, 0,  image.shape[1])
        ymin = np.clip((ymin - dh)/ratio, 0,  image.shape[0])
        xmax = np.clip((xmax - dw)/ratio, 0,  image.shape[1])
        ymax = np.clip((ymax - dh)/ratio, 0,  image.shape[0])

        # print resuilt
        print("%s @ (%d %d %d %d) %.6f" % (CLASSES[label], xmin, ymin, xmax, ymax, score))

        # draw resuilt
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), CLASS_COLORS[label], 2)
        if (int(xmin) <= 10 or int(ymin) <= 10):
            cv2.putText(image, '{0} {1:.3f}'.format(CLASSES[label], score),(int(xmin) + 6, int(ymin) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else :
            cv2.putText(image, '{0} {1:.3f}'.format(CLASSES[label], score),(int(xmin), int(ymin) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # save resuilt
    cv2.imwrite("result.jpg", image)

    rknn_lite.release()
