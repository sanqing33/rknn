# 写在开头

​	因为网上的关于PC端Ubuntu安装 rknn-toolkit2 与板端安装 rknn_toolkit_lite2 的教程有许多很详细的，所以在此只给出一些使用过程中遇到的问题与本项目的代码（github、gitee、百度网盘（有没有其他的速度快的分享办法！！）），后续发现其他问题会继续补充

​	**在阅读过程中发现有错误的欢迎指正，有其他的问题欢迎评论区讨论**

# 配置信息

- 开发板：`香橙派3b（Rockchip rk3566）`
- 板端系统：`Debian GNU/Linux 12 (bookworm)`
- 板端内核：`5.10.160-rockchip-rk356x`
- PC端系统：`Ubuntu 24.04.1 LTS`
- PC端内核：`5.15.167.4-microsoft-standard-WSL2`
- PC端与板端 python 版本：`3.9`
- PC端 rknn-toolkit2 与板端 rknn_toolkit_lite2 版本：`2.3.0`

# 项目地址

### 本文项目地址

包含本项目使用的文件与参考项目的文件，可按需查看使用

`www`

### 参考项目地址

- **模型训练**

​	模型训练使用的ultralytics官方的yolov5：

​		`https://github.com/ultralytics/yolov5`

- **模型转换**

​	pt转onnx使用的ultralytics官方yolov5的export.py（须按照`问题1`修改代码）

​	onnx转rknn使用的瑞芯微官方rknn-toolkit2或rknn_model_zoo：

​		`https://github.com/airockchip/rknn-toolkit2/tree/master`

​		`https://github.com/airockchip/rknn_model_zoo`

- **部署使用**

​	因为瑞芯微官方没给yolov5的班端使用例程，所以用的野火鲁班猫官方配套例程（须按照`问题4`修改代码，可直接使用问题4中修改后的完整代码）：

​		`https://gitee.com/LubanCat/lubancat_ai_manual_code`

# 问题参考

### 问题1

- 使用 rknn-toolkit2 将onnx模型转换成rknn模型时报错`ValueError: cannot reshape array of size 115200 into shape (3,newaxis,19200,6)`

在将pt模型转换成onnx模型时需要修改yolov5项目目录下的`models\yolo.py`文件中的`class Detect(nn.Module)`内的`def forward(self, x)`函数

（试用过瑞芯微官方的针对 RKNN 优化过的yolov5模型，不修改这个yolo.py文件依旧会报一样的错误，修改过后一切正常）

`注：修改这个文件之后无法进行模型训练，训练时候需要修改回来，推荐将源代码注释掉再添加上新代码`

```python
	def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split(
                        (2, 2, self.nc + 1, self.no - self.nc - 5), 4
                    )
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return (
            x
            if self.training
            else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        )
```

将以上代码修改为如下：

```python
	def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = torch.sigmoid(self.m[i](x[i]))
        return x
```

### 问题2

- 在板端使用鲁班猫的yolo例程（`lubancat_ai_manual_code/dev_env/rknn_toolkit_lite2/examples/yolov5_inference/test.py`）报错`Init runtime environment failed!`

原因是板端推理测试会调用 librknnrt.so 库，该库是一个板端的 runtime 库，但是系统自带的库版本低，需要更新，在瑞芯微官方rknn-toolkit2项目下有`rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so`，将其复制到系统的`/usr/lib/`目录下进行替换

### 问题3

- 在板端使用鲁班猫的yolo例程报错`TypeError: 'NoneType' object is not iterable`

需要修改`test.py`中的`if __name__ == '__main__':`的内容，将

```python
outputs = rknn_lite.inference(inputs=[img])
```

修改为

```python
outputs = rknn_lite.inference(inputs=[np.expand_dims(img, axis=0)])

# 我试过往上大部分教程所说的，如下添加一行"img = np.expand_dims(img, axis=0)"，但是会报错"段错误"，在翻遍了所有帖子后在某个犄角旮旯里看见一位大佬说的直接用一行代码才解决
# img = np.expand_dims(img, axis=0)
# outputs = rknn_lite.inference(inputs=[img])
```

### 问题4

- 在板端使用自己训练yolov5并转换得到的rknn模型进行推理时，推理出现很多目标检测结果，如下

<img src="C:\Users\13952\AppData\Roaming\Typora\typora-user-images\image-20241228181205420.png" alt="image-20241228181205420" style="zoom:33%;" />

根据帖子（[NullGogo：RK3588部署yolo模型实时单张或批量推理](https://blog.csdn.net/weixin_44377866/article/details/144233990)）所述原因没做同目标的对齐，需用使用`rknn_model_zoo/examples/yolov5/python/yolov5.py`内的函数对鲁班猫的yolo例程（`lubancat_ai_manual_code/dev_env/rknn_toolkit_lite2/examples/yolov5_inference/test.py`）进行修改替换，具体过程如下：

原鲁班猫例程代码：

```python
outputs = rknn_lite.inference(inputs=[np.expand_dims(img, axis=0)])

input0_data = outputs[0]
input1_data = outputs[1]
input2_data = outputs[2]

input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

input_data = list()
input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

boxes, classes, scores = yolov5_post_process(input_data)
```

修改后的代码：

```python
outputs = rknn_lite.inference(inputs=[np.expand_dims(img, axis=0)])
anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
boxes, classes, scores = post_process(outputs, anchors)
```

同时将`rknn_model_zoo/examples/yolov5/python/yolov5.py`中的函数`post_process`和`box_process`复制到`lubancat_ai_manual_code/dev_env/rknn_toolkit_lite2/examples/yolov5_inference/test.py`中，可以删除掉test.py中不使用的函数`sigmoid`、`xywh2xyxy`、`process`、`yolov5_post_process`

##### 修改过后的完整代码

```python
import urllib
import time
import sys
import numpy as np
import cv2
import platform
from rknnlite.api import RKNNLite

RK3566_RK3568_RKNN_MODEL = 'cat.rknn'
RK3588_RKNN_MODEL = 'yolov5s_for_rk3588.rknn'
RK3562_RKNN_MODEL = 'yolov5s_for_rk3562.rknn'
IMG_PATH = './cat.jpg'

OBJ_THRESH = 0.25 
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES = "cat"

# decice tree for rk356x/rk3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

def get_host():
    # get platform and device type
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
    

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE // grid_h, IMG_SIZE // grid_w]).reshape(
        1, 2, 1, 1
    )

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:, :2, :, :] * 2 - 0.5
    box_wh = pow(position[:, 2:4, :, :] * 2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2  # top left x
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2  # top left y
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2  # bottom right x
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2  # bottom right y

    return xyxy

def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []
    # 1*255*h*w -> 3*85*h*w
    input_data = [
        _in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data
    ]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:, :4, :, :], anchors[i]))
        scores.append(input_data[i][:, 4:5, :, :])
        classes_conf.append(input_data[i][:, 5:, :, :])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

if __name__ == '__main__':

    host_name = get_host()
    if host_name == 'RK3566_RK3568':
        rknn_model = RK3566_RK3568_RKNN_MODEL
    elif host_name == 'RK3562':
        rknn_model = RK3562_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)
        
    # Create RKNN object
    rknn_lite = RKNNLite()

	  # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    #img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Inference
    print('--> Running model')
    # outputs = rknn_lite.inference(inputs=[img])
    outputs = rknn_lite.inference(inputs=[np.expand_dims(img, axis=0)])
    
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    
    boxes, classes, scores = post_process(outputs, anchors)

    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img_1, boxes, scores, classes)

    # show output
    cv2.imwrite("out.jpg", img_1)

    rknn_lite.release()
```

