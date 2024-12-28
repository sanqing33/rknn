import cv2
import numpy as np
import platform
import os
from synset_label import labels
from rknnlite.api import RKNNLite

# decice tree for RK356x/RK3576/RK3588
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


INPUT_SIZE = 224

RK3566_RK3568_RKNN_MODEL = 'resnet18_for_rk3566_rk3568.rknn'
RK3588_RKNN_MODEL = 'resnet18_for_rk3588.rknn'
RK3562_RKNN_MODEL = 'resnet18_for_rk3562.rknn'
RK3576_RKNN_MODEL = 'resnet18_for_rk3576.rknn'


def show_top5(result):
    output = result[0].reshape(-1)
    # Softmax
    output = np.exp(output) / np.sum(np.exp(output))
    # Get the indices of the top 5 largest values
    output_sorted_indices = np.argsort(output)[::-1][:5]
    top5_str = 'resnet18\n-----TOP 5-----\n'
    for i, index in enumerate(output_sorted_indices):
        value = output[index]
        if value > 0:
            topi = '[{:>3d}] score:{:.6f} class:"{}"\n'.format(index, value, labels[index])
        else:
            topi = '-1: 0.0\n'
        top5_str += topi
    print(top5_str)
    return output_sorted_indices[0]  # 返回概率最大的类别索引，用于后续标注


if __name__ == '__main__':
    # 获取设备信息
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

    # 加载RKNN模型
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret!= 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # 输入图片所在目录，可根据实际情况修改
    input_image_dir = 'input_images'
    # 输出标注图片所在目录，若不存在则创建，可根据实际情况修改
    output_image_dir = 'output_images'
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # 获取输入目录下所有图片文件的列表（仅支持常见图片格式，可按需扩展）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for root, dirs, files in os.walk(input_image_dir):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    for image_file in image_files:
        # 读取图片
        ori_img = cv2.imread(image_file)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        # 初始化运行时环境
        print('--> Init runtime environment')
        # Run on RK356x / RK3576 / RK3588 with Debian OS, do not need specify target.
        if host_name in ['RK3576', 'RK3588']:
            # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
            ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        else:
            ret = rknn_lite.init_runtime()
        if ret!= 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        # 推理
        print('--> Running model')
        outputs = rknn_lite.inference(inputs=[img])

        # 展示分类结果并获取最可能类别的索引
        top_index = show_top5(outputs)
        most_likely_class = labels[top_index]

        # 设置标注文字的字体、字号、颜色、粗细等参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 255)  # 红色，可自行调整为喜欢的颜色，格式为(B, G, R)
        thickness = 2
        # 确定标注位置，这里示例放在图片左上角，可根据需求调整坐标
        position = (10, 30)
        # 在图片上添加标注文字
        cv2.putText(ori_img, most_likely_class, position, font, font_scale, font_color, thickness)

        # 获取输出文件名，保持原文件名，仅修改保存目录
        output_image_file = os.path.join(output_image_dir, os.path.basename(image_file))
        # 保存标注后的图片
        cv2.imwrite(output_image_file, ori_img)

        print(f'已处理并保存标注图片: {output_image_file}')

    rknn_lite.release()
