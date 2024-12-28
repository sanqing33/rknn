"""Exports a YOLOv5 *.pt model to ONNX  formats

Usage:
    $ copy export_onnx.py to yolov5-face
    $ export PYTHONPATH="$PWD" && python export.py --weights ./weights/yolov5s-face.pt  --img_size 640 
"""

import sys

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from models.experimental import attempt_load
import onnx
import argparse
import time
import types

def export_forward(self, x):
    z = []  # inference output
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        z.append(x[i])
    
    return z

def run(weights='yolov5s-face.pt', img_size=(640, 640)):
    t = time.time()
    model = attempt_load(weights, map_location=torch.device('cpu'))  # load FP32 model
    model.eval()

    # model
    model.model[-1].forward = types.MethodType(export_forward, model.model[-1])
    print(f'starting export {weights} to onnx...')

    save_path = weights.replace(".pt", ".onnx")
    model.fuse()  # only for ONNX
    output_names = ['output0']
    img = torch.zeros(1, 3, img_size[0], img_size[1])
    torch.onnx.export(
        model,  # --dynamic only compatible with cpu
        img,
        save_path,
        verbose=False,
        opset_version=12,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names)
    # Checks
    onnx_model = onnx.load(save_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print('ONNX export success, saved as %s' % save_path)
    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640])
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_opt()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    run(args.weights, args.img_size)