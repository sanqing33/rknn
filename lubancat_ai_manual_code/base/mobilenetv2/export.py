import torch
import os
from model import MobileNetV2


if __name__ == '__main__':

    # 模型
    model = MobileNetV2(num_classes=5)

    # 加载权重
    model.load_state_dict(torch.load("./MobileNetV2.pth"))

    model.eval()
    # 保存模型
    trace_model = torch.jit.trace(model, torch.Tensor(1, 3, 224, 224))
    trace_model.save('./MobileNetV2.pt')

