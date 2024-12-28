import torch
from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights

# model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()

output_name = 'vit_b_16_224.onnx'

torch.onnx.export(
    model,
    torch.rand(1, 3, 224, 224),
    output_name,
    opset_version=14,
)

print("generated onnx model named {}".format(output_name))
