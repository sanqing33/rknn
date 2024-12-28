import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights

import numpy as np
from PIL import Image

CLASS_LABEL_PATH = './model/synset.txt'

image = "./model/space_shuttle_224.jpg"

# 使用的是torchvision.transforms，需要更快转换速度可以更换成torchvision.transforms.v2
pretreatment = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

# labels
with open(CLASS_LABEL_PATH, 'r') as f:
    labels = [l.rstrip() for l in f]

# Model
# model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()

# pre
raw_image = Image.open(image).convert("RGB")
image = pretreatment(raw_image)
image = torch.unsqueeze(image, 0)

logits = model(image)
probs = F.softmax(logits, dim=1)

# print the top-5 inferences class
scores = np.squeeze(probs.detach().numpy())
a = np.argsort(scores)[::-1]

print('-----TOP 5-----')
for i in a[0:5]:
    print('[%d] score=%.6f class="%s"' % (i, scores[i], labels[i]))
print('done')

