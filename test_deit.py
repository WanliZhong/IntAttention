import requests
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from replace_tools import replace_qkv

torch.set_grad_enabled(False);

# Read the ImageNet categories
with open("imagenet_classes.txt", "r") as f:
    imagenet_categories = [s.strip() for s in f.readlines()]

# create the data transform that DeiT expects
transform = T.Compose([
    T.Resize(256, interpolation=3),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
# for name, module in model.named_modules():
#     print(name, module, "\n---------\n")

# TODO：修改model的部分层
from replace_tools import replace_qkv

replace_qkv(model, "IndexAttention", quant_bit=8)

import time
time_start = time.time()
for _ in range(1000):
    model.eval()
time_end = time.time()
print("Time cost of 1000 model.eval():", time_end - time_start)

# url = 'http://images.cocodataset.org/val2017/000000039770.jpg'
# im = Image.open(requests.get(url, stream=True).raw)
im = Image.open("000000039770.png")

# transform the original image and add a batch dimension
img = transform(im).unsqueeze(0)

# compute the predictions
out = model(img)

# and convert them into probabilities
scores = nn.functional.softmax(out, dim=-1)[0]

# finally get the index of the prediction with highest score
topk_scores, topk_label = torch.topk(scores, k=5, dim=-1)

for i in range(5):
  pred_name = imagenet_categories[topk_label[i]]
  print(f"Prediction index {i}: {pred_name:<25}, score: {topk_scores[i].item():.3f}")