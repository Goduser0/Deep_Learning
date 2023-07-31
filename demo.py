import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE


from sklearn import preprocessing
from sklearn.manifold import TSNE

import tqdm

# dir =  "My_Datasets/Classification/PCB-200/残铜/00001_0_01_05159_05657.bmp"
# img = Image.open(dir).convert('RGB')
# trans = T.Compose([
#     T.ToTensor(),
#     T.Resize(64)
#     ])
# img = trans(img)
# print(img.shape)


p = torch.randn([8, 64])
q = torch.randn([8, 64])

# 归一化处理，使其成为概率分布
p = F.softmax(p, dim=-1)
q = F.softmax(q, dim=-1)
print(p.sum(dim=1))
# 创建 KL 散度损失函数

# 计算 KL 散度损失
loss = F.kl_div(p.log(), q, reduction='batchmean')

print("KL 散度损失:", loss)

