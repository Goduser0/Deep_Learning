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


# 创建一个示例的 (3, 128, 128) 的 NumPy 数组
image_array = np.random.randint(0, 256, size=(3, 128, 128), dtype=np.uint8)

# 将 NumPy 数组转换为 PIL 图像
print(type(image_array), image_array.shape)
image = Image.fromarray(image_array.transpose(1, 2, 0))

# 保存图像
image.save('output_image.png')

