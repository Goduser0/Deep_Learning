import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

a = Image.open('./My_Datasets/Classification/NEU-CLS/Pa_299.bmp')

plt.imshow(a)
plt.show()