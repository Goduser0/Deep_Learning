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

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE


from sklearn import preprocessing
from sklearn.manifold import TSNE

dir =  "My_Datasets/Classification/PCB-200/残铜/00001_0_01_05159_05657.bmp"
img = Image.open(dir).convert('RGB')
trans = T.Compose([
    T.ToTensor(),
    T.Resize(64)
    ])
img = trans(img)
print(img.shape)