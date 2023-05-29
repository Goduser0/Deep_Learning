import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

a = [{"a":1, "b":1}, {"a":2, "b":2}]
a = pd.DataFrame(a)
a.to_csv('a.csv', index=False)
a = pd.read_csv('a.csv')

for i in range(1):
    df = pd.read_csv('a.csv')
    df.to_csv('a.csv')

a = pd.read_csv('a.csv')
print(a)
