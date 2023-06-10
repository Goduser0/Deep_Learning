import torch
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

file_list = os.listdir('./My_Datasets/Classification/Magnetic-Tile-Defect')
file_list = [filename for filename in file_list if filename[:2]=='MT']
print(file_list)