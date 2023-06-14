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

out = torch.randn(16, 3, 200, 200)

stddev = torch.randn(16, 1, 1, 3, 200, 200)
stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
stddev = stddev.repeat(16, 1, 200, 200)
stddev = torch.cat([out, stddev], 1)

print(stddev.detach().shape)