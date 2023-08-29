import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim

import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from TA_G import FeatureMatchGenerator


def traditional_aug():
    pass

def generate(z_dim, n, model_path, samples_save_path):

    z = torch.FloatTensor(np.random.normal(0, 1, (n, z_dim)))
    G = FeatureMatchGenerator(3, 128, 128, 64, 1e-2)
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint["model_state_dict"])
    imgs = G(z)
    i = 0
    for img in imgs:
        i+=1
        img = img.detach().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img.transpose(1, 2, 0))
        img.save(samples_save_path + f"/{i}.png")


generate(128, 
         100, 
         "./My_TAOD/TA/models/source/PCB_200 0 2023-08-28_20:11:31/100_net_g_source.pth",
         "./My_TAOD/TA/samples",
         )
