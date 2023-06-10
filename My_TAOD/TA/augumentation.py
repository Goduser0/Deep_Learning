import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim

import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from layers import pixel_norm, equal_conv2d, equal_linear, constant_input


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()
        
        self.size = size
        self.style_dim = style_dim
        
        layers = [pixel_norm()]
        
        for i in range(n_mlp):
            layers.append(
                equal_linear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu')
            )
            
        self.style = nn.Sequential(*layers)
        
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        
        self.intput = constant_input(self.channels[4])
        self.conv1 = 
        
        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        
        in_channel = self.channels[4]