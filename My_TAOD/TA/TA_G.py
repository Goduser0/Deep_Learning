import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torchinfo import summary

class Generator(nn.Module):
    def __init__(self, img_size=64, z_dim=128, conv_dim=64):
        self.img_size = img_size
        
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        