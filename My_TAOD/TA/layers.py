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

class pixel_norm(nn.Module):
    """像素归一化"""
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class equal_conv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None
            
    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias = self.bias,
            stride=self.stride,
            padding=self.padding
        )

        return out
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )
    
    
class equal_linear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None  
    ):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
            
        self.activation = activation
        
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        
    def forward(self, input):
        if self.activation:
            output = F.linear(input, self.weight * self.scale)
            # out = fused_leaky_relu(out, self.bias * self.lr_mul)
            
        else:
            output = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        
        return output
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
        

class scaled_leak_relu(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        
        self.negative_slope = negative_slope
        
    def forward(self, input):
        output = F.leaky_relu(input, negative_slope=self.negative_slope)
        
        return output * math.sqrt(2)
    

class noise_injection(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.weight = nn.Parameter(torch.zeros(1))
        
    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
            
        return image + self.weight * noise
    
    
class constant_input(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))
        
    def forward(self, input):
        batch = input.shape[0]
        output = self.input.repeat(batch, 1, 1, 1)
        
        return output
    

