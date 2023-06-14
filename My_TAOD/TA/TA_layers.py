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


##########################################################################################################
# CLASS: PixelNorm
##########################################################################################################
class PixelNorm(nn.Module):
    """像素归一化"""
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


##########################################################################################################
# CLASS: EqualConv2d
##########################################################################################################
class EqualConv2d(nn.Module):
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
    

##########################################################################################################
# CLASS: EqualLinear
##########################################################################################################
class EqualLinear(nn.Module):
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
            if self.bias:
                output = F.linear(
                    input, self.weight * self.scale, bias=self.bias * self.lr_mul
                )
            else:
                output = F.linear(
                    input, self.weight * self.scale
                )
        
        return output
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
        
        
##########################################################################################################
# CLASS: ScaledLeakRelu
##########################################################################################################
class ScaledLeakRelu(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        
        self.negative_slope = negative_slope
        
    def forward(self, input):
        output = F.leaky_relu(input, negative_slope=self.negative_slope)
        
        return output * math.sqrt(2)
    

##########################################################################################################
# CLASS: NoiseInjection
##########################################################################################################
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.weight = nn.Parameter(torch.zeros(1))
        
    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
            
        return image + self.weight * noise
    

##########################################################################################################
# CLASS: ConstantInput
##########################################################################################################
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))
        
    def forward(self, input):
        batch = input.shape[0]
        output = self.input.repeat(batch, 1, 1, 1)
        
        return output
    
    
#######################################################################################################
# CLASS: SelfAttention
#######################################################################################################
class SelfAttention(nn.Module):
    """Self Attention Layer"""
    def __init__(self, in_channels, activation='relu', k=8):
        super(SelfAttention, self).__init__()
        self.in_channels =  in_channels
        self.activation = activation
        
        self.W_query = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // k), kernel_size=1)
        self.W_key = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // k), kernel_size=1)
        self.W_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X):
        """
        Input:
            X: (B, C, W, H)
        Output:
            output: (B, C, W, H) self attention value + input feature
            attention: (B, N, N)
        """
        B, C, W, H = X.size()
        
        queries = self.W_query(X).view(B, -1, W*H).permute(0, 2, 1) 
        # (B, C//k, W, H) -> (B, C//k, W*H) -> (B, W*H, C//k) = (B, N, C')
        
        keys = self.W_key(X).view(B, -1, W*H)
        # (B, C//k, W, H) -> (B, C//k, W*H) = (B, C', N)
        
        values = self.W_value(X).view(B, -1 ,W*H)
        # (B, C, W, H) -> (B, C, W*H) = (B, C, N)

        qk = torch.bmm(queries, keys)
        # (B, N, C')*(B, C', N) = (B, N, N)
        
        attention = self.softmax(qk)
        # (B, N, N)
        
        output = torch.bmm(values, attention.permute(0, 2, 1))
        # (B, C, N)*(B, N, N) = (B, C, N)
        
        output = output.view(B, C, W, H)
        # (B, C, N) -> (B, C, W, H)
        
        output = self.gamma * output + X
        # (B, C, W, H)
        
        return output, attention


#######################################################################################################
# CLASS: MultiHeadSelfAttention
#######################################################################################################
class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Layer"""
    def __init__(self, in_channels, num_heads=4, k=2):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert (in_channels // k) % num_heads == 0
        assert (in_channels) % num_heads == 0
        
        self.W_query = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // k), kernel_size=1)
        self.W_key = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // k), kernel_size=1)
        self.W_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        
        self.num_heads = num_heads
        self.k = k
        
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X):
        """
        Input:
            X: (B, C, W, H)
        Output:
            output: (B, C, W, H) self attention value + input feature
            attention: (B, h, N, N)
        """
        B, C, W, H = X.size()
        
        h = self.num_heads
        k = self.k
        
        queries = self.W_query(X).reshape(B, h, C // k // h, W*H).permute(0, 1, 3, 2)
        # (B, C//k, W, H) -> (B, h, C//k//h, W*H) -> (B, h, W*H, C//k//h) = (B, h, N, C'//h)
        
        keys = self.W_key(X).reshape(B, h, C // k // h, W*H)
        # (B, C//k, W, H) -> (B, h, C//k//h, W*H) = (B, h, C'//h, N)
        
        values = self.W_value(X).reshape(B, h, C // h, W*H)
        # (B, C, W, H) -> (B, h, C//h, W*H) = (B, h, C//h, N)
        
        qk = torch.matmul(queries, keys)
        # (B, h, N, C'//h)*(B, h, C'//h, N) = (B, h, N, N)
        
        attention = self.softmax(qk)
        # (B, h, N, N)
        
        output = torch.matmul(values, attention.permute(0, 1, 3, 2))
        # (B, h, C//h, N)*(B, h, N, N) = (B, h, C//h, N)
        
        output = output.view(B, C, W, H)
        # (B, h, C//h, N) -> (B, C, W, H)

        output = self.gamma * output + X
        # (B, C, W, H)

        return output, attention 
    

#######################################################################################################
# CLASS: Extra
#######################################################################################################
class Extra(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.new_conv = nn.ModuleList()
        self.new_conv.append(nn.Conv2d(512, 1, 3))
        self.new_conv.append(nn.Conv2d(512, 1, 3))
        self.new_conv.append(nn.Conv2d(512, 1, 3))
        self.new_conv.append(nn.Conv2d(512, 1, 3))
        
    def forward(self, inp, index):
        output = self.new_conv[index](inp)
        return output
