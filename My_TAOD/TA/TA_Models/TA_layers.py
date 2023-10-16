import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.models import vgg16
from torch import optim

import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import warnings

import sys
sys.path.append("My_TAOD/TA/TA_Utils")
from TA_utils import *
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader, img_1to255, img_255to1
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_Augmentation import ImgAugmentation


##########################################################################################################
# CLASS: PixelNorm
##########################################################################################################
class PixelNorm(nn.Module):
    """像素归一化"""
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)

#######################################################################################################
# FUNCTION: fused_leaky_relu()
#######################################################################################################
def fused_leaky_relu(input, bias, negative_slope=0.2, scale= 2 ** 0.5):
    return scale * F.leaky_relu(
        input + bias.view((1, -1) + (1,) * (len(input.shape) - 2)), 
        negative_slope=negative_slope
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
            # 平替
            output = F.linear(input, self.weight * self.scale)
            output = fused_leaky_relu(output, self.bias * self.lr_mul)
            # 自定义
            #output = F.linear(input, self.weight * self.scale, self.bias * self.lr_mul)
            #output = F.leaky_relu(output, 0.2)
            
        else:
            output = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

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
# CLASS: StdDevNorm
#######################################################################################################
class StdDevNorm(nn.Module):
    def __init__(self, input_channl, stddev_feat=1, stddev_group=4):
        super().__init__()
        self.stddev_feat = stddev_feat
        self.stddev_group = stddev_group
        self.conv = nn.Conv2d(input_channl + 1, input_channl, 1)
        
    def forward(self, input):
        batch, channel, height, width = input.shape # (B, C, H, W)
        group = min(batch, self.stddev_group)
        stddev = input.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width) # ->(B, 1, 1, C, H, W)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        output = torch.cat([input, stddev], 1)
        output = self.conv(output)
        return output

#######################################################################################################
# CLASS: ResidualConv_original
#######################################################################################################
class ResidualConv_original(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv_original, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)
    
#######################################################################################################
# CLASS: ResidualConv
#######################################################################################################
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv_original, self).__init__()
        
        self.leakyrelu_slope = 0.1
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(self.leakyrelu_slope),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(self.leakyrelu_slope),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)
    
#######################################################################################################
# CLASS: KLDLoss
#######################################################################################################
class KLDLoss(nn.Module):
    def __init__(self, reduction):
        super(KLDLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = F.kl_div(q.log(), p, reduction=self.reduction)
        return loss

#######################################################################################################
# CLASS: PerceptualLoss
#######################################################################################################
class PerceptualLoss(nn.Module):
    def __init__(self, device, layer_indexs=None, loss=nn.MSELoss()):
        """
        return loss is "batchmean"
        Args:
            real_img (torch.float32)
            fake_img (torch.float32)
        """
        super(PerceptualLoss, self).__init__()
        self.criterion = loss
        self.device = device
        self.layer_indexs = layer_indexs
        
        self.vgg_model = vgg16(pretrained=True).features[:16]
        self.vgg_model = self.vgg_model.to(self.device)
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
        }
    
    
    def output_features(self, x):
        output = {}
        for name, module in self.vgg_model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
    
        return list(output.values())
    
    def forward(self, real_img, fake_img):
        loss = []
        real_img = real_img.to(self.device)
        fake_img = fake_img.to(self.device)
        real_img_features = self.output_features(real_img)
        fake_img_features = self.output_features(fake_img)
        for real_img_feature, fake_img_feature in zip(real_img_features, fake_img_features):
            loss.append(self.criterion(real_img_feature, fake_img_feature))
        return sum(loss) / len(loss)

#######################################################################################################
# CLASS: TEST
#######################################################################################################
def test():
    warnings.filterwarnings("ignore")
    PLloss = PerceptualLoss('cuda:0')
    
    trans = T.Compose(
        [
            T.ToTensor(), 
            T.Resize((128, 128)), # (0, 255)
        ]
    )
    data_iter_loader = get_loader('PCB_200', 
                              "./My_TAOD/dataset/PCB_200/0.7-shot/train/0.csv", 
                              32, 
                              4, 
                              shuffle=True, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
    imgAug = ImgAugmentation()
    
    for i, data in enumerate(data_iter_loader):
        raw_img = data[0] # [B, C, H, W] -1~1
        print(raw_img.shape)
        real_imgs_aug = imgAug.imgRealAug(img_1to255(raw_img))
        real_imgs_aug = img_255to1(real_imgs_aug)
        loss = PLloss(raw_img, real_imgs_aug)
        print(loss)
        break
    
if __name__ == "__main__":
    test()
    