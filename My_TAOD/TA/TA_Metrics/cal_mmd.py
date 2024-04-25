import torch
import torch.nn as nn
import torchvision.transforms as T

import pandas as pd
import cv2
import numpy as np

import sys
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_utils import load_img_for_mmd

class calculator_MMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(calculator_MMD, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = None
        
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """ with multi RBF-bandwith Kernel to calculate
        Args:
            source (_type_): (sample_size_1, feature_size)
            target (_type_): (sample_size_2, feature_size)
            kernel_mul (float, optional): For calculating bandwith. Defaults to 2.0.
            kernel_num (int, optional): multi Kernel nums. Defaults to 5.
            fix_sigma (_type_, optional): whether to use fix sigma. Defaults to None.
        Returns:
            (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2) matrix.
                [   K_ss K_st
                    K_ts K_tt ]
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)),
            int(total.size(0)),
            int(total.size(1)),
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)),
            int(total.size(0)),
            int(total.size(1)),
        )
        L2_distance = ((total0 - total1)**2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul**(kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        
        return sum(kernel_val)
        
    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        
        return loss
        
    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source=source, 
                target=target, 
                kernel_mul=self.kernel_mul, 
                kernel_num=self.kernel_num, 
                fix_sigma=self.fix_sigma
            )
            
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY -YX)
            
            return loss
        
def score_mmd(real_path, fake_path, batch_size):
    MMD = calculator_MMD('rbf')
    trans = T.Compose(
        [
            T.ToTensor(),
            T.Resize((256, 256)),
        ]
    )
    real_img = load_img_for_mmd(real_path, trans, batch_size)[0]
    fake_img = load_img_for_mmd(fake_path, trans, batch_size)[0]
    
    x = real_img.reshape(real_img.shape[0], -1)
    y = fake_img.reshape(fake_img.shape[0], -1)
    
    result = MMD(x, y)
    return result.numpy()
    
def test():
    real_path = "./My_TAOD/dataset/PCB_Crop/30-shot/train/0.csv"
    fake_path = "./My_TAOD/dataset/PCB_Crop/30-shot/train/1.csv"
    result = score_mmd(real_path, fake_path, 30)
    
    print(f"MMD: {result}")
  
if __name__ == '__main__':
    test()