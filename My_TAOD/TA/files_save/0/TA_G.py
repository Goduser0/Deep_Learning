import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torchinfo import summary

from TA_layers import SelfAttention, PixelNorm, EqualLinear

########################################################################################################
# CLASS: FeatureMatchGenerator()
########################################################################################################
class FeatureMatchGenerator(nn.Module):
    def __init__(self, n_mlp=None, img_size=128, z_dim=128, conv_dim=64, lr_mlp=0.01):
        super(FeatureMatchGenerator, self).__init__()
        
        self.img_size = img_size
        
        self.z_dim = z_dim
        
        mlp = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        if n_mlp:
            mlp.append(PixelNorm())
            for i in range(n_mlp):
                mlp.append(EqualLinear(z_dim, z_dim, lr_mul=lr_mlp, activation='leaky_relu'))
            
        repeat_num = int(np.log2(self.img_size)) - 3 # =4
        multi = 2 ** repeat_num # =16
        
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * multi, 8)))
        layer1.append(nn.BatchNorm2d(conv_dim * multi))
        layer1.append(nn.ReLU())
        
        curr_dim = conv_dim * multi
        
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(curr_dim // 2))
        layer2.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(curr_dim // 2))
        layer3.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        attn1_dim = curr_dim
        
        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(curr_dim // 2))
        layer4.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        attn2_dim = curr_dim
        
        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*mlp)
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attn1 = SelfAttention(attn1_dim)
        self.attn2 = SelfAttention(attn2_dim)
    
    def forward(self, z):
        z = self.mlp(z)
        z = z.reshape(z.shape[0], z.shape[1], 1, 1)
        output = self.layer1(z)
        output = self.layer2(output)
        output = self.layer3(output)
        output, p1 = self.attn1(output)
        output = self.layer4(output)
        output, p2 = self.attn2(output)
        output = self.last(output)
        
        return output
        
########################################################################################################
# CLASS: Generater TEST
########################################################################################################
if __name__ == "__main__":
    z = torch.randn(8, 128) # batchsize z_dim
    FMG = FeatureMatchGenerator(n_mlp=2)
    output = FMG(z)
    print(output.shape)
    summary(FMG, z.shape, device="cpu")