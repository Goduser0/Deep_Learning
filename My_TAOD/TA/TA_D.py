import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torchinfo import summary

from TA_layers import SelfAttention


##########################################################################################################
# CLASS: FeatureMatchDiscriminator()
#########################################################################################################
class FeatureMatchDiscriminator(nn.Module):
    def __init__(self, img_size=64, conv_dim=64):
        self.img_size = img_size
        
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        # layer1
        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim
        
        # layer2
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim * 2
        
        # layer3
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attention1_dim = curr_dim
        
        # layer4
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attention2_dim = curr_dim
        
        # last
        last.append(nn.Conv2d(curr_dim, 1, 4))
        
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attention1 = SelfAttention(attention1_dim)
        self.attention2 = SelfAttention(attention2_dim)
        
        # MLP Project Head
        self.MLP = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        
        # Upsample
        self.Upsample1 = nn.ConvTranspose2d(512, 512//2, 4, 2, 1)
        self.Upsample2 = nn.ConvTranspose2d(256, 256//2, 4, 2, 1)
        self.Upsample3 = nn.ConvTranspose2d(128, 128//2, 4, 2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.weights = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X):
        results = []
        
        output = self.layer1(X)
        feature1 = output
        results.append(feature1) # 1st Feature map layer 
        
        output = self.layer2(output)
        feature2 = output
        results.append(feature2) # 2nd Feature map layer
        
        output = self.layer3(output)
        output, p1 = self.attention1(output)
        feature3 = output
        results.append(feature3) # 3rd Feature map layer
        
        output = self.layer4(output)
        output, p2 = self.attention2(output)
        feature4 = output
        results.append(feature4) # 4th Feature map layer
        
        output = self.last(output)
        output = output.reshape(output.shape[0], -1)
        results.append(output) # Final output to calculate GAN loss
        
        output = self.MLP(output)
        results.append(output) # Project output to calculate MMD loss
        
        # Attention Module
        feature4_up = self.Upsample1(feature4)
        feature3 = feature3 + feature4_up
        
        feature3_up = self.Upsample2(feature3)
        feature2 = feature2 + feature3_up
        
        feature2_up = self.Upsample3(feature2)
        feature1 = feature1 + feature2_up
        
        feature_avg = self.avg(feature1).reshape(feature1.shape[0], -1)
        feature_attention = self.weights(feature_avg)
        feature_attention = self.softmax(feature_attention)
        
        for i in range(feature_attention.shape[1]):
            attention = feature_attention[:, i].reshape(feature_attention.shape[0], 1, 1, 1)
            results[i] = results[i] * attention
        
        return results
    
    
##########################################################################################################
# CLASS: FeatureMatchPatchDiscriminator()
#########################################################################################################
class FeatureMatchPatchDiscriminator(nn.Module):
    def __init__(self, img_size=64, conv_dim=64):
        self.img_size = img_size
        
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        # layer1
        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim
        
        # layer2
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim * 2
        
        # layer3
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attention1_dim = curr_dim
        
        # layer4
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attention2_dim = curr_dim
        
        # last
        last.append(nn.Conv2d(curr_dim, 1, 4))
        
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attention1 = SelfAttention(attention1_dim)
        self.attention2 = SelfAttention(attention2_dim)
        
        # MLP Project Head
        self.MLP = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        
        # Upsample
        self.Upsample1 = nn.ConvTranspose2d(512, 512//2, 4, 2, 1)
        self.Upsample2 = nn.ConvTranspose2d(256, 256//2, 4, 2, 1)
        self.Upsample3 = nn.ConvTranspose2d(128, 128//2, 4, 2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.weights = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X):
        results = []
        
        output = self.layer1(X)
        feature1 = output
        results.append(feature1) # 1st Feature map layer: (B, 3, W, H) -> (B, 64, W//2, H//2)
        
        output = self.layer2(output)
        feature2 = output
        results.append(feature2) # 2nd Feature map layer: -> (B, 128, W//4, H//4)
        
        output = self.layer3(output)
        output, p1 = self.attention1(output)
        feature3 = output
        results.append(feature3) # 3rd Feature map layer: -> (B, 256, W//8, H//8)
        
        output = self.layer4(output)
        output, p2 = self.attention2(output)
        feature4 = output
        results.append(feature4) # 4th Feature map layer: -> (B, 512, W//16, H//16)
        
        output = self.last(output) # -> (B, 1, 1, 1)
        output = output.reshape(output.shape[0], -1) # -> (B, 1)
        results.append(output) # Final output to calculate GAN loss
        
        output = self.MLP(output) # -> (B, 64)
        results.append(output) # Project output to calculate MMD loss
        
        # Attention Module
        feature4_up = self.Upsample1(feature4)
        feature3 = feature3 + feature4_up
        
        feature3_up = self.Upsample2(feature3)
        feature2 = feature2 + feature3_up
        
        feature2_up = self.Upsample3(feature2)
        feature1 = feature1 + feature2_up
        
        feature_avg = self.avg(feature1).reshape(feature1.shape[0], -1)
        feature_attention = self.weights(feature_avg)
        feature_attention = self.softmax(feature_attention)
        
        for i in range(feature_attention.shape[1]):
            attention = feature_attention[:, i].reshape(feature_attention.shape[0], 1, 1, 1)
            results[i] = results[i] * attention
        
        return results
        


        
        