import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
import torch.nn.functional as F
from torchinfo import summary

from TA_layers import SelfAttention, StdDevNorm


########################################################################################################
# CLASS: FeatureMatchDiscriminator()
########################################################################################################
class FeatureMatchDiscriminator(nn.Module):
    def __init__(self, img_size=128, conv_dim=64):
        super(FeatureMatchDiscriminator, self).__init__()
        self.img_size = img_size
        
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        stdnorm = []
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
        
        # stdnorm
        stdnorm.append(StdDevNorm(curr_dim))
        
        # last
        last.append(nn.Conv2d(curr_dim, 1, img_size // 16))
        
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.stdnorm = nn.Sequential(*stdnorm)
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
        
    def forward(self, X) -> list:
        features = []
        
        output = self.layer1(X)
        feature1 = output
        features.append(feature1) # 0: Feature map layer: (B, 3, W, H) -> (B, 64, W//2, H//2)
        
        output = self.layer2(output)
        feature2 = output
        features.append(feature2) # 1: Feature map layer: -> (B, 128, W//4, H//4)
        
        output = self.layer3(output)
        output, p1 = self.attention1(output)
        feature3 = output
        features.append(feature3) # 2: Feature map layer: -> (B, 256, W//8, H//8)
        
        output = self.layer4(output)
        output, p2 = self.attention2(output)
        feature4 = output
        features.append(feature4) # 3: Feature map layer: -> (B, 512, W//16, H//16)
        
        output = self.stdnorm(output)
        
        output = self.last(output) # -> (B, 1, 1, 1)
        output = output.reshape(output.shape[0], -1) # -> (B, 1)
        features.append(output) # 4: Final output to calculate GAN loss: ->(B, 1)
        
        output = self.MLP(output) # -> (B, 64)
        features.append(output) # 5: Project output to calculate MMD loss: ->(B, 64)
        
        # Attention Module
        feature4_up = self.Upsample1(feature4) # (B, 512, W//16, H//16) -> (B, 256, W//8, H//8)
        feature3_combine = feature3 + feature4_up
        
        feature3_up = self.Upsample2(feature3_combine) # (B, 256, W//8, H//8) -> (B, 128, W//4, H//4)
        feature2_combine = feature2 + feature3_up
        
        feature2_up = self.Upsample3(feature2_combine) # (B, 128, W//4, H//4) -> (B, 64, W//2, H//2)
        feature1_combine = feature1 + feature2_up
        
        feature_avg = self.avg(feature1_combine).reshape(feature1_combine.shape[0], -1) # (B, 64, W//4, H//4) -> (B, 64)
        feature_attention = self.weights(feature_avg) # (B, 64) -> (B, 4)
        feature_attention = self.softmax(feature_attention) # (B, 4) -> Softmax
        
        for i in range(feature_attention.shape[1]):
            attention = feature_attention[:, i].reshape(feature_attention.shape[0], 1, 1, 1)
            features[i] = features[i] * attention
        
        return features
    

#######################################################################################################
# CLASS: Extra
#######################################################################################################
class Extra(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.new_conv = nn.ModuleList()
        self.new_conv.append(nn.Conv2d(64, 1, 3))
        self.new_conv.append(nn.Conv2d(128, 1, 3))
        self.new_conv.append(nn.Conv2d(256, 1, 3))
        self.new_conv.append(nn.Conv2d(512, 1, 3))
        
        self.activater = nn.LeakyReLU()
        
    def forward(self, input, index):
        output = self.new_conv[index](input)
        output = self.activater(output)
        return output


########################################################################################################
# CLASS: FeatureMatchPatchDiscriminator()
########################################################################################################
class FeatureMatchPatchDiscriminator(nn.Module):
    def __init__(self, img_size=128, conv_dim=64):
        super(FeatureMatchPatchDiscriminator, self).__init__()
        self.img_size = img_size
        
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        stdnorm = []
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
        
        # stdnorm
        stdnorm.append(StdDevNorm(curr_dim))
        
        # last
        last.append(nn.Conv2d(curr_dim, 1, img_size // 16))
        
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.stdnorm = nn.Sequential(*stdnorm)
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
        
    def forward(self, X, flag=None, p_ind=None, extra=None):
        features = []
        
        output = self.layer1(X)
        feature1 = output
        features.append(feature1) # 0: Feature map layer: (B, 3, W, H) -> (B, 64, W//2, H//2)
        
        output = self.layer2(output)
        feature2 = output
        features.append(feature2) # 1: Feature map layer: -> (B, 128, W//4, H//4)
        
        output = self.layer3(output)
        output, p1 = self.attention1(output)
        feature3 = output
        features.append(feature3) # 2: Feature map layer: -> (B, 256, W//8, H//8)
        
        output = self.layer4(output)
        output, p2 = self.attention2(output)
        feature4 = output
        features.append(feature4) # 3: Feature map layer: -> (B, 512, W//16, H//16)
        
        output = self.stdnorm(output)
        
        output = self.last(output) # -> (B, 1, 1, 1)
        output = output.reshape(output.shape[0], -1) # -> (B, 1)
        features.append(output) # 4: Final output to calculate GAN loss: ->(B, 1)
        
        output = self.MLP(output) # -> (B, 64)
        features.append(output) # 5: Project output to calculate MMD loss: ->(B, 64)
        
        # Attention Module
        feature4_up = self.Upsample1(feature4) # (B, 512, W//16, H//16) -> (B, 256, W//8, H//8)
        feature3_combine = feature3 + feature4_up
        
        feature3_up = self.Upsample2(feature3_combine) # (B, 256, W//8, H//8) -> (B, 128, W//4, H//4)
        feature2_combine = feature2 + feature3_up
        
        feature2_up = self.Upsample3(feature2_combine) # (B, 128, W//4, H//4) -> (B, 64, W//2, H//2)
        feature1_combine = feature1 + feature2_up
        
        feature_avg = self.avg(feature1_combine).reshape(feature1_combine.shape[0], -1) # (B, 64, W//2, H//2) -> (B, 64)
        feature_attention = self.weights(feature_avg) # (B, 64) -> (B, 4)
        feature_attention = self.softmax(feature_attention) # (B, 4) -> Softmax
        
        for i in range(feature_attention.shape[1]):
            attention = feature_attention[:, i].reshape(feature_attention.shape[0], 1, 1, 1)
            features[i] = features[i] * attention
        
        # Patch Discriminator
        if (flag) and (flag>0):
            output = extra(features[p_ind], p_ind)
            return output
        
        # Image Discriminator
        return features

########################################################################################################
# Discriminator TEST
########################################################################################################
if __name__ == "__main__":
    D = FeatureMatchDiscriminator()
    X = torch.randn(8, 3, 128, 128) # (B, C, W, H)
    print(f"Input X: {X.shape}")
    Y = D(X)
    summary(D, X.shape, device="cpu")
        