import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
import torch.nn.functional as F
from torchinfo import summary

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_layers import SelfAttention, StdDevNorm


########################################################################################################
# CLASS: FeatureMatchDiscriminator()
########################################################################################################
class FeatureMatchDiscriminator(nn.Module):
    def __init__(self, img_size=128, conv_dim=64):
        """
        input: image
        output: [feat_map0, feat_map1, feat_map2, feat_map3, predict_label, feat_mmd]
        """
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
        # stdnorm.append(StdDevNorm(curr_dim))
        
        # last
        last.append(nn.Conv2d(curr_dim, 1, img_size // 16))
        
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        # self.stdnorm = nn.Sequential(*stdnorm)
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
        
        # output = self.stdnorm(output)
        
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


########################################################################################################
# CLASS: FirstResBlockDiscriminator()
########################################################################################################
class FirstResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.skip_conv.weight.data, np.sqrt(2))
        
        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2),
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            self.skip_conv,
        )
        
    def forward(self, x):
        return self.convs(x) + self.skip(x)
    
########################################################################################################
# CLASS: ResBlockDiscriminator()
########################################################################################################
class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        # convs
        if stride == 1:
            self.convs = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2,
            )
        else:
            self.convs = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        # skip
        self.skip = nn.Sequential()
        if stride != 1:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.skip_conv.weight.data, np.sqrt(2))
            
            self.skip = nn.Sequential(
                self.skip_conv,
                nn.AvgPool2d(2, stride=stride, padding=0),
            )
        
    def forward(self, x, feat=False):
        if feat:
            return self.convs(x)
        else:
            return self.convs(x) + self.skip(x)

########################################################################################################
# CLASS: PFS_Discriminator_patch()
########################################################################################################
class PFS_Discriminator_patch(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        """
        input: image
        output: [feature_map, predict_label]
        """
        super(PFS_Discriminator_patch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inputBatch = nn.BatchNorm2d(self.in_channels)
        self.model0 = FirstResBlockDiscriminator(self.in_channels, self.out_channels, stride=2)
        self.model1 = ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2)
        self.model2 = ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2)
        self.model3 = ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2)
        self.model4 = ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2)
        
        self.conv1 = nn.Conv2d(self.out_channels, 1, 1, 1, 0)
        self.ReLU = nn.ReLU()
        self.avg = nn.AvgPool2d(4)
        
        self.fc = nn.Linear(self.out_channels, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        
    def forward(self, x):
        x = self.model0(x)
        x = self.model1(x)
        x = self.model2(x)
        # x1
        x1 = self.conv1(x)
        # x2
        x = self.model3(x)
        x = self.model4(x)
        x = self.ReLU(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x2 = self.fc(x)
        
        return x1, x2

########################################################################################################
# CLASS: PFS_Discriminator()
########################################################################################################
class PFS_Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        """
        input: image
        output: predict_label
        """
        super(PFS_Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inputBatch = nn.BatchNorm2d(self.in_channels)
        self.model0 = FirstResBlockDiscriminator(in_channels, self.out_channels, stride=2)
        self.model1 = ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2)
        self.model2 = ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2)
        self.model3 = ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2)
        self.model4 = ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2)
        
        self.ReLU = nn.ReLU()
        self.avg = nn.AvgPool2d(4)
        
        self.fc = nn.Linear(self.out_channels, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
    
    def forward(self, x):
        f0 = self.model0(x)
        f1 = self.model1(f0)
        f2 = self.model2(f1)
        f3 = self.model3(f2)
        f4 = self.model4(f3)
        
        f4_r = self.ReLU(f4)
        f5 = self.avg(f4_r)
        f5_f = f5.view(-1, self.out_channels)
        out = self.fc(f5_f)
        return out
    
########################################################################################################
# CLASS: CoGAN_Discriminator()
########################################################################################################
class CoGAN_Discriminator(nn.Module):
    def __init__(self, channels=3, hidden_channels=128):
        """
        input: image, domain
        output: predict_label
        """
        super(CoGAN_Discriminator, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        
        self.front1 = nn.Sequential(
            FirstResBlockDiscriminator(channels, self.hidden_channels, stride=2),
            ResBlockDiscriminator(self.hidden_channels, self.hidden_channels, stride=2))
        self.front2 = nn.Sequential(
            FirstResBlockDiscriminator(channels, self.hidden_channels, stride=2),
            ResBlockDiscriminator(self.hidden_channels, self.hidden_channels, stride=2))

        self.back = nn.Sequential(
            ResBlockDiscriminator(self.hidden_channels, self.hidden_channels, stride=2),
            ResBlockDiscriminator(self.hidden_channels, self.hidden_channels, stride=2),
            ResBlockDiscriminator(self.hidden_channels, self.hidden_channels, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(4))
        self.fc = nn.Linear(self.hidden_channels, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)


    def forward(self, img, domain, feat=False):
        if domain == 'S':
            out = self.fc(self.back(self.front1(img)).view(-1, self.hidden_channels))
        elif domain == 'T':
            out = self.fc(self.back(self.front2(img)).view(-1, self.hidden_channels))
        return out

########################################################################################################
# Discriminator TEST
########################################################################################################
def test():
    # D = PFS_Discriminator()
    # D = PFS_Discriminator_patch()
    # D = CoGAN_Discriminator()
    D = FeatureMatchDiscriminator()
    X = torch.randn(8, 3, 128, 128) # (B, C, W, H)
    # summary(D, X.shape, device="cpu")
    
    print(f"Input X: {X.shape}")
    Y = D(X)
    if isinstance(Y, list):
        for i, y in enumerate(Y):
            print(f"Output {i}: {y.shape}")
    else:
        print(f"Output Y: {Y.shape}")
    
if __name__ == "__main__":
    test()