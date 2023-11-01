import numpy as np

import torch
import torch.nn as nn
import torch.nn.modules as M
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torchinfo import summary

import warnings
warnings.filterwarnings("ignore")

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
# CLASS: ResBlockGenerator()
########################################################################################################
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        
        self.convs = nn.Sequential(
            M.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            M.InstanceNorm2d(out_channels),
            nn.ReLU(),
            self.conv2,
        )
        
        self.skip = nn.Sequential()
        
        if stride != 1:
            self.skip = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        return self.convs(x) + self.skip(x)

########################################################################################################
# CLASS: PFS_Generator()
########################################################################################################
class PFS_Generator(nn.Module):
    def __init__(self, z_dim, hidden_channels=128, out_channels=3):
        super(PFS_Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        self.dense = nn.Linear(self.z_dim, 4 * 4 * self.hidden_channels)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        
        self.model0 = ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2)
        self.model1 = ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2)
        self.model2 = ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2)
        self.model3 = ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2)
        self.model4 = ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2)
        self.model5 = nn.BatchNorm2d(self.hidden_channels)
        self.model6 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.model6.weight.data, 1.)
        self.model = nn.Sequential(
            self.model0, 
            self.model1,
            self.model2, 
            self.model3,
            self.model4,
            self.model5,
            nn.ReLU(),
            self.model6,
            nn.Tanh(),
        )
        
    def forward(self, z):
        f1 = self.dense(z).view(-1, self.hidden_channels, 4, 4)
        out = self.model(f1)
        return out

########################################################################################################
# CLASS: CoGAN_Generator()
########################################################################################################
class CoGAN_Generator(nn.Module):
    def __init__(self, z_dim, hidden_channels=128, out_channels=3):
        super(CoGAN_Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        self.dense = nn.Linear(self.z_dim, 4 * 4 * self.hidden_channels)
        self.front = nn.Sequential(
            ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2),
            ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2),
            ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2))
        self.back1 = nn.Sequential(
            ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2),
            ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.out_channels, 3, stride=1, padding=1),
            nn.Tanh())
        self.back2 = nn.Sequential(
            ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2),
            ResBlockGenerator(self.hidden_channels, self.hidden_channels, stride=2),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.out_channels, 3, stride=1, padding=1),
            nn.Tanh())

        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.back1[4].weight.data, 1.)
        nn.init.xavier_uniform(self.back2[4].weight.data, 1.)

    def forward(self, z, domain):
        f = self.front(self.dense(z).view(-1, self.hidden_channels, 4, 4))
        if domain == 'S':
            return self.back1(f)
        elif domain == 'T':
            return self.back2(f)

########################################################################################################
# CLASS: Generater TEST
########################################################################################################
def test():
    # G = FeatureMatchGenerator(n_mlp=2)
    G = CoGAN_Generator(128)
    z = torch.randn(8, 128) # batchsize z_dim
    # summary(G, z.shape, device="cpu")
    
    print(f"Input z:{z.shape}")
    output = G(z, domain="T")
    print(f"Output Y:{output.shape}")
    
if __name__ == "__main__":
    test()