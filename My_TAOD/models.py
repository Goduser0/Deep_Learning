import sys

from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T


##########################################################################################################
# Resnet18
##########################################################################################################
# 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
            
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        
        if self.conv3:
            X = self.conv3(X)
        
        Y += X
        return F.relu(Y)
    
# Resnet 18
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)


def resnet_block(in_channels, out_channels, num_residual, first_block=False):
    blk=[]
    for i in range(num_residual):
        if i==0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

Resnet18 = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 6),
)

###########################################################################################################
# VGG-11
###########################################################################################################
# VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# VGG-11
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    out_channels = 1
    
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
                
    return nn.Sequential(
        *conv_blks,

        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), 
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 6)
    )
    

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
VGG11 = vgg(conv_arch)


##########################################################################################################
# Add other Nets
##########################################################################################################



##########################################################################################################
# FUNCTION:classsification_net_select
# Classification_net_select
##########################################################################################################
def classification_net_select(name):
    if name.lower() == 'resnet18':
        return Resnet18
    elif name.lower() == 'vgg11':
        return VGG11

    else:
        sys.exit(f"ERROR:\t({__name__}):The Net: '{name}' doesn't exist")
    