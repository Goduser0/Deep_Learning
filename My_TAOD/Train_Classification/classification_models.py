import sys
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import warnings
warnings.filterwarnings("ignore")


##########################################################################################################
# Resnet18
##########################################################################################################
class Residual(nn.Module):
    def __init__(self, input_channels, hidden_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, hidden_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
            
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))  # stride=1: [B, 3, H, W] -> [B, C, H, W] 
        Y = self.bn2(self.conv2(Y)) # stride=1: [B, C, H, W] -> [B, C, H, W] 
        
        if self.conv3:
            X = self.conv3(X)
        
        Y += X
        return F.relu(Y)
    
# 残差块
def resnet_block(in_channels, out_channels, num_residual, first_block=False):
    blk=[]
    for i in range(num_residual):
        if i==0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2)) # [B, C_in, H, W] -> [B, C_out, H/2, W/2]
        else:
            blk.append(Residual(out_channels, out_channels)) # [B, C_in, H, W] -> [B, C_out, H, W]
    return blk

# Resnet 18
class Resnet18(nn.Module):
    def __init__(self, num_class):
        super(Resnet18, self).__init__()
        self.num_classes = num_class
        self.net = self.net_arch(num_class)
        
    def net_arch(self, num_class):
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ) # [B, 3, H, W] -> [B, 64, H/2, W/2]
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)) # [B, 64, H/2, W/2] -> [B, 64, H/2, W/2]
        b3 = nn.Sequential(*resnet_block(64, 128, 2)) # [B, 64, H/2, W/2] -> [B, 128, H/4, W/4]
        b4 = nn.Sequential(*resnet_block(128, 256, 2)) # [B, 128, H/4, W/4] -> [B, 256, H/8, W/8]
        b5 = nn.Sequential(*resnet_block(256, 512, 2)) # [B, 256, H/8, W/8] -> [B, 512, H/16, W/16]
        net = nn.Sequential(
            b1, b2, b3, b4, b5, # [B, 3, H, W] -> [B, 512, H/16, W/16]
            nn.AdaptiveAvgPool2d((1, 1)), # [B, 512, H/16, W/16] -> [B, 512, 1, 1]
            nn.Flatten(), # [B, 512, 1, 1] -> [B, 512]
            nn.Linear(512, num_class), # num_class=5: [B, 512] -> [B, 5]
            )
        
        return net

    def forward(self, x):
        x = self.net(x)
        return x


##########################################################################################################
# Resnet50
##########################################################################################################
# 残差块:Bottleneck
class Bottleneck(nn.Module):
    """
    __init__:
        downsample: 在_make_layer函数中赋值, 用于控制shortcut图片下采样
    """
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
               
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
    
    def forward(self, x):
        skip = x
        if self.downsample:
            skip = self.downsample(x)
        
        out = self.conv1(x) # [B, C_in, H, W] -> [B, C_out, H, W] 
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out) # stride=1: [B, C_out, H, W] -> [B, C_out, H, W] stride=2: [B, C_out, H, W] -> [B, C_out, H/2, W/2]
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out) # stride=1: [B, C_out, H, W] -> [B, 4*C_out, H, W] stride=2: [B, C_out, H/2, W/2] -> [B, 4*C_out, H/2, W/2]
        out = self.bn3(out)
        # 残差连接
        out += skip
        out = self.relu(out)
        
        return out

class Resnet50(nn.Module):
    def __init__(self, img_channel, num_class, block_num=[3, 4, 6, 3], block=Bottleneck):
        super(Resnet50, self).__init__()
        self.num_class = num_class
        
        self.block = block
        self.block_num = block_num
        
        self.img_channel = img_channel
        # layer1的输出维度
        self.in_channel = 64
        
        self.net = self.net_arch(self.img_channel, self.num_class, self.in_channel, self.block, self.block_num)

    def _make_stage(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=channel*block.expansion)
            )

        layers = []
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample, stride=stride)) 
        self.in_channel = channel*block.expansion
        
        for _ in range(1, block_num):
            layers.append(block(in_channel=self.in_channel, out_channel=channel))
        
        return nn.Sequential(*layers)
        
    def net_arch(self, img_channel, num_class, in_channel, block, block_num):
        layer1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=in_channel, kernel_size=7, stride=2, padding=3, bias=False), # [B, 3, H, W] -> [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # [B, 64, H/2, W/2] -> [B, 64, H/4, W/4]
        )
        
        stage1 = self._make_stage(block=block, channel=64, block_num=block_num[0], stride=1) # [B, 64, H/4, W/4] -> [B, 256, H/4, W/4]
        stage2 = self._make_stage(block=block, channel=128, block_num=block_num[1], stride=2) # [B, 256, H/4, W/4] -> [B, 512, H/8, W/8]
        stage3 = self._make_stage(block=block, channel=256, block_num=block_num[2], stride=2) # [B, 512, H/8, W/8] -> [B, 1024, H/16, W/16]
        stage4 = self._make_stage(block=block, channel=512, block_num=block_num[3], stride=2) # [B, 1024, H/16, W/16] -> [B, 2048, H/32, W/32]
        
        return nn.Sequential(
            layer1,
            stage1,
            stage2,
            stage3,
            stage4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # [B, 2048, H/32, W/32] -> [B, 2048, 1, 1]
            nn.Flatten(), # [B, 2048, 1, 1] -> [B, 2048]
            nn.Linear(in_features=512*self.block.expansion, out_features=num_class), # [B, 2048] -> [B, 5]
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    
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
class Vgg11(nn.Module):
    def __init__(self, conv_arch, img_size, num_classes):
        super(Vgg11, self).__init__()
        self.conv_arch = conv_arch
        self.img_size = img_size
        self.num_classes = num_classes
        self.net = self.net_arch(conv_arch, num_classes)
            
    def net_arch(self, conv_arch, num_classes):
        conv_blks = []
        in_channels = 3
        
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
                    
        return nn.Sequential(
            *conv_blks, # [B, 3, H, W] -> [B, 512, H/32, W/32]

            nn.Flatten(),
            nn.Linear(out_channels * (self.img_size // 2**(len(self.conv_arch)))**2, 4096), 
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.net(x)
        return(x)


##########################################################################################################
# Add other Nets
##########################################################################################################


##########################################################################################################
# FUNCTION:classsification_net_select
##########################################################################################################
RESNET18 = Resnet18(num_class=5)

RESNET50 = Resnet50(img_channel=3, num_class=5)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
VGG11 = Vgg11(conv_arch, img_size=128, num_classes=5)

RESNET18_PRETRAINED = torchvision.models.resnet18(pretrained=True)
RESNET18_PRETRAINED.fc = nn.Linear(RESNET18_PRETRAINED.fc.in_features, 5)

RESNET50_PRETRAINED = torchvision.models.resnet50(pretrained=True)
RESNET50_PRETRAINED.fc = nn.Linear(RESNET50_PRETRAINED.fc.in_features, 5)

def classification_net_select(name, pretrained=False):
    if (name.lower() == 'resnet18') and (pretrained is False):
        return RESNET18
    elif (name.lower() == 'vgg11') and (pretrained is False):
        return VGG11
    elif (name.lower() == 'resnet50') and (pretrained is False):
        return RESNET50
    elif (name.lower() == "resnet18") and pretrained:
        return RESNET18_PRETRAINED
    elif (name.lower() == "resnet50") and pretrained:
        return RESNET50_PRETRAINED
    else:
        sys.exit(f"ERROR:\t({__name__}):The Net: '{name}' doesn't exist")
        
        
##########################################################################################################
# Test
##########################################################################################################  
def test():
    pass
    model1 = torchvision.models.resnet18(pretrained=True)
    model2 = torchvision.models.resnet50(pretrained=True)
    model2.fc = nn.Linear(model2.fc.in_features, 5)
    x = torch.randn(8, 3, 128, 128)
    print(f"Input x:{x.shape}")
    output1 = RESNET18(x)
    output2 = RESNET50(x)
    output3 = VGG11(x)
    output4 = RESNET18_PRETRAINED(x)
    output5 = RESNET50_PRETRAINED(x)
    print(f"RESNET18 Y:{output1.shape} \nRESNET50 Y:{output2.shape} \nVGG11 Y:{output3.shape} \nRESNET18_PRETRAINED Y:{output4.shape} \nRESNET50_PRETRAINED Y:{output5.shape}")

if __name__ == "__main__":
    test()
