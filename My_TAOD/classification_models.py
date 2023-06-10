import sys

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T


##########################################################################################################
# Resnet18
##########################################################################################################
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


# 残差块
def resnet_block(in_channels, out_channels, num_residual, first_block=False):
    blk=[]
    for i in range(num_residual):
        if i==0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return blk


# Resnet 18
class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()
        self.num_classes = num_classes
        self.net = self.net_arch(num_classes)
        
    def net_arch(self, num_classes):
        b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
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
        in_channel: 残差块输入通道
        out_channel: 残差块输出通道数
        stride: 卷积步长
        downsample: 在_make_layer函数中赋值, 用于控制shortcut图片下采样
    """
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
               
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)   # H,W不变。C: in_channel -> out_channel
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)  # stride=1:H,W不变; stride=2:W/2, H/2，C不变
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)   # H,W不变。C: out_channel -> 4*out_channel
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out

class resnet50(nn.Module):
    def __init__(self, img_channel, num_classes, block_num=[3, 4, 6, 3], block=Bottleneck):
        super(resnet50, self).__init__()
        self.num_classes = num_classes
        
        self.block = block
        self.block_num = block_num
        
        self.img_channel = img_channel
        # layer1的输出维度
        self.in_channel = 64
        
        self.net = self.net_arch(self.img_channel, self.num_classes, self.in_channel, self.block, self.block_num)

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
        
    def net_arch(self, img_channel, num_classes, in_channel, block, block_num):
        layer1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=in_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        )
        
        stage1 = self._make_stage(block=block, channel=64, block_num=block_num[0], stride=1) 
        stage2 = self._make_stage(block=block, channel=128, block_num=block_num[1], stride=2)
        stage3 = self._make_stage(block=block, channel=256, block_num=block_num[2], stride=2)
        stage4 = self._make_stage(block=block, channel=512, block_num=block_num[3], stride=2)
        
        return nn.Sequential(
            layer1,  
            stage1, 
            stage2, 
            stage3, 
            stage4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512*self.block.expansion, out_features=num_classes),
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
class vgg(nn.Module):
    def __init__(self, conv_arch, num_classes):
        super(vgg, self).__init__()
        self.conv_arch = conv_arch
        self.num_classes = num_classes
        self.net = self.net_arch(conv_arch, num_classes)
            
    def net_arch(self, conv_arch, num_classes):
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

            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.net(x)
        return(x)



##########################################################################################################
# Add other Nets
##########################################################################################################



##########################################################################################################
# FUNCTION:show_net_arch()
##########################################################################################################
def show_net_arch(input, net_class):
    print(input.shape)
    for layer in net_class.net:
        input = layer(input)
        print(f'{layer.__class__.__name__} out: \t {input.shape}')


##########################################################################################################
# FUNCTION:classsification_net_select
# Classification_net_select
##########################################################################################################
Resnet18 = resnet18(num_classes=6)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
VGG11 = vgg(conv_arch, num_classes=6)

Resnet50 = resnet50(img_channel=1, num_classes=6)

    
def classification_net_select(name):
    if name.lower() == 'resnet18':
        return Resnet18
    elif name.lower() == 'vgg11':
        return VGG11
    elif name.lower() == 'resnet50':
        return Resnet50
    else:
        sys.exit(f"ERROR:\t({__name__}):The Net: '{name}' doesn't exist")
