import sys
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from efficientnet_pytorch import EfficientNet

import warnings
warnings.filterwarnings("ignore")

##########################################################################################################
# FUNCTION:classsification_net_select
##########################################################################################################
NUM_ClASSES = 5

def classification_net_select(name):
    if (name.lower() == 'resnet18'):
        RESNET18 = torchvision.models.resnet18(pretrained=False)
        RESNET18.fc = nn.Linear(RESNET18.fc.in_features, NUM_ClASSES)
        return RESNET18
    
    elif (name.lower() == 'resnet50'):
        RESNET50 = torchvision.models.resnet50(pretrained=False)
        RESNET50.fc = nn.Linear(RESNET50.fc.in_features, NUM_ClASSES)
        return RESNET50
    
    elif (name.lower() == "resnet18_pretrained"):
        RESNET18_PRETRAINED = torchvision.models.resnet18(pretrained=True)
        RESNET18_PRETRAINED.fc = nn.Linear(RESNET18_PRETRAINED.fc.in_features, NUM_ClASSES)
        return RESNET18_PRETRAINED
    
    elif (name.lower() == "resnet50_pretrained"):
        RESNET50_PRETRAINED = torchvision.models.resnet50(pretrained=True)
        RESNET50_PRETRAINED.fc = nn.Linear(RESNET50_PRETRAINED.fc.in_features, NUM_ClASSES)
        return RESNET50_PRETRAINED
    
    elif (name.lower() == "efficientnet"):
        EFFICIENTNET = EfficientNet.from_name(model_name='efficientnet-b0', num_classes=NUM_ClASSES)
        return EFFICIENTNET
    
    elif (name.lower() == "efficientnet_pretrained"):
        EFFICIENTNET_PRETRAINED = EfficientNet.from_pretrained(model_name='efficientnet-b0', num_classes=NUM_ClASSES)
        return EFFICIENTNET_PRETRAINED
    
    elif (name.lower() == "vgg11"):
        VGG11 = torchvision.models.vgg11(num_classes=NUM_ClASSES)
        return VGG11
    
    elif (name.lower() == "vgg11_pretrained"):
        VGG11_PRETRAINED = torchvision.models.vgg11(pretrained=True)
        VGG11_PRETRAINED.classifier[-1] = nn.Linear(4096, NUM_ClASSES)
        return VGG11_PRETRAINED

    elif (name.lower() == "vgg19_pretrained"):
        VGG19_PRETRAINED = torchvision.models.vgg19(pretrained=True)
        VGG19_PRETRAINED.classifier[-1] = nn.Linear(4096, NUM_ClASSES)
        return VGG19_PRETRAINED
    
    elif (name.lower() == "mobilenet"):
        MOBILENET = torchvision.models.mobilenet_v2(num_classes=NUM_ClASSES)
        return MOBILENET
    
    elif (name.lower() == "mobilenet_pretrained"):
        MOBILENET_PRETRAINED = torchvision.models.mobilenet_v2(pretrained=True)
        MOBILENET_PRETRAINED.classifier[-1] = nn.Linear(MOBILENET_PRETRAINED.last_channel, NUM_ClASSES)
        return MOBILENET_PRETRAINED
    
    
    else:
        sys.exit(f"ERROR:\t({__name__}):The Net: '{name}' doesn't exist")
        
        
##########################################################################################################
# Test
##########################################################################################################  
def test():
    pass
    x = torch.randn(8, 3, 128, 128)
    print(f"Input x:{x.shape}")
    
    # print(RESNET18(x).shape)
    # print(RESNET18_PRETRAINED(x).shape)
    # print(RESNET50(x).shape)
    # print(RESNET50_PRETRAINED(x).shape)
    # print(EFFICIENTNET(x).shape)
    # print(EFFICIENTNET_PRETRAINED(x).shape)
    # print(VGG11(x).shape)
    # print(VGG11_PRETRAINED(x).shape)
    # print(MOBILENET(x).shape)
    # print(MOBILENET_PRETRAINED(x).shape)

if __name__ == "__main__":
    test()
