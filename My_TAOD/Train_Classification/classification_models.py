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
RESNET18 = torchvision.models.resnet18(pretrained=False)
RESNET18.fc = nn.Linear(RESNET18.fc.in_features, 5)

RESNET18_PRETRAINED = torchvision.models.resnet18(pretrained=True)
RESNET18_PRETRAINED.fc = nn.Linear(RESNET18_PRETRAINED.fc.in_features, 5)

RESNET50 = torchvision.models.resnet50(pretrained=False)
RESNET50.fc = nn.Linear(RESNET50.fc.in_features, 5)

RESNET50_PRETRAINED = torchvision.models.resnet50(pretrained=True)
RESNET50_PRETRAINED.fc = nn.Linear(RESNET50_PRETRAINED.fc.in_features, 5)

EFFICIENTNET = EfficientNet.from_name(model_name='efficientnet-b0', num_classes=5)
EFFICIENTNET_PRETRAINED = EfficientNet.from_pretrained(model_name='efficientnet-b0', num_classes=5)

def classification_net_select(name):
    if (name.lower() == 'resnet18'):
        return RESNET18
    elif (name.lower() == 'resnet50'):
        return RESNET50
    elif (name.lower() == "resnet18_pretrained"):
        return RESNET18_PRETRAINED
    elif (name.lower() == "resnet50_pretrained"):
        return RESNET50_PRETRAINED
    elif (name.lower() == "efficientnet"):
        return EFFICIENTNET
    elif (name.lower() == "efficientnet_pretrained"):
        return EFFICIENTNET_PRETRAINED
    
    else:
        sys.exit(f"ERROR:\t({__name__}):The Net: '{name}' doesn't exist")
        
        
##########################################################################################################
# Test
##########################################################################################################  
def test():
    pass
    x = torch.randn(8, 3, 128, 128)
    print(f"Input x:{x.shape}")
    output1 = RESNET18(x)
    output2 = RESNET50(x)
    output3 = RESNET18_PRETRAINED(x)
    output4 = RESNET50_PRETRAINED(x)
    output5 = EFFICIENTNET(x)
    output6 = EFFICIENTNET_PRETRAINED(x)

    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
    print(output4.shape)
    print(output5.shape)
    print(output6.shape)

if __name__ == "__main__":
    test()
