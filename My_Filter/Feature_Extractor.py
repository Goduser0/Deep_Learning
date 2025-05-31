import torch
from torch import nn
import torchvision
import torchvision.transforms as T

import warnings
warnings.filterwarnings('ignore')


OUT_FEATURES = 1024

def select_feature_extractor(fe_name):
    if fe_name.lower() == 'vgg19':
        VGG19_FE = torchvision.models.vgg19(pretrained=True)
        VGG19_FE.classifier[-1] = nn.Linear(4096, 5)
        
        checkpoint = torch.load("My_TAOD/Train_Classification/results/VGG19_FE/PCB_200 10-shot/models/60.pth")
        VGG19_FE.load_state_dict(checkpoint["model_state_dict"])
        
        VGG19_FE.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        VGG19_FE.classifier = nn.Sequential(
            # nn.Linear(in_features=512, out_features=OUT_FEATURES),
        )
        return VGG19_FE


if __name__ == '__main__':
    model = select_feature_extractor('vgg19')
    print(model)
    x = torch.randn(8, 3, 224, 224)
    y = model(x)
    print(y.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params:,}")  # 千位分隔符显示
    
    model = torchvision.models.vgg19(pretrained=True)
    print(model)
    x = torch.randn(8, 3, 224, 224)
    y = model(x)
    print(y.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params:,}")  # 千位分隔符显示