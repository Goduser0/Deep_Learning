import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import linalg
import os
import cv2
import random

import torchvision.models as models
import torchvision.transforms as T

import sys
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_utils import load_img_for_fid

class ConvNetFeatureExtract(object):
    def __init__(self, model='resnet34', workers=4):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''
        self.model = model
        self.workers = workers
        if self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).cuda().eval()
            self.trans = T.Compose([
                T.Resize(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True)
            resnet.cuda().eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).cuda().eval()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
            self.trans = T.Compose([
                T.Resize(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained=True, transform_input=False).cuda().eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              ).cuda().eval()
            self.inception = inception
            self.inception_feature = inception_feature
            self.trans = T.Compose([
                T.Resize((299, 299)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError

    def extractFeature(self, images_path, num):
        # build images dataset
        images = load_img_for_fid(images_path, trans=self.trans, batch_size=num)
        print(images.shape)
        print("Extracting Features...")
        with torch.no_grad():
            input = images.cuda()
            if self.model == 'vgg' or self.model == 'vgg16':
                fconv = self.vgg.features(input).view(input.size(0), -1)
                print(self.model + " feature shape:", fconv.shape)
            elif self.model.find('resnet') >= 0:
                fconv = self.resnet_feature(input).mean(3).mean(2).squeeze()
                print(self.model + " feature shape:", fconv.shape)
            elif self.model == 'inception' or self.model == 'inception_v3':
                fconv = self.inception_feature(input).mean(3).mean(2).squeeze()
                print(self.model + " feature shape:", fconv.shape)
            else:
                raise NotImplementedError
            feature_conv = fconv.data.cpu()
        return feature_conv
    

def calculator_FID(source, target):
    m = source.mean(0)
    m_w = target.mean(0)
    X_np = source.numpy()
    Y_np = target.numpy()
    
    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real
    
    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + np.trace(C + C_w - 2 * C_C_w_sqrt)
        
    return score

def score_fid(real_path, fake_path, num_samples, gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    convnet_feature_extract = ConvNetFeatureExtract(model="inception_v3", workers=4)
    real_feature = convnet_feature_extract.extractFeature(real_path, num_samples)
    fake_feature = convnet_feature_extract.extractFeature(fake_path, num_samples)
    result = calculator_FID(real_feature, fake_feature)
    return result    
    
def test():  
    real_path = "My_TAOD/dataset/DeepPCB_Crop/30-shot/test/0.csv"
    fake_path = "My_TAOD/dataset/DeepPCB_Crop/30-shot/test/1.csv"
    result = score_fid(real_path, fake_path, num_samples=100)
    print(f"FID: {result}")
    
if __name__ == '__main__':
    test()

