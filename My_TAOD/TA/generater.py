import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim

import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time

from TA_VAE import VAE
from TA_G import FeatureMatchGenerator


def traditional_aug():
    pass

def generate(z_dim, n, model_path, samples_save_path):
    local_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    
    z = torch.FloatTensor(np.random.normal(0, 1, (n, z_dim)))
    G = FeatureMatchGenerator(3, 128, 128, 64, 1e-2)
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint["model_state_dict"])
    imgs = G(z)
    
    model_path_parts = model_path.split("/")
    add_name = f"{model_path_parts[-2]}_{model_path_parts[-1][:-4]}"
    dirname = f"{add_name}_AT_{local_time}"
    
    img_label = model_path_parts[-2].split(" ")[1]
    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper', 'Missing_hole']
    img_class = img_classes[int(img_label)]
    
    os.makedirs(f"{samples_save_path}/{dirname}", exist_ok=False)
    
    img_save_list = []
    i = 0
    for img in imgs:
        i+=1
        img = img.detach().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img.transpose(1, 2, 0))
        
        filename = f"{i}.jpg"
        img_path = f"{samples_save_path}/{dirname}/{filename}"
        img.save(img_path)
        
        img_save_item = [img_label, img_class, img_path]
        img_save_list.append(img_save_item)
        
    # Save to csv    
    img_save_df = pd.DataFrame(img_save_list, columns=["Image_Label", "Image_Class", "Image_Path"])
    img_save_df.to_csv(f"{samples_save_path}/{dirname}/generate_imgs.csv")
    print("Generate Done!!!")    
    

# generate(128, 
#          10, 
#          "./My_TAOD/TA/models/source/PCB_200 0 2023-08-28_20:37:07/100_net_g_source.pth",
#          "./My_TAOD/TA/samples",
#          )

def img_translater(vae_common_params, vae_unique_params, g_params, img_path):
    
    data_iter_loader = get_loader(config.dataset_class, 
                              config.data_path, 
                              config.batch_size, 
                              config.num_workers, 
                              shuffle=True, 
                              trans=trans,
                              drop_last=True
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
    VAE_common = VAE(in_channels=3, latent_dim=64, input_size=128)
    VAE_unique = VAE(in_channels=3, latent_dim=64, input_size=128)
    G = FeatureMatchGenerator(n_mlp=3, img_size=128, z_dim=128, conv_dim=64, lr_mlp=1e-2)
    
    VAE_common.load_state_dict(torch.load(vae_common_params)["model_state_dict"])
    VAE_unique.load_state_dict(torch.load(vae_unique_params)["model_state_dict"])
    G.load_state_dict(torch.load(g_params)["model_state_dict"])
    
    VAE_common.eval()
    VAE_unique.eval()
    G.eval()
    
    img = img_path
    feat_common = VAE_common(img)
    feat_unique = VAE_unique(img)
    feat = torch.concat([feat_common, feat_unique], dim=1)
    output = G(feat)
    
img_translater
    
    
