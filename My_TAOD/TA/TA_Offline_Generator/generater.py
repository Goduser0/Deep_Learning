import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim
from torch.autograd import Variable

import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader, img_1to255, img_255to1
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_VAE import VariationalAutoEncoder
from TA_G import FeatureMatchGenerator, PFS_Generator


def traditional_aug():
    pass

def generate(z_dim, n, model, model_path, samples_save_path, domain):
    """加载生成器，随机生成"""
    local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    z = torch.FloatTensor(np.random.normal(0, 1, (n, z_dim)))

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if domain:
        imgs = model(z, domain)
    else:
        imgs = model(z)
    
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
    return f"{samples_save_path}/{dirname}/generate_imgs.csv"

# generate(128, 
#          32, 
#          "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/models/source/PCB_200 0 2023-09-04_11:18:31/500_net_g_source.pth",
#          "./My_TAOD/TA/samples",
#          )

def img_translater(vae_common_params, vae_unique_params, g_params, trans):
    """加载图像"""
    data_iter_loader = get_loader('PCB_200', 
                              "./My_TAOD/dataset/PCB_200/0.7-shot/train/0.csv", 
                              1,
                              4, 
                              shuffle=False, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True,
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
    VAE_common = VariationalAutoEncoder(in_channels=3, latent_dim=64, input_size=128)
    VAE_unique = VariationalAutoEncoder(in_channels=3, latent_dim=64, input_size=128)
    G = FeatureMatchGenerator(n_mlp=3, img_size=128, z_dim=128, conv_dim=64, lr_mlp=1e-2)
    
    VAE_common.load_state_dict(torch.load(vae_common_params)["model_state_dict"])
    VAE_unique.load_state_dict(torch.load(vae_unique_params)["model_state_dict"])
    G.load_state_dict(torch.load(g_params)["model_state_dict"])
    
    VAE_common.eval()
    VAE_unique.eval()
    G.eval()
    
    for i, data in enumerate(data_iter_loader):
        if i <= 10:
            img = data[0]
            
            results_common = VAE_common(img)
            results_unique = VAE_unique(img)
            [recon_img_common, _, mu_common, log_var_common] = [i for i in results_common]
            [recon_img_unique, _, mu_unique, log_var_unique] = [i for i in results_unique]    
            features_common = VAE_common.reparameterize(mu_common, log_var_common)
            features_unique = VAE_unique.reparameterize(mu_unique, log_var_unique)
            features = torch.concat([features_common, features_unique], dim=1)
            # add_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (1, 32))), requires_grad=False) # [1, 32]
            # features = torch.concat([features, add_z], dim=1)
            output = G(features)
            
            raw_img = img[0]
            raw_img = img_1to255(raw_img.numpy()).transpose(1, 2, 0)
            plt.imshow(raw_img)
            plt.savefig(f"test_raw_{i}.jpg")
            plt.close()
            
            gen_img = output[0]
            gen_img = img_1to255(gen_img.detach().numpy()).transpose(1, 2, 0)
            plt.imshow(gen_img)
            plt.savefig(f"test_gen_{i}.jpg")
            plt.close()
            
            # break        

def test():
    # img_translater(
    #     "./My_TAOD/TA/models/source/PCB_200 0 2023-09-13_12:02:21/500_net_VAE_common.pth",
    #     "./My_TAOD/TA/models/source/PCB_200 0 2023-09-13_12:02:21/500_net_VAE_unique.pth",
    #     "./My_TAOD/TA/models/source/PCB_200 0 2023-09-13_12:02:21/500_net_g_source.pth",
    #     trans=T.Compose([T.ToTensor(), T.Resize((128, 128))]),
    # )
    G = PFS_Generator(z_dim=128)
    model_path = "My_TAOD/TA/TA_Results/baseline_from_scratch/models/DeepPCB_Crop 0 2023-10-31_14-29-13/500_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/baseline_from_scratch/samples"

    generate(128, 200, G, model_path, img_save_path)
        
if __name__ == "__main__":
    test()