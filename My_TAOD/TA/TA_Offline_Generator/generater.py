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
from dataset_loader import get_loader, get_loader_ST, img_1to255, img_255to1
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_VAE import Encoder
from TA_G import FeatureMatchGenerator, PFS_Generator


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

def translate(n, Src_class, model_S2T, model_S2T_path, dataset_S_path, samples_save_path, batch_size):
    """加载图像翻译器, 加载一个domain的图像生成另外一个domain的图像"""
    local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    model_path_parts = model_S2T_path.split("/")
    add_name = f"{model_path_parts[-2]}_{model_path_parts[-1][:-4]}"
    dirname = f"{add_name}_AT_{local_time}"
    
    img_label = model_path_parts[-2].split(" ")[1]
    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper', 'Missing_hole']
    img_class = img_classes[int(img_label)]
    
    os.makedirs(f"{samples_save_path}/{dirname}", exist_ok=False)
    
    netG_S2T = model_S2T
    
    netG_S2T.load_state_dict(torch.load(model_S2T_path)["model_state_dict"])
    
    # Inputs & targets memory allocation
    Tensor = torch.Tensor
    input_A = Tensor(batch_size, 3, 128, 128)

    # Dataset loader
    trans = T.Compose([
        T.ToTensor(),
        T.Resize((128, 128)),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    
    data_iter_loader = get_loader(Src_class, dataset_S_path, batch_size, num_workers=4, shuffle=True, trans=trans, drop_last=True)
    
    
    img_save_list = []
    i = 0
    
    for i, batch in enumerate(data_iter_loader):
        print(batch[0].shape)
        real_Src = Variable(input_A.copy_(batch[0]))
        # Generate output
        fake_B = 0.5*(netG_S2T(real_Src).data + 1.0)
    
        for img in fake_B:
            i+=1
            img = img.detach().numpy()
            img = ((img + 1) / 2 * 255).astype(np.uint8)
            img = Image.fromarray(img.transpose(1, 2, 0))
            
            filename = f"{i}.jpg"
            img_path = f"{samples_save_path}/{dirname}/{filename}"
            img.save(img_path)
            
            img_save_item = [img_label, img_class, img_path]
            img_save_list.append(img_save_item)
            
            if len(img_save_list) >= n:
                # Save to csv    
                img_save_df = pd.DataFrame(img_save_list, columns=["Image_Label", "Image_Class", "Image_Path"])
                img_save_df.to_csv(f"{samples_save_path}/{dirname}/generate_imgs.csv")
                print("Generate Done!!!")    
                return f"{samples_save_path}/{dirname}/generate_imgs.csv"
        
    
      
def img_translater(G, vae_com_params, vae_uni_params, g_params, results_save_dir, trans):
    data_iter_loader = get_loader('PCB_Crop', 
                              "./My_TAOD/dataset/PCB_Crop/30-shot/test/0.csv", 
                              1,
                              4, 
                              shuffle=False, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True,
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
    VAE_com = Encoder(in_channels=3, latent_dim=64, input_size=128)
    VAE_uni = Encoder(in_channels=3, latent_dim=64, input_size=128)
    
    VAE_com.load_state_dict(torch.load(vae_com_params)["model_state_dict"])
    VAE_uni.load_state_dict(torch.load(vae_uni_params)["model_state_dict"])
    G.load_state_dict(torch.load(g_params)["model_state_dict"])
    
    VAE_com.eval()
    VAE_uni.eval()
    G.eval()
    
    for i, data in enumerate(data_iter_loader):
        if i <= 10:
            raw_img = data[0]
            
            mu_com, log_var_com, feat_com = VAE_com(raw_img)
            mu_uni, log_var_uni, feat_uni = VAE_uni(raw_img) 
            feat_recon = torch.concat([feat_com, feat_uni], dim=1)
            # add_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (1, 64))), requires_grad=False) # [1, 64]
            # feat_gen = torch.concat([features, add_z], dim=1)
            recon_img = G(feat_recon)
            
            fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=80)
            raw_img = raw_img[0]
            raw_img = img_1to255(raw_img.numpy()).transpose(1, 2, 0)
            axes[0].imshow(raw_img)
            axes[0].set_title("raw_img")
            
            recon_img = recon_img[0]
            recon_img = img_1to255(recon_img.detach().numpy()).transpose(1, 2, 0)
            axes[1].imshow(recon_img)
            axes[1].set_title("recon_img")

            plt.savefig(f"{results_save_dir}/test_recon_{i}.jpg")
            plt.close()        

def test():    
    # G = PFS_Generator(z_dim=128)
    # G_path = "My_TAOD/TA/TA_Results/PFS_baseline_from_scratch/models/DeepPCB_Crop 0 2023-10-31_14-29-13/1000_net_g.pth"
    # img_save_path = "My_TAOD/TA/TA_Results/PFS_baseline_from_scratch/samples"
    
    G = FeatureMatchGenerator(n_mlp=3)
    G_path = "My_TAOD/TA/TA_Results/S/models/PCB_Crop 0 2023-11-23_00-26-57/1000_net_g_source.pth"
    vae_com_path = "My_TAOD/TA/TA_Results/S/models/PCB_Crop 0 2023-11-23_00-26-57/1000_net_VAE_common.pth"
    vae_uni_path = "My_TAOD/TA/TA_Results/S/models/PCB_Crop 0 2023-11-23_00-26-57/1000_net_VAE_unique.pth"
    img_save_path = "My_TAOD/TA/TA_Results/S/samples"
    
    # generate(128, 200, G, G_path, img_save_path, domain=None)
    
    img_translater(
        G,
        vae_com_path,
        vae_uni_path,
        G_path,
        img_save_path,
        trans=T.Compose([T.ToTensor(), T.Resize((128, 128))]),
    )
if __name__ == "__main__":
    test()