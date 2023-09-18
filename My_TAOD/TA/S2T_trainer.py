import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn

import numpy as np

import os
import argparse
import random
import tqdm
import time

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_G import FeatureMatchGenerator
from TA_D import FeatureMatchDiscriminator
from TA_VAE import VAE
from TA_layers import KLDLoss
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import S2T_trainer_logger
from TA_utils import requires_grad, S_trainer_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument('--logs_dir', 
                    type=str, 
                    default='./My_TAOD/TA/logs/target')
parser.add_argument('--models_dir', 
                    type=str, 
                    default='./My_TAOD/TA/models/target')
parser.add_argument('--samples_dir', 
                    type=str, 
                    default='./My_TAOD/TA/samples')
parser.add_argument('--results_dir', 
                    type=str, 
                    default='./My_TAOD/TA/results/target')

# random seed
parser.add_argument("--random_seed", type=int, default=1)

# data loader
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--dataset_source",
                    type=str, 
                    default='PCB_200', 
                    choices=['PCB_Crop', 'PCB_200'])
parser.add_argument("--data_source_path", 
                    type=str, 
                    default="./My_TAOD/dataset/PCB_200/0.7-shot/train/0.csv")

parser.add_argument("--dataset_target",
                    type=str, 
                    default='PCB_Crop', 
                    choices=['PCB_Crop', 'PCB_200'])
parser.add_argument("--data_target_path", 
                    type=str, 
                    default="./My_TAOD/dataset/PCB_Crop/30-shot/train/0.csv")

parser.add_argument("--catagory",
                    type=str,
                    default="0")

# train
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--gpu_id", type=str, default="0")

# VAE
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--lr_vae", type=float, default=1e-4)

# vae_common
parser.add_argument("--vae_common_init_path",
                    type=str, 
                    default="./My_TAOD/TA/models/source/PCB_200 0 2023-08-28_20:37:07/100_net_VAE_common.pth")
# vae_unique
parser.add_argument("--vae_unique_init_path",
                    type=str, 
                    default="./My_TAOD/TA/models/source/PCB_200 0 2023-08-28_20:37:07/100_net_VAE_unique.pth")
# g_target
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--conv_dim", type=int, default=64)
parser.add_argument("--g_reg_every", type=int, default=4)
parser.add_argument("--lr_g", type=float, default=1e-4)

parser.add_argument("--n_mlp", type=int, default=3)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--lr_mlp", type=float, default=1e-2)

parser.add_argument("--g_target_init_path",
                    type=str, 
                    default="./My_TAOD/TA/models/source/PCB_200 0 2023-08-28_20:37:07/100_net_g_source.pth")

# Others
parser.add_argument("--log", type=bool, default=True)
parser.add_argument("--time", type=str, default=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

# config
config = parser.parse_args()

# 一致性检验
assert(
    config.dataset_source == config.data_source_path.split('/')[-4]
    and
    config.dataset_target == config.data_target_path.split('/')[-4]
    and
    config.catagory == config.data_source_path.split('/')[-1][0]
    and
    config.catagory == config.data_target_path.split('/')[-1][0]
    and
    config.vae_common_init_path.split('/')[-2] == config.g_target_init_path.split('/')[-2]
    and
    config.vae_unique_init_path.split('/')[-2] == config.g_target_init_path.split('/')[-2]
    and
    config.g_target_init_path.split('/')[-2].split(" ")[0] == config.dataset_source
    and
    config.g_target_init_path.split('/')[-2].split(" ")[1] == config.catagory
)
# logger
S2T_trainer_logger(config)

models_save_path = config.models_dir + '/' + config.dataset_target + ' ' + config.catagory + ' ' + config.time
os.makedirs(models_save_path, exist_ok=False)

######################################################################################################
#### Setting
######################################################################################################
# random seed
random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

# device
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

# source_loader
trans = T.Compose(
    [
        T.ToTensor(), 
        T.Resize((128, 128)), # (0, 255)
    ]
)

source_iter_loader = get_loader(config.dataset_source, 
                              config.data_source_path, 
                              config.batch_size, 
                              config.num_workers, 
                              shuffle=True, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
# target_loader
target_iter_loader = get_loader(config.dataset_target, 
                              config.data_target_path, 
                              config.batch_size, 
                              config.num_workers, 
                              shuffle=True, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True
                              ) # 像素值范围：（-1, 1）[B, C, H, W]

# model
VAE_common = VAE(
    in_channels=3, 
    latent_dim=config.latent_dim,
    input_size=config.img_size,
    ).cuda()
VAE_common.load_state_dict(torch.load(config.vae_common_init_path)["model_state_dict"])

VAE_unique = VAE(
    in_channels=3, 
    latent_dim=config.latent_dim,
    input_size=config.img_size,
    ).cuda()
VAE_unique.load_state_dict(torch.load(config.vae_unique_init_path)["model_state_dict"])

g_target = FeatureMatchGenerator(
    config.n_mlp, config.img_size, config.z_dim, config.conv_dim, config.lr_mlp
    ).cuda()
g_target.load_state_dict(torch.load(config.g_target_init_path)["model_state_dict"])

VAE_unique_optim = torch.optim.Adam(
    VAE_unique.parameters(), 
    lr=config.lr_vae,
    betas=[0.0, 0.9],
)

g_reg_ratio = config.g_reg_every / (config.g_reg_every + 1)
g_target_optim = torch.optim.Adam(
    g_target.parameters(),
    lr = config.lr_g * g_reg_ratio,
    betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
)

######################################################################################################
#### Train
######################################################################################################
kl_loss = KLDLoss(reduction='batchmean')

for epoch in tqdm.tqdm(range(1, config.num_epochs + 1), desc=f"On training"):
    VAE_unique.train()
    g_target.train()

    for i, target_data in enumerate(target_iter_loader):
        for j, source_data in enumerate(source_iter_loader):
            target_img = target_data[0].cuda()
            target_label = target_data[1].cuda()
            source_img = source_data[0].cuda()
            source_label = source_data[1].cuda()
            
            #-------------------------------------------------------------------
            # Train VAE_unique
            #-------------------------------------------------------------------
            VAE_unique_optim.zero_grad()
            
            results_common = VAE_common(source_img)
            results_unique = VAE_unique(target_img)
            [recon_img_common, _, mu_common, log_var_common] = [i for i in results_common]
            [recon_img_unique, _, mu_unique, log_var_unique] = [i for i in results_unique]    
            
            # 1.loss_kld_vae_unique
            loss_kld_vae_unique = VAE_common.loss_function(*results_unique, **{"recons_weight":0.0, "kld_weight": 1.0})["loss"]
            # Total Loss VAE_unique
            TotalLoss_vae_unique = loss_kld_vae_unique
            
            TotalLoss_vae_unique.backward()
            VAE_unique_optim.step()
            
            #-------------------------------------------------------------------
            # Train g_target
            #-------------------------------------------------------------------
            g_target_optim.zero_grad()
            
            results_common = VAE_common(source_img)
            results_unique = VAE_unique(target_img)
            [recon_img_common, _, mu_common, log_var_common] = [i for i in results_common]
            [recon_img_unique, _, mu_unique, log_var_unique] = [i for i in results_unique]  
            features_common = VAE_common.reparameterize(mu_common, log_var_common)
            features_unique= VAE_unique.reparameterize(mu_unique, log_var_unique)
            features = torch.concat([features_common, features_unique], dim=1)
            gen_img = g_target(features)
            
            results_unique_gen = VAE_unique(gen_img)
            [recon_img_unique_gen, _, mu_unique_gen, log_var_unique_gen] = [i for i in results_unique_gen]
            features_unique_gen = VAE_unique.reparameterize(mu_unique_gen, log_var_unique_gen)
            
            # 1.loss_featrecon_unique
            loss_featrecon_unique = kl_loss(features_unique, features_unique_gen)
            # Total Loss g_target
            TotalLoss_g_target = loss_featrecon_unique
            
            TotalLoss_g_target.backward()
            g_target_optim.step()
            
            # Show
            print(
                "[Epoch %d/%d] [Batch_Target %d/%d] [Batch_Source %d/%d] [TotalLoss_vae_unique: %f] [TotalLoss_g_target: %f]"
                %
                (epoch, config.num_epochs, i+1, len(target_iter_loader), j, len(source_iter_loader), TotalLoss_vae_unique, TotalLoss_g_target)
            )
            
            # Record Data
            S2T_trainer_record_data(config, 
                        {
                            "epoch": f"{epoch}",
                            "num_epochs": f"{config.num_epochs}",
                            "batch_target": f"{i+1}",
                            "batch_source": f"{j+1}",
                            "num_batchs_target": f"{len(target_iter_loader)}",
                            "num_batchs_source": f"{len(source_iter_loader)}",
                            "TotalLoss_vae_unique": f"{TotalLoss_vae_unique.item()}",
                            "TotalLoss_g_target": f"{TotalLoss_g_target.item()}", 
                        },
                        flag_plot=True,
                        )
    
    ##################################################################################
    ## Checkpoint
    ##################################################################################
    #-------------------------------------------------------------------
    # Save model G & VAE
    #-------------------------------------------------------------------
    if epoch % (config.num_epochs // 25) == 0:
        net_g_target_path = models_save_path + '/%d_net_g_target.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": g_target.state_dict(),
            "z_dim": config.z_dim,
        }, net_g_target_path)
        
        net_VAE_unique_path = models_save_path + '/%d_net_VAE_unique.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": VAE_unique.state_dict(),
            "z_dim": config.z_dim,
        }, net_VAE_unique_path)
        