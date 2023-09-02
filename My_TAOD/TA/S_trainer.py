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

from TA_G import FeatureMatchGenerator
from TA_D import FeatureMatchDiscriminator
from TA_VAE import VAE
from TA_utils import requires_grad, S_trainer_record_data
from TA_layers import KLDLoss
from TA_logger import S_trainer_logger

import sys
sys.path.append("./My_TAOD")
from dataset_loader import get_loader

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument('--logs_dir', 
                    type=str, 
                    default='./My_TAOD/TA/logs/source')
parser.add_argument('--models_dir', 
                    type=str, 
                    default='./My_TAOD/TA/models/source')
parser.add_argument('--samples_dir', 
                    type=str, 
                    default='./My_TAOD/TA/samples')
parser.add_argument('--results_dir', 
                    type=str, 
                    default='./My_TAOD/TA/results/source')

# random seed
parser.add_argument("--random_seed", type=int, default=1)

# data loader
parser.add_argument("--dataset_class",
                    type=str, 
                    default='PCB_200', 
                    choices=['PCB_Crop', 'PCB_200'])
parser.add_argument("--data_path", 
                    type=str, 
                    default="./My_TAOD/dataset/PCB_200/0.7-shot/train/0.csv")
parser.add_argument("--catagory",
                    type=str,
                    default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)

# train
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--gpu_id", type=str, default="0")

# g_source && d_source
parser.add_argument("--gan_type", type=str, default="")
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--conv_dim", type=int, default=64)

# g_source
parser.add_argument("--g_reg_every", type=int, default=4)
parser.add_argument("--lr_g", type=float, default=1e-4)

parser.add_argument("--n_mlp", type=int, default=3)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--lr_mlp", type=float, default=1e-2)

# d_source
parser.add_argument("--d_reg_every", type=int, default=4)
parser.add_argument("--lr_d", type=float, default=1e-5)

# VAE
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--lr_vae", type=float, default=1e-4)

# Others
parser.add_argument("--log", type=bool, default=True)
parser.add_argument("--time", type=str, default=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

# config
config = parser.parse_args()

# 一致性校验
assert(config.data_path.split('/')[-4]==config.dataset_class 
       and 
       config.data_path.split('/')[-1][0]==config.catagory)

# logger
S_trainer_logger(config)

models_save_path = config.models_dir + '/' + config.dataset_class + ' ' + config.catagory + ' ' + config.time
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

# data_loader
trans = T.Compose(
    [
        T.ToTensor(), 
        T.Resize((128, 128)), # (0, 255)
    ]
)
data_iter_loader = get_loader(config.dataset_class, 
                              config.data_path, 
                              config.batch_size, 
                              config.num_workers, 
                              shuffle=True, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True
                              ) # 像素值范围：（-1, 1）[B, C, H, W]

# model
g_source = FeatureMatchGenerator(
    config.n_mlp, config.img_size, config.z_dim, config.conv_dim, config.lr_mlp
).cuda()
d_source = FeatureMatchDiscriminator(
    config.img_size, config.conv_dim
).cuda()

VAE_common = VAE(
    in_channels=3, 
    latent_dim=config.latent_dim,
    input_size=config.img_size,
    ).cuda()
VAE_unique = VAE(
    in_channels=3, 
    latent_dim=config.latent_dim,
    input_size=config.img_size,
    ).cuda()


# optimizers
g_reg_ratio = config.g_reg_every / (config.g_reg_every + 1)
d_reg_ratio = config.d_reg_every / (config.d_reg_every + 1)

g_optim = torch.optim.Adam(
    g_source.parameters(),
    lr = config.lr_g * g_reg_ratio,
    betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
)
d_optim = torch.optim.Adam(
    d_source.parameters(),
    lr=config.lr_d * d_reg_ratio,
    betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
)

VAE_common_optim = torch.optim.Adam(
    VAE_common.parameters(), 
    lr=config.lr_vae,
    betas=[0.0, 0.9],
)
VAE_unique_optim = torch.optim.Adam(
    VAE_unique.parameters(), 
    lr=config.lr_vae,
    betas=[0.0, 0.9],
)

######################################################################################################
#### Train
######################################################################################################
kl_loss = KLDLoss(reduction='batchmean')
cos_sim = nn.CosineSimilarity()

for epoch in tqdm.tqdm(range(1, config.num_epochs + 1), desc=f"On training"):
    VAE_common.train()
    VAE_unique.train()
    g_source.train()
    d_source.train()
    
    for i, data in enumerate(data_iter_loader):
        raw_img = data[0].cuda() # [B, C, H, W] -1~1
        category_label = data[1].cuda() # [B]
        
        ##################################################################################
        ## Decoupling Branch
        ##################################################################################
        #-------------------------------------------------------------------
        # Train VAE
        #-------------------------------------------------------------------
        VAE_common_optim.zero_grad()
        VAE_unique_optim.zero_grad()
        
        # 原始图像解耦
        results_common_raw = VAE_common(raw_img)
        results_unique_raw = VAE_unique(raw_img)
        [recon_img_common_raw, _, mu_common_raw, log_var_common_raw] = [i for i in results_common_raw] #[B, C, H, W] [B, 64] [B, 64]
        [recon_img_unique_raw, _, mu_unique_raw, log_var_unique_raw] = [i for i in results_unique_raw] #[B, C, H, W] [B, 64] [B, 64]    
        features_common_raw = VAE_common.reparameterize(mu_common_raw, log_var_common_raw) # [B, 64]
        features_unique_raw = VAE_unique.reparameterize(mu_unique_raw, log_var_unique_raw) # [B, 64]
        features_raw = torch.concat([features_common_raw, features_unique_raw], dim=1) # [B, 128]
        
        # 图像生成
        gen_img = g_source(features_raw) # [B, 3, 128, 128]
        
        # 生成图像解耦
        VAE_common_optim.zero_grad()
        VAE_unique_optim.zero_grad()
        
        results_common_gen = VAE_common(gen_img.detach()) #[B, C, H, W] [B, 64] [B, 64]
        results_unique_gen = VAE_unique(gen_img.detach()) #[B, C, H, W] [B, 64] [B, 64]
        [recon_img_common_gen, _, mu_common_gen, log_var_common_gen] = [i for i in results_common_gen]
        [recon_img_unique_gen, _, mu_unique_gen, log_var_unique_gen] = [i for i in results_common_gen]

        # 1.loss_kld_vae_raw
        loss_kld_vae_common_raw = VAE_common.loss_function(*results_common_raw, **{"recons_weight":0.0, "kld_weight": 1.0})["loss"]
        loss_kld_vae_unique_raw = VAE_unique.loss_function(*results_unique_raw, **{"recons_weight":0.0, "kld_weight": 1.0})["loss"]
        loss_kld_vae_raw = 0.5 * (loss_kld_vae_common_raw + loss_kld_vae_unique_raw)
        # 2.loss_kld_vae_gen
        loss_kld_vae_common_gen = VAE_common.loss_function(*results_common_gen, **{"recons_weight":0.0, "kld_weight": 1.0})["loss"]
        loss_kld_vae_unique_gen = VAE_unique.loss_function(*results_unique_gen, **{"recons_weight":0.0, "kld_weight": 1.0})["loss"]
        loss_kld_vae_gen = 0.5 * (loss_kld_vae_common_gen + loss_kld_vae_unique_gen)
        # 3.loss_imgrecon
        loss_imgrecon = F.mse_loss(raw_img, gen_img)
        # Total Loss VAE
        TotalLoss_vae = loss_kld_vae_raw + loss_kld_vae_gen + 10.0*loss_imgrecon
        
        TotalLoss_vae.backward()
        VAE_common_optim.step()
        VAE_unique_optim.step()
        
        #-------------------------------------------------------------------
        # Train G
        #-------------------------------------------------------------------
        g_optim.zero_grad()
        
        results_common_raw = VAE_common(raw_img)
        results_unique_raw = VAE_unique(raw_img)
        [recon_img_common_raw, _, mu_common_raw, log_var_common_raw] = [i for i in results_common_raw] #[B, C, H, W] [B, 64] [B, 64]
        [recon_img_unique_raw, _, mu_unique_raw, log_var_unique_raw] = [i for i in results_unique_raw] #[B, C, H, W] [B, 64] [B, 64]    
        features_common_raw = VAE_common.reparameterize(mu_common_raw, log_var_common_raw) # [B, 64]
        features_unique_raw = VAE_unique.reparameterize(mu_unique_raw, log_var_unique_raw) # [B, 64]
        features_raw = torch.concat([features_common_raw, features_unique_raw], dim=1) # [B, 128]
        gen_img = g_source(features_raw) # [B, 3, 128, 128]
        results_common_gen = VAE_common(gen_img) #[B, C, H, W] [B, 64] [B, 64]
        results_unique_gen = VAE_unique(gen_img) #[B, C, H, W] [B, 64] [B, 64]
        [recon_img_common_gen, _, mu_common_gen, log_var_common_gen] = [i for i in results_common_gen]
        [recon_img_unique_gen, _, mu_unique_gen, log_var_unique_gen] = [i for i in results_common_gen]
        features_common_gen = VAE_common.reparameterize(mu_common_gen, log_var_common_gen) # [B, 64]
        features_unique_gen = VAE_unique.reparameterize(mu_unique_gen, log_var_unique_gen) # [B, 64]
        
        # 1.loss_featurerecon_common
        loss_featurerecon_common = kl_loss(features_common_raw, features_common_gen)
        # 2.loss_featurerecon_unique
        loss_featurerecon_unique = kl_loss(features_unique_raw, features_unique_gen)
        # 3.loss_imgrecon
        loss_imgrecon = F.mse_loss(raw_img, gen_img)
        # Total Loss G
        TotalLoss_g = loss_featurerecon_common + loss_featurerecon_unique + loss_imgrecon
        
        TotalLoss_g.backward()
        g_optim.step()
        
        ##################################################################################
        ## Discriminating Branch
        ##################################################################################
        #-------------------------------------------------------------------
        # Train G
        #-------------------------------------------------------------------
        g_optim.zero_grad()
        
        # Generate fake images
        D_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (raw_img.shape[0], 64))), requires_grad=False).cuda() # [B, 64]
        D_features = torch.concat([D_z, features_unique_raw.detach()], dim=1) # [B, 128]
        D_gen_img = g_source(D_features) # [B, 3, 128, 128]
        
        # Calculate G loss_adv
        D_raw_img_output = d_source(raw_img) # [B, 64, 64, 64] [B, 128, 32, 32] [B, 256, 16, 16] [B, 512, 8, 8] [B, 1] [B, 64]
        D_gen_img_output = d_source(D_gen_img) # [B, 64, 64, 64] [B, 128, 32, 32] [B, 256, 16, 16] [B, 512, 8, 8] [B, 1] [B, 64]
        pred_raw_img = D_raw_img_output[4] # [B, 1]
        pred_gen_img = D_gen_img_output[4] # [B, 1]
        labels_raw_img = torch.ones_like(pred_raw_img).cuda() # [B, 1]
        labels_gen_img = torch.zeros_like(pred_gen_img).cuda() # [B, 1]
        
        # 1.loss_PPL
        
        
        # 2.loss_FM
        criterionFeat = nn.L1Loss()
        loss_FM = torch.cuda.FloatTensor(1).fill_(0)
        num_features_maps = 4
        weights_features_maps = [1., 1., 1., 1.]
        for i_map in range(num_features_maps):
            i_loss_FM = criterionFeat(D_gen_img_output[i_map], D_raw_img_output[i_map])
            loss_FM += i_loss_FM * weights_features_maps[i_map]
        
        # 3.loss_adv
        loss_adv_raw_img = nn.BCEWithLogitsLoss()(pred_raw_img, labels_raw_img)
        loss_adv_gen_img = nn.BCEWithLogitsLoss()(pred_gen_img, labels_gen_img)
        loss_adv = 0.5 * (loss_adv_raw_img + loss_adv_gen_img)
        
        # D Total Loss G
        D_TotalLoss_g = 10.0 * loss_FM + loss_adv
        
        D_TotalLoss_g.backward()
        g_optim.step()
        
        #-------------------------------------------------------------------
        # Train D
        #-------------------------------------------------------------------
        d_optim.zero_grad()
        
        # Calculate D loss_adv
        D_gen_img = g_source(D_features)
        D_raw_img_output = d_source(raw_img) # [B, 64, 64, 64] [B, 128, 32, 32] [B, 256, 16, 16] [B, 512, 8, 8] [B, 1] [B, 64]
        D_gen_img_output = d_source(D_gen_img)
        pred_raw_img = D_raw_img_output[4] # [B, 1]
        pred_gen_img = D_gen_img_output[4] # [B, 1]
        labels_raw_img = torch.ones_like(pred_raw_img).cuda() # [B, 1]
        labels_gen_img = torch.zeros_like(pred_gen_img).cuda() # [B, 1]
        
        loss_adv_raw_img = nn.BCEWithLogitsLoss()(pred_raw_img, labels_raw_img)
        loss_adv_gen_img = nn.BCEWithLogitsLoss()(pred_gen_img, labels_gen_img)
        loss_adv = 0.5 * (loss_adv_raw_img + loss_adv_gen_img)
        D_TotalLoss_d = loss_adv
        
        D_TotalLoss_d.backward()
        d_optim.step()
        
        # Show
        print(
            "[Epoch %d/%d] [Batch %d/%d] [TotalLoss_vae: %f] [TotalLoss_g: %f] [D_TotalLoss_g: %f] [D_TotalLoss_d: %f]"
            %
            (epoch, config.num_epochs, i, len(data_iter_loader), TotalLoss_vae, TotalLoss_g, D_TotalLoss_g, D_TotalLoss_d)
        )
        
        # Record Data
        S_trainer_record_data(config, 
                    {
                        "epoch": f"{epoch}",
                        "num_epochs": f"{config.num_epochs}",
                        "batch": f"{i+1}",
                        "num_batchs": f"{len(data_iter_loader)}",
                        "TotalLoss_vae": f"{TotalLoss_vae.item()}",
                        "TotalLoss_g": f"{TotalLoss_g.item()}", 
                        "D_TotalLoss_g": f"{D_TotalLoss_g.item()}",                   
                        "D_TotalLoss_d": f"{D_TotalLoss_d.item()}",
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
        net_g_source_path = models_save_path + '/%d_net_g_source.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": g_source.state_dict(),
            "z_dim": config.z_dim,
        }, net_g_source_path)
        
        net_VAE_common_path = models_save_path + '/%d_net_VAE_common.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": VAE_common.state_dict(),
            "z_dim": config.z_dim,
        }, net_VAE_common_path)
        
        net_VAE_unique_path = models_save_path + '/%d_net_VAE_unique.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": VAE_unique.state_dict(),
            "z_dim": config.z_dim,
        }, net_VAE_unique_path)
    #-------------------------------------------------------------------
    # Save model D
    #-------------------------------------------------------------------
    if epoch == config.num_epochs:
        net_d_source_path = models_save_path + '/%d_net_d_source.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": d_source.state_dict(),
            "z_dim": config.z_dim,
        }, net_d_source_path)
        