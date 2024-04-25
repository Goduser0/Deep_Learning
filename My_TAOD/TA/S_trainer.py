import os
import argparse
import random
import tqdm
import time
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_G import FeatureMatchGenerator
from TA_D import FeatureMatchDiscriminator
from TA_VAE import Encoder
from TA_layers import KLDLoss, PerceptualLoss
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import S_trainer_logger
from TA_utils import requires_grad, S_trainer_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument('--logs_dir', 
                    type=str, 
                    default='./My_TAOD/TA/TA_Results/S/logs')
parser.add_argument('--models_dir', 
                    type=str, 
                    default='./My_TAOD/TA/TA_Results/S/models')
parser.add_argument('--samples_dir', 
                    type=str, 
                    default='./My_TAOD/TA/TA_Results/S/samples')
parser.add_argument('--results_dir', 
                    type=str, 
                    default='./My_TAOD/TA/TA_Results/S/results')

# random seed
parser.add_argument("--random_seed", type=int, default=12)

# data loader
parser.add_argument("--dataset_class",
                    type=str, 
                    default='PCB_Crop', 
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'])
parser.add_argument("--data_path", 
                    type=str, 
                    default="./My_TAOD/dataset/PCB_Crop/160-shot/train/0.csv")
parser.add_argument("--category",
                    type=str,
                    default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)

# train
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--gpu_id", type=str, default="1")

# g_source && d_source
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--conv_dim", type=int, default=64)

# g_source
parser.add_argument("--g_reg_every", type=float, default=4.0)
parser.add_argument("--lr_g", type=float, default=4e-5)

parser.add_argument("--n_mlp", type=int, default=3)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--lr_mlp", type=float, default=0.001)

# d_source
parser.add_argument("--d_reg_every", type=float, default=4.0)
parser.add_argument("--lr_d", type=float, default=1e-5)

# VAE
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--lr_vae", type=float, default=2e-5)

# Others
parser.add_argument("--log", type=bool, default=True)
parser.add_argument("--time", type=str, default=time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime()))

# config
config = parser.parse_args()

# 一致性校验
assert(config.data_path.split('/')[-4] == config.dataset_class 
       and 
       config.data_path.split('/')[-1][0] == config.category)

# logger
S_trainer_logger(config)

models_save_path = config.models_dir + '/' + config.dataset_class + ' ' + config.category + ' ' + config.time
os.makedirs(models_save_path, exist_ok=False)

######################################################################################################
#### Setting
######################################################################################################
# random seed
random.seed(config.random_seed)
# torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

# device
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

# data_loader
trans = T.Compose(
    [
        T.ToTensor(), 
        T.Resize((config.img_size, config.img_size)), # (0, 255)
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
g_src = FeatureMatchGenerator(
    config.n_mlp, config.img_size, config.z_dim, config.conv_dim, config.lr_mlp
).cuda()
d_src = FeatureMatchDiscriminator(
    config.img_size, config.conv_dim
).cuda()

VAE_com = Encoder(
    in_channels=3, 
    latent_dim=config.latent_dim,
    input_size=config.img_size,
    ).cuda()
VAE_uni = Encoder(
    in_channels=3, 
    latent_dim=config.latent_dim,
    input_size=config.img_size,
    ).cuda()

# optimizers
g_reg_ratio = config.g_reg_every / (config.g_reg_every + 1)
d_reg_ratio = config.d_reg_every / (config.d_reg_every + 1)

g_optim = torch.optim.Adam(
    g_src.parameters(),
    lr = config.lr_g * g_reg_ratio,
    betas=(0.0 ** g_reg_ratio, 0.9 ** g_reg_ratio),
)
d_optim = torch.optim.Adam(
    d_src.parameters(),
    lr=config.lr_d * d_reg_ratio,
    betas=(0.0 ** d_reg_ratio, 0.9 ** d_reg_ratio),
)

VAE_com_optim = torch.optim.Adam(
    VAE_com.parameters(), 
    lr=config.lr_vae,
    betas=[0.0, 0.9],
)
VAE_uni_optim = torch.optim.Adam(
    VAE_uni.parameters(), 
    lr=config.lr_vae,
    betas=[0.0, 0.9],
)

######################################################################################################
#### Train
######################################################################################################
for epoch in tqdm.tqdm(range(1, config.num_epochs + 1)):
    VAE_com.train()
    VAE_uni.train()
    g_src.train()
    d_src.train()
    
    loss_list = []
    
    for i, data in enumerate(data_iter_loader):
        raw_img = data[0].cuda() # [B, C, H, W] -1~1
        category_label = data[1].cuda() # [B,]
        num = raw_img.size(0)
        ##################################################################################
        ## Decoupling Branch
        ##################################################################################
        #-------------------------------------------------------------------
        # Train VAE&G
        #-------------------------------------------------------------------
        VAE_com_optim.zero_grad()
        VAE_uni_optim.zero_grad()
        
        # 原始图像解耦
        mu_com_raw, logvar_com_raw, feat_com_raw = VAE_com(raw_img) #[B, 64] [B, 64] [B, 64]
        mu_uni_raw, logvar_uni_raw, feat_uni_raw = VAE_uni(raw_img) #[B, 64] [B, 64] [B, 64]
        feat_raw = torch.concat([feat_com_raw, feat_uni_raw], dim=1) # [B, 128]
        # 图像重建
        recon_img = g_src(feat_raw) # [B, 3, 128, 128]
        # 重建图像解耦
        mu_com_recon, logvar_com_recon, _ = VAE_com(recon_img) #[B, 64] [B, 64] _
        mu_uni_recon, logvar_uni_recon, _ = VAE_uni(recon_img) #[B, 64] [B, 64] _

        # 1.loss_kld_vae_com
        loss_kld_vae_com_raw = KLDLoss()(mu_com_raw, logvar_com_raw)
        loss_kld_vae_com_recon = KLDLoss()(mu_com_recon, logvar_com_recon)
        loss_kld_com_recon = KLDLoss()(mu_com_raw, logvar_com_raw, mu_com_recon, logvar_com_recon)
        loss_kld_vae_com = loss_kld_vae_com_raw + loss_kld_vae_com_recon + loss_kld_com_recon
        # 2.loss_kld_vae_uni
        loss_kld_vae_uni_raw = KLDLoss()(mu_uni_raw, logvar_uni_raw)
        loss_kld_vae_uni_recon = KLDLoss()(mu_uni_recon, logvar_uni_recon)
        loss_kld_uni_recon = KLDLoss()(mu_uni_raw, logvar_uni_raw, mu_uni_recon, logvar_uni_recon)
        loss_kld_vae_uni = loss_kld_vae_uni_raw + loss_kld_vae_uni_recon + loss_kld_uni_recon
        # 3.loss_imgrecon
        loss_imgrecon = torch.mean((recon_img-raw_img)**2) + torch.mean(torch.abs(recon_img-raw_img))
        # TotalLoss_vae
        TotalLoss_vae = loss_kld_vae_com + loss_kld_vae_uni + loss_imgrecon
        
        TotalLoss_vae.backward()
        VAE_com_optim.step()
        VAE_uni_optim.step()
        
        #-------------------------------------------------------------------
        # Train G
        #-------------------------------------------------------------------
        g_optim.zero_grad()
        
        # 原始图像解耦
        mu_com_raw, logvar_com_raw, feat_com_raw = VAE_com(raw_img) #[B, 64] [B, 64] [B, 64]
        mu_uni_raw, logvar_uni_raw, feat_uni_raw = VAE_uni(raw_img) #[B, 64] [B, 64] [B, 64]
        feat_raw = torch.concat([feat_com_raw, feat_uni_raw], dim=1) # [B, 128]
        # 图像重建
        recon_img = g_src(feat_raw) # [B, 3, 128, 128]
        # 重建图像解耦
        mu_com_recon, logvar_com_recon, _ = VAE_com(recon_img) #[B, 64] [B, 64] _
        mu_uni_recon, logvar_uni_recon, _ = VAE_uni(recon_img) #[B, 64] [B, 64] _
        
        # 1.loss_featurerecon_common
        loss_kld_com_recon = KLDLoss()(mu_com_raw, logvar_com_raw, mu_com_recon, logvar_com_recon)
        # 2.loss_featurerecon_unique
        loss_kld_uni_recon = KLDLoss()(mu_uni_raw, logvar_uni_raw, mu_uni_recon, logvar_uni_recon)
        # 3.loss_imgrecon
        loss_imgrecon = torch.mean((recon_img-raw_img)**2) + torch.mean(torch.abs(recon_img-raw_img))
        # Total Loss G
        TotalLoss_g = loss_kld_com_recon + loss_kld_uni_recon + loss_imgrecon
        
        TotalLoss_g.backward()
        g_optim.step()
        
        ##################################################################################
        ## Discriminating Branch
        ##################################################################################
        #-------------------------------------------------------------------
        # Train G
        #-------------------------------------------------------------------
        g_optim.zero_grad()
        
        # 原始图像解耦
        _, _, feat_uni_raw = VAE_uni(raw_img) #[B, 64] [B, 64] [B, 64]
        # 生成图像
        z_add = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (raw_img.shape[0], 64))), requires_grad=False).cuda() # [B, 64]
        feat_add = torch.concat([z_add, feat_uni_raw], dim=1) # [B, 128]
        gen_img = g_src(feat_add) # [B, 3, 128, 128]
        z_rand = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (raw_img.shape[0], 128))), requires_grad=False).cuda() # [B, 128]
        gen_rand_img = g_src(z_rand)
        
        # Calculate G loss_adv
        raw_img_output = d_src(raw_img) # [B, 64, 64, 64] [B, 128, 32, 32] [B, 256, 16, 16] [B, 512, 8, 8] [B, 1] [B, 64]
        gen_img_output = d_src(gen_img) # [B, 64, 64, 64] [B, 128, 32, 32] [B, 256, 16, 16] [B, 512, 8, 8] [B, 1] [B, 64]
        pred_raw_img = raw_img_output[4] # [B, 1]
        pred_gen_img = gen_img_output[4] # [B, 1]
        labels_raw_img = torch.ones_like(pred_raw_img).cuda() # [B, 1]
        labels_gen_img = torch.zeros_like(pred_gen_img).cuda() # [B, 1]
        
        gen_rand_img_output = d_src(gen_rand_img)
        pred_gen_rand_img = gen_rand_img_output[4]
        labels_gen_rand_img = torch.zeros_like(pred_gen_rand_img).cuda() # [B, 1]
        
        # 1.loss_FM
        criterionFeat = nn.L1Loss()
        loss_FM = torch.cuda.FloatTensor(1).fill_(0)
        num_features_maps = 4
        weights_features_maps = [1., 1., 1., 1.]
        for i_map in range(num_features_maps):
            i_loss_FM = criterionFeat(gen_img_output[i_map], raw_img_output[i_map])
            loss_FM += i_loss_FM * weights_features_maps[i_map]
        loss_FM = loss_FM * 10.0
        # 2.loss_adv
        loss_adv_gen_img = nn.BCEWithLogitsLoss()(pred_gen_img, torch.ones_like(pred_gen_img)) 
        loss_adv_gen_rand_img = nn.BCEWithLogitsLoss()(pred_gen_rand_img, torch.ones_like(pred_gen_rand_img))
        loss_adv =  (loss_adv_gen_img + loss_adv_gen_rand_img) * 0.5
        # D Total Loss G
        D_TotalLoss_g = loss_FM + loss_adv
    
        D_TotalLoss_g.backward()
        g_optim.step()
        #-------------------------------------------------------------------
        # Train D
        #-------------------------------------------------------------------
        d_optim.zero_grad()

        # Images: raw_img, gen_img, gen_rand_img
        gen_img = gen_img.detach()
        gen_rand_img = gen_rand_img.detach()
        
        # 判别
        raw_img_output = d_src(raw_img) # [B, 64, 64, 64] [B, 128, 32, 32] [B, 256, 16, 16] [B, 512, 8, 8] [B, 1] [B, 64]
        gen_img_output = d_src(gen_img)
        gen_rand_img_output = d_src(gen_rand_img)
        # BCELOSS
        loss_adv_raw_img = nn.BCEWithLogitsLoss()(raw_img_output[4], torch.ones_like(raw_img_output[4]).cuda())
        loss_adv_gen_img = nn.BCEWithLogitsLoss()(gen_img_output[4], torch.zeros_like(gen_img_output[4]).cuda())
        loss_adv_gen_rand_img = nn.BCEWithLogitsLoss()(gen_rand_img_output[4], torch.zeros_like(gen_rand_img_output[4]).cuda())
        D_TotalLoss_d = loss_adv_raw_img + loss_adv_raw_img + loss_adv_gen_rand_img
        
        D_TotalLoss_d.backward()
        d_optim.step()
        
        # Show
        print(
            "[Epoch %d/%d] [Batch %d/%d] [TotalLoss_vae: %f] [TotalLoss_g: %f] [D_TotalLoss_g: %f] [D_TotalLoss_d: %f]"
            %
            (epoch, config.num_epochs, i+1, len(data_iter_loader), TotalLoss_vae, TotalLoss_g, D_TotalLoss_g, D_TotalLoss_d)
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
            "model_state_dict": g_src.state_dict(),
            "z_dim": config.z_dim,
        }, net_g_source_path)
        
        net_VAE_common_path = models_save_path + '/%d_net_VAE_common.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": VAE_com.state_dict(),
            "z_dim": config.z_dim,
        }, net_VAE_common_path)
        
        net_VAE_unique_path = models_save_path + '/%d_net_VAE_unique.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": VAE_uni.state_dict(),
            "z_dim": config.z_dim,
        }, net_VAE_unique_path)
    #-------------------------------------------------------------------
    # Save model D
    #-------------------------------------------------------------------
    if epoch == config.num_epochs:
        net_d_source_path = models_save_path + '/%d_net_d_source.pth' % epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": d_src.state_dict(),
            "z_dim": config.z_dim,
        }, net_d_source_path)
        