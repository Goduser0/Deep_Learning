# 基础的GAN从头训练
import os
import argparse
import tqdm
import time
import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_G import PFS_Generator, FeatureMatchGenerator
from TA_D import PFS_Discriminator_patch, PFS_Discriminator, FeatureMatchDiscriminator
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import baseline_from_scratch_logger
from TA_utils import requires_grad, baseline_from_scratch_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument("--logs_dir", type=str,
                    default="My_TAOD/TA/TA_Results/baseline_from_scratch/logs")
parser.add_argument("--models_dir", type=str,
                    default="My_TAOD/TA/TA_Results/baseline_from_scratch/models")
parser.add_argument("--samples_dir", type=str,
                    default="My_TAOD/TA/TA_Results/baseline_from_scratch/samples")
parser.add_argument("--results_dir", type=str,
                    default="My_TAOD/TA/TA_Results/baseline_from_scratch/results")

# random seed
parser.add_argument("--random_seed", type=int, default=42)

# data loader 
parser.add_argument("--dataset_class",
                    type=str,
                    default='PCB_200',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )
parser.add_argument("--data_path",
                    type=str,
                    default="My_TAOD/dataset/PCB_200/30-shot/train/0.csv")
parser.add_argument("--category",
                    type=str,
                    default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=30)

# G
parser.add_argument("--lr_g", type=float, default=1e-4)
parser.add_argument("--z_dim", type=int, default=128)

# D
parser.add_argument("--lr_d", type=float, default=1e-3)

# train
parser.add_argument("--num_epochs", type=int, default=2000)
parser.add_argument("--gpu_id", type=str, default="0")

parser.add_argument("--img_size", type=int, default=128)

# Others
parser.add_argument("--log", type=bool, default=True)
parser.add_argument("--time", type=str, default=time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime()))

# config
config = parser.parse_args()

# 一致性检验
assert(config.data_path.split('/')[-4]==config.dataset_class
       and
       config.data_path.split('/')[-1][0]==config.category)

# logger
baseline_from_scratch_logger(config)

models_save_path = config.models_dir + '/' + config.dataset_class + ' ' + config.category + ' ' + config.time
os.makedirs(models_save_path, exist_ok=False)

######################################################################################################
#### Setting
######################################################################################################
# random seed
random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

# device
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

# data_loader
trans = T.Compose(
    [
        T.ToTensor(),
        T.Resize((config.img_size, config.img_size)),
    ]
)

data_iter_loader = get_loader(
    config.dataset_class,
    config.data_path,
    config.batch_size,
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)

# model
G = FeatureMatchGenerator(n_mlp=3).cuda()
D = FeatureMatchDiscriminator().cuda()
# optimizers
optim_G = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.5, 0.9))
optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=config.lr_d, betas=(0.5, 0.9))
######################################################################################################
#### Train
######################################################################################################
for epoch in tqdm.trange(1, config.num_epochs+1, desc=f"On training"):
    batchsize_list = []
    G_adv_loss_list = []
    G_FM_loss_list = []
    G_loss_list = []
    D_real_adv_loss_list = []
    D_fake_adv_loss_list = []
    D_loss_list = []
    for batch_idx, (raw_img, category_label) in enumerate(data_iter_loader):
        optim_G.zero_grad()
        num = raw_img.size(0)
        batchsize_list.append(num)
        
        # train D
        raw_img, category_label = Variable(raw_img.cuda()), Variable(category_label.cuda())
        for _ in range(1):
            optim_D.zero_grad()
            
            z = Variable(torch.randn(num, config.z_dim).cuda())
            fake_img = G(z)
            real_img_out = D(raw_img)
            fake_img_out = D(fake_img)
            
            real_adv_loss = nn.BCEWithLogitsLoss()(real_img_out[-2], torch.ones_like(real_img_out[-2]))
            fake_adv_loss = nn.BCEWithLogitsLoss()(fake_img_out[-2], torch.zeros_like(fake_img_out[-2]))
            D_loss = real_adv_loss + fake_adv_loss
            D_real_adv_loss_list.append(real_adv_loss.item() * num)
            D_fake_adv_loss_list.append(fake_adv_loss.item() * num)
            D_loss_list.append(D_loss.item() * num)

            D_loss.backward()
            optim_D.step()
            
        # train G
        optim_G.zero_grad()
        
        real_img_out = D(raw_img)
        fake_img_out = D(G(z))
        
        # G_adv_loss
        G_adv_loss = nn.BCEWithLogitsLoss()(fake_img_out[-2], torch.ones_like(fake_img_out[-2]))
        # G_FM_loss
        G_FM_loss = torch.cuda.FloatTensor(1).fill_(0)
        num_feature_maps = 4
        lambda_fm_weight = [1., 1., 1., 1.]
        for i_map in range(num_feature_maps):
            fm_loss = torch.nn.L1Loss()(fake_img_out[i_map], real_img_out[i_map])
            G_FM_loss += fm_loss * lambda_fm_weight[i_map]
        G_loss = G_adv_loss * 0.05 + G_FM_loss * 1.      
        G_adv_loss_list.append(G_adv_loss.item() * 0.05 * num)
        G_FM_loss_list.append(G_FM_loss.item() * 1. * num)
        G_loss_list.append(G_loss.item() * num)
        
        G_loss.backward()
        optim_G.step()
        
    # Show
    print(
        "[Epoch %d/%d] [Batch %d/%d] [G_loss: %.5f] [D_loss: %.5f]"
        %
        (epoch, config.num_epochs, batch_idx+1, len(data_iter_loader), np.sum(G_loss_list) / np.sum(batchsize_list), np.sum(D_loss_list) / np.sum(batchsize_list))
         
    )
    
    # Record Data
    baseline_from_scratch_record_data(
        config,
        {
            "epoch": f"{epoch}",
            "num_epochs": f"{config.num_epochs}",
            "batch": f"{batch_idx+1}",
            "num_batchs": f"{len(data_iter_loader)}",
            "G_adv_loss": f"{np.sum(G_adv_loss_list) / np.sum(batchsize_list)}",
            "G_FM_loss": f"{np.sum(G_FM_loss_list) / np.sum(batchsize_list)}",
            "G_loss": f"{np.sum(G_loss_list) / np.sum(batchsize_list)}",
            "D_real_adv_loss": f"{np.sum(D_real_adv_loss_list) / np.sum(batchsize_list)}",
            "D_fake_adv_loss": f"{np.sum(D_fake_adv_loss_list) / np.sum(batchsize_list)}",
            "D_loss": f"{np.sum(D_loss_list) / np.sum(batchsize_list)}",
        },
        flag_plot=True,
    )
    
    ##################################################################################
    ## Checkpoint
    ##################################################################################
    #-------------------------------------------------------------------
    # Save model G
    #-------------------------------------------------------------------
    if epoch % (config.num_epochs // 20) == 0:
        net_G_path = models_save_path + "/%d_net_g.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": G.state_dict(),
            "z_dim": config.z_dim,
        }, net_G_path)
    #-------------------------------------------------------------------
    # Save model D
    #-------------------------------------------------------------------
    if epoch == config.num_epochs:
        net_D_path = models_save_path + "/%d_net_d.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": D.state_dict(),
            "img_size": config.img_size,
        }, net_D_path)