import warnings
warnings.filterwarnings("ignore")

import argparse
import random
import numpy as np
import os
import tqdm
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader, img_1to255
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_G import CoGAN_Generator
from TA_D import CoGAN_Discriminator

sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import cogan_logger
from TA_utils import cogan_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument("--logs_dir", type=str,
                    default="My_TAOD/TA/TA_Results/CoGAN/logs")
parser.add_argument("--models_dir", type=str,
                    default="My_TAOD/TA/TA_Results/CoGAN/models")
parser.add_argument("--samples_dir", type=str,
                    default="My_TAOD/TA/TA_Results/CoGAN/samples")
parser.add_argument("--results_dir", type=str,
                    default="My_TAOD/TA/TA_Results/CoGAN/results")

# random seed
parser.add_argument("--random_seed", type=int, default=42)

# data loader
parser.add_argument("--dataset_S_class",
                    type=str,
                    default='DeepPCB_Crop',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )
parser.add_argument("--dataset_S_path",
                    type=str,
                    default="My_TAOD/dataset/DeepPCB_Crop/160-shot/train/0.csv")

parser.add_argument("--dataset_T_class",
                    type=str,
                    default='PCB_Crop',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )
parser.add_argument("--dataset_T_path",
                    type=str,
                    default="My_TAOD/dataset/PCB_Crop/30-shot/train/0.csv")

parser.add_argument("--category",
                    type=str,
                    default="0")



parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--S_batch_size", type=int, default=64)
parser.add_argument("--T_batch_size", type=int, default=16)

# train
parser.add_argument("--num_epochs", type=int, default=30000)
parser.add_argument("--gpu_id", type=str, default="0")

# G
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--lr_g", type=float, default=2e-4)

# D
parser.add_argument("--lr_d", type=float, default=2e-4)
parser.add_argument("--D_iters", type=int, default=1)

# Others
parser.add_argument("--log", type=bool, default=True)
parser.add_argument("--time", type=str, default=time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime()))

# config
config = parser.parse_args()

# 一致性检验
assert(config.dataset_S_path.split('/')[-4]==config.dataset_S_class
       and
       config.dataset_S_path.split('/')[-1][0]==config.category
       and
       config.dataset_T_path.split('/')[-4]==config.dataset_T_class
       and
       config.dataset_T_path.split('/')[-1][0]==config.category
       )

# logger
cogan_logger(config)

models_save_path = config.models_dir + '/' + config.dataset_S_class + '2' + config.dataset_T_class + ' ' + config.category + ' ' + config.time
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
        T.Resize((config.img_size, config.img_size)),
    ]
)

S_iter_loader = get_loader(
    config.dataset_S_class,
    config.dataset_S_path,
    config.batch_size,
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)

T_iter_loader = get_loader(
    config.dataset_T_class,
    config.dataset_T_path,
    config.batch_size,
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)

# model
D = CoGAN_Discriminator().cuda()
G = CoGAN_Generator(config.z_dim).cuda()

optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=config.lr_d, betas=(0.0, 0.9))
optim_G = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.0, 0.9))

for epoch in tqdm.trange(1, config.num_epochs + 1, desc="On training"):
    S_D_loss_list = []
    S_G_loss_list = []
    T_D_loss_list = []
    T_G_loss_list = []
    
    for batch_idx, (T_imgs, T_labels) in enumerate(T_iter_loader):
        num = T_imgs.size(0)
        T_imgs = T_imgs.cuda()
        
        for _, (S_imgs, _) in enumerate(S_iter_loader):
            S_imgs = S_imgs.cuda()
            break
        
        for _ in range(config.D_iters):
            z = Variable(torch.randn(num, config.z_dim).cuda())
            randA, randB = G(z, domain='S'), G(z, domain='T')
            optim_D.zero_grad()
            
            S_D_loss = nn.ReLU()(1.0 - D(S_imgs, domain='S')).mean() + \
                            nn.ReLU()(1.0 + D(randA.detach(), domain='S')).mean()

            T_D_loss = nn.ReLU()(1.0 - D(T_imgs, domain='T')).mean() + \
                            nn.ReLU()(1.0 + D(randB.detach(), domain='T')).mean()          
            D_loss = (S_D_loss + T_D_loss)*0.5
            (D_loss).backward()
            optim_D.step()
            
        S_D_loss_list.append(S_D_loss.item())
        T_D_loss_list.append(T_D_loss.item())
        
        optim_G.zero_grad()
        randA, randB = G(z, domain='S'), G(z, domain='T')
        S_G_loss = -D(randA, domain='S').mean()
        T_G_loss = -D(randB, domain='T').mean()
        G_loss = (S_G_loss + T_G_loss)*0.5
        
        (G_loss).backward()
        optim_G.step()
        
        S_G_loss_list.append(S_G_loss.item())
        T_G_loss_list.append(T_G_loss.item())
    
    # Show
    print(
        "[Epoch %d/%d] [S_D_loss: %.5f] [T_D_loss: %.5f] [S_G_loss: %.5f] [T_G_loss: %.5f]"
        %
        (epoch, config.num_epochs, np.mean(S_D_loss_list), np.mean(T_D_loss_list), np.mean(S_G_loss_list), np.mean(T_G_loss_list))
    )
    # Record Data
    cogan_record_data(config,
                      {
                          "epoch": f"{epoch}",
                          "num_epochs": f"{config.num_epochs}",
                          "S_D_loss": f"{np.mean(S_D_loss_list)}",
                          "T_D_loss": f"{np.mean(T_D_loss_list)}",
                          "S_G_loss": f"{np.mean(S_G_loss_list)}",
                          "T_G_loss": f"{np.mean(T_G_loss_list)}",
                      },
                      flag_plot=True,
                      )
    
    ##################################################################################
    ## Checkpoint
    ##################################################################################
    #-------------------------------------------------------------------
    # Save model G
    #-------------------------------------------------------------------
    if epoch % (config.num_epochs // 10) == 0:
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
    
        