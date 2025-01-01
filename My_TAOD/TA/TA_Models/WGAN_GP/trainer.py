import argparse
import time
import tqdm
import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends
import torch.backends.cudnn
import torchvision.transforms as T
from torch.autograd import Variable

from model import WGAN_GP_Generator, WGAN_GP_Discriminator, Cal_Gradient_Penalty

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader

sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import WGAN_GP_logger
from TA_utils import WGAN_GP_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Save Directories
parser.add_argument("--save_rootdir", type=str, default="My_TAOD/TA/TA_Results/WGAN_GP")
# data loader
parser.add_argument("--dataset_path",
                    type=str,
                    default="My_TAOD/dataset/DeepPCB_Crop/10-shot/train/0.csv")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
# train
parser.add_argument("--num_epochs", type=int, default=10000)
parser.add_argument("--gpu_id", type=str, default="0")

parser.add_argument("--img_size", type=int, default=32)
parser.add_argument("--lr_g", type=float, default=2e-4)

parser.add_argument("--lr_d", type=float, default=2e-4)
# Others
parser.add_argument("--time", type=str, default=time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime()))

config = parser.parse_args()

# Create Directories
start_index = config.dataset_path.find("dataset/") + len("dataset/")
dataset_class = config.dataset_path[start_index:].split('/')[0]
category = config.dataset_path.split('/')[-1].split('.')[0]
save_dir = f"{config.save_rootdir}/{dataset_class} {category} {config.time}"
os.makedirs(save_dir, exist_ok=False)
os.makedirs(f"{save_dir}/results", exist_ok=False)
os.makedirs(f"{save_dir}/models", exist_ok=False)
os.makedirs(f"{save_dir}/samples", exist_ok=False)

######################################################################################################
#### Setting
######################################################################################################

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

trans = T.Compose([
    T.ToTensor(),
    T.Resize((config.img_size, config.img_size)),
])

train_loader = get_loader(
    config.dataset_path,
    config.batch_size, 
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)
    
G = WGAN_GP_Generator()
D = WGAN_GP_Discriminator()

device = 'cuda:' + config.gpu_id
G.to(device)
D.to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(D.parameters(), lr=config.lr_d, betas=(0.5, 0.999))


######################################################################################################
#### Train
######################################################################################################
# logger
WGAN_GP_logger(config, save_dir)
# train
for epoch in tqdm.trange(1, config.num_epochs + 1, desc=f"[Epoch:{config.num_epochs}]On training"):
    # loss_list
    batchsize_list = []

    D_real_loss_list = []
    D_fake_loss_list = []
    D_gp_list = []
    D_loss_list = []
    
    G_loss_list = [] 
    
    for batch_idx, (raw_img, category_label) in enumerate(train_loader):
        
        for i in D.parameters():
            i.requires_grad = True
        
        num = raw_img.size(0)
        batchsize_list.append(num)
        
        raw_img, category_label = Variable(raw_img).to(device), Variable(category_label).to(device)

        # train D
        optim_D.zero_grad()
        z = Variable(torch.randn(num, 100, 1, 1)).to(device)
        
        real_img_out = D(raw_img)
        fake_img = G(z)
        fake_img_out = D(fake_img)
        
        gp = Cal_Gradient_Penalty(D, raw_img, fake_img, device)
        D_real_loss = -torch.mean(real_img_out)
        D_fake_loss = torch.mean(fake_img_out)
        lambda_gp = 10.0
        D_loss = D_real_loss + D_fake_loss + gp * lambda_gp
        
        D_gp_list.append(gp.item() * num)
        D_real_loss_list.append(D_real_loss.item() * num)
        D_fake_loss_list.append(D_fake_loss.item() * num)
        D_loss_list.append(D_loss.item() * num)
        
        D_loss.backward()
        optim_D.step()
        
        # train G
        for i in D.parameters():
            i.requires_grad = False
            
        optim_G.zero_grad()
        
        fake_img_out = D(G(z))
        
        G_fake_loss = -torch.mean(fake_img_out)
        G_loss = G_fake_loss
        
        G_loss_list.append(G_loss.item() * num)
        
        G_loss.backward()
        optim_G.step()
        
        
    # Record Data
    WGAN_GP_record_data(
        save_dir,
        {
            "epoch": f"{epoch}",
            "num_epochs": f"{config.num_epochs}",
            "batch": f"{batch_idx+1}",
            "num_batchs": f"{len(train_loader)}",
            
            "D_real_loss": f"{np.sum(D_real_loss_list) / np.sum(batchsize_list)}",
            "D_fake_loss": f"{np.sum(D_fake_loss_list) / np.sum(batchsize_list)}",
            "D_gp": f"{np.sum(D_gp_list) / np.sum(batchsize_list)}",
            "D_loss": f"{np.sum(D_loss_list) / np.sum(batchsize_list)}",
            
            "G_loss": f"{np.sum(G_loss_list) / np.sum(batchsize_list)}",
            
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
        net_G_path = f"{save_dir}/models/{epoch}_net_g.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": G.state_dict(),
        }, net_G_path)
    #-------------------------------------------------------------------
    # Save model D
    #-------------------------------------------------------------------
    if epoch == config.num_epochs:
        net_D_path = f"{save_dir}/models/{epoch}_net_d.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": D.state_dict(),
            "img_size": config.img_size,
        }, net_D_path)