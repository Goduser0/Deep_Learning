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
from TA_G import PFS_Generator
from TA_D import PFS_Discriminator
from TA_VAE import PFS_Encoder
from TA_layers import vgg
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import stage1_logger
from TA_utils import stage1_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument("--logs_dir", type=str,
                    default="My_TAOD/TA/TA_Results/stage1/logs")
parser.add_argument("--models_dir", type=str,
                    default="My_TAOD/TA/TA_Results/stage1/models")
parser.add_argument("--samples_dir", type=str,
                    default="My_TAOD/TA/TA_Results/stage1/samples")
parser.add_argument("--results_dir", type=str,
                    default="My_TAOD/TA/TA_Results/stage1/results")

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
                    default="My_TAOD/dataset/PCB_200/160-shot/train/0.csv")
parser.add_argument("--category",
                    type=str,
                    default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)

# train
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--gpu_id", type=str, default="0")

# G
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--lr_g", type=float, default=2e-4)

# D
parser.add_argument("--lr_d", type=float, default=2e-4)
parser.add_argument("--D_iters", type=int, default=5)

# VAE
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--lr_vae", type=float, default=5e-4)

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
stage1_logger(config)

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
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

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
D = PFS_Discriminator().cuda()
G = PFS_Generator(config.z_dim).cuda()
Encoder_c = PFS_Encoder(latent_dim=64).cuda()
Encoder_s = PFS_Encoder(latent_dim=64).cuda()

v2 = vgg().cuda()

optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=config.lr_d, betas=(0.5, 0.9))
optim_G = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
optim_Encoder_c = torch.optim.Adam([{'params':Encoder_c.parameters()}], lr=config.lr_vae, betas=(0.9,0.999))
optim_Encoder_s = torch.optim.Adam([{'params':Encoder_s.parameters()}], lr=config.lr_vae, betas=(0.9,0.999))


flag = False

for epoch in tqdm.trange(1, config.num_epochs + 1, desc="On training"):
    D_loss_list = []
    G_loss_list = []
    KLD_c_list = []
    KLD_s_list = []
    Imgrecon_loss_list = []
    Srecon_loss_list = []
    Perceptual_loss_list = []
    
    for batch_idx, (raw_img, category_label) in enumerate(data_iter_loader):
        optim_Encoder_c.zero_grad()
        optim_Encoder_s.zero_grad()
        optim_G.zero_grad()
        v2.zero_grad()
        
        num = raw_img.size(0)
        raw_img, category_label = raw_img.cuda(), category_label.cuda()
        
        mu_c, logvar_c, zr_c = Encoder_c(raw_img)
        mu_s, logvar_s, zr_s = Encoder_s(raw_img)
        zr = torch.cat([zr_s, zr_c], 1)
        
        if epoch != 1 and flag:
            for i in range(config.D_iters):
                z = Variable(torch.randn(num, config.z_dim).cuda())
                optim_D.zero_grad()
                fake_img = G(z)
                
                D_loss = nn.ReLU()(1.0 - D(raw_img)).mean() + nn.ReLU()(1.0 + D(fake_img)).mean()*0.5 + nn.ReLU()(1.0 + D(G(zr).detach())).mean()*0.5
                D_loss_list.append(D_loss.item())
                (D_loss*0.01).backward()
                optim_D.step()
                
            optim_Encoder_s.zero_grad()
            optim_Encoder_c.zero_grad()
            optim_G.zero_grad()
            
            G_loss_v = -D(G(z)).mean()
            G_loss_r = -D(G(zr.detach())).mean()
            G_loss = (G_loss_v + G_loss_r)*0.5
            G_loss_list.append(G_loss.item())
            (G_loss*0.01).backward()
        
        z1 = Variable(torch.randn(num, config.latent_dim).cuda())
        
        mu_c, logvar_c, zr_c = Encoder_c(raw_img)
        mu_s, logvar_s, zr_s = Encoder_s(raw_img)
        
        zr = torch.cat([zr_s, zr_c], 1)
        zr2 = torch.cat([z1, zr_c], 1)
        
        recon_img = G(zr)
        recon_img2 = G(zr2)
        KLD_c = -0.5 * torch.mean(1 + logvar_c - mu_c.pow(2) - logvar_c.exp()) 
        KLD_s = -0.5 * torch.mean(1 + logvar_s - mu_s.pow(2) - logvar_s.exp()) 
        KLD_c_list.append(KLD_c.item())
        KLD_s_list.append(KLD_s.item())
        mu_s2, logvar_s2, zr_s2 = Encoder_s(recon_img2)   
        
        recon_img_loss = torch.mean((recon_img-raw_img)**2) + torch.mean(torch.abs(recon_img-raw_img))
        Imgrecon_loss_list.append(recon_img_loss.item())
        f_recon_img, f_recon_img2, f_data = v2(recon_img), v2(recon_img2), v2(raw_img)
        for i in range(3):
            recon_img_loss += torch.mean(torch.abs(f_recon_img[i]-f_data[i]))*0.01
        perceptual_loss = torch.mean((f_recon_img2[2]-f_data[2])**2)*0.01
        Perceptual_loss_list.append(perceptual_loss.item())
        recon_s_loss = torch.mean((z1-zr_s2)**2)
        Srecon_loss_list.append(recon_s_loss.item())
        if recon_img_loss.item() < 0.2:
            flag = True
            
        recon_loss = recon_img_loss*1 + recon_s_loss*0.1 + perceptual_loss
        KLD_loss = KLD_c + KLD_s
        
        (recon_loss+KLD_loss*0.1).backward()
        optim_Encoder_c.step()
        optim_Encoder_s.step()
        optim_G.step()
    
    images = raw_img.detach().to('cpu').numpy()[0]
    images = img_1to255(images)
    images = np.transpose(images, (1, 2, 0))
    plt.imshow(images)
    plt.savefig(f"raw_{config.dataset_class}.png")
    plt.close()
    images = recon_img.detach().to('cpu').numpy()[0]
    images = img_1to255(images)
    images = np.transpose(images, (1, 2, 0))
    plt.imshow(images)
    plt.savefig(f"gen1_{config.dataset_class}.png")
    plt.close()
    images = recon_img2.detach().to('cpu').numpy()[0]
    images = img_1to255(images)
    images = np.transpose(images, (1, 2, 0))
    plt.imshow(images)
    plt.savefig(f"gen2_{config.dataset_class}.png")
    plt.close()
        
    # Show
    print(
        "[Epoch %d/%d] [Batch %d/%d] [G_loss: %.5f] [D_loss: %.5f] [KLD_c: %.5f] [KLD_s: %.5f] [imgrecon: %.5f] [s_recon: %.5f] [Perceptual_loss: %.5f]"
        %
        (epoch, config.num_epochs, batch_idx+1, len(data_iter_loader), np.mean(G_loss_list), np.mean(D_loss_list), np.mean(KLD_c_list), np.mean(KLD_s_list), np.mean(Imgrecon_loss_list), np.mean(Srecon_loss_list), np.mean(Perceptual_loss_list))
    )
    
    # Record Data
    stage1_record_data(config,
                        {
                            "epoch": f"{epoch}",
                            "num_epochs": f"{config.num_epochs}",
                            "batch": f"{batch_idx+1}",
                            "num_batchs": f"{len(data_iter_loader)}",
                            "G_loss": f"{np.mean(G_loss_list)}",
                            "D_loss": f"{np.mean(D_loss_list)}",
                            "KLD_c": f"{np.mean(KLD_c_list)}",
                            "KLD_s": f"{np.mean(KLD_s_list)}",
                            "imgrecon": f"{np.mean(Imgrecon_loss_list)}",
                            "s_recon": f"{np.mean(Srecon_loss_list)}",
                            "Perceptual_loss": f"{np.mean(Perceptual_loss_list)}",
                        },
                        flag_plot=True,
                        )

    ##################################################################################
    ## Checkpoint
    ##################################################################################
    #-------------------------------------------------------------------
    # Save model G
    #-------------------------------------------------------------------
    if epoch % (config.num_epochs // 40) == 0:
        net_G_path = models_save_path + "/%d_net_g.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": G.state_dict(),
            "z_dim": config.z_dim,
        }, net_G_path)
        
        net_Encoder_c_path = models_save_path + "/%d_net_Encoder_c.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": Encoder_c.state_dict(),
            "latent_dim": config.latent_dim,
        }, net_Encoder_c_path)
        
        net_Encoder_s_path = models_save_path + "/%d_net_Encoder_s.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": Encoder_s.state_dict(),
            "latent_dim": config.latent_dim,
        }, net_Encoder_s_path)
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