import argparse
import time
import tqdm
import numpy as np
import os
import itertools

import torch
import torch.nn as nn
import torch.backends
import torch.backends.cudnn
import torchvision.transforms as T
from torch.autograd import Variable

from model import MyGAN_Encoder, MyGAN_Generator, MyGAN_Discriminator, Lambda_Learing_Rate, Weights_Init_Normal, Cal_Gradient_Penalty

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader

sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import MyGAN_logger
from TA_utils import MyGAN_record_data 

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Save Directories
parser.add_argument("--save_rootdir", type=str, default="My_TAOD/TA/TA_Results/MyGAN")
# Tar data loader
parser.add_argument("--Tar_dataset_path",
                    type=str,
                    default="My_TAOD/dataset/DeepPCB_Crop/10-shot/train/0.csv")
# Src data loader
parser.add_argument("--Src_dataset_path",
                    type=str,
                    default="My_TAOD/dataset/PCB_200/200-shot/train/0.csv")

parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=4)
# train
parser.add_argument("--num_epochs", type=int, default=10000)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--decay_epoch", type=int, default=5000)

parser.add_argument("--img_size", type=int, default=64)

parser.add_argument("--lr_g", type=float, default=2e-4)

parser.add_argument("--lr_d", type=float, default=2e-4)
# Others
parser.add_argument("--time", type=str, default=time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime()))

config = parser.parse_args()

# Create Directories
Src_name = f"{config.Src_dataset_path.split('/')[-4]}[{config.Src_dataset_path.split('/')[-3]}]"
Tar_name = f"{config.Tar_dataset_path.split('/')[-4]}[{config.Tar_dataset_path.split('/')[-3]}]"

Src_category = config.Src_dataset_path.split('/')[-1].split('.')[0]
Tar_category = config.Tar_dataset_path.split('/')[-1].split('.')[0]
assert Src_category == Tar_category

save_dir = f"{config.save_rootdir}/{Tar_name}<-{Src_name} {Tar_category} {config.time}"
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

train_loader_Src = get_loader(
    config.Src_dataset_path,
    config.batch_size, 
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)

train_loader_Tar = get_loader(
    config.Tar_dataset_path,
    config.batch_size, 
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
    num_expand=int(config.Src_dataset_path.split('/')[-3].split('-')[0]),
)

E_content = MyGAN_Encoder()
E_style = MyGAN_Encoder()
G_src = MyGAN_Generator()
G_tar = MyGAN_Generator()
D_Src = MyGAN_Discriminator()
D_Tar = MyGAN_Discriminator()

E_content.apply(Weights_Init_Normal)
E_style.apply(Weights_Init_Normal)
G_src.apply(Weights_Init_Normal)
G_tar.apply(Weights_Init_Normal)
D_Src.apply(Weights_Init_Normal)
D_Tar.apply(Weights_Init_Normal)

device = 'cuda:' + config.gpu_id
E_content.to(device)
E_style.to(device)
G_src.to(device)
G_tar.to(device)
D_Src.to(device)
D_Tar.to(device)

optim_E = torch.optim.Adam(itertools.chain(E_content.parameters(), E_style.parameters()), lr=config.lr_g, betas=(0.5, 0.999))
optim_G = torch.optim.Adam(itertools.chain(G_src.parameters(), G_tar.parameters()), lr=config.lr_g, betas=(0.5, 0.999))
optim_D_Src = torch.optim.Adam(D_Src.parameters(), lr=config.lr_d, betas=(0.5, 0.999))
optim_D_Tar = torch.optim.Adam(D_Tar.parameters(), lr=config.lr_d, betas=(0.5, 0.999))

lr_scheduler_E = torch.optim.lr_scheduler.LambdaLR(optim_E, lr_lambda=Lambda_Learing_Rate(config.num_epochs, config.start_epoch, config.decay_epoch).step)
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=Lambda_Learing_Rate(config.num_epochs, config.start_epoch, config.decay_epoch).step)
lr_scheduler_D_Src = torch.optim.lr_scheduler.LambdaLR(optim_D_Src, lr_lambda=Lambda_Learing_Rate(config.num_epochs, config.start_epoch, config.decay_epoch).step)
lr_scheduler_D_Tar = torch.optim.lr_scheduler.LambdaLR(optim_D_Tar, lr_lambda=Lambda_Learing_Rate(config.num_epochs, config.start_epoch, config.decay_epoch).step)

Tensor = torch.cuda.FloatTensor
######################################################################################################
#### Train
######################################################################################################
# logger
MyGAN_logger(config, save_dir)
# train
for epoch in tqdm.trange(config.start_epoch + 1, config.num_epochs + 1, desc=f"[Epoch:{config.num_epochs}]On training"):
    # loss_list
    batchsize_list = []
    
    G_adv_loss_list = []
    G_FM_loss_list = []
    G_cycle_loss_Src_list = []
    G_cycle_loss_Tar_list = []
    G_loss_list = []
    
    D_Src_real_adv_loss_list = []
    D_Src_fake_adv_loss_list = []
    D_Src_recovery_adv_loss_list = []
    D_Src_loss_list = []

    D_Tar_real_adv_loss_list = []
    D_Tar_fake_adv_loss_list = []
    D_Tar_recovery_adv_loss_list = []
    D_Tar_loss_list = []
    
    for batch_idx, ((raw_img_Src, category_label_Src), (raw_img_Tar, category_label_Tar)) in enumerate(zip(train_loader_Src, train_loader_Tar)):
        
        num = raw_img_Tar.size(0)
        batchsize_list.append(num)
        
        raw_img_Src, category_label_Src = Variable(raw_img_Src).to(device), Variable(category_label_Src).to(device).view(num)
        raw_img_Tar, category_label_Tar = Variable(raw_img_Tar).to(device), Variable(category_label_Tar).to(device).view(num)
        
        real_labels = Variable(Tensor(num).fill_(1.0), requires_grad=False).to(device)
        fake_labels = Variable(Tensor(num).fill_(0.0), requires_grad=False).to(device)
        
        # train E & G
        optim_E.zero_grad()
        optim_G.zero_grad()
        
        # adv_loss
        f_src_content = E_content(raw_img_Src).view(-1, 64)
        f_src_style = E_style(raw_img_Src).view(-1, 64)
        f_tar_content = E_content(raw_img_Tar).view(-1, 64)
        f_tar_style = E_style(raw_img_Tar).view(-1, 64)
        
        recovery_img_Src = G_src(torch.cat([f_src_content, f_src_style], dim=1))
        fake_img_Src = G_src(torch.cat([f_tar_content, f_src_style], dim=1))
        recovery_img_Tar = G_src(torch.cat([f_tar_content, f_tar_style], dim=1))
        fake_img_Tar = G_src(torch.cat([f_src_content, f_tar_style], dim=1))
        
        recovery_img_Src_out = D_Src(recovery_img_Src)
        fake_img_Src_out = D_Src(fake_img_Src)
        real_img_Src_out = D_Src(raw_img_Src)
        recovery_img_Tar_out = D_Tar(recovery_img_Tar)
        fake_img_Tar_out = D_Tar(fake_img_Tar)
        real_img_Tar_out = D_Tar(raw_img_Tar)
        
        fake_adv_loss = - torch.mean(fake_img_Src_out[-2]) - torch.mean(fake_img_Tar_out[-2]) 
        
        
        criterionFeat = nn.L1Loss()
        FM_loss = torch.FloatTensor(1).fill_(0).to(device)
        num_features_maps = 4
        lambda_fm_weight = [1., 1., 1., 1.]
        for i_map in range(num_features_maps):
            fm_loss = criterionFeat(fake_img_Src_out[i_map], real_img_Src_out[i_map])
            fm_aug_loss = criterionFeat(fake_img_Tar_out[i_map], real_img_Tar_out[i_map])
            FM_loss += (fm_loss + fm_aug_loss) * lambda_fm_weight[i_map]
        
        G_cycle_loss_Src = nn.L1Loss()(recovery_img_Src, raw_img_Src)
        G_cycle_loss_Tar = nn.L1Loss()(recovery_img_Tar, raw_img_Tar)
        
        G_loss = fake_adv_loss * 0.05 + FM_loss + G_cycle_loss_Src*10.0 + G_cycle_loss_Tar*10.0
        
        G_adv_loss_list.append(fake_adv_loss.item() * num)
        G_FM_loss_list.append(FM_loss.item() * num)
        G_cycle_loss_Src_list.append(G_cycle_loss_Src.item() * num)
        G_cycle_loss_Tar_list.append(G_cycle_loss_Tar.item() * num)
        G_loss_list.append(G_loss.item() * num)
        
        G_loss.backward()
        optim_E.step()
        optim_G.step()
        
        # train D_Src
        optim_D_Src.zero_grad()
        
        real_img_Src_out = D_Src(raw_img_Src)
        D_Src_real_adv_loss = -torch.mean(real_img_Src_out[-2])
        
        fake_img_Src_out = D_Src(fake_img_Src.detach())
        D_Src_fake_adv_loss = torch.mean(fake_img_Src_out[-2])
        
        recovery_img_Src_out = D_Src(recovery_img_Src.detach())
        D_Src_recovery_adv_loss = -torch.mean(recovery_img_Src_out[-2])

        GP = Cal_Gradient_Penalty(D_Src, raw_img_Src.data, fake_img_Src.data, device)
        
        D_Src_loss = D_Src_real_adv_loss + D_Src_fake_adv_loss + D_Src_recovery_adv_loss + GP*10.0
        
        D_Src_real_adv_loss_list.append(D_Src_real_adv_loss.item() * num)
        D_Src_fake_adv_loss_list.append(D_Src_fake_adv_loss.item() * num)
        D_Src_recovery_adv_loss_list.append(D_Src_recovery_adv_loss.item() * num)
        D_Src_loss_list.append(D_Src_loss.item() * num)
        
        D_Src_loss.backward()
        optim_D_Src.step()
        
        # train D_Tar
        optim_D_Tar.zero_grad()
        
        real_img_Tar_out = D_Tar(raw_img_Tar)
        D_Tar_real_adv_loss = -torch.mean(real_img_Tar_out[-2])
        
        fake_img_Tar_out = D_Tar(fake_img_Tar.detach())
        D_Tar_fake_adv_loss = torch.mean(fake_img_Tar_out[-2])
        
        recovery_img_Tar_out = D_Tar(recovery_img_Tar.detach())
        D_Tar_recovery_adv_loss = -torch.mean(recovery_img_Tar_out[-2])
        
        GP = Cal_Gradient_Penalty(D_Tar, raw_img_Tar.data, fake_img_Tar.data, device)
        
        D_Tar_loss = D_Tar_real_adv_loss + D_Tar_fake_adv_loss + D_Tar_recovery_adv_loss + GP*10.0
        
        D_Tar_real_adv_loss_list.append(D_Tar_real_adv_loss.item() * num)
        D_Tar_fake_adv_loss_list.append(D_Tar_fake_adv_loss.item() * num)
        D_Tar_recovery_adv_loss_list.append(D_Tar_recovery_adv_loss.item() * num)
        D_Tar_loss_list.append(D_Tar_loss.item() * num)
        
        D_Tar_loss.backward()
        optim_D_Tar.step()
    
    lr_scheduler_E.step()
    lr_scheduler_G.step()
    lr_scheduler_D_Src.step()
    lr_scheduler_D_Tar.step()

    # Record Data
    MyGAN_record_data(
        save_dir,
        {
            "epoch": f"{epoch}",
            "num_epochs": f"{config.num_epochs}",
            "batch": f"{batch_idx+1}",
            "num_batchs": f"{len(train_loader_Tar)}",
            
            "G_adv_loss": f"{np.sum(G_adv_loss_list) / np.sum(batchsize_list)}",
            "G_FM_loss": f"{np.sum(G_FM_loss_list) / np.sum(batchsize_list)}",
            "G_cycle_loss_Src": f"{np.sum(G_cycle_loss_Src_list) / np.sum(batchsize_list)}",
            "G_cycle_loss_Tar": f"{np.sum(G_cycle_loss_Tar_list) / np.sum(batchsize_list)}",
            "G_loss": f"{np.sum(G_loss_list) / np.sum(batchsize_list)}",
            
            "D_Src_real_adv_loss": f"{np.sum(D_Src_real_adv_loss_list) / np.sum(batchsize_list)}",
            "D_Src_fake_adv_loss": f"{np.sum(D_Src_fake_adv_loss_list) / np.sum(batchsize_list)}",  
            "D_Src_recovery_adv_loss": f"{np.sum(D_Src_recovery_adv_loss_list) / np.sum(batchsize_list)}", 
            "D_Src_loss": f"{np.sum(D_Src_loss_list) / np.sum(batchsize_list)}",  
            
            "D_Tar_real_adv_loss": f"{np.sum(D_Tar_real_adv_loss_list) / np.sum(batchsize_list)}",
            "D_Tar_fake_adv_loss": f"{np.sum(D_Tar_fake_adv_loss_list) / np.sum(batchsize_list)}", 
            "D_Tar_recovery_adv_loss": f"{np.sum(D_Tar_recovery_adv_loss_list) / np.sum(batchsize_list)}",  
            "D_Tar_loss": f"{np.sum(D_Tar_loss_list) / np.sum(batchsize_list)}",  
        },
        flag_plot=True,
    )
    
    ##################################################################################
    ## Checkpoint
    ##################################################################################
    #-------------------------------------------------------------------
    # Save model E&G
    #-------------------------------------------------------------------
    if epoch % (config.num_epochs // 10) == 0:
    # if True:
        # net_E_content = f"{save_dir}/models/{epoch}_net_e_content.pth"
        # torch.save({
        #     "epoch": epoch,
        #     "model_state_dict": E_content.state_dict(),
        # }, E_content)
        
        # net_E_style = f"{save_dir}/models/{epoch}_net_e_style.pth"
        # torch.save({
        #     "epoch": epoch,
        #     "model_state_dict": E_style.state_dict(),
        # }, net_E_style)
        
        net_G_Src = f"{save_dir}/models/{epoch}_net_g_src.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": G_src.state_dict(),
        }, net_G_Src)
        
        net_G_Tar = f"{save_dir}/models/{epoch}_net_g_tar.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": G_tar.state_dict(),
        }, net_G_Tar)
    
    #-------------------------------------------------------------------
    # Save model D
    #-------------------------------------------------------------------
    if epoch == config.num_epochs:
        net_D_Src_path = f"{save_dir}/models/{epoch}_net_d_src.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": D_Src.state_dict(),
            "img_size": config.img_size,
        }, net_D_Src_path)
        net_D_Tar_path = f"{save_dir}/models/{epoch}_net_d_tar.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": D_Tar.state_dict(),
            "img_size": config.img_size,
        }, net_D_Tar_path)