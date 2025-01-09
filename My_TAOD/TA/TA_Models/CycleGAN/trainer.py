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

from model import CycleGAN_Generator, CycleGAN_Discriminator, Weights_Init_Normal, Lambda_Learing_Rate, Replay_Buffer

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader

sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import CycleGAN_logger
from TA_utils import CycleGAN_record_data 

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Save Directories
parser.add_argument("--save_rootdir", type=str, default="My_TAOD/TA/TA_Results/CycleGAN")
# Tar data loader
parser.add_argument("--Tar_dataset_path",
                    type=str,
                    default="My_TAOD/dataset/DeepPCB_Crop/10-shot/train/0.csv")
# Src data loader
parser.add_argument("--Src_dataset_path",
                    type=str,
                    default="My_TAOD/dataset/PCB_200/200-shot/train/0.csv")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)
# train
parser.add_argument("--num_epochs", type=int, default=3000)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--decay_epoch", type=int, default=1500)

parser.add_argument("--img_size", type=int, default=32)

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

G_Src2Tar = CycleGAN_Generator(3, 3)
G_Tar2Src = CycleGAN_Generator(3, 3)
D_Src = CycleGAN_Discriminator(3)
D_Tar = CycleGAN_Discriminator(3)

G_Src2Tar.apply(Weights_Init_Normal)
G_Tar2Src.apply(Weights_Init_Normal)
D_Src.apply(Weights_Init_Normal)
D_Tar.apply(Weights_Init_Normal)

device = 'cuda:' + config.gpu_id
G_Src2Tar.to(device)
G_Tar2Src.to(device)
D_Src.to(device)
D_Tar.to(device)


optim_G = torch.optim.Adam(itertools.chain(G_Src2Tar.parameters(), G_Tar2Src.parameters()), lr=config.lr_g, betas=(0.5, 0.999))
optim_D_Src = torch.optim.Adam(D_Src.parameters(), lr=config.lr_d, betas=(0.5, 0.999))
optim_D_Tar = torch.optim.Adam(D_Tar.parameters(), lr=config.lr_d, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=Lambda_Learing_Rate(config.num_epochs, config.start_epoch, config.decay_epoch).step)
lr_scheduler_D_Src = torch.optim.lr_scheduler.LambdaLR(optim_D_Src, lr_lambda=Lambda_Learing_Rate(config.num_epochs, config.start_epoch, config.decay_epoch).step)
lr_scheduler_D_Tar = torch.optim.lr_scheduler.LambdaLR(optim_D_Tar, lr_lambda=Lambda_Learing_Rate(config.num_epochs, config.start_epoch, config.decay_epoch).step)


fake_Src_buffer = Replay_Buffer()
fake_Tar_buffer = Replay_Buffer()

Tensor = torch.cuda.FloatTensor


######################################################################################################
#### Train
######################################################################################################
# logger
CycleGAN_logger(config, save_dir)
# train
for epoch in tqdm.trange(config.start_epoch + 1, config.num_epochs + 1, desc=f"[Epoch:{config.num_epochs}]On training"):
    # loss_list
    batchsize_list = []

    G_adv_loss_Src2Tar_list = []
    G_adv_loss_Tar2Src_list = []
    G_cycle_loss_Src2Tar2Src_list = []
    G_cycle_loss_Tar2Src2Tar_list = []
    G_identity_loss_Src_list = []
    G_identity_loss_Tar_list = []
    G_loss_list = []
    
    D_Src_real_adv_loss_list = []
    D_Src_fake_adv_loss_list = []
    D_Src_loss_list = []
    
    D_Tar_real_adv_loss_list = []
    D_Tar_fake_adv_loss_list = []
    D_Tar_loss_list = []
    
    for batch_idx, ((raw_img_Src, category_label_Src), (raw_img_Tar, category_label_Tar)) in enumerate(zip(train_loader_Src, train_loader_Tar)):
        
        num = raw_img_Tar.size(0)
        batchsize_list.append(num)
        
        raw_img_Src, category_label_Src = Variable(raw_img_Src).to(device), Variable(category_label_Src).to(device).view(num)
        raw_img_Tar, category_label_Tar = Variable(raw_img_Tar).to(device), Variable(category_label_Tar).to(device).view(num)
        
        raw_img_Src = Tensor(num, 3, config.img_size, config.img_size).copy_(raw_img_Src)
        raw_img_Tar = Tensor(num, 3, config.img_size, config.img_size).copy_(raw_img_Tar)
        real_labels = Variable(Tensor(num).fill_(1.0), requires_grad=False)
        fake_labels = Variable(Tensor(num).fill_(0.0), requires_grad=False)
        
        # train G
        optim_G.zero_grad()
        
        # adv_loss
        fake_Tar = G_Src2Tar(raw_img_Src)
        pred_fake = D_Tar(fake_Tar)
        G_adv_loss_Src2Tar = nn.MSELoss()(pred_fake, real_labels)
        
        fake_Src = G_Tar2Src(raw_img_Tar)
        pred_fake = D_Src(fake_Src)
        G_adv_loss_Tar2Src = nn.MSELoss()(pred_fake, real_labels)
        
        # cycle_loss
        recovered_Src = G_Tar2Src(fake_Tar)
        G_cycle_loss_Src2Tar2Src = nn.L1Loss()(recovered_Src, raw_img_Src) * 10.0
        
        recovered_Tar = G_Src2Tar(fake_Src)
        G_cycle_loss_Tar2Src2Tar = nn.L1Loss()(recovered_Tar, raw_img_Tar) * 10.0 
        
        # identity_loss
        same_Src = G_Tar2Src(raw_img_Src)
        G_identity_loss_Src = nn.L1Loss()(same_Src, raw_img_Src) * 5.0
        
        same_Tar = G_Src2Tar(raw_img_Tar)
        G_identity_loss_Tar = nn.L1Loss()(same_Tar, raw_img_Tar) * 5.0
        
        G_loss = G_adv_loss_Src2Tar + G_adv_loss_Tar2Src + G_cycle_loss_Src2Tar2Src +  G_cycle_loss_Tar2Src2Tar + G_identity_loss_Src + G_identity_loss_Tar
        
        G_adv_loss_Src2Tar_list.append(G_adv_loss_Src2Tar.item() * num)
        G_adv_loss_Tar2Src_list.append(G_adv_loss_Tar2Src.item() * num)
        G_cycle_loss_Src2Tar2Src_list.append(G_cycle_loss_Src2Tar2Src.item() * num)
        G_cycle_loss_Tar2Src2Tar_list.append(G_cycle_loss_Tar2Src2Tar.item() * num)
        G_identity_loss_Src_list.append(G_identity_loss_Src.item() * num)
        G_identity_loss_Tar_list.append(G_identity_loss_Tar.item() * num)
        G_loss_list.append(G_loss.item() * num)
        
        G_loss.backward()
        optim_G.step()
        
        # train D_Src
        optim_D_Src.zero_grad()
        
        pred_real = D_Src(raw_img_Src)
        D_Src_real_adv_loss = nn.MSELoss()(pred_real, real_labels)
        
        fake_Src = fake_Src_buffer.push_and_pop(fake_Src)
        pred_fake = D_Src(fake_Src.detach())
        D_Src_fake_adv_loss = nn.MSELoss()(pred_fake, fake_labels)
        
        D_Src_loss = (D_Src_real_adv_loss + D_Src_fake_adv_loss) * 0.5
        
        D_Src_real_adv_loss_list.append(D_Src_real_adv_loss.item() * num)
        D_Src_fake_adv_loss_list.append(D_Src_fake_adv_loss.item() * num)
        D_Src_loss_list.append(D_Src_loss.item() * num)
        
        D_Src_loss.backward()
        optim_D_Src.step()
        
        # train D_Tar
        optim_D_Tar.zero_grad()
        
        pred_real = D_Tar(raw_img_Tar)
        D_Tar_real_adv_loss = nn.MSELoss()(pred_real, real_labels)
        
        fake_Tar = fake_Tar_buffer.push_and_pop(fake_Tar)
        pred_fake = D_Tar(fake_Tar.detach())
        D_Tar_fake_adv_loss = nn.MSELoss()(pred_fake, fake_labels)
        
        D_Tar_loss = (D_Tar_real_adv_loss + D_Tar_fake_adv_loss) * 0.5
        
        D_Tar_real_adv_loss_list.append(D_Tar_real_adv_loss.item() * num)
        D_Tar_fake_adv_loss_list.append(D_Tar_fake_adv_loss.item() * num)
        D_Tar_loss_list.append(D_Tar_loss.item() * num)
        
        D_Tar_loss.backward()
        optim_D_Tar.step()
        
    lr_scheduler_G.step()
    lr_scheduler_D_Src.step()
    lr_scheduler_D_Tar.step()
    
    # Record Data
    CycleGAN_record_data(
        save_dir,
        {
            "epoch": f"{epoch}",
            "num_epochs": f"{config.num_epochs}",
            "batch": f"{batch_idx+1}",
            "num_batchs": f"{len(train_loader_Tar)}",
            
            "G_adv_loss_Src2Tar": f"{np.sum(G_adv_loss_Src2Tar_list) / np.sum(batchsize_list)}",
            "G_adv_loss_Tar2Src": f"{np.sum(G_adv_loss_Tar2Src_list) / np.sum(batchsize_list)}",
            "G_cycle_loss_Src2Tar2Src": f"{np.sum(G_cycle_loss_Src2Tar2Src_list) / np.sum(batchsize_list)}",
            "G_cycle_loss_Tar2Src2Tar": f"{np.sum(G_cycle_loss_Tar2Src2Tar_list) / np.sum(batchsize_list)}",
            "G_identity_loss_Src": f"{np.sum(G_identity_loss_Src_list) / np.sum(batchsize_list)}",
            "G_identity_loss_Tar": f"{np.sum(G_identity_loss_Tar_list) / np.sum(batchsize_list)}",
            "G_loss": f"{np.sum(G_loss_list) / np.sum(batchsize_list)}",
            
            "D_Src_real_adv_loss": f"{np.sum(D_Src_real_adv_loss_list) / np.sum(batchsize_list)}",
            "D_Src_fake_adv_loss": f"{np.sum(D_Src_fake_adv_loss_list) / np.sum(batchsize_list)}",   
            "D_Src_loss": f"{np.sum(D_Src_loss_list) / np.sum(batchsize_list)}",  
            
            "D_Tar_real_adv_loss": f"{np.sum(D_Tar_real_adv_loss_list) / np.sum(batchsize_list)}",
            "D_Tar_fake_adv_loss": f"{np.sum(D_Tar_fake_adv_loss_list) / np.sum(batchsize_list)}",   
            "D_Tar_loss": f"{np.sum(D_Tar_loss_list) / np.sum(batchsize_list)}",  
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
        net_G_Src2Tar_path = f"{save_dir}/models/{epoch}_net_g_src2tar.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": G_Src2Tar.state_dict(),
        }, net_G_Src2Tar_path)
        net_G_Tar2Src_path = f"{save_dir}/models/{epoch}_net_g_tar2src.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": G_Tar2Src.state_dict(),
        }, net_G_Tar2Src_path)
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