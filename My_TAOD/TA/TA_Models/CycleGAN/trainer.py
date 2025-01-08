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
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

from model import CycleGAN_Generator, CycleGAN_Discriminator, Weights_Init_Normal

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

parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=4)
# train
parser.add_argument("--num_epochs", type=int, default=10000)
parser.add_argument("--gpu_id", type=str, default="0")

parser.add_argument("--img_size", type=int, default=64)

parser.add_argument("--lr_g", type=float, default=2e-4)
parser.add_argument("--weight_decay_g", type=float, default=5e-4)

parser.add_argument("--lr_d", type=float, default=2e-4)
parser.add_argument("--weight_decay_d", type=float, default=5e-4)
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


optim_G = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.5, 0.999), weight_decay=config.weight_decay_g)
optim_D = torch.optim.Adam(D.parameters(), lr=config.lr_d, betas=(0.5, 0.999), weight_decay=config.weight_decay_d)

######################################################################################################
#### Train
######################################################################################################
# logger
CycleGAN_logger(config, save_dir)
# train
for epoch in tqdm.trange(1, config.num_epochs + 1, desc=f"[Epoch:{config.num_epochs}]On training"):
    # loss_list
    batchsize_list = []

    D_real_adv_loss_list = []
    D_fake_adv_loss_list = []
    D_mse_loss_list = []
    D_cls_loss_list = []
    D_loss_list = []
    
    G_fake_adv_loss_list = []
    G_loss_list = [] 
    
    for batch_idx, ((raw_img_Src, category_label_Src), (raw_img_Tar, category_label_Tar)) in enumerate(zip(train_loader_Src, train_loader_Tar)):
        
        num = raw_img_Tar.size(0)
        batchsize_list.append(num)
        
        raw_img_Src, category_label_Src = Variable(raw_img_Src).to(device), Variable(category_label_Src).to(device).view(num)
        raw_img_Tar, category_label_Tar = Variable(raw_img_Tar).to(device), Variable(category_label_Tar).to(device).view(num)
        
        # train D
        z = Variable(torch.randn(num, 128)).to(device)
        
        optim_D.zero_grad()
        
        real_img_out, real_feat_Src, real_feat_Tar = D(raw_img_Src, raw_img_Tar)
        D_real_adv_loss = F.cross_entropy(real_img_out, Variable(torch.LongTensor(np.ones(num*2, dtype=np.int64))).to(device))
        _, real_img_pred = torch.max(real_img_out.data, 1)
        real_img_acc = (real_img_pred == 1).sum() / (1.0*real_img_pred.size(0))
              
        fake_img_Src, fake_img_Tar = G(z)
        fake_img_out, fake_feat_Src, fake_feat_Tar = D(fake_img_Src, fake_img_Tar)
        D_fake_adv_loss = F.cross_entropy(fake_img_out, Variable(torch.LongTensor(np.zeros(num*2, dtype=np.int64))).to(device))
        _, fake_img_pred = torch.max(fake_img_out.data, 1)
        fake_img_acc = (fake_img_pred == 0).sum() / (1.0*fake_img_pred.size(0))
        
        dummy_tensor = Variable(torch.zeros_like(fake_feat_Src).to(device))
        D_mse_loss = (nn.MSELoss()(fake_feat_Src - fake_feat_Tar, dummy_tensor)) * fake_feat_Src.size(1) * fake_feat_Src.size(2) * fake_feat_Src.size(3)
        
        cls_output = D.classify_a(raw_img_Src)
        D_cls_loss = F.cross_entropy(cls_output, category_label_Src)
        _, cls_pred = torch.max(cls_output.data, 1)
        cls_acc = (cls_pred == category_label_Src.data).sum() / (1.0*cls_pred.size(0))
        
        D_loss = D_real_adv_loss + D_fake_adv_loss + D_mse_loss*1.0 + D_cls_loss*1.0
        
        D_real_adv_loss_list.append(D_real_adv_loss.item() * num)
        D_fake_adv_loss_list.append(D_fake_adv_loss.item() * num)
        D_mse_loss_list.append(D_mse_loss.item() * num)
        D_cls_loss_list.append(D_cls_loss.item() * num)
        D_loss_list.append(D_loss.item() * num)
        
        D_loss.backward()
        optim_D.step()
        
        # train G
        z = Variable(torch.randn(num, 128)).to(device)
        
        optim_G.zero_grad()
        
        fake_img_Src, fake_img_Tar = G(z)
        fake_img_out, fake_feat_Src, fake_feat_Tar = D(fake_img_Src, fake_img_Tar)
        
        G_fake_adv_loss = F.cross_entropy(fake_img_out, Variable(torch.LongTensor(np.ones(num*2, dtype=np.int64))).to(device))
        
        G_loss = G_fake_adv_loss
        
        G_fake_adv_loss_list.append(G_fake_adv_loss.item() * num)
        G_loss_list.append(G_loss.item() * num)
        
        G_loss.backward()
        optim_G.step()
        

    # Record Data
    CoGAN_record_data(
        save_dir,
        {
            "epoch": f"{epoch}",
            "num_epochs": f"{config.num_epochs}",
            "batch": f"{batch_idx+1}",
            "num_batchs": f"{len(train_loader_Tar)}",
            
            "D_real_adv_loss": f"{np.sum(D_real_adv_loss_list) / np.sum(batchsize_list)}",
            "D_fake_adv_loss": f"{np.sum(D_fake_adv_loss_list) / np.sum(batchsize_list)}",
            "D_mse_loss": f"{np.sum(D_mse_loss_list) / np.sum(batchsize_list)}",
            "D_cls_loss": f"{np.sum(D_cls_loss_list) / np.sum(batchsize_list)}",
            "D_loss": f"{np.sum(D_loss_list) / np.sum(batchsize_list)}",
            
            "G_fake_adv_loss": f"{np.sum(G_fake_adv_loss_list) / np.sum(batchsize_list)}",
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