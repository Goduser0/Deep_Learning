# 用在源域上预训练的GAN
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
from TA_G import PFS_Generator
from TA_D import PFS_Discriminator_patch, PFS_Discriminator
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import baseline_finetuning_logger
from TA_utils import PFS_baseline_finetuning_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument("--logs_dir", type=str,
                    default="My_TAOD/TA/TA_Results/PFS_baseline_finetuning/logs")
parser.add_argument("--models_dir", type=str,
                    default="My_TAOD/TA/TA_Results/PFS_baseline_finetuning/models")
parser.add_argument("--samples_dir", type=str,
                    default="My_TAOD/TA/TA_Results/PFS_baseline_finetuning/samples")
parser.add_argument("--results_dir", type=str,
                    default="My_TAOD/TA/TA_Results/PFS_baseline_finetuning/results")

# random seed
parser.add_argument("--random_seed", type=int, default=1)

# data loader 
parser.add_argument("--dataset_class",
                    type=str,
                    default='PCB_Crop',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )#*
parser.add_argument("--data_path",
                    type=str,
                    default="My_TAOD/dataset/PCB_Crop/30-shot/train/0.csv")#*
parser.add_argument("--category",
                    type=str,
                    default="0")#*
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=30)

# G
parser.add_argument("--lr_g", type=float, default=2e-4)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--G_init_class", type=str, default="DeepPCB_Crop")#*
parser.add_argument("--G_init_path", type=str, default="My_TAOD/TA/TA_Results/baseline_from_scratch/models/DeepPCB_Crop 0 2023-10-31_14-29-13/1000_net_g.pth")#*

# D
parser.add_argument("--lr_d", type=float, default=2e-4)

# train
parser.add_argument("--num_epochs", type=int, default=500)
parser.add_argument("--gpu_id", type=str, default="1")

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
baseline_finetuning_logger(config)

models_save_path = config.models_dir + '/' + config.dataset_class + "_from_" + config.G_init_class + ' ' + config.category + ' ' + config.time
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
G = PFS_Generator(z_dim=config.z_dim)
checkpoint = torch.load(config.G_init_path)
G.load_state_dict(checkpoint["model_state_dict"])
G = G.cuda()
D = PFS_Discriminator_patch().cuda()
# optimizers
optim_G = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.5, 0.9))
optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=config.lr_d, betas=(0.5, 0.9))
######################################################################################################
#### Train
######################################################################################################
for epoch in tqdm.trange(1, config.num_epochs+1, desc=f"On training"):
    G_loss_list = []
    D_loss_list = []
    for batch_idx, (raw_img, category_label) in enumerate(data_iter_loader):
        optim_G.zero_grad()
        num = raw_img.size(0)
        
        raw_img, category_label = Variable(raw_img.cuda()), Variable(category_label.cuda())
        for _ in range(1):
            z = Variable(torch.randn(num, config.z_dim).cuda())
            optim_D.zero_grad()
            fake_img = G(z)
            real_img_loss, real_patch_loss = D(raw_img)
            fake_img_loss, fake_patch_loss = D(fake_img)
            
            D_img_loss = nn.ReLU()(1.0 - real_img_loss).mean() + nn.ReLU()(1.0 + fake_img_loss).mean()
            D_patch_loss = nn.ReLU()(1.0 - real_patch_loss).mean() + nn.ReLU()(1.0 + fake_patch_loss).mean()
            
            D_loss = (D_img_loss + D_patch_loss) * 0.5
            D_loss_list.append(D_loss.item())
            D_loss.backward()
            optim_D.step()
        
        optim_G.zero_grad()
        fake_img_loss, fake_patch_loss = D(G(z))
        G_img_loss, G_patch_loss = -fake_img_loss.mean(), -fake_patch_loss.mean()
        G_loss = (G_patch_loss + G_img_loss) * 0.5
        G_loss_list.append(G_loss.item())
        G_loss.backward()
        optim_G.step()
            
    # Show
    print(
        "[Epoch %d/%d] [Batch %d/%d] [G_loss: %.5f] [D_loss: %.5f]"
        %
        (epoch, config.num_epochs, batch_idx+1, len(data_iter_loader), np.mean(G_loss_list), np.mean(D_loss_list))
    )
    
    # Record Data
    PFS_baseline_finetuning_record_data(config,
                                        {
                                            "epoch": f"{epoch}",
                                            "num_epochs": f"{config.num_epochs}",
                                            "batch": f"{batch_idx+1}",
                                            "num_batchs": f"{len(data_iter_loader)}",
                                            "G_loss": f"{np.mean(G_loss_list)}",
                                            "D_loss": f"{np.mean(D_loss_list)}",
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