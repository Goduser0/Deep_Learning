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
from TA_G import UNIT_Generator
from TA_D import UNIT_Discriminator
from TA_VAE import UNIT_Encoder


sys.path.append("./My_TAOD/TA/TA_Utils")
# from TA_logger import cogan_logger
# from TA_utils import cogan_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument("--logs_dir", type=str,
                    default="My_TAOD/TA/TA_Results/UNIT/logs")
parser.add_argument("--models_dir", type=str,
                    default="My_TAOD/TA/TA_Results/UNIT/models")
parser.add_argument("--samples_dir", type=str,
                    default="My_TAOD/TA/TA_Results/UNIT/samples")
parser.add_argument("--results_dir", type=str,
                    default="My_TAOD/TA/TA_Results/UNIT/results")

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
                    default="My_TAOD/dataset/DeepPCB_Crop/1.0-shot/train/0.csv")

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
parser.add_argument("--S_batch_size", type=int, default=32)
parser.add_argument("--T_batch_size", type=int, default=10)

# train
parser.add_argument("--num_epochs", type=int, default=2000)
parser.add_argument("--gpu_id", type=str, default="0")

# G
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--lr_g", type=float, default=2e-4)

# D
parser.add_argument("--lr_d", type=float, default=2e-4)
parser.add_argument("--D_iters", type=int, default=1)

# Encoder
parser.add_argument("--lr_encoder", type=float, default=2e-4)
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
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

S_iter_loader = get_loader(
    config.dataset_S_class,
    config.dataset_S_path,
    config.S_batch_size,
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)

T_iter_loader = get_loader(
    config.dataset_T_class,
    config.dataset_T_path,
    config.T_batch_size,
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)

# model
D_Src = UNIT_Discriminator().cuda()
D_Tar = UNIT_Discriminator().cuda()
G = UNIT_Generator(config.z_dim).cuda()
Encoder = UNIT_Encoder().cuda()

optim_D_Src = torch.optim.Adam(filter(lambda p: p.requires_grad, D_Src.parameters()), lr=config.lr_d, betas=(0.0,0.9))
optim_D_Tar = torch.optim.Adam(filter(lambda p: p.requires_grad, D_Tar.parameters()), lr=config.lr_d, betas=(0.0,0.9))
optim_G = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.5,0.9))
optim_Encoder = torch.optim.Adam(Encoder.parameters(), lr=config.lr_encoder, betas=(0.5, 0.9))

for epoch in tqdm.trange(1, config.num_epochs + 1, desc="On training"):
    genS_loss_list = []
    discS_loss_list = []
    genT_loss_list = []
    discT_loss_list = []
    for batch_idx, (T_imgs, T_labels) in enumerate(T_iter_loader):
        num = T_imgs.size(0)
        