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
from TA_G import PFS_Generator
from TA_D import PFS_Discriminator_patch, PFS_Discriminator

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# random seed
parser.add_argument("--random_seed", type=int, default=1)

# data loader 
parser.add_argument("--dataset_class",
                    type=str,
                    default='PCB_200',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )
parser.add_argument("--data_path",
                    type=str,
                    default="./My_TAOD/dataset/PCB_200/0.7-shot/train/0.csv")
parser.add_argument("--catagory",
                    type=str,
                    default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=4)
# G
parser.add_argument("--lr_g", type=float, default=1e-4)
# D
parser.add_argument("--lr_d", type=float, default=1e-3)

# train
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--gpu_id", type=str, default="0")

parser.add_argument("--img_size", type=int, default=128)

# Others
parser.add_argument("--time", type=str, default=time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime()))

# config
config = parser.parse_args()

# 一致性检验
assert(config.data_path.split('/')[-4]==config.dataset_class
       and
       config.data_path.split('/')[-1][0]==config.catagory)

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
G = PFS_Generator(z_dim=128).cuda()
D = PFS_Discriminator_patch().cuda()
# optimizers
optim_G = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=(0.5, 0.9))
optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=config.lr_d, betas=(0.5, 0.9))
######################################################################################################
#### Train
######################################################################################################
for epoch in tqdm.trange(1, config.num_epochs+1, desc=f"On training"):
    pass 