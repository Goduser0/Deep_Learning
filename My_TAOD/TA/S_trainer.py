import torch
from torch.autograd import Variable
import torchvision.transforms as T
import torch.nn as nn

import numpy as np

import os
import argparse
import random

from TA_G import FeatureMatchGenerator
from TA_D import FeatureMatchPatchDiscriminator, Extra
from TA_utils import requires_grad, mix_noise, get_subspace

import sys
sys.path.append("/home/zhouquan/MyDoc/Deep_Learning/My_TAOD")
from dataset_loader import get_loader

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# random seed
parser.add_argument("--random_seed", type=int, default=1)

parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--gpu_id", type=str, default="0")

# data load
dataset_list = ['NEU_CLS', 'elpv', 'Magnetic_Tile']
parser.add_argument("--dataset_class", type=str, default='NEU_CLS', choices=dataset_list)
parser.add_argument("--data_path", 
                    type=str, 
                    default="/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/NEU_CLS/210-shot/train/0.csv")
parser.add_argument("--num_workers", type=int, default=2)

# g_source && d_source
parser.add_argument("--gan_type", type=str, default="")
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--conv_dim", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--subspace_freq", type=int, default=4)

# g_source
parser.add_argument("--n_train", type=int, default=210)
parser.add_argument("--n_sample", type=int, default=210)
parser.add_argument("--n_mlp", type=int, default=3)

parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--subspace_std", type=float, default=0.1)
parser.add_argument("--mix_prob", type=float, default=0.9)

parser.add_argument("--lr_mlp", type=float, default=0.01)
parser.add_argument("--g_reg_every", type=int, default=4)

# d_source
parser.add_argument("--d_reg_every", type=int, default=16)

# Saved Directories
parser.add_argument('--logs_dir', 
                    type=str, 
                    default='/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/logs')
parser.add_argument('--models_dir', 
                    type=str, 
                    default='/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/models')
parser.add_argument('--samples_dir', 
                    type=str, 
                    default='/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/samples')
parser.add_argument('--results_dir', 
                    type=str, 
                    default='/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/results')

# config
config = parser.parse_args()


########################################################################################################
#### Setting
########################################################################################################
# random seed
random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

# device
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

# data_loader
trans = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.ToTensor(), 
        T.Resize(64),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ]
)
data_iter_loader = get_loader(config.dataset_class, 
                              config.data_path, 
                              config.batch_size, 
                              config.num_workers, 
                              shuffle=True, 
                              transforms=trans,
                              img_type='PIL'
                              )

# model
g_source = FeatureMatchGenerator(
    config.n_mlp, config.img_size, config.z_dim, config.conv_dim, config.lr_mlp
).cuda()
d_source = FeatureMatchPatchDiscriminator(
    config.img_size, config.conv_dim
).cuda()
extra = Extra().cuda()

# optimizers
g_reg_ratio = config.g_reg_every / (config.g_reg_every + 1)
d_reg_ratio = config.d_reg_every / (config.d_reg_every + 1)

g_optim = torch.optim.Adam(
    g_source.parameters(),
    lr = config.lr * g_reg_ratio,
    betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
)

d_optim = torch.optim.Adam(
    d_source.parameters(),
    lr=config.lr * d_reg_ratio,
    betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
)

e_optim = torch.optim.Adam(
    extra.parameters(),
    lr=config.lr * d_reg_ratio,
    betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
)


########################################################################################################
#### Train
########################################################################################################
init_z = torch.randn(config.n_train, config.z_dim).cuda() # (n_train, z_dim)
sample_z = torch.randn(config.n_sample, config.z_dim).cuda() # (n_sample, z_dim)
sub_region_z = get_subspace(config, init_z.clone(), vis_flag=True) # (B, z_dim)


kl_loss = nn.KLDivLoss()
cos_sim = nn.CosineSimilarity()


for epoch in range(1, config.num_epochs + 1):
    for i, data in enumerate(data_iter_loader):
        z_flag = i % config.subspace_freq
        
        real_img = data[0].cuda()
        label = data[1]

        #########################################
        #### train generator()
        #########################################
        g_optim.zero_grad()
        if z_flag > 0:
            # sampling normally
            noise = mix_noise(config.batch_size, config.z_dim, config.mix_prob)
        else:
            # sampling from anchor
            noise = [get_subspace(config, init_z.clone())]
        noise = noise[0].cuda()
        fake_img = g_source(noise)
        
        real_features = d_source(real_img, extra=extra, flag=z_flag, p_ind=np.random.randint(0, 4))
        fake_features = d_source(fake_img, extra=extra, flag=z_flag, p_ind=np.random.randint(0, 4))
        break
    break