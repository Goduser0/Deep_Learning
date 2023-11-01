import warnings
warnings.filterwarnings("ignore")

import argparse
import random
import numpy as np
import os
import tqdm
import time
import matplotlib.pyplot as plt
from skimage import color

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
from TA_D import PFS_Discriminator_patch
from TA_VAE import PFS_Encoder
from TA_layers import vgg, PFS_Relation
sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import stage2_logger
from TA_utils import stage2_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument("--logs_dir", type=str,
                    default="My_TAOD/TA/TA_Results/stage2/logs")
parser.add_argument("--models_dir", type=str,
                    default="My_TAOD/TA/TA_Results/stage2/models")
parser.add_argument("--samples_dir", type=str,
                    default="My_TAOD/TA/TA_Results/stage2/samples")
parser.add_argument("--results_dir", type=str,
                    default="My_TAOD/TA/TA_Results/stage2/results")

# random seed
parser.add_argument("--random_seed", type=int, default=1)

# data loader
parser.add_argument("--dataset_T_class",
                    type=str,
                    default='PCB_200',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )
parser.add_argument("--train_T_path",
                    type=str,
                    default="My_TAOD/dataset/PCB_200/30-shot/train/0.csv")
parser.add_argument("--test_T_path",
                    type=str,
                    default="My_TAOD/dataset/PCB_200/30-shot/test/0.csv")

parser.add_argument("--dataset_S_class",
                    type=str,
                    default='DeepPCB_Crop',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )
parser.add_argument("--dataset_S_path",
                    type=str,
                    default="My_TAOD/dataset/DeepPCB_Crop/160-shot/train/0.csv")

parser.add_argument("--category",
                    type=str,
                    default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=30)


# train
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--gpu_id", type=str, default="0")

# G
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--z_dim", type=int, default=128)
parser.add_argument("--lr_g", type=float, default=2e-4)
parser.add_argument("--G_init_path", type=str, required=True)


# D
parser.add_argument("--lr_d", type=float, default=2e-4)
parser.add_argument("--D_iters", type=int, default=5)
# R
parser.add_argument("--lr_r", type=float, default=2e-4)

# VAE
parser.add_argument("--latent_dim", type=int, default=64)
parser.add_argument("--lr_vae", type=float, default=5e-4)
parser.add_argument("--Encoder_init_path", type=str, required=True)

# loss_ratio
parser.add_argument('--recon_ratio', type=float, default=1.0)
parser.add_argument('--gan_ratio', type=float, default=1.0)
parser.add_argument('--relation_ratio', type=float, default=1.0)

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
stage2_logger(config)

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

train_T_iter_loader = get_loader(
    config.dataset_T_class,
    config.train_T_path,
    config.batch_size,
    config.num_workers,
    shuffle=True,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)

test_T_iter_loader = get_loader(
    config.dataset_T_class,
    config.test_T_path,
    config.batch_size,
    config.num_workers,
    shuffle=False,
    trans=trans,
    img_type='ndarray',
    drop_last=False,
)

# model
T_G, S_G = torch.load(config.G_init_path), torch.load(config.G_init_path)
optim_G  = torch.optim.Adam(T_G.parameters(), lr=config.lr_g, betas=(0.5, 0.9))

Encoder_c, Encoder_s = torch.load(config.Encoder_init_path), torch.load(config.Encoder_init_path)
optim_Encoder_s = torch.optim.Adam(Encoder_s.parameters(), lr=config.lr_vae, betas=(0.5, 0.9))

# Discriminator
if config.gan_ratio != 0:
    T_D_patch = PFS_Discriminator_patch(in_channels=3).cuda()
    optim_D_patch = torch.optim.Adam(filter(lambda p: p.requires_grad, T_D_patch.parameters()), lr=config.lr_d, betas=(0.0,0.9))

# Relation
if config.relation_ratio != 0:
    R = PFS_Relation().cuda()
    R.set(Encoder_c)
    optim_R = torch.optim.Adam(R.model_bn1_B.parameters(), lr=config.lr_r, betas=(0.5, 0.9))

# Feature Extractor    
F =vgg().cuda().eval()

### Lab shuffle
lab = []
def preprocessing(data):
    new_data = data.clone()
    global iters, lab
    # Record min/max a/b
    if iters == 0:
        max_a, min_a, max_b, min_b = -10000, 10000, -10000, 10000
        for b in range(new_data.size(0)):
            data2 = new_data[b].cpu().data.numpy().transpose(1,2,0).astype(float) #*255
            data2 = color.rgb2lab(data2)
            max_a, min_a = max(max_a, np.max(data2[:,:,1])),  min(min_a, np.min(data2[:,:,1]))
            max_b, min_b = max(max_b, np.max(data2[:,:,2])),  min(min_b, np.min(data2[:,:,2]))
        lab = [[min_a, max_a], [min_b, max_b]]
    count = iters
    # Shuffle
    for b in range(new_data.size(0)):
        data2 = new_data[b].cpu().data.numpy().transpose(1,2,0).astype(float) #*255
        data2 = color.rgb2lab(data2)
        max_a, max_b = np.max(data2[:,:,1]), np.max(data2[:,:,2])
        min_a, min_b = np.min(data2[:,:,1]), np.min(data2[:,:,2])

        rand_a = (np.random.choice(201)*0.01-1)*(lab[0][1]-lab[0][0])*0.5
        rand_b = (np.random.choice(201)*0.01-1)*(lab[1][1]-lab[1][0])*0.5

        data2[:,:,1] = np.clip(data2[:,:,1] + rand_a, lab[0][0], lab[0][1])
        data2[:,:,2] = np.clip(data2[:,:,2] + rand_b, lab[1][0], lab[1][1])
        data2 = color.lab2rgb(data2) 
        new_data[b] = (torch.Tensor(data2.transpose(2,0,1)).cuda()-0.5)*2

    return torch.clamp(new_data, -1.0, 1.0)

iters = 0

z_fixed = torch.randn(64, 128).cuda()
zc = torch.randn(64, 64)
z_same_zs1, z_same_zs2 = torch.cat([torch.randn(1, 64).repeat(64,1), zc], 1).cuda(), torch.cat([torch.randn(1,64).repeat(64,1), zc], 1).cuda()
