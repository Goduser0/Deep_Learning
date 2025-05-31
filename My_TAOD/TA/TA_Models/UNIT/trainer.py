import argparse
import time
import tqdm
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNIT_Generator, UNIT_Discriminator, UNIT_Encoder

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader

sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import UNIT_logger
from TA_utils import UNIT_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Save Directories
parser.add_argument("--save_rootdir", type=str, default="My_TAOD/TA/TA_Results/UNIT")
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

parser.add_argument("--img_size", type=int, default=28)

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