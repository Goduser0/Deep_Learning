import os
import sys
import time
import argparse

import torch
import torchvision.transforms as T
from torch.backends import cudnn

from My_TAOD.TA.TA_utils import init_random_seed

def main(config):
    # Init random seed
    init_random_seed(config.manual_seed)
    
    # For fast training on GPUs
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    # Create directories if not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir) 
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer Augumation")
    
    # Directories
    parser.add_argument('--log_dir', type=str, default='./My_TAOD/TA/logs')
    parser.add_argument('--model_save_dir', type=str, default='./My_TAOD/TA/models')
    parser.add_argument('--sample_dir', type=str, default='./My_TAOD/TA/samples')
    parser.add_argument('--result_dir', type=str, default='./My_TAOD/TA/results')
    
    # Others
    parser.add_argument('--time', type=str, default=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    parser.add_argument('--manual_seed', type=int, default=None)
    
    # Parser
    TA_config  = parser.parse_args()
    
    main(TA_config)