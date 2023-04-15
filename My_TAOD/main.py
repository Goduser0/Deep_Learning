import os
import argparse
from torch.backends import cudnn

# torch.backends.cudnn:提供对Nvidia cuDNN的支持，可加速深度学习模型在GPU上的计算速度

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    cudnn.benchmark = True
    
    # 创建文件夹
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir) 
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model Configuration
    
    # Training Configuration
    parser.add_argument('--dataset', type=str, default='NEU_CLS', choices=['NEU_CLS'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training')
    
    # Test Configuration
    parser.add_argument('--test_iter', type=int, )
    
    # Directories
    parser.add_argument('--log_dir', type=str, default='./My_TAOD/logs')
    parser.add_argument('--model_save_dir', type=str, default='./My_TAOD/models')
    parser.add_argument('--sample_dir', type=str, default='./My_TAOD/samples')
    parser.add_argument('--result_dir', type=str, default='./My_TAOD/results')
    
    config = parser.parse_args()
    print(config)
    main(config)
    
    