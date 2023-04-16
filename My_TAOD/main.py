import os
import argparse
from torch.backends import cudnn
from data_loader import get_loader
from train import trainer
from models import Resnet18

# torch.backends.cudnn:提供对Nvidia cuDNN的支持，可加速深度学习模型在GPU上的计算速度

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training on GPUs
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

    # Data loader           
    if config.dataset in ['NEU_CLS']:
        train_iter_loader, test_iter_loader = get_loader(config.batch_size, config.dataset, config.mode, config.num_workers, config.tt_rate)

    if config.mode == 'train':
        if config.dataset in ['NEU_CLS']:
            trainer(Resnet18, train_iter_loader, test_iter_loader, config.epochs, config.lr, config.device)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model Configuration
    parser.add_argument('--net', type=str, default='Resnet18', choices=['Resnet18'])
    
    # Training Configuration
    parser.add_argument('--dataset', type=str, default='NEU_CLS', choices=['NEU_CLS'], help="Choose datasets")
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--tt_rate', type=float, default=0.7, help="Ratio of training set")
    
    # Others Configuration
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    
    # Directories
    parser.add_argument('--log_dir', type=str, default='./My_TAOD/logs')
    parser.add_argument('--model_save_dir', type=str, default='./My_TAOD/models')
    parser.add_argument('--sample_dir', type=str, default='./My_TAOD/samples')
    parser.add_argument('--result_dir', type=str, default='./My_TAOD/results')
    
    config = parser.parse_args()
    print(config)
    
    main(config)
    
    