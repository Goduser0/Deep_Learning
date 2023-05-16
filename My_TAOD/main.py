import os
import argparse
from torch.backends import cudnn
from dataset_loader import get_loader
from train import trainer
from models import Resnet18
from logger import logger

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

    train_iter_loader = None
    test_iter_loader = None
    # Data loader           
    if config.dataset_class in dataset_list:
        train_iter_loader = get_loader(config.dataset_class, config.dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True)
        test_iter_loader = get_loader(config.dataset_class, config.dataset_test_dir, config.test_batch_size, config.num_workers, shuffle=False)
        
    if config.mode == 'train':
        if config.dataset_class in dataset_list:
            trainer(Resnet18, train_iter_loader, test_iter_loader, config.epochs, config.lr, config.device)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model Configuration
    parser.add_argument('--net', type=str, default='Resnet18', choices=['Resnet18'])
    
    # Training Configuration
    dataset_list = ['NEU_CLS', 'elpv']
    parser.add_argument('--dataset_class', type=str, default='NEU_CLS', choices=dataset_list, help="Choose datasets")
    parser.add_argument('--dataset_train_dir', default='/home/zhouquan/MyDoc/DL_Learning/My_TAOD/dataset/NEU_CLS/210-shot/train.csv', type=str)
    parser.add_argument('--dataset_test_dir', default='/home/zhouquan/MyDoc/DL_Learning/My_TAOD/dataset/NEU_CLS/210-shot/test.csv', type=str)
    parser.add_argument('--train_batch_size', type=int, default=64, help='Mini-batch size of train')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Mini-batch size of test')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    
    # Others Configuration
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    
    # Directories
    parser.add_argument('--log_dir', type=str, default='./My_TAOD/logs')
    parser.add_argument('--model_save_dir', type=str, default='./My_TAOD/models')
    parser.add_argument('--sample_dir', type=str, default='./My_TAOD/samples')
    parser.add_argument('--result_dir', type=str, default='./My_TAOD/results')
    
    # Logger
    parser.add_argument('--log', type=bool, default=True)
    
    config = parser.parse_args()
    print(config)
    
    main(config)
    logger(config)