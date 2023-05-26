import os
import sys
import argparse

from torch.backends import cudnn
# torch.backends.cudnn:提供对Nvidia cuDNN的支持，可加速深度学习模型在GPU上的计算速度

from dataset_loader import get_loader
from train import classification_trainer
from models import classification_net_select
from logger import logger


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
    
    # 参数校验
    if config.dataset_train_dir.split("/")[-3] != config.dataset_class or config.dataset_test_dir.split("/")[-3] != config.dataset_class:
        sys.exit(f"ERROR:\t({__name__}):The dataset address doesn't match the dataset name")

    train_iter_loader = None
    test_iter_loader = None
    # Data loader           
    if config.dataset_class in dataset_list:
        train_iter_loader = get_loader(config.dataset_class, config.dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True)
        test_iter_loader = get_loader(config.dataset_class, config.dataset_test_dir, config.test_batch_size, config.num_workers, shuffle=False)
        
    if config.mode == 'train':
        if config.dataset_class in dataset_list:
            net = classification_net_select(config.classification_net)
            classification_trainer(net, train_iter_loader, test_iter_loader, config.epochs, config.lr, config.device)
    elif config.mode == 'test':
        if config.dataset_class in dataset_list:
            pass #**#
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model Configuration
    parser.add_argument('--classification_net', type=str, default='Resnet18', choices=['Resnet18', 'VGG11'])
    
    # Training Configuration
    dataset_list = ['NEU_CLS', 'elpv']
    parser.add_argument('--dataset_class', type=str, default='NEU_CLS', choices=dataset_list, help="Choose datasets")
    parser.add_argument('--dataset_train_dir', type=str, default=f'/home/zhouquan/MyDoc/DL_Learning/My_TAOD/dataset/NEU_CLS/30-shot/train.csv')
    parser.add_argument('--dataset_test_dir', type=str, default=f'/home/zhouquan/MyDoc/DL_Learning/My_TAOD/dataset/NEU_CLS/30-shot/test.csv')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Mini-batch size of train')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Mini-batch size of test')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs")
    
    # Others Configuration
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # Directories
    parser.add_argument('--log_dir', type=str, default='./My_TAOD/logs')
    parser.add_argument('--model_save_dir', type=str, default='./My_TAOD/models')
    parser.add_argument('--sample_dir', type=str, default='./My_TAOD/samples')
    parser.add_argument('--result_dir', type=str, default='./My_TAOD/results')
    
    # Parser
    parser.add_argument('--log', type=bool, default=True)
    config = parser.parse_args()
    
    # Logger
    logger(config)
    
    # Main
    main(config)
    