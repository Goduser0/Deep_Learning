import os
import sys
import time
import argparse

import torch
import torchvision.transforms as T
from torch.backends import cudnn
# torch.backends.cudnn:提供对Nvidia cuDNN的支持，可加速深度学习模型在GPU上的计算速度

import sys
sys.path.append("My_TAOD/dataset")
from dataset_loader import get_loader
from trainer import classification_trainer
from evaluator import classification_evaluator
from classification_models import classification_net_select
from logger import classification_logger

def str2bool(v):
    return v.lower() in ('true')


def main(config):
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
    

    train_iter_loader = None
    test_iter_loader = None
    eval_iter_loader = None
    # Data loader           
    if config.dataset_class in dataset_list:
        trans = T.Compose([T.ToTensor(), T.Resize(224)])
        train_iter_loader = get_loader(config.dataset_class, config.dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True, trans=trans)
        trans = T.Compose([T.ToTensor(), T.Resize(224)])
        test_iter_loader = get_loader(config.dataset_class, config.dataset_test_dir, config.test_batch_size, config.num_workers, shuffle=False, trans=trans)
        
    if config.mode == 'train':
        # 数据集一致性校验
        if config.dataset_train_dir.split("/")[7] != config.dataset_class or config.dataset_test_dir.split("/")[7] != config.dataset_class:
            sys.exit(f"ERROR:\t({__name__}):The dataset address doesn't match the dataset name")
            
        net = classification_net_select(config.classification_net)
        classification_trainer(config, net, train_iter_loader, test_iter_loader, config.epochs, config.lr, config.device)
    
    elif config.mode == 'eval':
        # 数据集一致性校验
        if config.dataset_eval_dir.split("/")[7] != config.dataset_class or config.dataset_eval_dir.split("/")[7] != config.eval_model_path.split("/")[-1].split(" ")[2]:
            sys.exit(f"ERROR:\t({__name__}):The dataset address doesn't match the dataset name")
        
        trans = T.Compose([T.ToTensor(), T.Resize(224)])
        eval_iter_loader = get_loader(config.dataset_class, config.dataset_eval_dir, config.eval_batch_size, config.num_workers, shuffle=False, transforms=trans)
        classification_evaluator(config, eval_iter_loader)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification")
    
    # Model Configuration
    classification_net_list = ['Resnet18', 'VGG11', 'Resnet50']
    parser.add_argument('--classification_net', type=str, default='Resnet18', choices=classification_net_list)
    
    # Training Configuration
    dataset_list = ['NEU_CLS', 'elpv', 'Magnetic_Tile']
    parser.add_argument('--dataset_class', type=str, default='NEU_CLS', choices=dataset_list, help="Choose datasets")
    parser.add_argument('--dataset_train_dir', type=str, default=f'/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/NEU_CLS/30-shot/train.csv')
    parser.add_argument('--dataset_test_dir', type=str, default=f'/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/NEU_CLS/30-shot/test.csv')
    parser.add_argument('--train_batch_size', type=int, default=256, help='Mini-batch size of train')
    parser.add_argument('--test_batch_size', type=int, default=256, help='Mini-batch size of test')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=200, help="Training epochs")
    
    # Augumentation Configuration
    
    # Evaluator Configuration
    parser.add_argument('--eval_model_path', type=str, default="./My_TAOD/Train_Classification/models/Classification Resnet18 NEU_CLS 2023-06-02_00:21:18.pt")
    parser.add_argument('--dataset_eval_dir', type=str, default="./My_TAOD/dataset/NEU_CLS/210-shot/test.csv")
    parser.add_argument('--eval_batch_size', type=int, default=540)
    
    # Others Configuration
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    
    # Directories
    parser.add_argument('--log_dir', type=str, default='./My_TAOD/Train_Classification/logs')
    parser.add_argument('--model_save_dir', type=str, default='./My_TAOD/Train_Classification/models')
    parser.add_argument('--sample_dir', type=str, default='./My_TAOD/Train_Classification/samples')
    parser.add_argument('--result_dir', type=str, default='./My_TAOD/Train_Classification/results')
    
    # Others
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--time', type=str, default=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    # Parser
    config = parser.parse_args()
    
    # Logger
    classification_logger(config)
    
    # Main
    main(config)
    