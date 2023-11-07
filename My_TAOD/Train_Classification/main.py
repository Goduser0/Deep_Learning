import os
import sys
import time
import argparse

import torch
import torchvision.transforms as T

import sys
sys.path.append("My_TAOD/dataset")
from dataset_loader import get_loader
from trainer import classification_trainer
from tester import classification_tester
from classification_models import classification_net_select
from logger import classification_logger


def main(config):
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
    trans = T.Compose([T.ToTensor(), T.Resize((224, 224))])
    train_iter_loader = get_loader(config.dataset_class, config.dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True, trans=trans)
    test_iter_loader = get_loader(config.dataset_class, config.dataset_test_dir, config.test_batch_size, config.num_workers, shuffle=False, trans=trans)
    
    if config.mode == "train":
        net = classification_net_select(config.classification_net)
        classification_trainer(config, net, train_iter_loader, None, config.epochs, config.lr)
    
    elif config.mode == "train_with_validation":
        validation_iter_loader = get_loader(config.dataset_class, config.dataset_validation_dir, config.validation_batch_size, config.num_workers, shuffle=False, trans=trans)
        net = classification_net_select(config.classification_net)
        classification_trainer(config, net, train_iter_loader, validation_iter_loader, config.epochs, config.lr)
    
    elif config.mode == "test":
        classification_tester(config, test_iter_loader)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument("--mode", type=str, default="train_with_validation", choices=["train", "train_with_validation", "test"])
    
    # Model Configuration
    classification_net_list = ["Resnet18", "VGG11", "Resnet50"]
    parser.add_argument("--classification_net", type=str, default="Resnet18", choices=classification_net_list)
    
    # Training Configuration
    dataset_list = ["NEU_CLS", "elpv", "Magnetic_Tile", "PCB_200", "PCB_Crop", "DeepPCB_Crop"]
    parser.add_argument("--dataset_class", type=str, default="PCB_200", choices=dataset_list, help="Choose datasets")
    parser.add_argument("--dataset_train_dir", type=str, default=f"My_TAOD/dataset/PCB_200/0.7-shot/train.csv")
    parser.add_argument("--dataset_validation_dir", type=str, default=f"My_TAOD/dataset/PCB_200/0.7-shot/validation.csv")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Mini-batch size of train")
    parser.add_argument("--validation_batch_size", type=int, default=256, help="Mini-batch size of validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    
    # Test Configuration
    parser.add_argument("--test_model_path", type=str, default="My_TAOD/Train_Classification/models/Classification Resnet18 NEU_CLS 2023-06-02_00:21:18.pt")
    parser.add_argument("--dataset_test_dir", type=str, default="My_TAOD/dataset/NEU_CLS/210-shot/test.csv")
    parser.add_argument("--test_batch_size", type=int, default=540)
    
    # Others Configuration
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=1)
    
    # Directories
    parser.add_argument("--log_dir", type=str, default="./My_TAOD/Train_Classification/logs")
    parser.add_argument("--model_save_dir", type=str, default="./My_TAOD/Train_Classification/models")
    parser.add_argument("--sample_dir", type=str, default="./My_TAOD/Train_Classification/samples")
    parser.add_argument("--result_dir", type=str, default="./My_TAOD/Train_Classification/results")
    
    # Others
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--time", type=str, default=time.strftime(f"%Y%m%d_%H%M%S", time.localtime()))
    # Parser
    config = parser.parse_args()
    
    # 一致性校验
    assert (config.dataset_train_dir.split("/")[-3] == config.dataset_class
            and
            config.dataset_test_dir.split("/")[-3] == config.dataset_class
            )
    
    # Logger
    classification_logger(config)
    
    # Main
    main(config)
    