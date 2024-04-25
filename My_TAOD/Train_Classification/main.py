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
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
         
    trans = T.Compose([T.ToTensor(), T.Resize((config.img_size, config.img_size))])
    
    if config.mode == "train":
        train_iter_loader = get_loader(config.dataset_class, config.dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True, trans=trans)
        net = classification_net_select(config.classification_net)
        classification_trainer(config, net, train_iter_loader, None, config.epochs, config.lr)
    
    elif config.mode == "train_with_validation":
        train_iter_loader = get_loader(config.dataset_class, config.dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True, trans=trans)
        validation_iter_loader = get_loader(config.dataset_class, config.dataset_validation_dir, config.validation_batch_size, config.num_workers, shuffle=False, trans=trans)
        net = classification_net_select(config.classification_net)
        classification_trainer(config, net, train_iter_loader, validation_iter_loader, config.epochs, config.lr)
    
    elif config.mode == "test":
        test_iter_loader = get_loader(config.dataset_class, config.dataset_test_dir, config.test_batch_size, config.num_workers, shuffle=False, trans=trans)
        classification_tester(config, test_iter_loader)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification")
    
    parser.add_argument("--mode", type=str, choices=["train", "train_with_validation", "test"])
    
    # Model Configuration
    classification_net_list = ["VGG11", "Resnet18", "Resnet50", "Resnet18_Pretrained", "Resnet50_Pretrained"]
    parser.add_argument("--classification_net", type=str, default="VGG11", choices=classification_net_list)
    
    # Training Configuration
    dataset_list = ["NEU_CLS", "elpv", "Magnetic_Tile", "PCB_200", "PCB_Crop", "DeepPCB_Crop"]
    parser.add_argument("--dataset_class", type=str, default="DeepPCB_Crop", choices=dataset_list, help="Choose datasets")
    parser.add_argument("--dataset_train_dir", type=str, default=f"My_TAOD/dataset/DeepPCB_Crop/30-shot" + "/train.csv")
    parser.add_argument("--dataset_validation_dir", type=str, default=f"My_TAOD/dataset/DeepPCB_Crop/30-shot" + "/validation.csv")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Mini-batch size of train")
    parser.add_argument("--validation_batch_size", type=int, default=16, help="Mini-batch size of validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--img_size", type=int, default=128)
    
    # Test Configuration
    parser.add_argument("--test_model_path", type=str, default="My_TAOD/Train_Classification/models/DeepPCB_Crop Resnet50 20231108_231254/100.pth")
    parser.add_argument("--dataset_test_dir", type=str, default="My_TAOD/dataset/DeepPCB_Crop/30-shot" + "/test.csv")
    parser.add_argument("--test_batch_size", type=int, default=512)
    
    # Others Configuration
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Directories
    parser.add_argument("--log_dir", type=str, default="./My_TAOD/Train_Classification/logs")
    parser.add_argument("--model_dir", type=str, default="./My_TAOD/Train_Classification/models")
    parser.add_argument("--result_dir", type=str, default="./My_TAOD/Train_Classification/results")
    
    # Others
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--time", type=str, default=time.strftime(f"%Y%m%d_%H%M%S", time.localtime()))
    # Parser
    config = parser.parse_args()
    
    # 一致性校验
    assert (((config.mode in ["train"]) and (config.dataset_train_dir.split('/')[-3] == config.dataset_class))
            or
            ((config.mode in ["train_with_validation"]) and (config.dataset_train_dir.split('/')[-3] == config.dataset_class) and (config.dataset_validation_dir.split('/')[-3] == config.dataset_class))
            or
            ((config.mode in ["test"]) and (config.dataset_test_dir.split('/')[-3] == config.dataset_class) and (config.test_model_path.split('/')[-2].split(" ")[0] == config.dataset_test_dir.split('/')[-3]))
            )
    
    # Logger
    if config.mode in ["train", "train_with_validation"]:
        classification_logger(config)

    # Main
    main(config)
    
    print("!!!Done!!!")