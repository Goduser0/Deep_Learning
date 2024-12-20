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
    save_dir = f"./My_TAOD/Train_Classification/results/{config.mode}/{config.dataset_class} {config.classification_net} {config.time}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Logger
    classification_logger(config, save_dir)
    
    dataset_train_dir = f"My_TAOD/dataset/{config.dataset_class}/{config.dataset_ratio}/train.csv"
    dataset_validation_dir = f"My_TAOD/dataset/{config.dataset_class}/{config.dataset_ratio}/validation.csv"
    dataset_test_dir = f"My_TAOD/dataset/{config.dataset_class}/{config.dataset_ratio}/test.csv"
    
    trans = T.Compose([T.ToTensor(), T.Resize((config.img_size, config.img_size))])
    
    if config.mode == "train":
        train_iter_loader = get_loader(dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True, trans=trans)
        net = classification_net_select(config.classification_net)
        classification_trainer(config, save_dir, net, train_iter_loader, None, config.epochs, config.lr)
    
    elif config.mode == "train_with_validation":
        train_iter_loader = get_loader(dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True, trans=trans)
        validation_iter_loader = get_loader(dataset_validation_dir, config.validation_batch_size, config.num_workers, shuffle=False, trans=trans)
        net = classification_net_select(config.classification_net)
        classification_trainer(config, save_dir, net, train_iter_loader, validation_iter_loader, config.epochs, config.lr)
    
    elif config.mode == "test":
        test_iter_loader = get_loader(dataset_test_dir, config.test_batch_size, config.num_workers, shuffle=False, trans=trans)
        classification_tester(config, save_dir, test_iter_loader)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification")
    
    parser.add_argument("--mode", type=str, choices=["train", "train_with_validation", "test"], required=True)
    
    # Model Configuration
    classification_net_list = ["Resnet18_Pretrained", "Resnet50_Pretrained", "EfficientNet_Pretrained", "VGG11_Pretrained", "MobileNet_Pretrained"]
    parser.add_argument("--classification_net", type=str, choices=classification_net_list, required=True)
    
    # Training Configuration
    dataset_class_choices = ["NEU_CLS", "elpv", "Magnetic_Tile", "PCB_200", "PCB_Crop", "DeepPCB_Crop"]
    parser.add_argument("--dataset_class", type=str, choices=dataset_class_choices, required=True)
    parser.add_argument("--dataset_ratio", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--validation_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128)
    
    # Test Configuration
    parser.add_argument("--test_model_path", type=str)
    parser.add_argument("--test_batch_size", type=int, default=512)
    
    # Others Configuration
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Others
    parser.add_argument("--time", type=str, default=time.strftime(f"%Y%m%d_%H%M%S", time.localtime()))
    
    # Parser
    config = parser.parse_args()
    
    # 一致性校验
    # /home/zhouquan/MyDoc/Deep_Learning/My_TAOD/Train_Classification/results/train/DeepPCB_Crop Resnet18 20240612_162617/models/10.pth
    if config.mode in ["test"]:
        model_name = config.test_model_path.split('/')[-3]
        train_log_path = "./My_TAOD/Train_Classification/results/train/" + model_name + '/log.txt'
        train_log = {}
        with open(train_log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    key, value = line.split(':', 1)
                    
                    train_log[key.strip()] = value.strip().strip("'")
        f.close()

        assert (
            (train_log["classification_net"] == config.classification_net) # 模型一致
            and
            (train_log["dataset_class"] == config.dataset_class) # 数据集种类一致
            and
            (train_log["dataset_ratio"] == config.dataset_ratio) # 数据集比例一致
            )
    
    # Main
    main(config)
    
    print("!!!Done!!!")