import os
import sys
import time
import argparse
import ast

import torch
import torchvision.transforms as T
import pandas as pd

import sys
sys.path.append("My_TAOD/dataset")
from dataset_loader import get_loader
from trainer import classification_trainer
from tester import classification_tester
from classification_models import classification_net_select
from logger import classification_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification")
    
    # Model Configuration
    classification_net_list = ["Resnet18_Pretrained", "Resnet50_Pretrained", "EfficientNet_Pretrained", "VGG11_Pretrained", "MobileNet_Pretrained", "VGG19_Pretrained"]
    parser.add_argument("--classification_net", type=str, choices=classification_net_list, default="VGG19_Pretrained") #
    
    # Training Configuration
    
    # My_TAOD/dataset/diff_num/ConGAN/DeepPCB_Crop[10-shot]/30.csv
    # My_TAOD/dataset/diff_num/ConGAN/DeepPCB_Crop[10-shot]/70.csv
    
    # My_TAOD/dataset/diff_num/CycleGAN/DeepPCB_Crop[10-shot]<-PCB_200[200-shot]/30.csv
    # My_TAOD/dataset/diff_num/CycleGAN/DeepPCB_Crop[10-shot]<-PCB_200[200-shot]/70.csv
    
    # parser.add_argument("--add_train_csv", type=str, required=False, default="[\"My_TAOD/dataset/diff_num/ConGAN/DeepPCB_Crop[10-shot]/30.csv\",\"My_TAOD/dataset/diff_num/ConGAN/DeepPCB_Crop[10-shot]/70.csv\"]") #
    parser.add_argument("--add_train_csv", type=str, required=False, default="[\"My_TAOD/dataset/diff_num/CycleGAN/DeepPCB_Crop[10-shot]<-PCB_200[200-shot]/30.csv\"]") #
    # parser.add_argument("--add_train_csv", type=str, required=True) #
    
    parser.add_argument("--train_batch_size", type=int, default=512)
    
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--model_init_path", type=str, default=None)
    
    parser.add_argument("--test_batch_size", type=int, default=512)
    
    # Others Configuration
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Others
    parser.add_argument("--time", type=str, default=time.strftime(f"%Y%m%d_%H%M%S", time.localtime()))
    
    # Parser
    config = parser.parse_args()
    
    add_train_csv = ast.literal_eval(config.add_train_csv)
    config.gan_type = add_train_csv[0].split('/')[-3]
    config.dataset_class = add_train_csv[0].split('/')[-2].split('[')[0]
    config.dataset_ratio = add_train_csv[0].split('/')[-2].split(']')[0].split('[')[-1]
    config.dataset_shot = 0
    
    config.train_csv = f"My_TAOD/dataset/{config.dataset_class}/{config.dataset_ratio}/train.csv"
    config.test_csv = f"My_TAOD/dataset/{config.dataset_class}/{config.dataset_ratio}/test.csv"
    
    df = pd.read_csv(config.train_csv)
    for item in add_train_csv:
        config.dataset_shot += int(item.split('/')[-1].split('.')[0])
        add_df = pd.read_csv(item)
        df = pd.concat([df, add_df])
    
    # Create directories if not exist
    save_dir = f"./My_TAOD/Train_Classification/results/diff_num/{config.gan_type}/{add_train_csv[0].split('/')[-2]}/{config.classification_net}/{config.dataset_shot}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    df.to_csv(f"{save_dir}/all_train.csv")
    
    # Logger
    classification_logger(config, save_dir)
    
    trans = T.Compose([T.ToTensor(), T.Resize((config.img_size, config.img_size))])
    
    dataset_train_dir = f"{save_dir}/all_train.csv"
    dataset_test_dir = config.test_csv
    
    # Train
    train_iter_loader = get_loader(dataset_train_dir, config.train_batch_size, config.num_workers, shuffle=True, dataset_class=config.dataset_class, trans=trans)
    net = classification_net_select(config.classification_net)
    classification_trainer(config, save_dir, net, train_iter_loader, None, config.epochs, config.lr)
    
    # Test
    config.test_model_path = f"{save_dir}/models/{config.epochs}.pth"
    test_iter_loader = get_loader(dataset_test_dir, config.test_batch_size, config.num_workers, shuffle=False, dataset_class=config.dataset_class, trans=trans)
    classification_tester(config, save_dir, test_iter_loader)
    
    