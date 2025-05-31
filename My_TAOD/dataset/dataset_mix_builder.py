import os
import sys
import random
import shutil

import numpy as np
import pandas as pd
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as T

###########################################################################################################
# FUNCTION: dataset_by_label
###########################################################################################################
def dataset_by_label(df, filepath, mode):
    label_list = df['Image_Label'].unique().tolist() 
    os.makedirs(filepath + '/' + mode, exist_ok=True)
    for label in label_list:
        label_dir = filepath + '/' + mode + '/' + str(label) + '.csv'
        df_class = df.loc[df["Image_Label"] == label]
        df_class.to_csv(label_dir)

###########################################################################################################
# FUNCTION: bulid_mix_dataset
# 将原始数据集每类抽样划分为适用于fewshot的训练集和测试集，并存为对应的csv文件
# 之后的Dataloader将直接从对应的csv文件中读取数据的地址
###########################################################################################################
target_dir = '/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset'

def build_mix_dataset(tar_dataset, src_dataset_csv, num_addin):
    # tar_dataset: /home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/PCB_Crop/10-shot
    # src_dataset: /home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/DeepPCB_Crop.csv
    # 确保加入数据集与原始数据集不为同一个数据集
    assert tar_dataset.split('/')[-2] != src_dataset_csv.split('/')[-1].split('.')[0], f"tar and src can't be the same dataset({src_dataset_csv.split('/')[-1].split('.')[0]})"
    
    # 创建保存文件夹
    save_name = f"{tar_dataset.split('/')[-1]} + {src_dataset_csv.split('/')[-1].split('.')[0]}({num_addin})"
    dir_dataset = f"{target_dir}/{tar_dataset.split('/')[-2]}/{save_name}"
    os.makedirs(dir_dataset, exist_ok=False)
    
    # 复制target的所有test文件
    shutil.copy(f"{tar_dataset}/test.csv", dir_dataset)
    shutil.copytree(f"{tar_dataset}/test", f"{dir_dataset}/test")
    
    # 取出source中指定数目的添加文件
    src_df = pd.read_csv(src_dataset_csv)
    src_label_list = src_df["Image_Label"].unique().tolist()
    df_addin = pd.DataFrame()
    for label in src_label_list:
        # 取出source中标签为label的所有样本
        src_df_class = src_df.loc[src_df["Image_Label"] == label]
        src_sample_choose = src_df_class.sample(num_addin)
        df_addin = pd.concat([df_addin, src_sample_choose])
    
    # 将抽取出的src文件插入tar的train文件中
    df_train = pd.read_csv(f"{tar_dataset}/train.csv")
    df_train = pd.concat([df_train, df_addin])
    
    df_train.to_csv(f"{dir_dataset}/train.csv")
    dataset_by_label(df_train, dir_dataset, 'train')
    
    print(f'({__name__}):Dataset:{save_name}\tBuild Successfully!!!')


###########################################################################################################
# 运行函数：创建Dataset
###########################################################################################################
if __name__ == "__main__":
    DPC_csv = "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/DeepPCB_Crop.csv"
    P2_csv = "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/PCB_200.csv"
    PC_csv = "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/PCB_Crop.csv"
    
    tar_DPC = "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/DeepPCB_Crop/10-shot"
    tar_P2 = "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/PCB_200/10-shot"
    tar_PC = "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset/PCB_Crop/10-shot"
    
    NUM_ADDIN = 150
    
    build_mix_dataset(tar_DPC, P2_csv, NUM_ADDIN)
    build_mix_dataset(tar_DPC, PC_csv, NUM_ADDIN)
    
    build_mix_dataset(tar_P2, DPC_csv, NUM_ADDIN)
    build_mix_dataset(tar_P2, PC_csv, NUM_ADDIN)
    
    build_mix_dataset(tar_PC, DPC_csv, NUM_ADDIN)
    build_mix_dataset(tar_PC, P2_csv, NUM_ADDIN)
