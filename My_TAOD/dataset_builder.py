import os
import sys
import pandas as pd

import torch
import torchvision
import random
from torch.utils import data
from PIL import Image
import numpy as np
import os
import torchvision.transforms as T

# __getitem__()将图片转为Tensor

def bmp2ndarray(file_path):
    """将BMP文件转为ndarray"""
    img = Image.open(file_path).convert('L')
    img_array = np.array(img)
    return img_array

def png2ndarray(file_path):
    """将PNG文件转为ndarray"""
    img = Image.open(file_path).convert('L')
    img_array = np.array(img)
    return img_array


###########################################################################################################
# DATASET CLASS: NEU_CLS
###########################################################################################################
class NEU_CLS(data.Dataset):
    """Dataset class for NEU_CLS"""
    
    def __init__(self, root_dir='./My_Datasets/Classification/NEU-CLS', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self.load_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples.loc[idx, 'Image_Path']
        img_content = bmp2ndarray(img_path)
        img_class = self.samples.loc[idx, 'Image_Class']
        # 开裂(Crazing)
        # 点蚀表面(Pitted Surface)
        # 内含物(Inclusion)
        # 轧制氧化皮(Rolled-in Scale)
        # 划痕(Scratches)
        # 斑块(Patches)
        img_label = self.samples.loc[idx, 'Image_Label']
        if self.transform:
            img_content = self.transform(img_content)
        return img_content, img_label

    def load_samples(self):
        sample_list = []
        for filename in os.listdir(self.root_dir):
            filename_list = filename.split('.')
            if filename_list[1] == 'bmp':
                # image_path
                img_path = os.path.join(self.root_dir, filename)
                # image_class
                img_class = filename_list[0][:2]
                # image_label
                img_classes = ['Cr', 'PS', 'In', 'RS', 'Sc', 'Pa']
                img_label = img_classes.index(img_class)
                
                image_item = [img_label, img_class, img_path]
                sample_list.append(image_item)   
        return pd.DataFrame(sample_list, columns=['Image_Label', 'Image_Class', 'Image_Path'])
    
    def save_csv(self):
        self.samples.to_csv("NEU_CLS.csv")
        
###########################################################################################################
# DATASET: elpv
###########################################################################################################
class elpv(data.Dataset):
    """Dataset class for elpv"""
    
    def __init__(self, root_dir='./My_Datasets/Classification/elpv-dataset-master', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.label_path = root_dir+'/labels.csv'
        self.labels = ["mono", "poly"]
        self.informations = pd.read_csv(self.label_path)
        self.samples = self.load_samples()
        
    def len(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = self.samples.loc[idx, 'Image_Content']
        prob = self.samples.loc[idx, 'probs']
        type = self.samples.loc[idx, 'types']
        if self.transform:
            img = self.transform(img)
        return img, prob, type
        
    def load_samples(self):
        data = np.genfromtxt(self.label_path, dtype=['|S19', '<f8', '|S4'], names=['path', 'probability', 'type'])
        image_fnames = np.char.decode(data['path'])
        probs = data['probability']
        types = np.char.decode(data['type'])
        img_classes = [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
        img_label = [img_classes.index(prob) for prob in probs]
        
        sample_list = zip(img_label, probs, types, image_fnames)
        df = pd.DataFrame(sample_list, columns=['Image_Label', 'probs', 'types', 'Image_Path'])
        df['Image_Path'] = '/home/zhouquan/MyDoc/DL_Learning/My_Datasets/Classification/elpv-dataset-master/' + df['Image_Path']
        return df

    def save_csv(self):
        self.samples.to_csv("elpv.csv")

target_dir = '/home/zhouquan/MyDoc/DL_Learning/My_TAOD/dataset'

###########################################################################################################
# DATASET:DeepPCB
###########################################################################################################

###########################################################################################################
# DATASET:PKUPCB
###########################################################################################################

###########################################################################################################
# DATASET: add other dataset
###########################################################################################################

###########################################################################################################
# FUNCTION: bulid_dataset
# 将原始数据集每类抽样划分为适用于fewshot的训练集和测试集，并存为对应的csv文件
###########################################################################################################
def build_dataset(dataset):
    if dataset == 'NEU_CLS':
        dataset_origin = NEU_CLS()
        df = dataset_origin.samples
        label_list = df['Image_Label'].unique().tolist()
        dataset_train_size_list = [1, 5, 10, 30, int(300*0.7)]
        
    elif dataset == 'elpv':
        dataset_origin = elpv()
        df = dataset_origin.samples
        label_list = df['probs'].unique().tolist()
        dataset_train_size_list = [1, 5, 10, 30]
        
    else:
        sys.exit(f"The dataset '{dataset}' doesn't exist")
    
    dir_dataset = target_dir+'/'+dataset
    if not os.path.exists(dir_dataset):
        os.makedirs(dir_dataset, exist_ok=True)
    
    for dataset_train_size in dataset_train_size_list:
        dir_class = dir_dataset + f'/{dataset_train_size}-shot'
        os.makedirs(dir_class, exist_ok=True)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for label in label_list:
            if dataset == 'NEU_CLS':
                df_class = df.loc[df["Image_Label"] == label]
            elif dataset == 'elpv':
                df_class = df.loc[df["probs"] == label]
            else:
                return -1
            sample_choose = df_class.sample(dataset_train_size)
            df_train = pd.concat([df_train, sample_choose])
            sample_rest = df_class.drop(sample_choose.index)
            df_test = pd.concat([df_test, sample_rest])            
        df_train.to_csv(dir_class + f'/train.csv')
        df_test.to_csv(dir_class + f'/test.csv')

###########################################################################################################
###########################################################################################################
build_dataset('NEU_CLS')
build_dataset('elpv')
