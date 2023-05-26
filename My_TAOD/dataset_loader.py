import torch
import torchvision
import random
from torch.utils import data
from PIL import Image
import numpy as np
import os
import pandas as pd
import torchvision.transforms as T
import sys

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
# FUNCTION: get_loader
###########################################################################################################
def get_loader(dataset_class, dataset_dir, batch_size, num_workers, shuffle):
    df = pd.read_csv(dataset_dir)
    
    if dataset_class.lower() == 'neu_cls':
        class MyDataset(data.Dataset): # type: ignore
            # 依据csv文件创建dataset
            def __init__(self, df, transforms=None):
                self.df = df
                self.transforms = transforms
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, index):
                image_path = self.df.loc[index, "Image_Path"]
                image = bmp2ndarray(image_path)
                if self.transforms:
                    image = self.transforms(image)
                label = self.df.loc[index, "Image_Label"]
                return image, label
    elif dataset_class.lower() == 'elpv':
        class MyDataset(data.Dataset):
            # 依据csv文件创建dataset
            def __init__(self, df, transforms=None):
                self.df = df
                self.transforms = transforms
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, index):
                image_path = self.df.loc[index, "Image_Path"]
                image = png2ndarray(image_path)
                if self.transforms:
                    image = self.transforms(image)
                label = self.df.loc[index, "Image_Label"]
                return image, label
        
    else:
        sys.exit(f"ERROR:\t({__name__}):The dataset_class '{dataset_class}' doesn't exist")
    
    # 添加trans
    trans = [T.ToTensor(), T.Resize(224)]
    trans = T.Compose(trans)
    
    # 返回dataloader
    dataset_iter_loader = data.DataLoader(
        dataset=MyDataset(df, transforms=trans), # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        )

    return dataset_iter_loader
