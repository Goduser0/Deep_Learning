import os
import sys
import random
import itertools

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms as T

# png/jpg/bmp -> ndarray
def bmp2ndarray(file_path):
    """将bmp/png/jpg文件转为ndarray"""
    img = Image.open(file_path).convert('RGB')
    img_array = np.array(img).astype(np.uint8)
    return img_array

# [0, 255] -> [-1, 1]
def img_255to1(img):
    """ 
    [0, 255] -> [-1, 1]
    np.uint8 -> np.float32
    torch.uint8 -> torch.float32
    """
    if img.dtype == np.uint8:
        return ((img / 127.5) - 1.).astype(np.float32)
    elif img.dtype == torch.uint8:
        return ((img / 127.5) - 1.).to(torch.float32)
    else:
        raise TypeError

# [-1, 1] -> [0, 255]
def img_1to255(img):
    """
    [-1, 1] -> [0, 255]
    np.float32 -> np.uint8
    torch.float32 -> torch.uint8
    """
    if img.dtype == np.float32:
        return ((img + 1.) * 127.5).astype(np.uint8)
    elif img.dtype == torch.float32:
        return ((img + 1.) * 127.5).to(torch.uint8)
    else:
        raise TypeError
    
def expand_dataframe(df, num_expand):
    """"
    将dataframe循环扩展至num_expand行
    """
    length = len(df)
    df_combined = df
    for _ in range((num_expand // length) - 1):
        df_combined = pd.concat([df_combined, df], axis=0, ignore_index=True)
    df_combined = pd.concat([df_combined, df.head(num_expand%length)], axis=0, ignore_index=True)
    return df_combined

###########################################################################################################
# FUNCTION: get_loader
###########################################################################################################
def get_loader(dataset_dir, batch_size, num_workers, shuffle, dataset_class=None, trans=None, img_type='ndarray', drop_last=False, num_expand=0, require_path=False):
    """_summary_

    Args:
        dataset_class (str)
        dataset_dir (str)
        batch_size (int)
        num_workers (int)
        shuffle (bool)
        trans (_type_, optional): Defaults to None.
        img_type (str, optional): Defaults to 'ndarray'.
        drop_last (bool, optional): Defaults to False.

    Returns:
        [images, labels]: images(-1~1, if totensor -> torch.float32
                                       else -> np.float32)
    """
    if dataset_class:
        pass
    else:
        start_index = dataset_dir.find("dataset/") + len("dataset/")
        dataset_class = dataset_dir[start_index:].split('/')[0]
    
    df = pd.read_csv(dataset_dir)
    if num_expand != 0:
        df = expand_dataframe(df, num_expand)
    
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
                if img_type.lower() == 'ndarray':
                    image = bmp2ndarray(image_path)
                    image = img_255to1(image)
                else:
                    raise "img_type should be ndarray"
                
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
                if img_type.lower() == 'ndarray':
                    image = bmp2ndarray(image_path)
                    image = img_255to1(image)
                else:
                    raise "img_type should be ndarray"
                
                if self.transforms:
                    image = self.transforms(image)
                    
                label = self.df.loc[index, "Image_Label"]
                return image, label

    elif dataset_class.lower() == 'magnetic_tile':
        class MyDataset(data.Dataset):
            # 依据csv文件创建dataset
            def __init__(self, df, transforms=None):
                self.df = df
                self.transforms = transforms
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, index):
                image_path = self.df.loc[index, "Image_Path"]
                if img_type.lower() == 'ndarray':
                    image = bmp2ndarray(image_path)
                    image = img_255to1(image)
                else:
                    raise "img_type should be ndarray"
                
                if self.transforms:
                    image = self.transforms(image)
                    
                label = self.df.loc[index, "Image_Label"]
                return image, label
            
    elif dataset_class.lower() == 'pcb_crop':
        class MyDataset(data.Dataset):
            # 依据csv文件创建dataset
            def __init__(self, df, transforms=None):
                self.df = df
                self.transforms = transforms
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, index):
                image_path = self.df.loc[index, "Image_Path"]
                if img_type.lower() == 'ndarray':
                    image = bmp2ndarray(image_path)
                    image = img_255to1(image)
                else:
                    raise "img_type should be ndarray"
                
                if self.transforms:
                    image = self.transforms(image)
        
                label = self.df.loc[index, "Image_Label"] # torch.int64
                
                if require_path:
                    return image, label, image_path
                return image, label
            
    elif dataset_class.lower() == 'pcb_200':
        class MyDataset(data.Dataset):
            # 依据csv文件创建dataset
            def __init__(self, df, transforms=None):
                self.df = df
                self.transforms = transforms
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, index):
                image_path = self.df.loc[index, "Image_Path"]
                if img_type.lower() == 'ndarray':
                    image = bmp2ndarray(image_path)
                    image = img_255to1(image)
                else:
                    raise "img_type should be ndarray"
    
                if self.transforms:
                    image = self.transforms(image)
                
                label = self.df.loc[index, "Image_Label"] # torch.int64
                if require_path:
                    return image, label, image_path 
                return image, label
    
    elif dataset_class.lower() == 'deeppcb_crop':
        class MyDataset(data.Dataset):
            # 依据csv文件创建dataset
            def __init__(self, df, transforms=None):
                self.df = df
                self.transforms = transforms
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, index):
                image_path = self.df.loc[index, "Image_Path"]
                if img_type.lower() == 'ndarray':
                    image = bmp2ndarray(image_path)
                    image = img_255to1(image)
                else:
                    raise "img_type should be ndarray"
    
                if self.transforms:
                    image = self.transforms(image)
                
                label = self.df.loc[index, "Image_Label"] # torch.int64
                
                if require_path:
                    return image, label, image_path      
                return image, label
            
    else:
        sys.exit(f"ERROR:\t({__name__}):The dataset_class '{dataset_class}' doesn't exist")
    

    # 返回dataloader
    dataset_iter_loader = data.DataLoader(
        dataset=MyDataset(df, transforms=trans), # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        )

    return dataset_iter_loader

###########################################################################################################
# FUNCTION: get_loader_ST
###########################################################################################################
def get_loader_ST(Src_dataset_dir, Tar_dataset_dir, batch_size, num_workers, shuffle, trans=None, unaligned=False, img_type='ndarray', drop_last=False):
    """_summary_

    Args:
        dataset_class (str)
        dataset_dir (str)
        batch_size (int)
        num_workers (int)
        shuffle (bool)
        trans (_type_, optional): Defaults to None.
        img_type (str, optional): Defaults to 'ndarray'.
        drop_last (bool, optional): Defaults to False.

    Returns:
        [images, labels]: images(-1~1, if totensor -> torch.float32
                                       else -> np.float32)
    """
    Src_df = pd.read_csv(Src_dataset_dir)
    Tar_df = pd.read_csv(Tar_dataset_dir)
    class ImageDataset(data.Dataset):
        def __init__(self, Src_df, Tar_df, transforms=None, unaligned=False):
            self.Src_df = Src_df
            self.Tar_df = Tar_df
            self.transforms = transforms
            self.unaligned = unaligned
        
        def __getitem__(self, index):
            item_Src = self.Src_df.loc[index % len(self.Src_df), "Image_Path"]
            
            if self.unaligned:
                item_Tar = self.Tar_df.loc[random.randint(0, len(self.Tar_df)-1) , "Image_Path"]
            else:
                item_Tar = self.Tar_df.loc[index % len(self.Tar_df), "Image_Path"]
                
            if img_type.lower() == 'ndarray':
                    item_Src = bmp2ndarray(item_Src)
                    item_Src = img_255to1(item_Src)
                    item_Tar = bmp2ndarray(item_Tar)
                    item_Tar = img_255to1(item_Tar)
            else:
                raise "img_type should be ndarray"
        
            if self.transforms:
                    item_Src = self.transforms(item_Src)
                    item_Tar = self.transforms(item_Tar)
            
            return {'Src': item_Src, 'Tar': item_Tar}
        
        def __len__(self):
            return max(len(self.Src_df), len(self.Tar_df))
    
    dataset_iter_loader = data.DataLoader(
        dataset=ImageDataset(Src_df, Tar_df, transforms=trans, unaligned=unaligned), # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        )

    return dataset_iter_loader


#######################################################################################################
## Test
#######################################################################################################
def test():
    trans = T.Compose(
                    [ 
                    T.ToTensor(),
                    ]
    )
    data_loader_Tar = get_loader(
                              "./My_TAOD/dataset/DeepPCB_Crop/10-shot/train/0.csv", 
                              batch_size=32,
                              num_workers=4, 
                              shuffle=True, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=False,
                              num_expand=50,
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
    data_loader_Src = get_loader(
                              "./My_TAOD/dataset/PCB_Crop/50-shot/train/0.csv", 
                              batch_size=32,
                              num_workers=4, 
                              shuffle=True, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=False,
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
       
    for batch_idx, ((raw_img_Src, category_label_Src), (raw_img_Tar, category_label_Tar)) in enumerate(zip(data_loader_Src, data_loader_Tar)):
        print(f"{batch_idx}: [{category_label_Src.shape}] [{category_label_Tar.shape}]")
        
if __name__ == '__main__':
    test()