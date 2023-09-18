import os
import sys
import random

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
    """将BMP文件转为ndarray"""
    img = Image.open(file_path).convert('RGB')
    img_array = np.array(img).astype(np.float32)
    return img_array

# # png/jpg/bmp -> PIL.Image
# def bmp2PIL(file_path):
#     """将BMP文件转为PIL Image"""
#     img = Image.open(file_path).convert('RGB')
#     return img

# [0, 255] -> [-1, 1]
def img_255to1(img_uint):
    """ [0, 255] -> [-1, 1]
        Input: ndarray
        Output: ndarray
    """
    return ((img_uint / 127.5) - 1.).astype(np.float32)

# [-1, 1] -> [0, 255]
def img_1to255(img_tensor):
    """ [-1, 1] -> [0, 255]
        Input: ndarray
        Output: ndarray
    """
    return ((img_tensor + 1.) * 127.5).astype(np.uint8)

###########################################################################################################
# FUNCTION: get_loader
###########################################################################################################
def get_loader(dataset_class, dataset_dir, batch_size, num_workers, shuffle, trans=None, img_type='ndarray', drop_last=False):
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
                
                label = self.df.loc[index, "Image_Label"]
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
            
                label = self.df.loc[index, "Image_Label"]
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
            
                label = self.df.loc[index, "Image_Label"]
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

#######################################################################################################
## Test
#######################################################################################################
def test():
    a = cv2.imread("/home/zhouquan/MyDoc/Deep_Learning/My_Datasets/Classification/PCB-Crop/Short/01_short_01_0.jpg")
    b = bmp2ndarray("/home/zhouquan/MyDoc/Deep_Learning/My_Datasets/Classification/PCB-Crop/Short/01_short_01_0.jpg")
    c = np.array([[[255., 255., 255.], [0., 0., 0.]], [[255., 255., 255.], [0., 0., 0.]]])
    
    trans = T.Compose(
    [
        T.ToTensor(), 
        T.Resize((128, 128)),
    ])
    result = trans(img_255to1(b))
    # print(result, result.shape)
    
    data_iter_loader = get_loader('PCB_200', 
                              "./My_TAOD/dataset/PCB_200/0.7-shot/train/0.csv", 
                              1,
                              4, 
                              shuffle=False, 
                              trans=trans,
                              img_type='ndarray',
                              drop_last=True,
                              ) # 像素值范围：（-1, 1）[B, C, H, W]
    for i, data in enumerate(data_iter_loader):
        raw_img = data[0][0] # [B, C, H, W] -1~1
        category_label = data[1] # [B]
        raw_img = raw_img.numpy()
        raw_img = img_1to255(raw_img)
        raw_img = raw_img.transpose(1, 2, 0)
        # print(raw_img, raw_img.shape)
        plt.imshow(raw_img)
        plt.savefig("test_test.jpg")
        plt.close()
        break
    
if __name__ == '__main__':
    test()