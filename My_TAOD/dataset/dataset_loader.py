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

###########################################################################################################
# FUNCTION: get_loader
###########################################################################################################
def get_loader(dataset_class, dataset_dir, batch_size, num_workers, shuffle, trans=None, img_type='ndarray', drop_last=False):
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
                
                label = self.df.loc[index, "Image_Label"] # torch.int64
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
    # test_1
    # path_1 = "/home/zhouquan/MyDoc/Deep_Learning/My_Datasets/Classification/PCB-Crop/Short/01_short_01_0.jpg"
    # path_2 = "/home/zhouquan/MyDoc/Deep_Learning/My_Datasets/Classification/PCB-200/Mouse_bite/000001_0_01_04736_11571.bmp"
    # a = cv2.imread(path_1) # H*W*C
    # b = bmp2ndarray(path_2)
    # c = np.array([[[0, 0, 0], [50, 50, 50]], [[100, 100, 100], [150, 150, 150]], [[200, 200, 200], [250, 250, 250]], [[255, 255, 255], [255, 255, 255]]], dtype=np.uint8)
    # print(f"a.shape:{a.shape} | b.shape:{b.shape} | c.shape:{c.shape}")
    # trans = T.Compose(
    # [
    #     T.ToTensor(), 
    #     T.Resize((128, 128)),
    # ])
    # result = trans(img_255to1(c))
    # print(np.max(c), np.min(c), c.shape, c.dtype)
    # print(torch.max(result), torch.min(result), result.shape, result.dtype)
    
    # test2
    # data_iter_loader = get_loader('PCB_Crop', 
    #                           "./My_TAOD/dataset/PCB_Crop/0.7-shot/train/2.csv", 
    #                           8,
    #                           4, 
    #                           shuffle=True, 
    #                           trans=trans,
    #                           img_type='ndarray',
    #                           drop_last=True,
    #                           ) # 像素值范围：（-1, 1）[B, C, H, W]
    # for i, data in enumerate(data_iter_loader):
    #     raw_img = data[0][0] # [B, C, H, W] -1~1
    #     category_label = data[1] # [B]
    #     print(category_label.shape)
    #     print(raw_img.dtype)
    #     raw_img = torch.autograd.Variable(raw_img)
    #     print(raw_img.dtype)
    #     raw_img = raw_img.numpy()
    #     print(raw_img.dtype)
    #     raw_img = img_1to255(raw_img)
    #     raw_img = raw_img.transpose(1, 2, 0)
    #     print(raw_img.dtype)
    #     # print(raw_img, raw_img.shape)
    #     plt.imshow(raw_img)
    #     plt.savefig("test_test.jpg")
    #     plt.close()
    #     break

    # test3
    trans = T.Compose(
                    [ 
                    T.ToTensor(),
                    T.Resize(int(128*1.12), Image.BICUBIC), 
                    T.RandomCrop(128), 
                    T.RandomHorizontalFlip(),
                    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    )

    data_iter_loader = get_loader_ST(
        "./My_TAOD/dataset/DeepPCB_Crop/1.0-shot/train/0.csv",
        "./My_TAOD/dataset/PCB_Crop/30-shot/train/0.csv",
        8,
        4,
        True,
        trans=trans,
        unaligned=True,
        img_type="ndarray",
        drop_last=False,
        )

    for i, data in enumerate(data_iter_loader):
        Src_img = data['Tar'] # [B, C, H, W] -1~1
        Tar_img = data['Src']     
        print(Src_img.shape)
        print(Tar_img.shape)
        
        # Src_img = torch.autograd.Variable(Src_img[0])
        # Tar_img = torch.autograd.Variable(Tar_img[0])
        # print(Src_img.dtype)
        # print(Tar_img.dtype)
        
        Src_img = img_1to255(Src_img[0].numpy()).transpose(1, 2, 0)
        Tar_img = img_1to255(Tar_img[0].numpy()).transpose(1, 2, 0)
        
        plt.imshow(Src_img)
        plt.savefig("test_Src.jpg")
        plt.close()
        plt.imshow(Tar_img)
        plt.savefig("test_Tar.jpg")
        plt.close()
        
        break
        
    
if __name__ == '__main__':
    test()