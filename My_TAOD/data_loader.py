import torch
import torchvision
from torch.utils import data
from PIL import Image
import numpy as np
import os
import pandas as pd
import torchvision.transforms as T


def bmp2ndarray(file_path):
    """将BMP文件转为ndarray"""
    img = Image.open(file_path).convert('L')
    img_array = np.array(img)
    return img_array

def jpeg2jpg(path_in, path_out):
    img = Image.open(path_in)
    img.save(path_out, "JPG", optimize=True, progressive=True)
    
def bmp2jpg(path_in, path_out):
    img = Image.open(path_in)
    img.save(path_out, "JPG", optimize=True, progressive=True)


class NEU_CLS(data.Dataset):
    """Dataset class for NEU_CLS"""
    
    def __init__(self, root_dir='./My_Datasets/Classification/NEU-CLS', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self.load_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = self.samples.loc[idx, 'Image_Content']
        label = self.samples.loc[idx, 'Image_Class']
        labels = ['Cr', 'PS', 'In', 'RS', 'Sc', 'Pa']
        for i in range(6):
            if label == labels[i]:
                label = i
        label = np.array(label)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def load_samples(self):
        sample_list = []
        for filename in os.listdir(self.root_dir):
            filename_list = filename.split('.')
            if filename_list[1] == 'bmp':
                # image_filename
                img_filename = filename
                # image_path
                img_path = os.path.join(self.root_dir, filename)
                # image(ndarray)
                img = bmp2ndarray(img_path)
                # image_class
                img_class = filename_list[0][:2]
                img_classID = filename_list[0][3:]
                image_item = [img, img_class, img_filename, img_path, img_classID]
                sample_list.append(image_item)
            else:
                continue
        return pd.DataFrame(sample_list, columns=['Image_Content', 'Image_Class', 'Image_Filename', 'Image_Path', 'Image_ClassID'])


def get_loader(batch_size, dataset, mode, num_workers, tt_rate):
    
    trans = []
    trans.append(T.ToTensor())
    trans = T.Compose(trans)
    
    if dataset == 'NEU_CLS':
        dataset = NEU_CLS(transform=trans)
        
        train_size = int(tt_rate*len(dataset))
        test_size = len(dataset) - train_size
        dataset_train, dataset_test = data.random_split(dataset, [train_size, test_size]) # type: ignore
        
        
    train_iter_loader = data.DataLoader(
        dataset=dataset_train, # type: ignore
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        )
    
    test_iter_loader = data.DataLoader(
        dataset=dataset_test, # type: ignore
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        )
    
    return train_iter_loader, test_iter_loader
