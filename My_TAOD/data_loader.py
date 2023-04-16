import torch
import torchvision
from torch.utils import data
from PIL import Image
import numpy as np
import os
import pandas as pd
import torchvision.transforms as T


def bmp_to_ndarray(file_path):
    """将BMP文件转为ndarray"""
    img = Image.open(file_path).convert('L')
    img_array = np.array(img)
    return img_array


class NEU_CLS(data.Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
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
                img = bmp_to_ndarray(img_path)
                # image_class
                img_class = filename_list[0][:2]
                img_classID = filename_list[0][3:]
                image_item = [img, img_class, img_filename, img_path, img_classID]
                sample_list.append(image_item)
            else:
                continue
        return pd.DataFrame(sample_list, columns=['Image_Content', 'Image_Class', 'Image_Filename', 'Image_Path', 'Image_ClassID'])


trans = T.ToTensor()
# iamge_size[1, 200, 200]
My_NEUCLS = NEU_CLS('./My_Datasets/Classification/NEU-CLS', transform=trans)
# My_NEUCLS.samples.to_csv('My_NEUCLS.csv')

def get_loader(image_dir, attr):
    pass


from ResNet import net

batch_size = 128
lr = 0.05
num_epochs = 100

train_size = int(0.7*len(My_NEUCLS))
test_size = len(My_NEUCLS) - train_size
nue_train, nue_test = data.random_split(My_NEUCLS, [train_size, test_size])

train_iter = data.DataLoader(nue_train, batch_size, shuffle=True)
test_iter = data.DataLoader(nue_test, batch_size, shuffle=False)

from train import train

train(net, train_iter, test_iter, num_epochs, lr, torch.device('cuda:1'))
