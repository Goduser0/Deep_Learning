import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


##########################################################################################################
# FUNCTION:show_images
##########################################################################################################
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Args:
        imgs (_type_): a list of images
        num_rows (int)
        num_cols (int)
        titles (list, optional): a list of images'titles. Defaults to None.
        scale (float, optional): Defaults to 1.5.

    Returns:
        A figure
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


##########################################################################################################
# Ploting Result
########################################################################################################## 
if __name__ == "__main__":
    df1 = pd.read_csv('./My_TAOD/dataset/PCB_Crop/5-shot/train.csv')
    df2 = pd.read_csv('./My_TAOD/dataset/PCB_200/5-shot/train.csv')
    df = pd.concat([df1, df2])

    images = []
    names = []
    for i in zip(df['Image_Path'], df['Image_Class']):
        images.append(Image.open(i[0]).convert('RGB'))
        names.append(str(i[1]))
        
    # image_transform
    images_trans = []
    trans = T.Compose([
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ])

    for image in images:
        images_trans.append(trans(image))

    # show image
    num_rows = 3
    axes = show_images(images, num_rows, int(len(images)/num_rows)+1, names, scale=3)
    plt.savefig("image_viewer_plot.png")
    plt.close()
