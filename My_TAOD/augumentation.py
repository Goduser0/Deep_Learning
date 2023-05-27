import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torch import optim

from PIL import Image
import matplotlib.pyplot as plt

##########################################################################################################
# FUNCTION:show_images
##########################################################################################################
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
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


class Generator(nn.Module):
    def __init__(self):
        pass
    
    def forward(self,):
        pass
    
class Discriminator(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
if __name__ == "__main__":
    G = Generator()
    D = Discriminator()
    
a = Image.open('./My_Datasets/Classification/NEU-CLS/Pa_299.bmp')
b = Image.open('./My_Datasets/Classification/NEU-CLS/Pa_299.bmp')
axes = show_images([a, b], 1, 2)
plt.imshow(a)
