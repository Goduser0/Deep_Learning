import cv2
import numpy as np
import torch
import pandas as pd
import torchvision.transforms as T

import sys
sys.path.append("./My_TAOD")
from dataset_loader import bmp2ndarray, bmp2PIL

def load_images(csv_path, trans=None):
    df = pd.read_csv(csv_path)
    
    nums = len(df)
    catagory_lables = df["Image_Label"]
    image_path = df["Image_Path"]
    
    images = []
    for i in image_path:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        if trans:
            img = trans(img)
        else:
            img = img.transpose(2, 0, 1)
        images.append(img)
    
    if trans:
        images = torch.cat(images, dim=0)
    else:
        images = torch.FloatTensor(np.array(images))

    return images, catagory_lables, nums
    
# test
def test():
    trans = T.Compose(
        [
            T.ToTensor(), 
            T.Resize((256, 256)),
        ]
    )
    a, _, _ = load_images("./My_TAOD/dataset/PCB_200/30-shot/train/0.csv", trans=trans)
   
# test()
    
    