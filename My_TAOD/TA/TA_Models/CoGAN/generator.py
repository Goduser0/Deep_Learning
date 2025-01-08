import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.autograd import Variable

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time
import tqdm

from model import CoGAN_Generator

def CoGAN_SampleGenerator(G_path, batch_size=10000):
    # G_path = "My_TAOD/TA/TA_Results/CoGAN/DeepPCB_Crop[10-shot]<-PCB_200[200-shot] 0 2025-01-02_22-50-11/models/10000_net_g.pth"    
    G = CoGAN_Generator()
    G_time = G_path.split('/')[-3]
    G_epoch = int(G_path.split('/')[-1].split('_')[0])
    checkpoint = torch.load(G_path)
    G.load_state_dict(checkpoint["model_state_dict"])

    local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    dirname = f"{G_epoch}epoch {local_time}"
    os.makedirs(f"My_TAOD/TA/TA_Results/CoGAN/{G_time}/samples/{dirname}", exist_ok=False)
    
    img_label = G_time.split(" ")[1]
    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper', 'Missing_hole']
    img_class = img_classes[int(img_label)]
    
    z = torch.randn(batch_size, 128)
    Src_img, Tar_img = G(z)
    
    img_save_list = []
    i = 0
    for img in tqdm.tqdm(Tar_img):
        i+=1
        img = img.detach().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img.transpose(1, 2, 0))
        
        filename = f"{i}.jpg"
        img_path = f"My_TAOD/TA/TA_Results/CoGAN/{G_time}/samples/{dirname}/{filename}"
        img.save(img_path)
        
        img_save_item = [img_label, img_class, img_path]
        img_save_list.append(img_save_item)
        
    # Save to csv    
    img_save_df = pd.DataFrame(img_save_list, columns=["Image_Label", "Image_Class", "Image_Path"])
    img_save_df.to_csv(f"My_TAOD/TA/TA_Results/CoGAN/{G_time}/samples/{dirname}/generate_imgs.csv")
    print("Generate Done!!!")
    
if __name__ == "__main__":
    G_path = "My_TAOD/TA/TA_Results/CoGAN/DeepPCB_Crop[10-shot]<-PCB_200[200-shot] 0 2025-01-02_22-50-11/models/10000_net_g.pth" 
    CoGAN_SampleGenerator(G_path)
    