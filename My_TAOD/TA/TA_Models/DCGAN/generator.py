import torch

import os
import numpy as np
import pandas as pd
from PIL import Image
import time
import tqdm

from model import DCGAN_Generator

def DCGAN_SampleGenerator(G_path, batch_size=100):
    #  G_path = "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/TA_Results/DCGAN/DeepPCB_Crop 0 2024-12-30_16-52-43/models/10000_net_g.pth"    
    device = "cuda:0"
    G = DCGAN_Generator()
    G_time = G_path.split('/')[-3]
    G_epoch = int(G_path.split('/')[-1].split('_')[0])
    checkpoint = torch.load(G_path)
    G.load_state_dict(checkpoint["model_state_dict"])
    G.to(device)
    G.eval()
    
    local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    dirname = f"{G_epoch}epoch {local_time}"
    os.makedirs(f"My_TAOD/TA/TA_Results/DCGAN/{G_time}/samples/{dirname}", exist_ok=False)
    
    img_label = G_time.split(" ")[1]
    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper', 'Missing_hole']
    img_class = img_classes[int(img_label)]
    
    z = torch.randn(batch_size, 100, 1, 1).to(device)
    imgs = G(z)
    
    img_save_list = []
    i = 0
    for img in tqdm.tqdm(imgs):
        i+=1
        img = img.detach().cpu().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img.transpose(1, 2, 0))
        
        filename = f"{i}.jpg"
        img_path = f"My_TAOD/TA/TA_Results/DCGAN/{G_time}/samples/{dirname}/{filename}"
        img.save(img_path)
        
        img_save_item = [img_label, img_class, img_path]
        img_save_list.append(img_save_item)
        
    # Save to csv    
    img_save_df = pd.DataFrame(img_save_list, columns=["Image_Label", "Image_Class", "Image_Path"])
    img_save_csv = f"My_TAOD/TA/TA_Results/DCGAN/{G_time}/samples/{dirname}/generate_imgs.csv"
    img_save_df.to_csv(img_save_csv)
    print("Generate Done!!!")
    return img_save_csv
    
if __name__ == "__main__":
    root_path = 'My_TAOD/TA/TA_Results/DCGAN'
    G_path_list = [os.path.join(root_path, folder) for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]
    for G_path in G_path_list:
        DCGAN_SampleGenerator(f"{G_path}/models/10000_net_g.pth")