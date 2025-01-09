import torch
import torchvision.transforms as T

import os
import numpy as np
import pandas as pd
from PIL import Image
import time
import tqdm

from model import CycleGAN_Generator

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader

def CycleGAN_SampleGenerator(G_Src2Tar_path, batch_size=200):
    # G_Src2Tar_path = "My_TAOD/TA/TA_Results/CycleGAN/DeepPCB_Crop[10-shot]<-PCB_200[200-shot] 0 2025-01-08_22-17-34/models/3000_net_g_src2tar.pth"
    G_Src2Tar = CycleGAN_Generator(3, 3)
    G_time = G_Src2Tar_path.split('/')[-3]
    G_epoch = int(G_Src2Tar_path.split('/')[-1].split('_')[0])
    checkpoint = torch.load(G_Src2Tar_path)
    G_Src2Tar.load_state_dict(checkpoint["model_state_dict"])
    G_Src2Tar.eval()
    
    local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    dirname = f"{G_epoch}epoch_src2tar {local_time}"
    os.makedirs(f"My_TAOD/TA/TA_Results/CycleGAN/{G_time}/samples/{dirname}", exist_ok=False)
    
    img_label = G_time.split(" ")[1]
    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper', 'Missing_hole']
    img_class = img_classes[int(img_label)]
    
    Src_name = G_Src2Tar_path.split('/')[-3].split(' ')[0].split('<-')[-1].split('[')[0]
    Src_shot = G_Src2Tar_path.split('/')[-3].split(' ')[0].split('<-')[-1].split('[')[1][:-1]
    Src_label = G_Src2Tar_path.split('/')[-3].split(' ')[1]
    Src_dataset_path = f"My_TAOD/dataset/{Src_name}/{Src_shot}/train/{Src_label}.csv"
    
    trans = T.Compose([
        T.ToTensor(),
        T.Resize((32, 32)),
    ])
    
    loader_Src = get_loader(
        Src_dataset_path,
        batch_size, 
        4,
        shuffle=True,
        trans=trans,
        img_type='ndarray',
        drop_last=False,
    )
    
    for _, (Src_img, _) in enumerate(loader_Src):                    
        fake_Tar_img = G_Src2Tar(Src_img)
        break
        
    img_save_list = []
    i = 0
    for img in tqdm.tqdm(fake_Tar_img):
        i += 1
        img = img.detach().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img.transpose(1, 2, 0))
        
        filename = f"{i}.jpg"
        img_path = f"My_TAOD/TA/TA_Results/CycleGAN/{G_time}/samples/{dirname}/{filename}"
        img.save(img_path)
        
        img_save_item = [img_label, img_class, img_path]
        img_save_list.append(img_save_item)
        
    img_save_df = pd.DataFrame(img_save_list, columns=["Image_Label", "Image_Class", "Image_Path"])
    img_save_csv = f"My_TAOD/TA/TA_Results/CycleGAN/{G_time}/samples/{dirname}/generate_imgs.csv"
    img_save_df.to_csv(img_save_csv)
    print("Generate Done!!!")
    return img_save_csv

if __name__ == "__main__":
    G_Src2Tar_path = "My_TAOD/TA/TA_Results/CycleGAN/DeepPCB_Crop[10-shot]<-PCB_200[200-shot] 0 2025-01-08_22-17-34/models/1500_net_g_src2tar.pth"
    CycleGAN_SampleGenerator(G_Src2Tar_path)