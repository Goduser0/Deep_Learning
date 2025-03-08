import torch

import os
import numpy as np
import pandas as pd
from PIL import Image
import time
import tqdm

from model import WGAN_GP_Generator
import sys
sys.path.append("/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/TA_Metrics")
from cal_fid import score_fid
from cal_mmd import score_mmd

def WGAN_GP_SampleGenerator(G_path, sample_size=100):
    #  G_path = "/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/TA_Results/WGAN_GP/DeepPCB_Crop 0 2024-12-31_16-16-01/models/10000_net_g.pth"    
    device = "cuda:0"
    G = WGAN_GP_Generator()
    G_time = G_path.split('/')[-3]
    G_epoch = int(G_path.split('/')[-1].split('_')[0])
    checkpoint = torch.load(G_path)
    G.load_state_dict(checkpoint["model_state_dict"])
    G.to(device)
    G.eval()
    
    local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    dirname = f"{G_epoch}epoch {local_time}"
    os.makedirs(f"My_TAOD/TA/TA_Results/WGAN_GP/{G_time}/samples/{dirname}", exist_ok=False)
    
    img_label = G_time.split(" ")[1]
    img_classes = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper', 'Missing_hole']
    img_class = img_classes[int(img_label)]
    
    split_list = [100] * (sample_size//(100)) + [i for i in [sample_size%100] if i != 0]

    img_save_list = []
    i = 0
    for batch_size in split_list:
        z = torch.randn(batch_size, 100, 1, 1).to(device)
        imgs = G(z)
    
        for img in tqdm.tqdm(imgs):
            i+=1
            img = img.detach().cpu().numpy()
            img = ((img + 1) / 2 * 255).astype(np.uint8)
            img = Image.fromarray(img.transpose(1, 2, 0))
            
            filename = f"{i}.jpg"
            img_path = f"My_TAOD/TA/TA_Results/WGAN_GP/{G_time}/samples/{dirname}/{filename}"
            img.save(img_path)
            
            img_save_item = [img_label, img_class, img_path]
            img_save_list.append(img_save_item)
        
    # Save to csv    
    img_save_df = pd.DataFrame(img_save_list, columns=["Image_Label", "Image_Class", "Image_Path"])
    img_save_csv = f"My_TAOD/TA/TA_Results/WGAN_GP/{G_time}/samples/{dirname}/generate_imgs.csv"
    img_save_df.to_csv(img_save_csv)
    print("Generate Done!!!")
    return img_save_csv
    
if __name__ == "__main__":
    root_path = 'My_TAOD/TA/TA_Results/WGAN_GP'
    G_path_list = [os.path.join(root_path, folder) for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]
    
    mean_times = 5
    
    for G_path in G_path_list:
        fid_list = []
        mmd_list = []
        for _ in range(mean_times):
            fake_path = WGAN_GP_SampleGenerator(f"{G_path}/models/10000_net_g.pth", 100)
            # print(fake_path)
            real_path = f"My_TAOD/dataset/{G_path.split('/')[-1].split(' ')[0]}/10-shot/test/{G_path.split('/')[-1].split(' ')[1]}.csv"
            # print(real_path)
            fid_list.append(score_fid(real_path, 100, fake_path, 100))
            mmd_list.append(score_mmd(real_path, fake_path, 50))

        fid = sum(fid_list) / len(fid_list)
        mmd = sum(mmd_list) / len(mmd_list)
        with open(os.path.dirname(os.path.dirname(fake_path)) + '/' + 'score.txt', 'a') as f:
            f.write(f"fid: {fid}\nmmd: {mmd}\n")