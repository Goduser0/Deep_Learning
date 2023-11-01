import numpy as np
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("My_TAOD/TA/TA_Metrics")
from cal_fid import score_fid

sys.path.append("My_TAOD/TA/TA_Offline_Generator")
from generater import generate

sys.path.append("./My_TAOD/TA/TA_Models")
from TA_G import PFS_Generator, CoGAN_Generator

def aug_fid(z_dim, num_samples, model, model_path, img_save_path, real_path, num_fid_samples, domain=None, gpu_id="0"):
    fake_csv_path = generate(z_dim, num_samples, model, model_path, img_save_path, domain)
    result = score_fid(real_path, fake_csv_path, num_fid_samples)
    return result
# 
def fid_baseline_from_scratch():
    z_dim = 128
    G = PFS_Generator(z_dim)
    model_path = "My_TAOD/TA/TA_Results/baseline_from_scratch/models/PCB_Crop 4 2023-10-31_15-48-13/1000_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/baseline_from_scratch/samples"
    real_path = "My_TAOD/dataset/PCB_Crop/30-shot/test/4.csv"
    num_fid_samples = 100
    
    score_list = []
    for i in range(3):
        score_list.append(aug_fid(z_dim, 200, G, model_path, img_save_path, real_path, num_fid_samples))
    print(f"FID:{np.mean(score_list)}")
    
def fid_cogan():
    z_dim = 128
    G = CoGAN_Generator(z_dim)
    model_path = "My_TAOD/TA/TA_Results/CoGAN/models/PCB_Crop2DeepPCB_Crop 0 2023-11-02_05-04-25/1000_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/CoGAN/samples"
    real_path = "My_TAOD/dataset/DeepPCB_Crop/30-shot/test/0.csv"
    num_fid_samples = 100
    
    score_list = []
    for i in range(3):
        score_list.append(aug_fid(z_dim, 100, G, model_path, img_save_path, real_path, num_fid_samples, domain="T"))
    print(f"FID:{np.mean(score_list)}")

if __name__ == "__main__":
    # fid_baseline_from_scratch()
    fid_cogan()