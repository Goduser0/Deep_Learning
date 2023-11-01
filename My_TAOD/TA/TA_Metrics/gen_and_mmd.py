import numpy as np
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("My_TAOD/TA/TA_Metrics")
from cal_mmd import score_mmd

sys.path.append("My_TAOD/TA/TA_Offline_Generator")
from generater import generate

sys.path.append("./My_TAOD/TA/TA_Models")
from TA_G import PFS_Generator

def aug_mmd(z_dim, num_samples, model, model_path, img_save_path, real_path, batch_size):
    fake_csv_path = generate(z_dim, num_samples, model, model_path, img_save_path)
    result = score_mmd(real_path, fake_csv_path, batch_size)
    return result

def mmd_baseline_from_scratch():
    z_dim = 128
    G = PFS_Generator(z_dim)
    model_path = "My_TAOD/TA/TA_Results/baseline_from_scratch/models/PCB_Crop 4 2023-10-31_15-48-13/1000_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/baseline_from_scratch/samples"
    real_path = "My_TAOD/dataset/PCB_Crop/30-shot/test/4.csv"
    
    score_list = []
    for i in range(3):
        score_list.append(aug_mmd(z_dim, 100, G, model_path, img_save_path, real_path, batch_size=50))
    print(f"MMD:{np.mean(score_list)}")
    
if __name__ == "__main__":
    mmd_baseline_from_scratch()