import numpy as np
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("My_TAOD/TA/TA_Metrics")
from cal_mmd import score_mmd

sys.path.append("My_TAOD/TA/TA_Offline_Generator")
from generater import generate

sys.path.append("./My_TAOD/TA/TA_Models")
from TA_G import PFS_Generator, CoGAN_Generator

def aug_mmd(z_dim, num_samples, model, model_path, img_save_path, real_path, batch_size, domain=None):
    fake_csv_path = generate(z_dim, num_samples, model, model_path, img_save_path, domain)
    result = score_mmd(real_path, fake_csv_path, batch_size)
    return result

def mmd_baseline_from_scratch(sampling_times=3):
    z_dim = 128
    G = PFS_Generator(z_dim)
    model_path = "My_TAOD/TA/TA_Results/PFS_baseline_from_scratch/models/PCB_Crop 4 2023-10-31_15-48-13/1000_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/PFS_baseline_from_scratch/samples"
    real_path = "My_TAOD/dataset/PCB_Crop/30-shot/test/4.csv"
    
    score_list = []
    for _ in range(sampling_times):
        score_list.append(aug_mmd(z_dim, 100, G, model_path, img_save_path, real_path, batch_size=16))
    print(f"MMD:{np.mean(score_list)}")

def mmd_baseline_finetuning(sampling_times=3):
    z_dim = 128
    G = PFS_Generator(z_dim)
    model_path = "My_TAOD/TA/TA_Results/PFS_baseline_finetuning/models/PCB_200_from_DeepPCB_Crop 4 2024-04-26_06-34-57/492_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/PFS_baseline_finetuning/samples"
    real_path = "My_TAOD/dataset/PCB_200/30-shot/test/4.csv"
    
    score_list = []
    for _ in range(sampling_times):
        score_list.append(aug_mmd(z_dim, 100, G, model_path, img_save_path, real_path, batch_size=16))
    print(f"MMD:{np.mean(score_list)}")
    
def mmd_cogan(sampling_times=3):
    z_dim = 128
    G = CoGAN_Generator(z_dim)
    model_path = "My_TAOD/TA/TA_Results/CoGAN/models/PCB_2002DeepPCB_Crop 4 2024-04-23_19-40-16/2000_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/CoGAN/samples"
    real_path = "My_TAOD/dataset/DeepPCB_Crop/30-shot/test/4.csv"
    
    score_list = []
    for _ in range(sampling_times):
        score_list.append(aug_mmd(z_dim, 100, G, model_path, img_save_path, real_path, batch_size=16, domain="T"))
    print(f"MMD:{np.mean(score_list)}")
    
if __name__ == "__main__":
    # mmd_baseline_from_scratch()
    # mmd_baseline_finetuning()
    mmd_cogan()
    