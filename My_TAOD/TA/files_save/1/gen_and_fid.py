import numpy as np
import warnings
warnings.filterwarnings("ignore")

import tqdm

import sys
sys.path.append("/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/TA_Metrics")
from cal_fid import score_fid

sys.path.append("/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/TA/TA_Offline_Generator")
from generater import generate, translate

sys.path.append("/home/zhouquan/MyDoc/Deep_Learning//My_TAOD/TA/TA_Models")
from TA_G import PFS_Generator, CoGAN_Generator, CycleGAN_Generator

def aug_fid(z_dim, num_samples, model, model_path, img_save_path, real_path, num_fid_samples, sampling_times, domain=None, gpu_id="0"):
    fake_csv_path = generate(z_dim, num_samples, model, model_path, img_save_path, domain)
    
    score_list = []
    for _ in tqdm.trange(1, sampling_times+1, desc=f"Mean FID({sampling_times}):"):
        result = score_fid(real_path, num_fid_samples, fake_csv_path, num_fid_samples)
        score_list.append(result)
    return np.mean(score_list)
# 
def fid_baseline_from_scratch(sampling_times=10):
    z_dim = 128
    G = PFS_Generator(z_dim).eval()
    model_path = "My_TAOD/TA/TA_Results/PFS_baseline_from_scratch/models/PCB_Crop 4 2023-10-31_15-48-13/1000_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/PFS_baseline_from_scratch/samples"
    real_path = "My_TAOD/dataset/PCB_Crop/30-shot/test/4.csv"
    num_fid_samples = 100
    
    score = aug_fid(z_dim, 100, G, model_path, img_save_path, real_path, num_fid_samples, sampling_times)
    print(f"FID:{score}")
    
def fid_baseline_finetuning(sampling_times=10):
    z_dim = 128
    G = PFS_Generator(z_dim).eval()
    model_path = "My_TAOD/TA/TA_Results/PFS_baseline_finetuning/models/PCB_200_from_DeepPCB_Crop 4 2024-04-26_06-34-57/492_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/PFS_baseline_finetuning/samples"
    real_path = "My_TAOD/dataset/PCB_200/30-shot/test/4.csv"
    num_fid_samples = 100
    
    score = aug_fid(z_dim, 100, G, model_path, img_save_path, real_path, num_fid_samples, sampling_times)
    print(f"FID:{score}")
    
def fid_cogan(sampling_times=10):
    z_dim = 128
    G = CoGAN_Generator(z_dim).eval()
    model_path = "My_TAOD/TA/TA_Results/CoGAN/models/PCB_2002DeepPCB_Crop 4 2024-04-23_19-40-16/2000_net_g.pth"
    img_save_path = "My_TAOD/TA/TA_Results/CoGAN/samples"
    real_path = "My_TAOD/dataset/DeepPCB_Crop/30-shot/test/4.csv"
    num_fid_samples = 100
    
    score = aug_fid(z_dim, 100, G, model_path, img_save_path, real_path, num_fid_samples, sampling_times, domain="T")
    print(f"FID:{score}")

def fid_cyclegan(sampling_times=10):
    
    G = CycleGAN_Generator(3, 3).eval()
    img_save_path = "My_TAOD/TA/TA_Results/Cycle/samples"
    
    real_path = "My_TAOD/dataset/DeepPCB_Crop/30-shot/test/0.csv"
    model_path = "My_TAOD/TA/TA_Results/CycleGAN/models/PCB_200_2_DeepPCB_Crop 0 2024-04-25_16-38-38/200_net_G_A2B.pth"
    Src_path = "My_TAOD/dataset/PCB_200/160-shot/train/0.csv"
    
    fake_csv_path = translate(100, G, model_path, Src_path, img_save_path, 32)
    score_list = []
    
    for _ in tqdm.trange(1, sampling_times+1, desc=f"Mean FID({sampling_times}):"):
        result = score_fid(real_path, 100, fake_csv_path, 100)
        score_list.append(result)
        
    print(f"FID:{np.mean(score_list)}")
    

if __name__ == "__main__":
    # fid_baseline_from_scratch()
    # fid_baseline_finetuning()
    # fid_cogan()
    fid_cyclegan()