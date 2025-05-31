import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import InstanceHardnessThreshold, RepeatedEditedNearestNeighbours, CondensedNearestNeighbour
from scipy.spatial import cKDTree

import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False 

sys.path.append("/home/zhouquan/MyDoc/Deep_Learning/My_TAOD/dataset")
from dataset_loader import get_loader
sys.path.append("/home/zhouquan/MyDoc/Deep_Learning/My_Filter")
from Feature_Extractor import select_feature_extractor

def remove_redundant_samples(X, y, r=None, method='radius'): # r
    """
    基于 k 近邻或 r 邻域的增强样本去重方法。
    
    参数：
    X - 样本特征矩阵 (numpy array, shape: [n_samples, n_features])
    y - 样本标签 (numpy array, shape: [n_samples])
    k - 近邻数量 (默认 5)
    r - 邻域半径 (默认 None，仅在 method='radius' 时有效)
    method - 'knn' 使用 k 近邻, 'radius' 使用 r 邻域
    
    返回：
    过滤后的 X 和 y
    """
    tree = cKDTree(X)
    kill_indices = set()
    
    for i in range(len(X)):
        if i in kill_indices:
            continue
        
        if method == 'knn':
            # _, neighbors = tree.query(X[i], k=k + 1)
            # neighbors = neighbors[1:]  # 排除自身
            pass
        elif method == 'radius':
            neighbors = tree.query_ball_point(X[i], r)
            neighbors.remove(i)
            neighbors = [neighbor for neighbor in neighbors if y[neighbor] == y[i]]
            
        if len(neighbors) > 0:
            kill_indices.update(neighbors)
        
    return np.delete(X, list(kill_indices), axis=0), np.delete(y, list(kill_indices))

def main(config):
    FE = select_feature_extractor('vgg19')
    FE.eval()

    trans = T.Compose([T.ToTensor(), T.Resize((224, 224))])
    # data_path = "My_TAOD/dataset/filter/CycleGAN/PCB_200[10-shot]<-DeepPCB_Crop[200-shot]/1500.csv"
    # data_path = "My_TAOD/dataset/filter/CycleGAN/PCB_200[10-shot]<-PCB_Crop[200-shot]/1500.csv"
    data_path = "My_TAOD/dataset/filter/ConGAN/PCB_200[10-shot]/1500.csv"
    data_loader = get_loader(data_path, 256, 4, shuffle=False, trans=trans, dataset_class="PCB_200", require_path=True)

    device = "cuda:1"

    FE.to(device)

    start_time = time.time()

    for batch_idx, (X, y, _) in enumerate(data_loader):
        
        X = X.to(device)
        batch_X_data = FE(X)
        batch_X_data = batch_X_data.detach().cpu()
        batch_y = y
        
        if batch_idx == 0:
            data_X = batch_X_data
            data_y = batch_y
        else:
            data_X = torch.cat((data_X, batch_X_data), dim=0)
            data_y = torch.cat((data_y, batch_y), dim=0)

    cost_time = time.time() - start_time

    data_X = data_X.numpy()
    data_y = data_y.numpy()
    data_origin_X = data_X
    data_origin_y = data_y

    original_indices = np.arange(len(data_origin_X))
    
    # RENN-IHT
    RENN = RepeatedEditedNearestNeighbours(sampling_strategy='all', n_neighbors=config.k) # n_neighbors
    start_time = time.time()
    data_X, data_y = RENN.fit_resample(data_X, data_y)
    cost_time1 = time.time() - start_time

    kept_indices_renn = np.array([np.where((data_origin_X == x).all(axis=1))[0][0] for x in data_X])
    
    IHT = InstanceHardnessThreshold(sampling_strategy='all', estimator=LogisticRegression(), cv=10, random_state=42) # cv
    start_time = time.time()
    data_X, data_y = IHT.fit_resample(data_X, data_y)
    cost_time2 = time.time() - start_time

    kept_indices_renn_iht = np.array([np.where((data_origin_X == x).all(axis=1))[0][0] for x in data_X])
    
    # RNN
    start_time = time.time()
    data_X, data_y = remove_redundant_samples(data_X, data_y, r=config.r, method='radius')
    cost_time3 = time.time() - start_time
    
    kept_indices_renn_iht_rnn = np.array([np.where((data_origin_X == x).all(axis=1))[0][0] for x in data_X])
    
    # print(f"经过 RENN + IHT + RNN 后: 样本数：{len(kept_indices_renn_iht_rnn)} 耗时:{cost_time3:.3f}")
    # print(f"{(cost_time1 + cost_time2 + cost_time3):.3f}s")
    # print(f"\
    # 类别为 0 的样本数: {np.count_nonzero(data_y == 0)}\n\
    # 类别为 1 的样本数: {np.count_nonzero(data_y == 1)}\n\
    # 类别为 2 的样本数: {np.count_nonzero(data_y == 2)}\n\
    # 类别为 3 的样本数: {np.count_nonzero(data_y == 3)}\n\
    # 类别为 4 的样本数: {np.count_nonzero(data_y == 4)}\n\
    #     ")
    
    df = pd.read_csv(data_path)
    # df_1 = df.iloc[kept_indices_renn]
    # df_2 = df.iloc[kept_indices_renn_iht]
    df_3 = df.iloc[kept_indices_renn_iht_rnn]
    # df_1.to_csv(f"{os.path.dirname(data_path)}/{data_path.split('/')[-1].split('.')[0]}_renn.csv", index=False)
    # df_2.to_csv(f"{os.path.dirname(data_path)}/{data_path.split('/')[-1].split('.')[0]}_renn_iht.csv", index=False)
    df_3.to_csv(f"{os.path.dirname(data_path)}/{data_path.split('/')[-1].split('.')[0]}_renn_iht_rnn.csv", index=False)
    
    
    # save_dir = f"My_TAOD/Train_Classification/results/filter/ConGAN/PCB_200[10-shot]/Resnet18_Pretrained/{config.k}_{config.r}"
    save_dir = f"My_TAOD/Train_Classification/results/filter/ablation/{config.k}_{config.r}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(save_dir + '/fliter.txt', 'w') as f:
        # f.write(f"{len(kept_indices_renn)}\n")
        f.write(f"{len(kept_indices_renn_iht)}\n")
        # f.write(f"{len(kept_indices_renn_iht_rnn)}\n")
        f.write(f"{(cost_time1+cost_time2):.3f}\n")
        # f.write(f"{(cost_time1 + cost_time2 + cost_time3):.3f}\n")
    f.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fliter")
    parser.add_argument("--r", type=float, default=0.2, required=False)
    parser.add_argument("--k", type=int, default=35, required=False)
    config = parser.parse_args()
    main(config)