import os
import random
import argparse
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as T

#######################################################################################################
# FUNCTION: init_random_seed()
#######################################################################################################
def init_random_seed(manual_seed):
    """Init the random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print(f'Use random seed:{seed}')
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#######################################################################################################
# FUNCTION: make_noise()
#######################################################################################################
def make_noise(batch_size, latent_dim, num_noise):
    if num_noise == 1:
        noises = [torch.randn(batch_size, latent_dim)] # [1, B, z_dim]
    else:
        noises = torch.randn(num_noise, batch_size, latent_dim).unbind(0)
        
    return noises


#######################################################################################################
# FUNCTION: mix_noise()
#######################################################################################################
def mix_noise(batch_size, latent_dim, prob):
    """
    prob
    """
    if prob > 0 and random.random() < prob:
        return make_noise(batch_size, latent_dim, 2)
    else:
        return make_noise(batch_size, latent_dim, 1)


#######################################################################################################
# FUNCTION: get_subspace()
#######################################################################################################
def get_subspace(config, init_z, vis_flag=False):
    # 从n_train个样本中采样batch_size个样本，形成高斯子空间
    std = config.subspace_std
    batch_size = config.batch_size if not vis_flag else config.n_sample
    index = np.random.randint(0, init_z.size(0), size=batch_size)
    z = init_z[index]
    
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    
    return z


#######################################################################################################
# FUNCTION: fused_leaky_relu()
#######################################################################################################
def fused_leaky_relu(input, bias, negative_slope=0.2, scale= 2 ** 0.5):
    return scale * F.leaky_relu(
        input + bias.view((1, -1) + (1,) * (len(input.shape) - 2)), 
        negative_slope=negative_slope
                                )


#######################################################################################################
# FUNCTION: upfirdn2d_native()
#######################################################################################################
def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    input = input.permute(0, 2, 3, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    
    output = input.view(-1, in_h, 1, in_w, 1, minor)
    output = F.pad(output, [0, 0, 0, up_x-1, 0, 0, 0, up_y -1])
    output = output.view(-1, in_h * up_y, in_w * up_x, minor)
    
    output = F.pad(output, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    
    output = output[
        :,
        max(-pad_y0, 0) : output.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : output.shape[2] - max(-pad_x1, 0),
        :,
    ]
    
    output = output.permute(0, 3, 1, 2)
    output = output.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    output = F.conv2d(output, w)
    output = output.reshape(
        -1, 
        minor, 
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    
    return output[:, :, ::down_y, ::down_x]
    
    
#######################################################################################################
# FUNCTION: upfirdn2d()
#######################################################################################################
def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    output = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    
    return output

#######################################################################################################
# FUNCTION: make_kernel()
#######################################################################################################
def make_kernel(ksize):
    ksize = torch.tensor(ksize, dtype=torch.float32)
    
    if ksize.ndim == 1:
        kernel = ksize[None, :] * ksize[:, None]
    else:
        kernel = ksize
        
    kernel /= kernel.sum()
    
    return kernel

#######################################################################################################
#### FUNCTION: requires_grad()
#######################################################################################################
def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

#######################################################################################################
#### FUNCTION: plt_tsne()
#######################################################################################################
def plt_tsne2d(X, Y, labels=None):
    # tsne
    tsne = TSNE(n_components=2, random_state=501)
    X_tsne = tsne.fit_transform(X)
    # 归一化
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_norm = scaler.fit_transform(X_tsne)
    
    # plot
    plt.figure(figsize=(15, 15))
    scatter = plt.scatter(X_norm[:,0], X_norm[:,1], c=Y, s=50)
    if labels:
        pass
    else:
        labels = range(len(scatter.legend_elements()[0]))
    plt.legend(handles=scatter.legend_elements()[0],labels=labels,title="classes")
                
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    save_path = "./My_TAOD/TA/results/" + str(t) + ".png"
    plt.savefig(save_path)
    plt.close()
    
#######################################################################################################
#### FUNCTION: S_trainer_record_data()
#######################################################################################################
def S_trainer_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + ' ' + config.catagory + ' ' + config.time
    filepath = config.results_dir + '/' + filename + '.csv'
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
    
    if not os.path.exists(f"{config.results_dir}/{filename}"):
        os.makedirs(f"{config.results_dir}/{filename}")
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        epoch = results["epoch"]
        num_epochs = results["num_epochs"]
        batch = results["batch"]
        num_batchs = results["num_batchs"]
        
        TotalLoss_vae = results["TotalLoss_vae"]
        TotalLoss_g = results["TotalLoss_g"]
        D_TotalLoss_g = results["D_TotalLoss_g"]
        D_TotalLoss_d = results["D_TotalLoss_d"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in TotalLoss_vae], label="TotalLoss_vae")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/TotalLoss_vae.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in TotalLoss_g], label="TotalLoss_g")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/TotalLoss_g.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_TotalLoss_g], label="D_TotalLoss_g")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_TotalLoss_g.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in D_TotalLoss_d], label="D_TotalLoss_d")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/D_TotalLoss_d.jpg')
        plt.close()
        

#######################################################################################################
#### FUNCTION: S2T_trainer_record_data()
#######################################################################################################
def S2T_trainer_record_data(config, content, flag_plot=True):
    """Save Data"""
    assert os.path.exists(config.results_dir)
    filename = config.dataset_class + ' ' + config.catagory + ' ' + config.time
    filepath = config.results_dir + '/' + filename + '.csv'
    content = pd.DataFrame.from_dict(content, orient="index").T
    
    if not os.path.exists(filepath):
        content.to_csv(filepath, index=False)
    else:
        results = pd.read_csv(filepath)
        results = pd.concat([results, content], axis=0, ignore_index=True)
        results.to_csv(filepath, index=False)
    
    if not os.path.exists(f"{config.results_dir}/{filename}"):
        os.makedirs(f"{config.results_dir}/{filename}")
        
    if flag_plot:
        results = pd.read_csv(filepath)
        
        epoch = results["epoch"]
        num_epochs = results["num_epochs"]
        
        TotalLoss_vae_unique = results["TotalLoss_vae_unique"]
        TotalLoss_g_target = results["TotalLoss_g_target"]
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in TotalLoss_vae_unique], label="TotalLoss_vae_unique")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/TotalLoss_vae_unique.jpg')
        plt.close()
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=80)
        ax1.plot([y for y in TotalLoss_g_target], label="TotalLoss_g_target")
        ax1.legend()
        fig.tight_layout()
        plt.savefig(f'{config.results_dir}/{filename}/TotalLoss_g_target.jpg')
        plt.close()
        
    
#######################################################################################################
# Function Test
#######################################################################################################
def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subspace_std", default=0.1)
    parser.add_argument("--batch_size", default=20)
    parser.add_argument("--n_sample", default=120)
    parser.add_argument("--latent_dim", default=10)
    config = parser.parse_args()

    # init_z = torch.randn(config.n_sample, config.latent_dim, device=torch.device('cuda'))
    # print(init_z.detach().shape)
    # z = get_subspace(config, init_z)
    # print(z.detach().shape)

    df = pd.read_csv("./My_TAOD/dataset/PCB-Crop/30-shot/train.csv")
    image_path = df["Image_Path"]
    image_label = df["Image_Label"]
    image_path = list(image_path)

    image = []
    for path in image_path:
        img = Image.open(path).convert('RGB')
        img_array = np.asarray(img)
        
        trans = T.Compose([T.ToTensor(), T.Resize((224, 224))])
        img_array = trans(img_array)
        
        image.append(img_array)
    

    image = torch.stack(image)
    image = image.view(len(image), -1)
    image_label = torch.Tensor(image_label)
    plt_tsne2d(image, image_label)

# test()

