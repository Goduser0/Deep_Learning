import os
import random
import argparse

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

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
def make_noise(batch_size, latent_dim, num_noise, device):
    if num_noise == 1:
        noises = [torch.randn(batch_size, latent_dim, device=device)]
    else:
        noises = torch.randn(num_noise, batch_size, latent_dim, device=device).unbind(0)
        
    return noises


#######################################################################################################
# FUNCTION: mix_noise()
#######################################################################################################
def mix_noise(batch_size, latent_dim, prob, device):
    """
    prob
    """
    if prob > 0 and random.random() < prob:
        return make_noise(batch_size, latent_dim, 2, device)
    else:
        return make_noise(batch_size, latent_dim, 1, device)


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
# Function Test
#######################################################################################################
# parser = argparse.ArgumentParser()
# parser.add_argument("--subspace_std", default=0.1)
# parser.add_argument("--batch_size", default=20)
# parser.add_argument("--n_sample", default=120)
# parser.add_argument("--latent_dim", default=10)
# config = parser.parse_args()

# init_z = torch.randn(config.n_sample, config.latent_dim, device=torch.device('cuda'))
# print(init_z.detach().shape)
# z = get_subspace(config, init_z)
# print(z.detach().shape)


# a = mix_noise(16, 10, 0.9, torch.device('cuda'))
# for i in a:
#     print(i.detach().shape)
