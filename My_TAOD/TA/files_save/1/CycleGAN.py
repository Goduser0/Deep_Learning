import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import random
import numpy as np
import os
import tqdm
import time
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

import sys
sys.path.append("./My_TAOD/dataset")
from dataset_loader import get_loader_ST
sys.path.append("./My_TAOD/TA/TA_Models")
from TA_G import CycleGAN_Generator
from TA_D import CycleGAN_Discriminator
from TA_layers import CycleGAN_ReplayBuffer, CycleGAN_weights_init_normal, CycleGAN_LambdaLR, img_1to255

sys.path.append("./My_TAOD/TA/TA_Utils")
from TA_logger import cyclegan_logger
from TA_utils import cyclegan_record_data

########################################################################################################
#### Config
########################################################################################################
parser = argparse.ArgumentParser()

# Saved Directories
parser.add_argument("--logs_dir", type=str,
                    default="My_TAOD/TA/TA_Results/CycleGAN/logs")
parser.add_argument("--models_dir", type=str,
                    default="My_TAOD/TA/TA_Results/CycleGAN/models")
parser.add_argument("--samples_dir", type=str,
                    default="My_TAOD/TA/TA_Results/CycleGAN/samples")
parser.add_argument("--results_dir", type=str,
                    default="My_TAOD/TA/TA_Results/CycleGAN/results")

# random seed
parser.add_argument("--random_seed", type=int, default=42)

# data loader
parser.add_argument("--dataset_S_class",
                    type=str,
                    default='DeepPCB_Crop',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )
parser.add_argument("--dataset_S_path",
                    type=str,
                    default="My_TAOD/dataset/DeepPCB_Crop/30-shot/train/0.csv")

parser.add_argument("--dataset_T_class",
                    type=str,
                    default='PCB_Crop',
                    choices=['PCB_Crop', 'PCB_200', 'DeepPCB_Crop'],
                    )
parser.add_argument("--dataset_T_path",
                    type=str,
                    default="My_TAOD/dataset/PCB_Crop/30-shot/train/0.csv")

parser.add_argument("--category",
                    type=str,
                    default="0")

parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument('--batch_size', type=int, default=10, help='size of the batches')

# train
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument("--gpu_id", type=str, default="0")


parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')


# Others
parser.add_argument("--log", type=bool, default=True)
parser.add_argument("--time", type=str, default=time.strftime(f"%Y-%m-%d_%H-%M-%S", time.localtime()))

# config
config = parser.parse_args()

# 一致性检验
assert(config.dataset_S_path.split('/')[-4]==config.dataset_S_class
       and
       config.dataset_S_path.split('/')[-1][0]==config.category
       and
       config.dataset_T_path.split('/')[-4]==config.dataset_T_class
       and
       config.dataset_T_path.split('/')[-1][0]==config.category
       )

# logger
cyclegan_logger(config)

models_save_path = config.models_dir + '/' + config.dataset_S_class + '_2_' + config.dataset_T_class + ' ' + config.category + ' ' + config.time
os.makedirs(models_save_path, exist_ok=False)

######################################################################################################
#### Setting
######################################################################################################
# random seed
random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

# device
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

# data_loader
transforms_ = T.Compose([
    T.ToTensor(),
    T.Resize(int(config.size*1.12), Image.BICUBIC),
    T.RandomCrop(config.size), 
    T.RandomHorizontalFlip(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

data_iter_loader = get_loader_ST(config.dataset_S_path, config.dataset_T_path, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, trans=transforms_, unaligned=True, drop_last=True)

# model
netG_A2B = CycleGAN_Generator(config.input_nc, config.output_nc).cuda()
netG_B2A = CycleGAN_Generator(config.output_nc, config.input_nc).cuda()
netD_A = CycleGAN_Discriminator(config.input_nc).cuda()
netD_B = CycleGAN_Discriminator(config.output_nc).cuda()
      
netG_A2B.apply(CycleGAN_weights_init_normal)
netG_B2A.apply(CycleGAN_weights_init_normal)
netD_A.apply(CycleGAN_weights_init_normal)
netD_B.apply(CycleGAN_weights_init_normal)

# loss
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# optim & LR schedulers
optim_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=config.lr, betas=(0.5, 0.999))
optim_D_A = torch.optim.Adam(netD_A.parameters(), lr=config.lr, betas=(0.5, 0.999))
optim_D_B = torch.optim.Adam(netD_B.parameters(), lr=config.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=CycleGAN_LambdaLR(config.num_epochs, config.start_epoch, config.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optim_D_A, lr_lambda=CycleGAN_LambdaLR(config.num_epochs, config.start_epoch, config.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optim_D_B, lr_lambda=CycleGAN_LambdaLR(config.num_epochs, config.start_epoch, config.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(config.batch_size, config.input_nc, config.size, config.size)
input_B = Tensor(config.batch_size, config.output_nc, config.size, config.size)
target_real = Variable(Tensor(config.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(config.batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = CycleGAN_ReplayBuffer()
fake_B_buffer = CycleGAN_ReplayBuffer()

for epoch in tqdm.trange(config.start_epoch+1, config.num_epochs+1, desc="On Training"):
    loss_G_list = []
    loss_G_identity_list = []
    loss_G_GAN_list = []
    loss_G_cycle_list = []
    loss_D_list = []
    
    for i, batch in enumerate(data_iter_loader):
        real_Src = Variable(input_A.copy_(batch['Src']))
        real_Tar = Variable(input_B.copy_(batch['Tar']))
        
        ###### Generators A2B and B2A ######
        optim_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_Tar)
        loss_identity_B = criterion_identity(same_B, real_Tar)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_Src)
        loss_identity_A = criterion_identity(same_A, real_Src)*5.0

        # GAN loss
        fake_B = netG_A2B(real_Src)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_Tar)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_Src)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_Tar)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optim_G.step()
        ###################################
        
        ###### Discriminator A ######
        optim_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_Src)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optim_D_A.step()
        ###################################
        
        ###### Discriminator B ######
        optim_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_Tar)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optim_D_B.step()
        ###################################
        
        loss_G_list.append(loss_G.item())
        loss_G_identity_list.append(loss_identity_A.item() + loss_identity_B.item())
        loss_G_GAN_list.append(loss_GAN_A2B.item() + loss_GAN_B2A.item())
        loss_G_cycle_list.append(loss_cycle_ABA.item() + loss_cycle_BAB.item())
        loss_D_list.append(loss_D_A.item() + loss_D_B.item())
        
    show_real_Src = img_1to255(real_Src[0].cpu().numpy()).transpose(1, 2, 0)
    show_real_Tar = img_1to255(real_Tar[0].cpu().numpy()).transpose(1, 2, 0)
    show_fake_A = img_1to255(fake_A[0].cpu().numpy()).transpose(1, 2, 0)
    show_fake_B = img_1to255(fake_B[0].cpu().numpy()).transpose(1, 2, 0)
    plt.imshow(show_real_Src)
    plt.savefig("cyclegan_real_Src.jpg")
    plt.close()
    plt.imshow(show_real_Tar)
    plt.savefig("cyclegan_real_Tar.jpg")
    plt.close()
    plt.imshow(show_fake_A)
    plt.savefig("cyclegan_fake_A.jpg")
    plt.close()
    plt.imshow(show_fake_B)
    plt.savefig("cyclegan_fake_B.jpg")
    plt.close()
        
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    cyclegan_record_data(config,
                         {
                             "epoch": f"{epoch}",
                             "num_epochs": f"{config.num_epochs}",
                             "loss_G": f"{np.mean(loss_G_list)}",
                             "loss_G_identity": f"{np.mean(loss_G_identity_list)}",
                             "loss_G_GAN": f"{np.mean(loss_G_GAN_list)}",
                             "loss_G_cycle": f"{np.mean(loss_G_cycle_list)}",
                             "loss_D": f"{np.mean(loss_D_list)}",
                                 
                         },
                         flag_plot = True,
                         )
    
    ##################################################################################
    ## Checkpoint
    ##################################################################################
    #-------------------------------------------------------------------
    # Save model G
    #-------------------------------------------------------------------
    if epoch % (config.num_epochs // 10) == 0:
        net_G_A2B_path = models_save_path + "/%d_net_G_A2B.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": netG_A2B.state_dict(),
        }, net_G_A2B_path)
        
        net_G_B2A_path = models_save_path + "/%d_net_G_B2A.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": netG_B2A.state_dict(),
        }, net_G_B2A_path)
        
    #-------------------------------------------------------------------
    # Save model D
    #-------------------------------------------------------------------
    if epoch == config.num_epochs:
        net_D_A_path = models_save_path + "/%d_net_D_A.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": netD_A.state_dict(),
            "img_size": config.size,
        }, net_D_A_path)
        
        net_D_B_path = models_save_path + "/%d_net_D_B.pth"%epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": netD_B.state_dict(),
            "img_size": config.size,
        }, net_D_B_path)
        
        
        
    